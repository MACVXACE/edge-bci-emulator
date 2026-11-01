"""
bci_core.py - Phase 1: signal-processing core for an edge-compute MI-BCI node.

What this file does
-------------------
1. Generates a 1 s burst of single-channel motor-cortex EEG (C3) with:
     - ground-truth neural signal: 1/f background + mu (10 Hz) + beta (21 Hz),
       with an ERD (event-related desynchronisation) dip at motor-imagery onset
     - EMG from a jaw clench (20-100 Hz band-limited, burst envelope)
     - electrode baseline drift (offset + sub-Hz wander)
     - optional EOG blinks (off by default; the Phase 2 slider turns it on)
   Every stochastic term (EMG, 1/f background) is random *by physics*, drawn
   from a seeded generator. Nothing in this file returns a random *metric*.

2. Digitises it with a 9-bit ADC (the bandwidth spec), then removes drift with a
   multiplier-free DC blocker (shift + add only).

3. Removes EMG with adaptive noise cancellation (ANC): the primary channel d(n)
   = EEG + h*s(n); a reference electrode on the masseter supplies x(n) ~ s(n).
   The adaptive filter learns h, and the cleaned EEG is the error e(n).
   Three interchangeable filters, each with an instrumented operation counter:
     - LMS        (float64)  - honesty baseline: 2M+1 multiplies, slow convergence
     - RLS        (float64)  - textbook form, measured 4M^2 + 4M + 1
     - RLS-DCD    (int64 fixed point) - transversal RLS-DCD (Zakharov, White &
                  Liu, IEEE TSP 2008). The DCD solver uses only >>, +, compare.

How operations are counted
--------------------------
Counts are *measured*, not looked up: every arithmetic line in every filter goes
through a helper that does the operation and charges the counter using the
operands' actual shapes. verify_complexity() then checks the measured totals
against the closed-form expressions for a sweep of M. For the fixed-point
filter, the per-sample adds/shifts/compares depend on the data (how many DCD
step-size halvings occur), so we report mean and worst case.

Run:  python bci_core.py            (report + self-checks)
      python bci_core.py --plot     (also writes phase1_check.png)
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field

import numpy as np
from scipy import signal as sps


# =============================================================================
# 0. Node / link constants (the bandwidth constraint, Phase-2 UI will read these)
# =============================================================================
@dataclass(frozen=True)
class NodeConfig:
    fs: int = 256                 # samples/s per channel
    n_channels: int = 16
    adc_bits: int = 9
    adc_fsr_uv: float = 800.0     # ASSUMPTION: +/-400 uV full scale (confirm vs. your AFE spec)
    link_cap_bps: float = 10_000  # EQS-HBC node capacity
    link_power_w: float = 415e-9  # EQS-HBC node power at that rate

    @property
    def lsb_uv(self) -> float:
        return self.adc_fsr_uv / 2 ** self.adc_bits

    @property
    def raw_bitrate_bps(self) -> int:
        return self.n_channels * self.fs * self.adc_bits


def link_budget(cfg: NodeConfig = NodeConfig()) -> dict:
    raw = cfg.raw_bitrate_bps
    return {
        "raw_bps": raw,                                   # 16*256*9 = 36,864 b/s
        "cap_bps": cfg.link_cap_bps,
        "overload_ratio": raw / cfg.link_cap_bps,         # >1 means raw streaming is impossible
        "energy_per_bit_J": cfg.link_power_w / cfg.link_cap_bps,
        "max_channels_raw": int(cfg.link_cap_bps // (cfg.fs * cfg.adc_bits)),
    }


# =============================================================================
# 1. Operation counter
# =============================================================================
@dataclass
class OpCounter:
    mul: int = 0
    div: int = 0
    add: int = 0     # add/subtract/negate
    shift: int = 0   # arithmetic shift by a constant or small integer
    cmp: int = 0     # magnitude compare / sign test

    def snapshot(self) -> np.ndarray:
        return np.array([self.mul, self.div, self.add, self.shift, self.cmp], dtype=np.int64)

    FIELDS = ("mul", "div", "add", "shift", "cmp")


# ---- counted float kernels (each charges exactly what numpy executes) --------
def _dot(a, b, ops):
    ops.mul += a.size; ops.add += a.size - 1
    return float(a @ b)

def _matvec(A, x, ops):
    r, c = A.shape; ops.mul += r * c; ops.add += r * (c - 1)
    return A @ x

def _vecmat(x, A, ops):
    r, c = A.shape; ops.mul += r * c; ops.add += c * (r - 1)
    return x @ A

def _outer(a, b, ops):
    ops.mul += a.size * b.size
    return np.outer(a, b)

def _scale(a, s, ops):
    ops.mul += np.size(a)
    return a * s

def _add(a, b, ops):
    ops.add += np.size(a)
    return a + b

def _sub(a, b, ops):
    ops.add += np.size(a)
    return a - b


# =============================================================================
# 2. Signal generation
# =============================================================================
@dataclass
class BurstConfig:
    duration_s: float = 1.0
    # neural ground truth at C3
    bg_uv: float = 5.0            # 1/f background, rms
    mu_uv: float = 12.0           # mu amplitude (peak)
    mu_hz: float = 10.0
    beta_uv: float = 5.0
    beta_hz: float = 21.0
    erd_onset_s: float = 0.5      # motor-imagery onset -> mu/beta power drop
    erd_depth: float = 0.5        # fraction of amplitude lost during ERD
    # EMG (jaw clench)
    emg_uv: float = 40.0          # rms at C3 during the clench
    clench_on_s: float = 0.30
    clench_off_s: float = 0.75
    emg_tonic: float = 0.10       # resting muscle tone, relative to clench level
    emg_ref_gain: float = 1.5     # masseter reference amplitude / C3 EMG amplitude
    # drift
    drift_uv: float = 60.0
    drift_offset_uv: float = 25.0
    drift_hz: float = 0.15
    # EOG (off by default in Phase 1)
    blink_uv: float = 0.0         # blink peak as seen at C3
    blink_times_s: tuple = (0.6,)
    eog_ref_gain: float = 6.0     # frontal (EOG ref) / C3 blink ratio
    # front-end
    sensor_noise_uv: float = 1.0  # electrode + amplifier noise, rms


# Volume-conduction / electrode path from the masseter source to C3.
# Unknown to the filters; known here so we can measure weight misalignment.
H_EMG = np.array([0.0, 0.55, 0.30, -0.15, 0.08])


def _pink(n, fs, rng):
    f = np.fft.rfftfreq(n, 1 / fs)
    spec = rng.standard_normal(f.size) + 1j * rng.standard_normal(f.size)
    spec[0] = 0.0
    spec[1:] /= np.sqrt(f[1:])                # power ~ 1/f
    x = np.fft.irfft(spec, n)
    return x / x.std()


def _raised_cosine_gate(t, on, off, edge_s=0.04):
    g = np.zeros_like(t)
    g[(t >= on + edge_s) & (t <= off - edge_s)] = 1.0
    rise = (t >= on) & (t < on + edge_s)
    fall = (t > off - edge_s) & (t <= off)
    g[rise] = 0.5 - 0.5 * np.cos(np.pi * (t[rise] - on) / edge_s)
    g[fall] = 0.5 + 0.5 * np.cos(np.pi * (t[fall] - (off - edge_s)) / edge_s)
    return g


def _blink_template(t, t0, tau=0.06):
    """Gamma-shaped monophasic blink, peak = 1 at t0 + 2*tau (~120 ms rise)."""
    u = np.clip(t - t0, 0, None) / tau
    b = u ** 2 * np.exp(-u)
    return b / (4 * np.exp(-2))               # max of u^2 e^-u is 4e^-2 at u=2


def adc_quantize(uv, cfg: NodeConfig):
    lo, hi = -(1 << (cfg.adc_bits - 1)), (1 << (cfg.adc_bits - 1)) - 1
    code = np.round(uv / cfg.lsb_uv)
    n_clipped = int(((code < lo) | (code > hi)).sum())
    return np.clip(code, lo, hi).astype(np.int64), n_clipped


@dataclass
class Burst:
    t: np.ndarray
    truth_uv: np.ndarray              # neural signal only
    primary_uv: np.ndarray            # what the C3 electrode sees (analog)
    refs_uv: dict                     # reference electrodes (analog)
    components_uv: dict               # each additive term at C3, for plotting
    primary_code: np.ndarray          # 9-bit ADC codes
    ref_codes: dict
    clipped: dict
    clench_mask: np.ndarray
    w_opt_emg: np.ndarray             # true EMG path in reference units (unpadded)


def generate_burst(bc: BurstConfig = BurstConfig(), cfg: NodeConfig = NodeConfig(),
                   seed: int = 0) -> Burst:
    rng = np.random.default_rng(seed)
    fs = cfg.fs
    n = int(round(bc.duration_s * fs))
    t = np.arange(n) / fs

    # --- neural ground truth ---
    erd = 1.0 - bc.erd_depth * _raised_cosine_gate(t, bc.erd_onset_s, t[-1] + 1.0, edge_s=0.15)
    mu = bc.mu_uv * np.sin(2 * np.pi * bc.mu_hz * t + rng.uniform(0, 2 * np.pi))
    beta = bc.beta_uv * np.sin(2 * np.pi * bc.beta_hz * t + rng.uniform(0, 2 * np.pi))
    truth = bc.bg_uv * _pink(n, fs, rng) + erd * (mu + beta)

    # --- EMG source: white -> 4th-order Butterworth 20-100 Hz, with burn-in ---
    burn = 4 * fs
    sos = sps.butter(4, [20, 100], btype="bandpass", fs=fs, output="sos")
    s = sps.sosfilt(sos, rng.standard_normal(n + burn))[burn:]
    s /= s.std()
    clench = _raised_cosine_gate(t, bc.clench_on_s, bc.clench_off_s)
    s_env = s * (bc.emg_tonic + (1 - bc.emg_tonic) * clench)
    mask = clench > 0.99

    path = sps.lfilter(H_EMG, [1.0], s_env)
    c = bc.emg_uv / np.sqrt(np.mean(path[mask] ** 2))
    emg_c3 = c * path
    g = bc.emg_ref_gain * bc.emg_uv
    emg_ref = g * s_env
    w_opt = (c / g) * H_EMG if g > 0 else np.zeros_like(H_EMG)   # emg_c3 = w_opt * emg_ref

    # --- drift (independent at each electrode) ---
    def drift(scale):
        return scale * (bc.drift_offset_uv
                        + bc.drift_uv * np.sin(2 * np.pi * bc.drift_hz * t + rng.uniform(0, 2 * np.pi)))
    drift_c3 = drift(1.0)

    # --- EOG ---
    blink = sum((_blink_template(t, tb) for tb in bc.blink_times_s), np.zeros(n))
    eog_c3 = bc.blink_uv * blink

    noise = lambda: bc.sensor_noise_uv * rng.standard_normal(n)
    primary = truth + emg_c3 + drift_c3 + eog_c3 + noise()
    refs = {
        "emg": emg_ref + drift(0.5) + noise(),
        "eog": bc.eog_ref_gain * bc.blink_uv * blink + drift(0.5) + noise(),
    }

    p_code, p_clip = adc_quantize(primary, cfg)
    r_codes, clipped = {}, {"primary": p_clip}
    for k, v in refs.items():
        r_codes[k], clipped[k] = adc_quantize(v, cfg)

    return Burst(t, truth, primary, refs,
                 {"eeg": truth, "emg": emg_c3, "drift": drift_c3, "eog": eog_c3},
                 p_code, r_codes, clipped, mask, w_opt)


# =============================================================================
# 3. Drift removal: multiplier-free DC blocker (shift + add)
#    y[n] = x[n] - x[n-1] + a*y[n-1],  a = 1 - 2^-k  ->  a*y = y - (y >> k)
#    Two cascaded sections, k=5 at 256 S/s: fc ~ (2^-5)*256/(2*pi) ~ 1.3 Hz each.
#    Why two: a single section at 0.32 Hz leaves 16 uV rms of a 0.15 Hz drift in a
#    1 s window - more than the EEG itself. The cascade gives |H| = 0.014 at 0.15 Hz
#    and 1.006 at 8 Hz (mu/beta band 8-30 Hz is untouched). Cost: 0 multiplies.
# =============================================================================
DC_K, DC_STAGES = 5, 2


def dc_block_int(code: np.ndarray, k: int = DC_K, stages: int = DC_STAGES, acc_frac: int = 8,
                 out_frac: int = 4, ops: OpCounter | None = None) -> np.ndarray:
    """Integer DC blocker. Input: ADC codes. Output: Q(out_frac) LSB units.
    Internal state keeps acc_frac fractional bits so the leak (acc >> k) does not
    truncate to zero. Initial state x[-1] = x[0] -> no start-up transient."""
    ops = ops if ops is not None else OpCounter()
    v = code.astype(np.int64) << acc_frac
    ops.shift += code.size
    for _ in range(stages):
        out = np.empty_like(v)
        x1 = int(v[0]); acc = 0
        for i, x in enumerate(v.tolist()):
            acc = (x - x1) + acc - (acc >> k)                # 3 add, 1 shift
            x1 = x
            out[i] = acc
        ops.add += 3 * v.size; ops.shift += v.size
        v = out
    rnd = 1 << (acc_frac - out_frac - 1)
    ops.add += v.size; ops.shift += v.size
    return (v + rnd) >> (acc_frac - out_frac)


def dc_block_float(x: np.ndarray, k: int = DC_K, stages: int = DC_STAGES) -> np.ndarray:
    """Same difference equation in float, used only for the ground-truth reference."""
    a = 1.0 - 2.0 ** -k
    y = np.asarray(x, dtype=float)
    for _ in range(stages):
        y = sps.lfilter([1.0, -1.0], [1.0, -a], y - y[0])
    return y


# =============================================================================
# 4. Adaptive filters
#    Convention: x(n) = [x_n, x_{n-1}, ..., x_{n-M+1}] (tapped delay line, pre-windowed),
#    y(n) = w^T x(n), e(n) = d(n) - y(n). e(n) is the cleaned EEG.
#    The delay-line shift is a memory operation (circular buffer pointer) - not counted.
# =============================================================================
class LMS:
    """w <- w + mu e x.  Measured: 2M + 1 multiplies/sample."""
    name = "LMS"

    def __init__(self, M: int, mu: float = 2.0 ** -14):
        self.M, self.mu = M, mu
        self.w = np.zeros(M); self.x = np.zeros(M); self.ops = OpCounter()

    @staticmethod
    def formula_mul(M):  # multiplies + divides per sample
        return 2 * M + 1

    def weights(self):
        return self.w.copy()

    def step(self, d: float, x_n: float, adapt: bool = True) -> float:
        ops = self.ops
        self.x = np.concatenate(([x_n], self.x[:-1]))
        y = _dot(self.w, self.x, ops)                         # M
        e = d - y; ops.add += 1
        if adapt:
            g = self.mu * e; ops.mul += 1                     # 1
            self.w = _add(self.w, _scale(self.x, g, ops), ops)  # M
        return e


class RLS:
    """Exponentially weighted RLS, textbook (Haykin) form without exploiting
    the symmetry of P - i.e. x^T P is computed separately from P x.

        pi    = P x                    M^2
        xP    = x^T P                  M^2
        gamma = lam + x^T pi           M
        inv   = 1 / gamma              1 division
        k     = pi * inv               M
        y     = w^T x                  M
        e     = d - y
        w     = w + k e                M
        P     = (P - k xP) * (1/lam)   M^2 (outer) + M^2 (scale)
        ----------------------------------------------------
        total = 4M^2 + 4M multiplies + 1 division = 4M^2 + 4M + 1

    Note: reusing pi^T for x^T P (P is symmetric) gives 3M^2 + 4M + 1, so the
    4M^2 figure is the naive baseline. Kept in float64 because the P recursion
    is numerically fragile in fixed point.
    """
    name = "RLS"

    def __init__(self, M: int, lam: float = 1 - 2.0 ** -7, reg: float = 1.0):
        self.M, self.lam, self.inv_lam = M, lam, 1.0 / lam   # 1/lam is a design-time constant
        self.P = np.eye(M) / reg
        self.w = np.zeros(M); self.x = np.zeros(M); self.ops = OpCounter()

    @staticmethod
    def formula_mul(M):
        return 4 * M * M + 4 * M + 1

    def weights(self):
        return self.w.copy()

    def step(self, d: float, x_n: float, adapt: bool = True) -> float:
        """adapt=False (gate closed): output only, w and P frozen -> M multiplies."""
        ops, P = self.ops, self.P
        self.x = x = np.concatenate(([x_n], self.x[:-1]))
        y = _dot(self.w, x, ops)
        e = d - y; ops.add += 1
        if not adapt:
            return e
        pi = _matvec(P, x, ops)
        xP = _vecmat(x, P, ops)
        gamma = self.lam + _dot(x, pi, ops); ops.add += 1
        inv = 1.0 / gamma; ops.div += 1
        k = _scale(pi, inv, ops)
        self.w = _add(self.w, _scale(k, e, ops), ops)
        self.P = _scale(_sub(P, _outer(k, xP, ops), ops), self.inv_lam, ops)
        return e


class RLSDCD:
    """Transversal exponentially weighted RLS-DCD, fixed point (int64).

    Per sample (Zakharov, White & Liu 2008, leading-DCD variant):
      1. R(n) = lam R(n-1) + x x^T.  For a tapped delay line,
         R(n)[i+1, j+1] = R(n-1)[i, j] exactly, so only the first column
         is computed:  R(n)[:,0] = lam R(n-1)[:,0] + x_n x(n)    -> M mul
         (the rest is an address offset in hardware; copied here, not counted)
      2. y = (W . x) >> F                                         -> M mul
      3. e = d - y
      4. p0 = lam r(n-1) + e x(n)                                 -> M mul
      5. Solve R dW = p0 by N_u leading-DCD iterations: step sizes are
         powers of two, so  alpha*R[:,p] = R[:,p] >> m.           -> 0 mul
         The DCD residual becomes r(n).
      6. W += dW (done inside the DCD, one add per update)

    lam = 1 - 2^-B.  lambda_mode:
      "shift"    : lam*v = v - (v >> B)       -> 3M multiplies/sample total
      "multiply" : lam*v = (v * L) >> Q       -> 5M multiplies/sample total
    The O(M^2) -> O(M) reduction comes from steps 1 and 5 together (shift
    structure + no matrix inverse); bit-shifts are what make step 5 multiplier-free.

    Number formats: x, d, e in Q(G) LSB units (G guard bits from the DC blocker);
    R, r in Q(2G) LSB^2; W in Q(F) dimensionless with |w| < H = 1 per update.
    """
    name = "RLS-DCD"

    def __init__(self, M: int, lam_shift: int = 7, frac_bits: int = 15, n_u: int = 4,
                 reg: int = 1 << 8, lambda_mode: str = "shift", lambda_q: int = 16):
        assert lambda_mode in ("shift", "multiply")
        self.M, self.B, self.F, self.Nu, self.Mb = M, lam_shift, frac_bits, n_u, frac_bits
        self.mode, self.Q = lambda_mode, lambda_q
        self.L = (1 << lambda_q) - (1 << (lambda_q - lam_shift))   # lam in Q(lambda_q), exact
        self.R = np.eye(M, dtype=np.int64) * reg
        self.r = np.zeros(M, dtype=np.int64)
        self.W = np.zeros(M, dtype=np.int64)
        self.x = np.zeros(M, dtype=np.int64)
        self.ops = OpCounter()
        self.dcd_updates = 0
        self.peak = {"R": 0, "r": 0, "acc": 0, "W": 0}

    def formula_mul(self, M=None):
        M = self.M if M is None else M
        return 3 * M if self.mode == "shift" else 5 * M

    @property
    def name_full(self):
        return f"RLS-DCD (lam {self.mode})"

    def weights(self):
        return self.W / float(1 << self.F)

    def _lam(self, v):
        ops = self.ops
        if self.mode == "shift":
            ops.shift += v.size; ops.add += v.size
            return v - (v >> self.B)
        ops.mul += v.size; ops.shift += v.size
        return (v * self.L) >> self.Q

    def _dcd(self, R, r):
        ops, M, F = self.ops, self.M, self.F
        r = r.copy()
        m = 1                                       # alpha = H/2 = 2^-1
        for _ in range(self.Nu):
            a = np.abs(r); ops.cmp += M             # sign tests
            p = int(np.argmax(a)); ops.cmp += M - 1
            rp = int(r[p]); arp = abs(rp)
            ops.cmp += 1
            if arp == 0:
                break
            while True:                             # find largest alpha that reduces cost
                ops.shift += 1; ops.cmp += 1
                if arp > (int(R[p, p]) >> (m + 1)):  # |r_p| > (alpha/2) R_pp
                    break
                m += 1
                ops.cmp += 1
                if m > self.Mb:
                    return r
            step = 1 << (F - m); ops.shift += 1
            col = R[:, p] >> m; ops.shift += M      # alpha * R[:,p]
            if rp > 0:
                self.W[p] += step; r -= col
            else:
                self.W[p] -= step; r += col
            ops.add += 1 + M
            self.dcd_updates += 1
        return r

    def step(self, d: int, x_n: int, adapt: bool = True) -> int:
        """adapt=False (gate closed): R keeps updating (its shift structure must
        stay continuous), output is computed with frozen W, the pending residual
        r is cleared, and steps 4-6 are skipped -> 2M multiplies, no DCD."""
        ops, M, F = self.ops, self.M, self.F
        R = self.R
        self.x = x = np.concatenate(([x_n], self.x[:-1]))
        # 1. first column of R(n); rest by shift structure
        col = self._lam(R[:, 0]) + x_n * x
        ops.mul += M; ops.add += M
        Rn = np.empty_like(R)
        Rn[1:, 1:] = R[:-1, :-1]
        Rn[:, 0] = col; Rn[0, :] = col
        # 2-3. output and error
        acc = int(self.W @ x); ops.mul += M; ops.add += M - 1
        y = (acc + (1 << (F - 1))) >> F; ops.add += 1; ops.shift += 1
        e = int(d) - y; ops.add += 1
        if not adapt:
            self.R = Rn
            self.r = np.zeros(M, dtype=np.int64)     # register clear, no arithmetic
            return e
        # 4. right-hand side
        p0 = self._lam(self.r) + e * x
        ops.mul += M; ops.add += M
        # 5-6. DCD solve, W update in place
        self.r = self._dcd(Rn, p0)
        self.R = Rn
        pk = self.peak
        pk["R"] = max(pk["R"], int(np.abs(Rn).max()))
        pk["r"] = max(pk["r"], int(np.abs(p0).max()))
        pk["acc"] = max(pk["acc"], abs(acc))
        pk["W"] = max(pk["W"], int(np.abs(self.W).max()))
        return e

    def datapath_bits(self) -> dict:
        """Signed register widths actually exercised in this run."""
        return {k: int(v).bit_length() + 1 for k, v in self.peak.items()}


# =============================================================================
# 5. Running the canceller and scoring it against ground truth
# =============================================================================
@dataclass
class ANCResult:
    name: str
    out_uv: np.ndarray
    w_hist: np.ndarray                    # (N, M) weights after each sample
    ops_per_sample: np.ndarray            # (N, 5) per-sample op deltas
    filt: object = field(repr=False)

    def ops_summary(self) -> dict:
        ps = self.ops_per_sample
        return {f: {"mean": float(ps[:, i].mean()), "max": int(ps[:, i].max())}
                for i, f in enumerate(OpCounter.FIELDS)}

    @property
    def mul_per_sample(self) -> int:
        """multiplies + divides per sample; constant when no gate is used."""
        mm = self.ops_per_sample[:, 0] + self.ops_per_sample[:, 1]
        assert mm.min() == mm.max(), "multiply count varies (a gate is active): use mul_adapting / mul_mean"
        return int(mm[0])


class RefGate:
    """Reference-power gate. Shift-add leaky mean-|x| estimator:
        A <- A - (A >> k) + |x|         (A ~ 2^k * mean|x|, time constant 2^k samples)
        adapt  <=>  A > theta * 2^k     (theta*2^k is a design-time constant)
    Cost per sample: 2 compares (sign test, threshold), 2 adds (+1 when negating), 1 shift.
    Works on ints (Q(G) units, true >>) or floats (exact power-of-two scaling)."""

    def __init__(self, theta: float, k: int = 4):
        self.k = k
        self.th = theta * (1 << k)
        self.A = 0

    def update(self, x, ops: OpCounter) -> bool:
        ops.cmp += 1
        if x < 0:
            x = -x; ops.add += 1
        if isinstance(self.A, (int, np.integer)) and isinstance(x, (int, np.integer)):
            self.A = int(self.A) - (int(self.A) >> self.k) + int(x)
        else:
            self.A = self.A - self.A * 2.0 ** -self.k + float(x)
        ops.shift += 1; ops.add += 2
        ops.cmp += 1
        return self.A > self.th


def run_anc(filt, d, x, to_uv: float, gate: RefGate | None = None) -> ANCResult:
    n = len(d)
    native = np.empty(n); w_hist = np.empty((n, filt.M)); per = np.empty((n, 5), dtype=np.int64)
    open_ = np.ones(n, dtype=bool)
    prev = filt.ops.snapshot()
    for i in range(n):
        if gate is not None:
            open_[i] = gate.update(x[i], filt.ops)
            native[i] = filt.step(d[i], x[i], adapt=bool(open_[i]))
        else:
            native[i] = filt.step(d[i], x[i])
        w_hist[i] = filt.weights()
        cur = filt.ops.snapshot(); per[i] = cur - prev; prev = cur
    name = getattr(filt, "name_full", filt.name)
    res = ANCResult(name, native * to_uv, w_hist, per, filt)
    res.out_native = native          # filter's own units, for chaining stages
    res.gate_open = open_
    return res


def snr_db(ref, est, mask=None):
    m = slice(None) if mask is None else mask
    return 10 * np.log10(np.sum(ref[m] ** 2) / np.sum((est[m] - ref[m]) ** 2))


def misalignment_db(w_hist, w_opt):
    M = w_hist.shape[1]
    wo = np.zeros(M); wo[:min(M, w_opt.size)] = w_opt[:M]
    return 10 * np.log10(np.sum((w_hist - wo) ** 2, axis=1) / np.sum(wo ** 2))


def clean_emg(burst: Burst, M: int = 8, cfg: NodeConfig = NodeConfig(), guard_bits: int = 4,
              dcd_kwargs: dict | None = None, lms_mu: float = 2.0 ** -14) -> dict:
    """Full Phase-1 pipeline on one channel: DC-block -> ANC (EMG reference)."""
    pre_ops = OpCounter()
    d_q = dc_block_int(burst.primary_code, out_frac=guard_bits, ops=pre_ops)
    x_q = dc_block_int(burst.ref_codes["emg"], out_frac=guard_bits, ops=pre_ops)
    scale = 1.0 / (1 << guard_bits)                        # Q(G) -> LSB
    d_f, x_f = d_q * scale, x_q * scale
    to_uv = cfg.lsb_uv

    results = {
        "LMS": run_anc(LMS(M, mu=lms_mu), d_f, x_f, to_uv),        # mu in 1/LSB^2
        "RLS": run_anc(RLS(M), d_f, x_f, to_uv),
        "RLS-DCD": run_anc(RLSDCD(M, **(dcd_kwargs or {})), d_q, x_q, to_uv * scale),
    }
    truth_ref = dc_block_float(burst.truth_uv)             # fair reference: same HP as the pipeline
    return {
        "truth_ref_uv": truth_ref,
        "input_uv": d_f * to_uv,
        "results": results,
        "pre_ops_per_sample_per_channel": {f: getattr(pre_ops, f) / (2 * len(d_q)) for f in OpCounter.FIELDS},
    }


# =============================================================================
# 5b. Multi-stage pipeline (Phase 2): DC block -> ANC(EMG ref) [-> ANC(EOG ref)]
# =============================================================================
FILTERS = {
    "LMS": lambda M, kw: LMS(M, mu=kw.get("lms_mu", 2.0 ** -14)),
    "RLS": lambda M, kw: RLS(M),
    "RLS-DCD": lambda M, kw: RLSDCD(M, **kw.get("dcd", {})),
}
INT_FILTERS = {"RLS-DCD"}


@dataclass
class PipelineResult:
    name: str
    out_uv: np.ndarray
    ops_per_sample: np.ndarray        # (N, 5) summed over all ANC stages, ONE channel
    w_hist_emg: np.ndarray
    stage_filters: list = field(repr=False)
    gate_open: dict = field(default_factory=dict)   # stage -> bool array

    def mean_ops(self) -> dict:
        return {f: float(self.ops_per_sample[:, i].mean()) for i, f in enumerate(OpCounter.FIELDS)}

    def max_ops(self) -> dict:
        return {f: int(self.ops_per_sample[:, i].max()) for i, f in enumerate(OpCounter.FIELDS)}

    @property
    def mul_adapting(self) -> int:
        """multiplies + divides on a sample where every stage adapts (the algorithmic cost)."""
        return int((self.ops_per_sample[:, 0] + self.ops_per_sample[:, 1]).max())

    @property
    def mul_mean(self) -> float:
        return float((self.ops_per_sample[:, 0] + self.ops_per_sample[:, 1]).mean())

    def mul_by_state(self) -> dict:
        """Measured multiplies on samples where every stage adapts / every stage is
        frozen (None if that state never occurred in this burst)."""
        mm = self.ops_per_sample[:, 0] + self.ops_per_sample[:, 1]
        g = np.vstack(list(self.gate_open.values())) if self.gate_open else np.ones((1, mm.size), bool)
        out = {}
        for k, m in (("adapting", g.all(axis=0)), ("frozen", ~g.any(axis=0))):
            out[k] = int(mm[m][0]) if m.any() else None
        out["open_frac"] = float(g.all(axis=0).mean())
        return out

    @property
    def total_ops_mean(self) -> float:
        return float(self.ops_per_sample.sum(axis=1).mean())


def clean(burst: Burst, M: int = 8, cfg: NodeConfig = NodeConfig(), use_eog: bool | None = None,
          filters=("LMS", "RLS", "RLS-DCD"), guard_bits: int = 4, filter_kwargs: dict | None = None,
          eog_first: bool = False, eog_taps: int = 1, gate_uv: float | None = 3.0,
          gate_k: int = 4) -> dict:
    """Run every requested filter through the same stage cascade:
        DC blocker -> EMG canceller (M taps) -> [EOG canceller, eog_taps]
    EOG reaches C3 by volume conduction, which is instantaneous at EEG frequencies,
    so its path is a single gain (as in classic EOG regression): eog_taps = 1.
    A longer EOG filter is ill-conditioned (a blink is a ~1-3 Hz signal, so its
    delayed copies are nearly collinear) and fits EEG with reference noise while
    the eye is still. Order (measured over 6 seeds, blink inside a clench): EMG
    first gives +4.0 dB settled-clench SNR vs -0.5 dB EOG-first. The EMG reference
    is band-limited to 20-100 Hz, so a 1-3 Hz blink barely perturbs the EMG fit;
    the reverse order leaves ~40 uV of EMG as noise for the single EOG gain.

    gate_uv: each stage adapts only while its reference's mean |x| (shift-add
    estimate, time constant 2^gate_k samples) exceeds gate_uv. With the reference
    at its noise floor, an adapting canceller fits the EEG with that noise and
    degrades the output; the gate freezes it instead. None disables the gate."""
    kw = filter_kwargs or {}
    if use_eog is None:
        use_eog = bool(np.any(burst.components_uv["eog"] != 0))
    refs = (["eog", "emg"] if eog_first else ["emg", "eog"]) if use_eog else ["emg"]

    dc_ops = OpCounter()
    d_q = dc_block_int(burst.primary_code, out_frac=guard_bits, ops=dc_ops)
    n = d_q.size
    dc_per_channel = {f: getattr(dc_ops, f) / n for f in OpCounter.FIELDS}
    x_q = {r: dc_block_int(burst.ref_codes[r], out_frac=guard_bits) for r in refs}
    scale = 1.0 / (1 << guard_bits)

    results = {}
    for name in filters:
        is_int = name in INT_FILTERS
        sig = d_q if is_int else d_q * scale
        to_uv = cfg.lsb_uv * (scale if is_int else 1.0)
        ops = np.zeros((n, 5), dtype=np.int64); stage_filters = []; w_emg = None; gates = {}
        for r in refs:
            f = FILTERS[name](eog_taps if r == "eog" else M, kw)
            x = x_q[r] if is_int else x_q[r] * scale
            gate = None
            if gate_uv is not None:
                theta = gate_uv / cfg.lsb_uv                      # LSB units
                gate = RefGate(int(round(theta * (1 << guard_bits))) if is_int else theta, gate_k)
            res = run_anc(f, sig, x, to_uv, gate)
            gates[r] = res.gate_open
            sig = res.out_native.astype(np.int64) if is_int else res.out_native
            ops += res.ops_per_sample; stage_filters.append(f)
            if r == "emg":
                w_emg = res.w_hist
        results[name] = PipelineResult(name, sig * to_uv, ops, w_emg, stage_filters, gates)

    return {
        "truth_ref_uv": dc_block_float(burst.truth_uv),   # truth through the same 1.3 Hz HP
        "input_uv": d_q * scale * cfg.lsb_uv,             # after ADC + DC blocker
        "raw_adc_uv": burst.primary_code * cfg.lsb_uv,    # what the ADC delivers
        "results": results,
        "dc_ops_per_sample_per_channel": dc_per_channel,
        "refs": refs,
    }


# =============================================================================
# 5c. Energy and bandwidth models (Phase 2)
# =============================================================================
@dataclass(frozen=True)
class EnergyModel:
    """90 nm, 1.0 V energy-per-operation approximations (user-specified).
    Caveat: real energy scales with operand width (RLS-DCD's R/acc registers are
    ~27 b) and float64 RLS would cost far more than 1.5 pJ per multiply, so
    charging RLS integer-multiply energy *flatters* the RLS baseline."""
    mul_pj: float = 1.5     # also charged for each division
    add_pj: float = 0.1
    cmp_pj: float = 0.1
    shift_pj: float = 0.1
    static_uw: float = 0.5  # leakage floor (assumption)
    csp_lda_uw: float = 0.4  # dynamic power of CSP log-var + LDA, per 1 s window (user-specified,
                             # from published low-power implementations; see csp_lda_opcount_uw)

    def pj(self, ops: dict) -> float:
        return ((ops["mul"] + ops["div"]) * self.mul_pj + ops["add"] * self.add_pj
                + ops["cmp"] * self.cmp_pj + ops["shift"] * self.shift_pj)

    def pj_array(self, per_sample: np.ndarray) -> np.ndarray:
        w = np.array([self.mul_pj, self.mul_pj, self.add_pj, self.shift_pj, self.cmp_pj])
        return per_sample @ w


def filtering_power(res: PipelineResult, dc_ops: dict, n_refs: int, em: EnergyModel = EnergyModel(),
                    cfg: NodeConfig = NodeConfig()) -> dict:
    """P = (pJ per sample-instant across the node) x fs.
    Node per sample-instant: n_channels x (DC blocker + ANC stages), plus one DC
    blocker per shared reference electrode. Ops measured on C3, applied per channel."""
    anc_pj = em.pj(res.mean_ops())
    anc_pj_max = float(em.pj_array(res.ops_per_sample).max())
    dc_pj = em.pj(dc_ops)
    node_pj = cfg.n_channels * (anc_pj + dc_pj) + n_refs * dc_pj
    node_pj_max = cfg.n_channels * (anc_pj_max + dc_pj) + n_refs * dc_pj
    dyn_uw = node_pj * cfg.fs * 1e-6           # pJ/s -> uW
    return {
        "anc_pj_per_channel": anc_pj,
        "dc_pj_per_channel": dc_pj,
        "node_pj_per_sample": node_pj,
        "dynamic_uw": dyn_uw,
        "dynamic_uw_worst": node_pj_max * cfg.fs * 1e-6,
        "static_uw": em.static_uw,
        "total_uw": dyn_uw + em.static_uw,
        "breakdown_pj": {   # per channel, ANC only
            "mul+div": (res.mean_ops()["mul"] + res.mean_ops()["div"]) * em.mul_pj,
            "add": res.mean_ops()["add"] * em.add_pj,
            "shift": res.mean_ops()["shift"] * em.shift_pj,
            "cmp": res.mean_ops()["cmp"] * em.cmp_pj,
        },
    }


def csp_lda_opcount_uw(em: EnergyModel = EnergyModel(), cfg: NodeConfig = NodeConfig(),
                       n_filters: int = 6, biquads: int = 2, window_s: float = 1.0) -> dict:
    """Cross-check for the CSP + LDA power penalty: arithmetic-only estimate with
    the same energy model, for a *streaming* implementation:
      per sample : project 16 ch -> n_filters (n_filters*16 MAC), band-pass each
                   CSP output (biquads x 5 mul, 4 add), square + accumulate
      per window : n_filters logs (not costed), LDA dot product + threshold
    Excludes memory access, control and clocking, which a real implementation pays."""
    C, K = cfg.n_channels, n_filters
    mul_s = K * C + K * biquads * 5 + K
    add_s = K * (C - 1) + K * biquads * 4 + K
    mul_w, add_w, cmp_w = K, K, 1
    pj_per_s = (mul_s * em.mul_pj + add_s * em.add_pj) * cfg.fs \
        + (mul_w * em.mul_pj + add_w * em.add_pj + cmp_w * em.cmp_pj) / window_s
    return {"mul_per_sample": mul_s, "add_per_sample": add_s, "dynamic_uw": pj_per_s * 1e-6}


# Frame layouts for one 8-bit class label. Field sizes in bits.
FRAME_PRESETS = {
    "802.15.6 MAC on the EQS PHY (112 b)": {
        "Preamble": 24, "PHY header": 8, "MAC header (7 octets)": 56, "Label": 8, "FCS (CRC-16)": 16},
    "Minimal custom MAC (72 b)": {
        "Preamble": 24, "PHY header": 8, "MAC header": 16, "Label": 8, "CRC-16": 16},
}


def bandwidth_chain(cfg: NodeConfig = NodeConfig(), frame: dict | None = None, window_s: float = 1.0) -> dict:
    """Bit rate at each pipeline stage. Cleaning does not compress: the filtered
    stream has the same rate as the raw one. On-node CSP log-variance + LDA emits
    one label per window, which must travel inside a frame."""
    frame = frame or next(iter(FRAME_PRESETS.values()))
    raw = cfg.raw_bitrate_bps
    frame_bits = sum(frame.values())
    label_bits = next(v for k, v in frame.items() if k.lower().startswith("label"))
    link_bps = frame_bits / window_s
    return {
        "raw_bps": raw, "filtered_bps": raw, "label_bps": label_bits / window_s, "link_bps": link_bps,
        "frame_bits": frame_bits, "label_bits": label_bits, "frame": frame,
        "cap_bps": cfg.link_cap_bps,
        "raw_util": raw / cfg.link_cap_bps, "link_util": link_bps / cfg.link_cap_bps,
        "tx_ms": frame_bits / cfg.link_cap_bps * 1e3,          # time on air at the link rate
        "link_pj_per_bit": cfg.link_power_w / cfg.link_cap_bps * 1e12,
        "link_power_cap_nw": cfg.link_power_w * 1e9,
        "link_power_proportional_nw": link_bps * cfg.link_power_w / cfg.link_cap_bps * 1e9,
    }


def node_power(filt_power: dict, em: EnergyModel, bw: dict) -> dict:
    """Whole-node budget: static + filtering dynamic + CSP/LDA + link.
    The link term is a range: bit-proportional energy (low) up to the radio's
    rated 415 nW at full capacity (high), since idle/wake-up overhead is unknown."""
    base = filt_power["total_uw"] + em.csp_lda_uw
    return {"static_uw": em.static_uw, "filtering_dyn_uw": filt_power["dynamic_uw"],
            "csp_lda_uw": em.csp_lda_uw,
            "link_low_uw": bw["link_power_proportional_nw"] * 1e-3, "link_high_uw": bw["link_power_cap_nw"] * 1e-3,
            "total_low_uw": base + bw["link_power_proportional_nw"] * 1e-3,
            "total_high_uw": base + bw["link_power_cap_nw"] * 1e-3}


def latency_budget(window_s: float, classify_ms: float, tx_ms: float, fes_ms: float) -> list[dict]:
    """Sequential stages from intent onset to muscle response (ms)."""
    stages = [("Biological intent window", window_s * 1e3, "biology"),
              ("CSP + LDA classification", classify_ms, "silicon"),
              ("EQS-HBC frame transmission", tx_ms, "silicon"),
              ("FES actuator delay", fes_ms, "actuation")]
    out, t = [], 0.0
    for name, dur, kind in stages:
        out.append({"stage": name, "start_ms": t, "end_ms": t + dur, "dur_ms": dur, "kind": kind})
        t += dur
    return out


# =============================================================================
# 6. Verification of the complexity claims
# =============================================================================
def verify_complexity(Ms=(2, 4, 8, 16, 32), n=64, seed=1) -> list[dict]:
    rng = np.random.default_rng(seed)
    d = rng.standard_normal(n) * 50; x = rng.standard_normal(n) * 50
    rows = []
    for M in Ms:
        row = {"M": M}
        for f, dd, xx in ((LMS(M), d, x), (RLS(M), d, x),
                          (RLSDCD(M, lambda_mode="shift"), (d * 16).astype(np.int64), (x * 16).astype(np.int64)),
                          (RLSDCD(M, lambda_mode="multiply"), (d * 16).astype(np.int64), (x * 16).astype(np.int64))):
            res = run_anc(f, dd, xx, 1.0)
            measured = res.mul_per_sample
            expected = f.formula_mul(M)
            assert measured == expected, f"{res.name} M={M}: measured {measured} != {expected}"
            row[res.name] = measured
        row["spec 4M^2+4M+1"] = 4 * M * M + 4 * M + 1
        rows.append(row)
    return rows


# =============================================================================
# 7. Report
# =============================================================================
def _report(M=8, seed=0, plot=False):
    cfg, bc = NodeConfig(), BurstConfig()
    lb = link_budget(cfg)
    print("=" * 78)
    print("LINK BUDGET")
    print(f"  raw EEG: {cfg.n_channels} ch x {cfg.fs} S/s x {cfg.adc_bits} b = {lb['raw_bps']:,} b/s")
    print(f"  EQS-HBC cap: {lb['cap_bps']:,.0f} b/s  ->  raw stream is {lb['overload_ratio']:.2f}x over capacity")
    print(f"  link energy: {lb['energy_per_bit_J']*1e12:.1f} pJ/bit;  raw streaming fits only "
          f"{lb['max_channels_raw']} channels")

    burst = generate_burst(bc, cfg, seed)
    out = clean_emg(burst, M=M, cfg=cfg)
    ref, mask = out["truth_ref_uv"], burst.clench_mask
    quiet = ~mask & (burst.t > 0.1)

    print("=" * 78)
    print(f"BURST  ({bc.duration_s:.1f} s, fs={cfg.fs}, LSB={cfg.lsb_uv:.4f} uV, seed={seed})")
    print(f"  ADC clipping: {burst.clipped}")
    print(f"  input SNR vs truth: clench {snr_db(ref, out['input_uv'], mask):6.2f} dB | "
          f"quiet {snr_db(ref, out['input_uv'], quiet):6.2f} dB")
    print(f"  DC blocker cost/sample/channel: {out['pre_ops_per_sample_per_channel']}")

    settled = mask & (burst.t >= bc.clench_on_s + 0.1)      # clench, after 100 ms to learn the path
    print("=" * 78)
    print(f"ANC RESULTS (M={M}, true EMG path length {H_EMG.size})")
    print("  SNR = 10log10(|truth|^2/|output-truth|^2); 'settled' = clench minus its first 100 ms")
    hdr = f"  {'filter':22s} {'mul/samp':>8s} {'add avg/max':>12s} {'shift avg/max':>14s} " \
          f"{'clench':>8s} {'settled':>8s} {'quiet':>7s} {'misalign@off':>12s}"
    print(hdr)
    off_idx = int(np.searchsorted(burst.t, bc.clench_off_s)) - 1
    for r in out["results"].values():
        s = r.ops_summary()
        mis = misalignment_db(r.w_hist, burst.w_opt_emg)[off_idx]
        print(f"  {r.name:22s} {r.mul_per_sample:8d} "
              f"{s['add']['mean']:6.0f}/{s['add']['max']:<5d} {s['shift']['mean']:7.0f}/{s['shift']['max']:<6d} "
              f"{snr_db(ref, r.out_uv, mask):6.2f}dB {snr_db(ref, r.out_uv, settled):6.2f}dB "
              f"{snr_db(ref, r.out_uv, quiet):5.2f}dB {mis:10.1f}dB")
    # bound: the true path applied to the same quantised, DC-blocked reference
    wo = np.zeros(M); wo[:min(M, H_EMG.size)] = burst.w_opt_emg[:M]
    x_uv = dc_block_int(burst.ref_codes["emg"]) / 16 * cfg.lsb_uv
    oracle = out["input_uv"] - sps.lfilter(wo, [1.0], x_uv)
    print(f"  {'[bound: w = true path]':22s} {'':8s} {'':12s} {'':14s} "
          f"{snr_db(ref, oracle, mask):6.2f}dB {snr_db(ref, oracle, settled):6.2f}dB "
          f"{snr_db(ref, oracle, quiet):5.2f}dB")
    dcd = out["results"]["RLS-DCD"].filt
    print(f"  RLS-DCD datapath widths used (signed bits): {dcd.datapath_bits()}")
    print(f"  RLS-DCD mean DCD updates/sample: {dcd.dcd_updates / len(burst.t):.2f} (N_u={dcd.Nu})")

    print("=" * 78)
    print("COMPLEXITY SWEEP (multiplies+divides per sample, measured; asserts == closed form)")
    rows = verify_complexity()
    keys = list(rows[0].keys())
    print("  " + " ".join(f"{k:>22s}" if i else f"{k:>4s}" for i, k in enumerate(keys)))
    for row in rows:
        print("  " + " ".join(f"{row[k]:>22d}" if i else f"{row[k]:>4d}" for i, k in enumerate(keys)))
    print("  all measured counts match their closed forms.")

    if plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        t = burst.t
        fig, ax = plt.subplots(3, 1, figsize=(11, 8.5), sharex=True)
        ax[0].plot(t, burst.primary_uv, lw=0.8, color="0.6", label="C3 electrode (analog, with drift)")
        ax[0].plot(t, out["input_uv"], lw=0.8, color="tab:red", label="after 9-bit ADC + DC blocker")
        ax[0].plot(t, ref, lw=1.2, color="k", label="ground-truth EEG (same HP)")
        ax[0].axvspan(bc.clench_on_s, bc.clench_off_s, color="tab:orange", alpha=0.1, label="jaw clench")
        ax[0].set_ylabel("uV"); ax[0].legend(loc="upper right", fontsize=8)
        ax[1].plot(t, ref, lw=1.4, color="k", label="ground truth")
        for r, c in zip(out["results"].values(), ("tab:green", "tab:blue", "tab:purple")):
            ax[1].plot(t, r.out_uv, lw=0.9, color=c, label=f"{r.name} output")
        ax[1].axvspan(bc.clench_on_s, bc.clench_off_s, color="tab:orange", alpha=0.1)
        ax[1].set_ylabel("uV"); ax[1].legend(loc="upper right", fontsize=8)
        for r, c in zip(out["results"].values(), ("tab:green", "tab:blue", "tab:purple")):
            ax[2].plot(t, misalignment_db(r.w_hist, burst.w_opt_emg), color=c, label=r.name)
        ax[2].axvspan(bc.clench_on_s, bc.clench_off_s, color="tab:orange", alpha=0.1)
        ax[2].set_ylabel("weight misalignment (dB)"); ax[2].set_xlabel("time (s)")
        ax[2].legend(loc="upper right", fontsize=8)
        fig.suptitle(f"Phase 1 check: ANC of jaw-clench EMG at C3, M={M}")
        fig.tight_layout(); fig.savefig("phase1_check.png", dpi=130)
        print("wrote phase1_check.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--M", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--plot", action="store_true")
    a = ap.parse_args()
    _report(a.M, a.seed, a.plot)
