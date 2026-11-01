"""
score.py - the BDN-04 explainer soundtrack, synthesised from the film's own cue sheet.

Layers (all generated here with NumPy/SciPy, no samples):
  drone    A1/A2 pedal (55 + 110 Hz), detuned band-limited saws through a slowly breathing low-pass
  pads     i-VI-III-VII in A minor (Am9, Fmaj9, Cadd9, G6), eight beats each at 60 BPM, over the A pedal
  shimmer  high partials with slow random swells, mostly reverb
  pulse    muted eighth-note bass in the high-energy chapters (arithmetic, power)
  clock    a dry mechanical 1 Hz tick wherever the film is about time (chapters 03 and 08)
  flam     the latency cue: a tick on 1,000 ms and a second click exactly 52.2 ms later
  hits     sub impacts where the raw stream breaches the conduit; whooshes on the grid collapse

The intensity of each layer follows a per-chapter energy curve; the latency chapter drops almost to
silence so the flam is heard against the clock alone.

Usage:  python score.py <scene_dir> <cue_dir> <out.wav>
        (scene_dir holds the rendered chapter MP4s; durations come from the files themselves)
Then normalise to -14 LUFS / -1 dBTP with ffmpeg loudnorm (build_film.py does both).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import wave

import numpy as np
from scipy.signal import butter, lfilter, oaconvolve, sosfilt

SR = 48_000
ORDER = ["C00_Title", "C01_Idea", "C02_Constraints", "C03_Bandwidth", "C04_Noise", "C04b_Chain",
         "C05_Decision", "C06_Arithmetic", "C07_Power", "C08_Latency", "C09_Dashboard", "C10_Built",
         "C11_End"]
ENERGY = {"C00_Title": 0.25, "C01_Idea": 0.35, "C02_Constraints": 0.45, "C03_Bandwidth": 0.5,
          "C04_Noise": 0.58, "C04b_Chain": 0.6, "C05_Decision": 0.68, "C06_Arithmetic": 0.82,
          "C07_Power": 0.88, "C08_Latency": 0.18, "C09_Dashboard": 0.62, "C10_Built": 0.78,
          "C11_End": 0.42}
CHORDS = [[45, 52, 57, 60, 64, 71],      # Am9
          [41, 48, 57, 60, 64, 67],      # Fmaj9 (A pedal underneath)
          [48, 55, 60, 64, 67, 74],      # Cadd9
          [43, 50, 55, 59, 64, 66]]      # G6 add #4 colour
BEATS_PER_CHORD = 8
RNG = np.random.default_rng(2026)


def hz(m):
    return 440.0 * 2 ** ((m - 69) / 12)


def duration(path):
    return float(subprocess.check_output(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                                          "-of", "csv=p=0", path]))


# ---------------------------------------------------------------------------
# Timeline
# ---------------------------------------------------------------------------
def timeline(scene_dir, cue_dir):
    t, chapters, cues = 0.0, [], []
    for name in ORDER:
        d = duration(os.path.join(scene_dir, f"{name}.mp4"))
        meta = json.load(open(os.path.join(cue_dir, f"{name}.json")))
        assert abs(meta["duration"] - d) < 0.1, (name, meta["duration"], d)
        chapters.append((name, t, d))
        for c in meta["cues"]:
            c = dict(c)
            for k in ("t", "anchor", "start", "end"):
                if k in c:
                    c[k] += t
            c["scene"] = name
            cues.append(c)
        t += d
    return chapters, cues, t


# ---------------------------------------------------------------------------
# DSP helpers
# ---------------------------------------------------------------------------
def saw(freq, n, phase=0.0):
    """PolyBLEP band-limited sawtooth."""
    dt = freq / SR
    ph = (phase + dt * np.arange(n)) % 1.0
    y = 2 * ph - 1
    m = ph < dt
    x = ph[m] / dt
    y[m] -= x + x - x * x - 1
    m = ph > 1 - dt
    x = (ph[m] - 1) / dt
    y[m] -= x * x + x + x + 1
    return y


def varying_lowpass(x, cutoff, q=0.6, block=512):
    """Biquad low-pass whose cutoff follows a per-sample curve (updated per block)."""
    y = np.empty_like(x)
    zi = np.zeros(2)
    for i in range(0, len(x), block):
        fc = float(np.clip(cutoff[min(i + block // 2, len(x) - 1)], 30, SR * 0.45))
        w = 2 * np.pi * fc / SR
        alpha = np.sin(w) / (2 * q)
        cw = np.cos(w)
        b = np.array([(1 - cw) / 2, 1 - cw, (1 - cw) / 2]) / (1 + alpha)
        a = np.array([1, -2 * cw / (1 + alpha), (1 - alpha) / (1 + alpha)])
        y[i:i + block], zi = lfilter(b, a, x[i:i + block], zi=zi)
    return y


def reverb_ir(seconds=3.4, seed=7):
    n = int(seconds * SR)
    rng = np.random.default_rng(seed)
    t = np.arange(n) / SR
    env = np.exp(-6.9 * t / seconds)
    ir = rng.standard_normal((n, 2)) * env[:, None]
    sos = butter(1, 5500, "low", fs=SR, output="sos")
    ir = sosfilt(sos, ir, axis=0)
    ir[: int(0.018 * SR)] = 0
    return ir / np.sqrt(np.sum(ir ** 2, axis=0))


def reverb(st, ir):
    return np.stack([oaconvolve(st[:, c], ir[:, c])[: len(st)] for c in range(2)], axis=1)


def pan(mono, p):
    """Equal-power pan, p in [-1, 1]."""
    a = (p + 1) * np.pi / 4
    return np.stack([mono * np.cos(a), mono * np.sin(a)], axis=1)


def tick(kind="tick"):
    n = int(0.06 * SR)
    t = np.arange(n) / SR
    noise = RNG.standard_normal(n)
    sos = butter(2, [2200, 7500], "band", fs=SR, output="sos")
    click = sosfilt(sos, noise) * np.exp(-t / 0.0016)
    f1, f2 = (1650, 3300) if kind == "tick" else (2350, 4700)
    body = 0.55 * np.sin(2 * np.pi * f1 * t) * np.exp(-t / 0.009) + 0.25 * np.sin(2 * np.pi * f2 * t) * np.exp(-t / 0.004)
    y = click / (np.max(np.abs(click)) + 1e-9) + body
    return y / np.max(np.abs(y))


def boom(strength=1.0):
    n = int(1.6 * SR)
    t = np.arange(n) / SR
    f = 38 + 34 * np.exp(-t / 0.18)
    ph = 2 * np.pi * np.cumsum(f) / SR
    sub = np.sin(ph) * np.exp(-t / 0.55)
    sos = butter(2, 900, "low", fs=SR, output="sos")
    thud = sosfilt(sos, RNG.standard_normal(n)) * np.exp(-t / 0.05) * 0.6
    y = sub + thud
    return strength * y / np.max(np.abs(y))


def whoosh(dur, rising=True):
    n = int(dur * SR)
    t = np.arange(n) / SR
    x = RNG.standard_normal(n)
    cut = 300 + 5000 * (t / dur if rising else 1 - t / dur) ** 2
    y = varying_lowpass(x, cut, q=1.4)
    env = np.sin(np.pi * np.clip(t / dur, 0, 1)) ** 2
    return y * env / (np.max(np.abs(y * env)) + 1e-9)


def blip():
    n = int(0.09 * SR)
    t = np.arange(n) / SR
    f = 1250 - 400 * t / t[-1]
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / 0.025)


def add(buf, sig, t0, gain=1.0, p=0.0):
    i = int(round(t0 * SR))
    if i >= len(buf) or i + len(sig) <= 0:
        return
    s = pan(sig * gain, p)
    j0, k0 = max(i, 0), max(0, -i)
    j1 = min(len(buf), i + len(s))
    buf[j0:j1] += s[k0:k0 + j1 - j0]


# ---------------------------------------------------------------------------
# The score
# ---------------------------------------------------------------------------
def energy_curve(chapters, cues, total):
    ctl = 100                                     # control rate, Hz
    n = int(total * ctl) + 1
    e = np.zeros(n)
    for name, t0, d in chapters:
        e[int(t0 * ctl):int((t0 + d) * ctl) + 1] = ENERGY[name]
    k = np.hanning(int(3.0 * ctl))
    e = np.convolve(np.pad(e, (len(k), len(k)), mode="edge"), k / k.sum(), mode="same")[len(k):-len(k)]
    # latency: hold near-silence until the flam, then swell back
    for c in cues:
        if c["kind"] == "flam":
            name, t0, d = next(ch for ch in chapters if ch[0] == c["scene"])
            a, f, b = int(t0 * ctl), int((c["t"] + c["gap"] + 0.05) * ctl), int((t0 + d) * ctl)
            e[a + 150:f] = np.minimum(e[a + 150:f], 0.12)
            ramp = np.clip((np.arange(b - f) / ctl) / 2.5, 0, 1)
            e[f:b] = 0.12 + (0.62 - 0.12) * np.sin(ramp * np.pi / 2) ** 2
    tt = np.arange(int(total * SR)) / SR
    return np.interp(tt, np.arange(n) / ctl, e)


def compose(chapters, cues, total):
    N = int(total * SR)
    t = np.arange(N) / SR
    E = energy_curve(chapters, cues, total)
    fade = np.clip(t / 2.5, 0, 1) * np.clip((total - t) / 4.0, 0, 1)

    # drone: A pedal
    drone = (saw(55.0, N) + saw(55.0 * 1.003, N, 0.3) + 0.6 * saw(110.0 * 0.998, N, 0.6)) / 2.6
    drone += 0.5 * np.sin(2 * np.pi * 55.0 * t)
    cut = 140 + 520 * E + 70 * np.sin(2 * np.pi * t / 23.0)
    drone = varying_lowpass(drone, cut, q=0.9) * (1 + 0.12 * np.sin(2 * np.pi * t / 11.0))
    drone_st = np.stack([drone * 0.98, drone], axis=1)

    # pads: chords on the 60 BPM grid, crossfaded
    pads = np.zeros((N, 2))
    seg = BEATS_PER_CHORD * 1.0
    xf = 2.5
    k = 0
    start = 0.0
    while start < total:
        chord = CHORDS[k % len(CHORDS)] if total - start > seg else CHORDS[0]
        a, b = max(0.0, start - xf / 2), min(total, start + seg + xf / 2)
        ia, ib = int(a * SR), int(b * SR)
        n = ib - ia
        u = (np.arange(n) / SR + a - (start - xf / 2)) / (seg + xf)
        env = np.clip(np.minimum(u, 1 - u) * (seg + xf) / xf, 0, 1)
        env = np.sin(env * np.pi / 2) ** 2
        for j, m in enumerate(chord):
            f = hz(m)
            v = (saw(f, n, RNG.random()) + saw(f * 1.004, n, RNG.random()) + saw(f * 0.996, n, RNG.random())) / 3
            g = 0.16 if j == 0 else 0.10 * (0.8 ** (j - 1))           # upper voices taper
            pads[ia:ib] += pan(v * env * g, (j / (len(chord) - 1) - 0.5) * 0.9)
        start += seg
        k += 1
    pcut = 380 + 2100 * E ** 1.5
    for c in range(2):
        pads[:, c] = varying_lowpass(pads[:, c], pcut, q=0.7)
    pads *= (0.15 + 0.85 * E)[:, None] * 0.9

    # shimmer
    shim = np.zeros((N, 2))
    for f, p in ((659.26, -0.6), (880.0, 0.5), (987.77, -0.2), (1318.5, 0.7), (1760.0, -0.8)):
        lfo = 0.5 + 0.5 * np.sin(2 * np.pi * t / RNG.uniform(7, 17) + RNG.uniform(0, 6.28))
        trem = 1 + 0.25 * np.sin(2 * np.pi * RNG.uniform(4.5, 6.5) * t)
        shim += pan(np.sin(2 * np.pi * f * t + RNG.uniform(0, 6.28)) * lfo ** 3 * trem * 0.08, p)
    air = sosfilt(butter(2, [3500, 9000], "band", fs=SR, output="sos"), RNG.standard_normal((N, 2)), axis=0)
    air *= (0.5 + 0.5 * np.sin(2 * np.pi * t / 13.0))[:, None] ** 2 * 0.05
    shim += air
    shim *= (0.25 + 0.75 * np.clip(E * 1.2, 0, 1))[:, None]
    for g in np.arange(3.0, total - 3.0, 4.7):                      # sparse glints
        add(shim, np.sin(2 * np.pi * hz(RNG.choice([76, 81, 83, 88])) * np.arange(int(0.6 * SR)) / SR)
            * np.exp(-np.arange(int(0.6 * SR)) / SR / 0.12), g + RNG.uniform(-0.8, 0.8), 0.05, RNG.uniform(-0.8, 0.8))

    # pulse: eighth notes on the root, only where energy is high
    pulse = np.zeros((N, 2))
    pl = int(0.22 * SR)
    pt = np.arange(pl) / SR
    for i, beat in enumerate(np.arange(0, total, 0.5)):
        lvl = np.interp(beat, t[::SR // 100], E[::SR // 100])
        g = np.clip((lvl - 0.6) / 0.3, 0, 1)
        if g <= 0:
            continue
        root = CHORDS[int(beat // seg) % len(CHORDS)][0] - 12
        note = saw(hz(root), pl) * np.exp(-pt / 0.07)
        note = lfilter(*butter(2, 420, "low", fs=SR), note)
        add(pulse, note, beat, 0.32 * g * (1.0 if i % 2 == 0 else 0.7))

    # clock, flam, hits
    fx_dry = np.zeros((N, 2))
    fx_wet = np.zeros((N, 2))
    tk = tick("tick")
    for c in cues:
        if c["kind"] == "ticks":
            k0 = int(np.ceil((c["start"] - c["anchor"]) - 1e-6))
            k1 = int(np.floor((c["end"] - c["anchor"]) + 1e-6))
            for kk in range(k0, k1 + 1):
                add(fx_dry, tk, c["anchor"] + kk, 0.45, 0.15)
        elif c["kind"] == "flam":
            add(fx_dry, tick("click"), c["t"] + c["gap"], 0.5, -0.15)   # the tick at t is on the 1 Hz grid
        elif c["kind"] == "hit":
            b = boom(c.get("strength", 1.0))
            add(fx_dry, b, c["t"], 0.75)
            add(fx_wet, b, c["t"], 0.3)
        elif c["kind"] == "swell":
            w = whoosh(c["dur"], True)
            add(fx_wet, w, c["t"], 0.10)
        elif c["kind"] == "collapse":
            w = whoosh(c["dur"], False)
            add(fx_wet, w, c["t"], 0.12)
            add(fx_dry, boom(0.45), c["t"] + c["dur"] * 0.92, 0.6)
        elif c["kind"] == "shift":
            add(fx_dry, blip(), c["t"] + 0.05, 0.10, 0.2)

    ir = reverb_ir()
    wet_bus = 0.40 * pads + 0.85 * shim + 0.12 * drone_st
    music = drone_st * 1.0 + pads * 0.45 + shim * 0.25 + pulse + 0.55 * reverb(wet_bus, ir)
    music *= (np.clip(E, 0.05, 1) ** 0.8)[:, None]           # the whole bed breathes with the story
    mix = (music + fx_dry + 0.3 * reverb(fx_wet, ir)) * fade[:, None]
    hp = butter(2, 28, "high", fs=SR, output="sos")
    mix = sosfilt(hp, mix, axis=0)
    return mix / np.max(np.abs(mix)) * 0.5


def write_wav(path, st):
    pcm = (np.clip(st, -1, 1) * 32767).astype(np.int16)
    with wave.open(path, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())


if __name__ == "__main__":
    scene_dir, cue_dir, out = sys.argv[1:4]
    chapters, cues, total = timeline(scene_dir, cue_dir)
    write_wav(out, compose(chapters, cues, total))
    json.dump({"total": total, "chapters": chapters,
               "flam": [c for c in cues if c["kind"] == "flam"]}, open(out + ".json", "w"), indent=1)
    print(f"{total:.2f} s, {len(cues)} cues")
