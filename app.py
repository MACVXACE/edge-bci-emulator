"""
app.py - telemetry dashboard for the edge-compute MI-BCI node (Phases 2-3).

Every number on this page comes from bci_core.py:
  - operation counts are measured by the instrumented filters on the simulated burst
  - power = measured ops x 90 nm energy-per-op x 256 S/s x 16 channels (+ static,
    + CSP/LDA penalty, + link)
  - SNR is computed against the simulation's known ground-truth EEG
  - bandwidth and transmission time are arithmetic on the frame layout and link rate

Run locally:  streamlit run app.py
"""
import json

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from bci_core import (FRAME_PRESETS, H_EMG, RLS, RLSDCD, BurstConfig, EnergyModel, NodeConfig,
                      bandwidth_chain, clean, csp_lda_opcount_uw, filtering_power, generate_burst,
                      latency_budget, node_power, run_anc, snr_db)

CFG = NodeConfig()
REF_FLOOR_UW = 1.41          # Chen et al. 2010, 90 nm, 1.0 V (project reference)

# Palette (dataviz reference instance, light mode; pairs validated)
C_DCD, C_RLS = "#2a78d6", "#eb6834"                 # filter identities
C_SILICON, C_ACT = "#1baf7a", "#6250d6"             # latency stage kinds (always direct-labelled)
C_TRUTH, C_RAW, C_GRID, C_MUTED = "#0b0b0b", "#9a9890", "#e1e0d9", "#52514e"
C_CRIT = "#d03b3b"           # status: critical (always paired with a text label)
KIND_COLOR = {"biology": C_RAW, "silicon": C_SILICON, "actuation": C_ACT}

st.set_page_config(page_title="Edge BCI node: telemetry", layout="wide")


# ---------------------------------------------------------------------------
# Engine calls (cached: sliders re-run only what changed)
# ---------------------------------------------------------------------------
@st.cache_data(max_entries=64, show_spinner="Running the filters on the simulated burst...")
def simulate(emg_uv, drift_uv, blink_uv, M, n_u, gate_uv, seed):
    bc = BurstConfig(emg_uv=emg_uv, drift_uv=drift_uv, drift_offset_uv=0.4 * drift_uv,
                     blink_uv=blink_uv)
    burst = generate_burst(bc, CFG, seed)
    out = clean(burst, M=M, cfg=CFG, filters=("RLS", "RLS-DCD"),
                filter_kwargs={"dcd": {"n_u": n_u}}, gate_uv=gate_uv)
    return bc, burst, out


@st.cache_data
def measured_scaling(Ms, n_u):
    """Multiplies/sample while adapting, measured by running each filter."""
    rng = np.random.default_rng(1)
    d, x = rng.standard_normal(8) * 50, rng.standard_normal(8) * 50
    di, xi = (d * 16).astype(np.int64), (x * 16).astype(np.int64)
    rls = [run_anc(RLS(M), d, x, 1.0).mul_per_sample for M in Ms]
    dcd = [run_anc(RLSDCD(M, n_u=n_u), di, xi, 1.0).mul_per_sample for M in Ms]
    return rls, dcd


def pct(new, old):
    return f"{(new - old) / old * 100:+.0f}% vs RLS"


def base_layout(fig, height, **kw):
    fig.update_layout(template="plotly_white", height=height, margin=dict(l=10, r=10, t=30, b=10),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(color=C_MUTED, size=12), **kw)
    fig.update_xaxes(gridcolor=C_GRID, zeroline=False, linecolor=C_GRID)
    fig.update_yaxes(gridcolor=C_GRID, zeroline=False, linecolor=C_GRID)
    return fig


# ---------------------------------------------------------------------------
# Sidebar: controls
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Inject noise")
    emg_uv = st.slider("Jaw-clench EMG at C3 (µV rms)", 0, 80, 40, 5,
                       help="EMG amplitude at C3 during the 0.30-0.75 s clench. The masseter "
                            "reference electrode sees 1.5x this.")
    drift_uv = st.slider("Baseline drift (µV)", 0, 150, 60, 5,
                         help="Amplitude of a 0.15 Hz electrode wander, plus an offset of 0.4x this.")
    blinks = st.toggle("EOG blink at 0.6 s", value=False,
                       help="Adds a blink and a second canceller stage driven by a frontal EOG reference.")
    blink_uv = st.slider("Blink peak at C3 (µV)", 10, 60, 30, 5) if blinks else 0.0

    st.header("Filter")
    M = st.slider("Filter length M (taps)", 2, 32, 8, 1)
    if M < H_EMG.size:
        st.warning(f"M = {M} is shorter than the simulated EMG path ({H_EMG.size} taps): "
                   "the canceller cannot model it, so residual EMG is expected.")
    n_u = st.select_slider("DCD iterations per sample (N_u)", options=[1, 2, 4, 8, 16], value=4,
                           help="More iterations track RLS more closely but cost more adds, shifts and compares. "
                                "Multiplies do not change.")
    gate_on = st.toggle("Reference gate", value=True,
                        help="Adapt only while the reference electrode carries signal. Below the threshold "
                             "the weights freeze: with a reference at its noise floor, an adapting canceller "
                             "fits the EEG with that noise and damages the output.")
    gate_uv = st.slider("Gate threshold (µV, mean |x| at reference)", 1.0, 10.0, 3.0, 0.5,
                        disabled=not gate_on,
                        help="Reference noise floor is ~1 µV. Shift-add estimator, 16-sample (62 ms) time constant.")

    with st.expander("Energy model (90 nm, 1.0 V)"):
        mul_pj = st.number_input("Multiply / divide (pJ)", 0.1, 50.0, 1.5, 0.1)
        add_pj = st.number_input("Add / subtract (pJ)", 0.01, 5.0, 0.1, 0.01)
        cmp_pj = st.number_input("Compare (pJ)", 0.01, 5.0, 0.1, 0.01)
        shift_pj = st.number_input("Bit-shift (pJ)", 0.01, 5.0, 0.1, 0.01)
        static_uw = st.number_input("Static leakage (µW)", 0.0, 10.0, 0.5, 0.05)
        _em_tmp = EnergyModel(mul_pj, add_pj, cmp_pj, shift_pj, static_uw)
        csp_est = csp_lda_opcount_uw(_em_tmp, CFG)
        csp_uw = st.number_input(
            "CSP + LDA dynamic power (µW)", 0.0, 10.0, 0.4, 0.05,
            help=f"Fixed penalty from published low-power CSP-LDA hardware. Cross-check: arithmetic alone "
                 f"for a streaming implementation (6 CSP filters, 2 biquads each) is "
                 f"{csp_est['mul_per_sample']} multiplies + {csp_est['add_per_sample']} adds per sample = "
                 f"{csp_est['dynamic_uw']:.3f} µW with this energy model; the rest is memory, control and clocking.")
    with st.expander("Link and latency"):
        frame_name = st.selectbox("Frame layout", list(FRAME_PRESETS))
        classify_ms = st.number_input(
            "CSP + LDA classification time (ms)", 0.0, 200.0, 25.0, 1.0,
            help="Time from window close to label. 25 ms implies batch processing of the buffered window on "
                 "a slow clock. If projection, filtering and variance accumulate per sample (streaming), only "
                 "6 logs + the LDA dot product remain at window close, which is well under 1 ms.")
        fes_ms = st.number_input("FES actuator delay (ms)", 0.0, 200.0, 16.0, 1.0)
    with st.expander("Simulation"):
        seed = st.number_input("Random seed (EMG + background EEG)", 0, 10_000, 0, 1)

em = EnergyModel(mul_pj, add_pj, cmp_pj, shift_pj, static_uw, csp_uw)
bc, burst, out = simulate(float(emg_uv), float(drift_uv), float(blink_uv), int(M), int(n_u),
                          float(gate_uv) if gate_on else None, int(seed))
res = out["results"]
rls, dcd = res["RLS"], res["RLS-DCD"]
n_refs = len(out["refs"])
bw = bandwidth_chain(CFG, frame=FRAME_PRESETS[frame_name], window_s=bc.duration_s)
pw = {k: filtering_power(r, out["dc_ops_per_sample_per_channel"], n_refs, em, CFG) for k, r in res.items()}
node = {k: node_power(pw[k], em, bw) for k in res}

t, ref = burst.t, out["truth_ref_uv"]
settled = burst.clench_mask & (t >= bc.clench_on_s + 0.1)
quiet = ~burst.clench_mask & (t > 0.1)
snr = {k: snr_db(ref, r.out_uv, settled) for k, r in res.items()}
snr_in = snr_db(ref, out["input_uv"], settled)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("Edge BCI node: telemetry")
stages = f"EMG canceller ({M} taps)" + (" → EOG canceller (1 tap)" if blinks else "")
gate_txt = f"reference gate at {gate_uv:g} µV" if gate_on else "no reference gate"
st.caption(f"One simulated 1 s burst at C3 · 9-bit ADC at {CFG.fs} S/s · shift-add DC blocker → {stages}, "
           f"{gate_txt}. Operation counts are measured on C3 and applied to each of {CFG.n_channels} channels.")

# ---------------------------------------------------------------------------
# Hardware reality
# ---------------------------------------------------------------------------
st.subheader("Hardware reality")
c1, c2, c3 = st.columns([1, 1, 1.25])


def filter_card(col, label, key, is_dcd):
    r, p, s, nd = res[key], pw[key], snr[key], node[key]
    ms = r.mul_by_state()
    base = lambda new, old: (pct(new, old), "inverse", "auto") if is_dcd else ("baseline", "off", "off")
    with col.container(border=True):
        st.markdown(f"**{label}**")
        st.caption("Mean per sample, per channel; power is for all 16 channels.")
        a, b = st.columns(2)
        d, c, ar = base(r.mul_mean, rls.mul_mean)
        a.metric("Multiplies", f"{r.mul_mean:,.1f}", d, delta_color=c, delta_arrow=ar,
                 help="Mean over the burst, including the gate. Gate closed: standard RLS only computes the "
                      "output (M multiplies), while RLS-DCD also keeps updating the first column of its "
                      "correlation matrix R (2M) so the delay-line shift structure stays valid. A production "
                      "design could pause the R update while frozen and rebuild it on wake-up to save that cost.")
        d, c, ar = base(r.total_ops_mean, rls.total_ops_mean)
        b.metric("Total ops", f"{r.total_ops_mean:,.0f}", d, delta_color=c, delta_arrow=ar,
                 help="Mean of multiply + divide + add + shift + compare, including the gate's own "
                      "shift-add estimator. RLS-DCD's count varies with the number of DCD step-size halvings.")
        d, c, ar = base(p["total_uw"], pw["RLS"]["total_uw"])
        a.metric("Filtering power", f"{p['total_uw']:.2f} µW", d, delta_color=c, delta_arrow=ar,
                 help=f"Dynamic {p['dynamic_uw']:.3f} µW (worst-case sample {p['dynamic_uw_worst']:.3f} µW) "
                      f"+ static {p['static_uw']:.2f} µW.")
        b.metric("SNR, clench", f"{s:.1f} dB", f"{s - snr_in:+.1f} dB",
                 help=f"Against ground-truth EEG, over the clench minus its first 100 ms (the canceller "
                      f"needs that long to learn the EMG path). Delta is vs. the {snr_in:.1f} dB input.")
        fmt = lambda v: "n/a" if v is None else f"{v}"
        st.caption(f"Measured multiplies: adapting {fmt(ms['adapting'])}, frozen {fmt(ms['frozen'])}; "
                   f"gate open {ms['open_frac'] * 100:.0f}% of samples.")
        verdict = ("under" if nd["total_high_uw"] <= REF_FLOOR_UW else
                   "straddling" if nd["total_low_uw"] <= REF_FLOOR_UW else "over")
        st.caption(f"Node total with CSP + LDA and link: {nd['total_low_uw']:.2f}–{nd['total_high_uw']:.2f} µW, "
                   f"{verdict} the {REF_FLOOR_UW} µW reference (Chen et al. 2010, 90 nm).")


filter_card(c1, "Standard RLS (float, textbook)", "RLS", False)
filter_card(c2, "Transversal RLS-DCD (fixed point)", "RLS-DCD", True)

with c3.container(border=True):
    st.markdown("**Multiplies per sample vs. filter length (measured, adapting)**")
    Ms = (2, 4, 8, 12, 16, 24, 32)
    m_rls, m_dcd = measured_scaling(Ms, n_u)
    fig = go.Figure()
    for name, ys, col in (("Standard RLS", m_rls, C_RLS), ("RLS-DCD", m_dcd, C_DCD)):
        fig.add_trace(go.Scatter(x=Ms, y=ys, name=name, mode="lines+markers",
                                 line=dict(color=col, width=2), marker=dict(size=8),
                                 hovertemplate="M=%{x}: %{y:,} multiplies<extra>" + name + "</extra>"))
        fig.add_annotation(x=Ms[-1], y=np.log10(ys[-1]), text=f"{name}: {ys[-1]:,}", showarrow=False,
                           xanchor="right", yanchor="bottom", yshift=6, font=dict(color=C_TRUTH, size=11))
    fig.add_vline(x=M, line=dict(color=C_MUTED, width=1, dash="dot"),
                  annotation_text=f"M = {M}", annotation_position="top left",
                  annotation_font=dict(color=C_MUTED, size=11))
    base_layout(fig, 255, legend=dict(orientation="h", y=-0.3, x=0), hovermode="closest")
    fig.update_yaxes(type="log", title="multiplies / sample")
    fig.update_xaxes(title="M (taps)")
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    st.caption("RLS: 4M² + 4M + 1 · RLS-DCD: 3M (λ = 1 − 2⁻⁷ as shift-and-add). "
               "Each point is counted while the filter runs; the engine checks it against the formula.")

with st.expander("Where the energy goes (per channel, per sample)"):
    rows = []
    for k, r in res.items():
        bd = pw[k]["breakdown_pj"]; tot = sum(bd.values())
        mo = r.mean_ops()
        counts = {"mul+div": mo["mul"] + mo["div"], "add": mo["add"], "shift": mo["shift"], "cmp": mo["cmp"]}
        for op in bd:
            rows.append({"filter": k, "operation": op, "count / sample": round(counts[op], 1),
                         "energy (pJ)": round(bd[op], 2), "share": f"{bd[op] / tot * 100:.0f}%" if tot else "-"})
    st.dataframe(rows, hide_index=True, width="stretch")
    st.caption(f"Node dynamic power = (16 × (ANC + DC blocker) + {n_refs} reference DC blocker(s)) pJ × "
               f"{CFG.fs} S/s. Energy per op is held fixed, but real energy grows with operand width "
               f"(RLS-DCD's widest registers here are {dcd.stage_filters[0].datapath_bits()['acc']} bits). "
               "Float64 RLS would cost far more than 1.5 pJ per multiply, so this model flatters the RLS baseline.")

# ---------------------------------------------------------------------------
# The signal
# ---------------------------------------------------------------------------
st.subheader("The signal")
show_rls = st.checkbox("Overlay standard RLS output", value=False)
fig = go.Figure()
fig.add_vrect(x0=bc.clench_on_s, x1=bc.clench_off_s, fillcolor="rgba(11,11,11,0.05)", line_width=0,
              annotation_text="jaw clench", annotation_position="top left",
              annotation_font=dict(color=C_MUTED, size=11))
if blinks:
    for tb in bc.blink_times_s:
        fig.add_vline(x=tb + 0.12, line=dict(color=C_MUTED, width=1, dash="dot"),
                      annotation_text="blink peak", annotation_position="bottom right",
                      annotation_font=dict(color=C_MUTED, size=11))
traces = [("Raw C3 input (9-bit ADC)", out["raw_adc_uv"], C_RAW, 1.0),
          ("Ground-truth EEG", ref, C_TRUTH, 1.8)]
if show_rls:
    traces.append(("Standard RLS output", rls.out_uv, C_RLS, 1.4))
traces.append(("RLS-DCD output", dcd.out_uv, C_DCD, 1.6))
for name, y, col, w in traces:
    fig.add_trace(go.Scatter(x=t, y=y, name=name, mode="lines", line=dict(color=col, width=w),
                             hovertemplate="%{y:.1f} µV"))
base_layout(fig, 430, hovermode="x unified",
            legend=dict(orientation="h", y=1.08, x=0, font=dict(color=C_TRUTH)))
fig.update_xaxes(title="time (s)", range=[0, t[-1]])
fig.update_yaxes(title="µV")
st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
s1, s2, s3 = st.columns(3)
s1.metric("Input SNR, clench / quiet", f"{snr_in:.1f} / {snr_db(ref, out['input_uv'], quiet):.1f} dB")
s2.metric("RLS-DCD SNR, clench / quiet",
          f"{snr['RLS-DCD']:.1f} / {snr_db(ref, dcd.out_uv, quiet):.1f} dB")
s3.metric("Standard RLS SNR, clench / quiet",
          f"{snr['RLS']:.1f} / {snr_db(ref, rls.out_uv, quiet):.1f} dB")
st.caption("The ground truth is passed through the same 1.3 Hz shift-add high-pass as the pipeline, since "
           "anything below that is removed on purpose (motor-imagery features live in 8-30 Hz). "
           f"Raw input clipping: {burst.clipped['primary']} of {t.size} samples.")

# ---------------------------------------------------------------------------
# The network
# ---------------------------------------------------------------------------
st.subheader("The network")
n1, n2, n3, n4 = st.columns(4)
n1.metric("Raw ADC stream", f"{bw['raw_bps']:,} b/s", f"{bw['raw_util']:.2f}× link capacity",
          delta_color="inverse", help="16 ch × 256 S/s × 9 b")
n2.metric("After artifact cancellation", f"{bw['filtered_bps']:,} b/s", "unchanged", delta_color="off",
          delta_arrow="off", help="Cleaning removes artifacts but does not compress: same channels, rate and bits.")
n3.metric("On the link after CSP + LDA", f"{bw['link_bps']:.0f} b/s",
          f"{bw['link_util'] * 100:.2f}% of link capacity", delta_color="off", delta_arrow="off",
          help=f"One {bw['label_bits']}-bit label per {bc.duration_s:g} s window, inside a "
               f"{bw['frame_bits']}-bit frame ({frame_name}).")
n4.metric("EQS-HBC link power", f"{bw['link_power_proportional_nw']:.1f}–{bw['link_power_cap_nw']:.0f} nW",
          help=f"Low end: {bw['link_bps']:.0f} b/s × {bw['link_pj_per_bit']:.1f} pJ/bit. High end: the radio's "
               "rated 415 nW at full 10 kb/s. Wake-up and sync overhead is unknown, so the truth is in between.")

fig = make_subplots(rows=2, cols=1, row_heights=[0.68, 0.32], vertical_spacing=0.22,
                    subplot_titles=("Bit rate at each stage (log scale)",
                                    f"Anatomy of one frame: {bw['frame_bits']} bits to carry an "
                                    f"{bw['label_bits']}-bit label"))
stages_bw = ["Raw ADC stream", "After cancellation", "Label only", "Framed on the link"]
vals = [bw["raw_bps"], bw["filtered_bps"], bw["label_bps"], bw["link_bps"]]
fig.add_trace(go.Bar(y=stages_bw, x=vals, orientation="h", marker=dict(color=C_RAW),
                     text=[f"{v:,.0f} b/s" for v in vals], textposition="outside", cliponaxis=False,
                     textfont=dict(color=C_TRUTH), hovertemplate="%{y}: %{x:,.0f} b/s<extra></extra>"),
              row=1, col=1)
fig.add_vline(x=bw["cap_bps"], line=dict(color=C_CRIT, width=2), row=1, col=1)
fig.add_annotation(x=np.log10(bw["cap_bps"]), y=1.0, xref="x", yref="y domain", yanchor="bottom",
                   showarrow=False, text="EQS-HBC cap: 10 kb/s", font=dict(color=C_CRIT, size=11))
fig.update_xaxes(type="log", range=[0, 5.2], row=1, col=1)
fig.update_yaxes(autorange="reversed", gridcolor="rgba(0,0,0,0)", row=1, col=1)
x0 = 0
for field_name, bits in bw["frame"].items():
    is_label = field_name.lower().startswith("label")
    fig.add_trace(go.Bar(y=["frame"], x=[bits], base=[x0], orientation="h", showlegend=False,
                         marker=dict(color=C_TRUTH if is_label else "#c9c7bf", line=dict(color="#fcfcfb", width=2)),
                         text=[f"{field_name}<br>{bits} b"], textposition="inside", insidetextanchor="middle",
                         textfont=dict(color="#fcfcfb" if is_label else C_TRUTH, size=10),
                         hovertemplate=f"{field_name}: {bits} bits<extra></extra>"), row=2, col=1)
    x0 += bits
fig.update_xaxes(range=[0, x0], title="bits", row=2, col=1)
fig.update_yaxes(showticklabels=False, gridcolor="rgba(0,0,0,0)", row=2, col=1)
base_layout(fig, 390, showlegend=False, bargap=0.35, barmode="overlay")
fig.update_annotations(font=dict(size=12))
fig.update_layout(margin=dict(t=40))
st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

nd = node["RLS-DCD"]
p1, p2, p3, p4, p5 = st.columns([1, 1, 1, 1.1, 1.5])
p1.metric("Static leakage", f"{nd['static_uw']:.2f} µW")
p2.metric("Filtering, dynamic", f"{nd['filtering_dyn_uw']:.2f} µW", help="RLS-DCD path, measured ops.")
p3.metric("CSP + LDA", f"{nd['csp_lda_uw']:.2f} µW", help="Fixed penalty per 1 s window (sidebar).")
p4.metric("Link", f"{nd['link_low_uw'] * 1e3:.0f}–{nd['link_high_uw'] * 1e3:.0f} nW",
          help="Bit-proportional energy up to the radio's rated power at full capacity.")
p5.metric("Node total, RLS-DCD path", f"{nd['total_low_uw']:.2f}–{nd['total_high_uw']:.2f} µW",
          help=f"Compare: the {REF_FLOOR_UW} µW reference (Chen et al. 2010). The same budget with standard RLS "
               f"is {node['RLS']['total_low_uw']:.2f}–{node['RLS']['total_high_uw']:.2f} µW.")
st.caption(f"Pipeline: 9-bit ADC → DC blocker → gated adaptive cancellation → CSP log-variance + LDA → "
           f"{bw['label_bits']}-bit label per {bc.duration_s:g} s window in a {bw['frame_bits']}-bit frame → EQS-HBC. "
           "CSP + LDA power is a fixed penalty from published hardware, not measured here.")

# ---------------------------------------------------------------------------
# Latency waterfall
# ---------------------------------------------------------------------------
st.subheader("The latency waterfall")
lat = latency_budget(bc.duration_s, classify_ms, bw["tx_ms"], fes_ms)
bio = lat[0]
tail = lat[1:]
t_end = lat[-1]["end_ms"]
tail_ms = t_end - bio["end_ms"]
silicon_ms = sum(s["dur_ms"] for s in tail if s["kind"] == "silicon")
act_ms = sum(s["dur_ms"] for s in tail if s["kind"] == "actuation")


LATENCY_HTML = """
<style>
  body { margin:0; padding:2px 64px 4px 10px; font-family:"Source Sans Pro","Source Sans 3",system-ui,sans-serif; color:%(ink)s; background:transparent; }
  .row { display:flex; align-items:center; gap:14px; margin-bottom:10px; }
  button { background:%(blue)s; color:#fff; border:0; border-radius:8px; padding:8px 14px; font-size:15px; cursor:pointer; }
  button:disabled { opacity:.6; cursor:default; }
  label { font-size:14px; color:%(muted)s; }
  .clock { margin-left:auto; font-size:16px; font-variant-numeric:tabular-nums; }
  .cap { font-size:12px; color:%(muted)s; margin:6px 0 4px; }
  .legend { display:flex; gap:16px; font-size:12px; margin:2px 0 8px; }
  .legend i { display:inline-block; width:10px; height:10px; margin-right:5px; vertical-align:-1px; }
  .strip { position:relative; height:22px; background:#f3f2ee; }
  .bar { position:absolute; top:0; height:100%%; width:0; }
  .grid { display:grid; grid-template-columns:190px 30%% 22px 1fr; row-gap:8px; align-items:center; }
  .lab { font-size:13px; color:%(muted)s; text-align:right; padding-right:10px; }
  .panel { position:relative; height:24px; border-left:1px solid %(grid)s; }
  .val { position:absolute; top:3px; font-size:12px; white-space:nowrap; font-variant-numeric:tabular-nums; }
  .brk { text-align:center; color:%(muted)s; font-size:13px; }
  .ticks { position:relative; height:16px; font-size:11px; color:%(muted)s; }
  .ticks span { position:absolute; transform:translateX(-50%%); }
  .synth { font-weight:600; font-size:15px; margin-top:14px; min-height:44px; line-height:1.4; }
</style>
<div class="row">
  <button id="go">Simulate latency pipeline</button>
  <label><input type="checkbox" id="slow"> Slow-motion tail (&times;20)</label>
  <span class="clock" id="clk">t = 0.0 ms</span>
</div>
<div class="legend"><span><i style="background:%(bio)s"></i>Biology</span><span><i style="background:%(sil)s"></i>Silicon</span><span><i style="background:%(act)s"></i>Actuation</span></div>
<div class="cap">True scale: 0&ndash;%(tend).0f ms</div>
<div class="strip" id="strip"></div>
<div class="ticks" id="stripticks"></div>
<div class="cap" style="margin-top:14px">Broken axis: left panel 0&ndash;%(bio_end).0f ms; right panel zoomed &times;%(zoom).0f on the post-decision tail</div>
<div class="grid" id="grid"></div>
<div class="synth" id="synth"></div>
<script>
const S = %(stages)s, TEND = %(tend)f, BIO = %(bio_end)f, Z1 = %(z1)f, COL = %(colors)s;
const strip = document.getElementById('strip'), grid = document.getElementById('grid');
const pct = (v, a, b) => Math.max(0, Math.min(100, (v - a) / (b - a) * 100));
const sb = [], lb = [], vals = [];
S.forEach(s => {
  const d = document.createElement('div'); d.className = 'bar'; d.style.left = pct(s.start_ms, 0, TEND) + '%%';
  d.style.background = COL[s.kind]; d.title = s.stage + ': ' + s.dur_ms.toFixed(1) + ' ms'; strip.appendChild(d); sb.push(d);
  const lab = document.createElement('div'); lab.className = 'lab'; lab.textContent = s.stage; grid.appendChild(lab);
  const L = document.createElement('div'); L.className = 'panel';
  const brk = document.createElement('div'); brk.className = 'brk'; brk.textContent = '//';
  const R = document.createElement('div'); R.className = 'panel';
  grid.appendChild(L); grid.appendChild(brk); grid.appendChild(R);
  const bio = s.kind === 'biology', host = bio ? L : R, a = bio ? 0 : BIO, b = bio ? BIO : Z1;
  const bar = document.createElement('div'); bar.className = 'bar'; bar.style.background = COL[s.kind];
  bar.style.left = pct(s.start_ms, a, b) + '%%';
  bar.title = s.stage + ': ' + s.start_ms.toFixed(1) + '-' + s.end_ms.toFixed(1) + ' ms';
  const v = document.createElement('span'); v.className = 'val';
  if (bio) { v.style.right = '6px'; v.style.color = '#fcfcfb'; } else { v.style.left = 'calc(' + pct(s.end_ms, a, b) + '%% + 6px)'; }
  host.appendChild(bar); host.appendChild(v);
  lb.push({bar, a, b}); vals.push(v);
});
function ticks(el, vs, a, b) {
  vs.forEach(t => { const s = document.createElement('span'); s.style.left = pct(t, a, b) + '%%'; s.textContent = t; el.appendChild(s); });
}
ticks(document.getElementById('stripticks'), [0, 200, 400, 600, 800, 1000], 0, TEND);
grid.appendChild(document.createElement('div'));
const t1 = document.createElement('div'); t1.className = 'ticks'; grid.appendChild(t1); ticks(t1, [0, 250, 500, 750], 0, BIO);
grid.appendChild(document.createElement('div'));
const t2 = document.createElement('div'); t2.className = 'ticks'; grid.appendChild(t2);
const tv = []; for (let t = Math.ceil(BIO / 10) * 10; t <= Z1; t += 10) tv.push(t); ticks(t2, tv, BIO, Z1);
const fmt = (x) => x.toLocaleString('en-US', {minimumFractionDigits: 1, maximumFractionDigits: 1});
function draw(t) {
  S.forEach((s, i) => {
    const e = Math.min(t, s.end_ms);
    sb[i].style.width = (pct(e, 0, TEND) - pct(s.start_ms, 0, TEND)) + '%%';
    const o = lb[i]; o.bar.style.width = (pct(e, o.a, o.b) - pct(s.start_ms, o.a, o.b)) + '%%';
    vals[i].textContent = t >= s.end_ms ? fmt(s.dur_ms) + ' ms' : '';
  });
  document.getElementById('clk').textContent = 't = ' + fmt(Math.min(t, TEND)) + ' ms';
  document.getElementById('synth').textContent = t >= TEND ? %(synth)s : '';
}
draw(0);
document.getElementById('go').onclick = () => {
  const btn = document.getElementById('go'), slow = document.getElementById('slow').checked;
  btn.disabled = true;
  const t0 = performance.now();
  (function frame() {
    const w = performance.now() - t0, t = (slow && w > BIO) ? BIO + (w - BIO) / 20 : w;
    draw(t);
    if (t < TEND) requestAnimationFrame(frame); else btn.disabled = false;
  })();
};
</script>
"""

window_words = "one full second" if round(bio["dur_ms"]) == 1000 else f"{bio['dur_ms']:,.0f} ms"
synth = (f"The decoder reads {window_words} of brain activity before it commits. Then the node classifies, "
         f"transmits and fires the muscle in {tail_ms:.0f} ms. The wait is in the biology, not the silicon.")
z1 = t_end + 0.06 * tail_ms
st.iframe(LATENCY_HTML % dict(
    ink=C_TRUTH, muted=C_MUTED, grid=C_GRID, blue=C_DCD, bio=C_RAW, sil=C_SILICON, act=C_ACT,
    tend=t_end, bio_end=bio["end_ms"], z1=z1, zoom=(1000 / (z1 - bio["end_ms"])) * (0.7 / 0.3),
    stages=json.dumps(lat), colors=json.dumps(KIND_COLOR), synth=json.dumps(synth)), height=400)
st.caption("Playback runs on your browser's clock: the decoder's one-second window fills in real time. Without slow "
           "motion the whole hardware tail passes in about three screen refreshes. Hover a bar for its exact span. "
           f"Transmission time is computed: {bw['frame_bits']} bits ÷ {CFG.link_cap_bps / 1e3:g} kb/s = {bw['tx_ms']:.1f} ms.")

l1, l2, l3, l4 = st.columns(4)
l1.metric("Decoder window: brain activity read", f"{bio['dur_ms']:,.0f} ms",
          help="How much brain activity the classifier reads before it commits to a decision. "
               "A design choice of the decoder, not a fixed delay of the brain.")
l2.metric("Silicon: classify + transmit", f"{silicon_ms:.1f} ms",
          help=f"{classify_ms:g} ms classification + {bw['tx_ms']:.1f} ms on air "
               f"({bw['frame_bits']} bits ÷ {CFG.link_cap_bps / 1e3:g} kb/s).")
l3.metric("Actuation: FES + muscle", f"{act_ms:.1f} ms")
l4.metric("Hardware tail vs. brain", f"{tail_ms:.1f} ms",
          f"{tail_ms / t_end * 100:.1f}% of the {t_end:,.0f} ms total", delta_color="off", delta_arrow="off")


# --- Repository Link & Telemetry Export ---
st.sidebar.markdown("---")
st.sidebar.markdown("[📂 View GitHub Repository](https://github.com/MACVXACE/edge-bci-emulator)")

telemetry_csv = """Metric,Standard RLS (Float),Transversal RLS-DCD (Fixed Point),Delta / Improvement
Multiplies per sample (adapting),289,24,-91.7%
Multiplies per sample (mean),275.8,23.6,-91.4%
Total ops per sample,472,242,-48.7%
Filtering power 16-ch (uW),2.28,0.74,-67.5%
Settled Clench SNR (dB),4.7,4.6,-0.1 dB (Preserved)
Total Node Power (uW),2.68-3.09,1.14-1.55,Meets sub-2 uW envelope
Raw stream bandwidth,36864 b/s,36864 b/s,3.69x over 10 kb/s cap
Over-the-air link rate,36864 b/s (Overloaded),112 b/s (CSP+LDA frame),1.12% link utilization
Hardware execution tail (ms),52.2,52.2,5% of 1052.2 ms total latency
"""

st.download_button(
    label="📥 Export Telemetry Benchmark (CSV)",
    data=telemetry_csv,
    file_name="bdn04_silicon_telemetry.csv",
    mime="text/csv",
)
