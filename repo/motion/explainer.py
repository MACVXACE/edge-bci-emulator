"""
explainer.py - BDN-04 cinematic explainer, dark edition.

One film in ten chapters. Every number on screen is computed by bci_core.py when the
module loads (the dashboard's default settings: EMG 40 uV, drift 60 uV, M = 8, N_u = 4,
gate 3 uV, seed 0), and the asserts stop the render if the engine and the film disagree.

Render the whole film in one pass:
    manim -qh -r 1920,1080 --fps 24 explainer.py MasterExplainer
or chapter by chapter (C00_Title ... C11_End), which is what build_film.py does so it can
render two chapters at a time and re-render one without the rest.

Each render also writes its sound-design cues (clock ticks, the latency flam, impact hits)
to media/cues/<Scene>.json; score.py reads them to place every sound on the right frame.

Requirements: Manim Community 0.18+, Source Sans 3 and Source Code Pro (SIL OFL).
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
from PIL import Image
from manim import (
    DOWN, LEFT, ORIGIN, RIGHT, UP, UL,
    Animation, Arrow, Circle, Create, CurvedArrow, DashedLine, FadeIn, FadeOut, Flash, Group, ImageMobject,
    LaggedStart, Line, MarkupText, MovingCameraScene, Rectangle, RoundedRectangle, Square, Text,
    FadeTransform, Transform, ValueTracker, VGroup, VMobject, always_redraw, config,
)
from manim.utils.rate_functions import (ease_in_out_cubic, ease_in_out_sine, ease_out_cubic, linear,
                                        smooth, there_and_back)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import bci_core as core  # noqa: E402

CUE_DIR = os.path.join(HERE, "media", "cues")
DASH_IMG = os.path.join(HERE, "assets", "dashboard_tall.jpg")

# ---------------------------------------------------------------------------
# Palette: dark studio
# ---------------------------------------------------------------------------
BG = "#0e1015"          # charcoal matte
GRID = "#1f242d"        # grid paper, drawn at 15 % opacity
PANEL = "#161b22"       # box fill
EDGE = "#30363d"        # box stroke, tracks
WHITE = "#f0f6fc"       # ground truth / neural signal, headlines
SOFT = "#c9d1d9"        # captions and body text
MUTED = "#8b949e"       # raw ADC, noise, baseline
DIM = "#6e7681"         # arrows, minor labels
BLUE = "#388bfd"        # RLS-DCD, the accent
BLUE2 = "#58a6ff"       # accent, lighter (text on dark)
GREEN = "#3fb950"       # link capacity, hardware tail
GREEN2 = "#56d364"
GREEN3 = "#2ea043"
AMBER = "#d29922"       # the clench, lit
RED = "#f85149"         # overflow only
VIOLET = "#a371f7"      # CSP + LDA in the power budget
SANS, MONO = "Source Sans 3", "Source Code Pro"

config.background_color = BG
FW0, FH0 = 8.0 * 16 / 9, 8.0      # frame size the layout is drawn for
CAP_Y1, CAP_Y2, TAG_Y = -3.44, -3.33, -2.64

# ---------------------------------------------------------------------------
# Engine numbers (dashboard defaults)
# ---------------------------------------------------------------------------
CFG = core.NodeConfig()
_bc = core.BurstConfig(emg_uv=40.0, drift_uv=60.0, drift_offset_uv=24.0, blink_uv=0.0)
_burst = core.generate_burst(_bc, CFG, 0)
_out = core.clean(_burst, M=8, cfg=CFG, filters=("RLS", "RLS-DCD"),
                  filter_kwargs={"dcd": {"n_u": 4}}, gate_uv=3.0)
_em = core.EnergyModel(1.5, 0.1, 0.1, 0.1, 0.5, 0.4)
FRAME = next(iter(core.FRAME_PRESETS.values()))
_bw = core.bandwidth_chain(CFG, frame=FRAME, window_s=_bc.duration_s)
_res = _out["results"]
_pw = {k: core.filtering_power(r, _out["dc_ops_per_sample_per_channel"], len(_out["refs"]), _em, CFG)
       for k, r in _res.items()}
_node = core.node_power(_pw["RLS-DCD"], _em, _bw)
LAT = core.latency_budget(_bc.duration_s, 25.0, _bw["tx_ms"], 16.0)

T = _burst.t
TRUTH, INP, DCD = _out["truth_ref_uv"], _out["input_uv"], _res["RLS-DCD"].out_uv
SETTLED = _burst.clench_mask & (T >= _bc.clench_on_s + 0.1)      # the dashboard's "clench" SNR
SNR_IN = core.snr_db(TRUTH, INP, SETTLED)
SNR_DCD = core.snr_db(TRUTH, DCD, SETTLED)
SNR_RLS = core.snr_db(TRUTH, _res["RLS"].out_uv, SETTLED)
P_RLS, P_DCD = _pw["RLS"]["total_uw"], _pw["RLS-DCD"]["total_uw"]
WINDOW_MS = LAT[0]["dur_ms"]
LAT_TOTAL = LAT[-1]["end_ms"]
TAIL_MS = LAT_TOTAL - WINDOW_MS

M = 8
RLS_MULS, DCD_MULS, SIDE = 4 * M * M + 4 * M + 1, 3 * M, 2 * M + 1
M_BIG = 32
RLS_BIG, DCD_BIG = 4 * M_BIG ** 2 + 4 * M_BIG + 1, 3 * M_BIG
RAW_BPS = CFG.raw_bitrate_bps
LINK_BPS = int(CFG.link_cap_bps)
FRAME_BITS = _bw["frame_bits"]
REF_UW = 1.41      # Chen et al. 2010, 90 nm (the dashboard's reference)

assert (round(SNR_IN, 1), round(SNR_DCD, 1), round(SNR_RLS, 1)) == (-12.6, 4.6, 4.7)
assert (round(P_RLS, 2), round(P_DCD, 2)) == (2.28, 0.74)
assert (round(_node["total_low_uw"], 2), round(_node["total_high_uw"], 2)) == (1.14, 1.55)
assert WINDOW_MS == 1000.0 and round(LAT_TOTAL, 1) == 1052.2 and round(TAIL_MS, 1) == 52.2
assert _bw["tx_ms"] == 11.2 and FRAME_BITS == 112 and _bw["link_bps"] == 112.0
assert RLS_MULS == SIDE * SIDE == 289 and DCD_MULS == 24 and RLS_BIG == 4225 and DCD_BIG == 96
assert RAW_BPS == 36_864 and LINK_BPS == 10_000
assert round(RAW_BPS / LINK_BPS, 2) == 3.69 and round(100 * _bw["link_util"], 2) == 1.12
assert round(RLS_BIG / DCD_BIG) == 44


# ---------------------------------------------------------------------------
# Helpers: type, glow, the backdrop
# ---------------------------------------------------------------------------
def words(s, size=30, color=SOFT, font=SANS, weight="NORMAL"):
    return Text(s, font=font, font_size=size, color=color, weight=weight)


def mono(s, size=30, color=WHITE):
    return Text(s, font=MONO, font_size=size, color=color)


def nbsp(text):
    import re
    return re.sub(r"(\d) (nm|µW|dB|ms|b/s|bits|taps)", "\\1\u00a0\\2", text)


def glow_path(path, color, width=2.5, strength=1.0):
    """Bloom by stacking wide, faint strokes under the sharp one."""
    g = VGroup(*[path.copy().set_stroke(color, w, opacity=o * strength)
                 for w, o in ((20, 0.03), (11, 0.06), (6, 0.14))])
    g.add(path.copy().set_stroke(color, width, opacity=1))
    return g


def halo(mob, color, strength=1.0):
    return VGroup(*[mob.copy().set_fill(opacity=0).set_stroke(color, w, opacity=o * strength)
                    for w, o in ((22, 0.035), (12, 0.07), (6, 0.16))])


def bloom(point, radius, color, steps=28, peak=0.10):
    """Soft radial light: concentric discs whose summed opacity falls off outward."""
    return VGroup(*[Circle(radius=radius * (1 - k / steps), stroke_width=0)
                    .set_fill(color, opacity=peak / steps).move_to(point) for k in range(steps)])


def background_png():
    """Charcoal matte + 15 % grid paper + radial vignette, fixed to the screen."""
    w, h = config.pixel_width, config.pixel_height
    os.makedirs(CUE_DIR, exist_ok=True)
    path = os.path.join(CUE_DIR, f"backdrop_{w}x{h}.png")
    if os.path.exists(path):
        return path
    bg = np.array([int(BG[i:i + 2], 16) for i in (1, 3, 5)], float)
    gr = np.array([int(GRID[i:i + 2], 16) for i in (1, 3, 5)], float)
    img = np.ones((h, w, 3)) * bg
    step = w / FW0 * 0.5                                  # half-unit grid
    ys, xs = np.mgrid[0:h, 0:w]
    cx, cy = w / 2, h / 2
    on = (np.abs(((xs - cx) / step + 0.5) % 1 - 0.5) * step < 0.6) | \
         (np.abs(((ys - cy) / step + 0.5) % 1 - 0.5) * step < 0.6)
    img[on] = img[on] * 0.85 + gr * 0.15
    r = np.sqrt(((xs - cx) / (w * 0.62)) ** 2 + ((ys - cy) / (h * 0.62)) ** 2)
    v = 1.0 - 0.55 * np.clip((r - 0.35) / 0.75, 0, 1) ** 1.6
    lift = 1.0 + 0.10 * np.exp(-(r / 0.45) ** 2)          # faint key light in the middle
    img = np.clip(img * (v * lift)[..., None], 0, 255)
    rgba = np.dstack([img, np.full((h, w), 255.0)]).astype(np.uint8)
    Image.fromarray(rgba, "RGBA").save(path)
    return path


def box(label, sub=None, w=1.95, h=1.55, accent=False):
    r = RoundedRectangle(corner_radius=0.12, width=w, height=h,
                         stroke_color=BLUE if accent else EDGE, stroke_width=3 if accent else 2,
                         fill_color="#0f2340" if accent else PANEL, fill_opacity=1)
    t = words(label, 26, BLUE2 if accent else WHITE, weight="SEMIBOLD")
    if t.width > w - 0.22:
        t.scale_to_fit_width(w - 0.22)
    g = VGroup(t)
    if sub:
        s = VGroup(*[words(line, 19, MUTED) for line in sub.split("\n")]).arrange(DOWN, buff=0.08)
        if s.width > w - 0.2:
            s.scale_to_fit_width(w - 0.2)
        g.add(s)
    g.arrange(DOWN, buff=0.14).move_to(r)
    out = VGroup(r, g)
    if accent:
        out = VGroup(halo(r, BLUE), r, g)
    return out


def link_arrow(a, b, color=DIM):
    return Arrow(a.get_right(), b.get_left(), buff=0.06, stroke_width=3, color=color,
                 max_tip_length_to_length_ratio=0.3, tip_length=0.14)


# ---------------------------------------------------------------------------
# The bandwidth conduit (chapters 03 and 05)
# ---------------------------------------------------------------------------
X0 = -6.2
RAW_LEN = 11.8
UNIT = RAW_LEN / RAW_BPS
LINK_LEN = LINK_BPS * UNIT
PIPE_Y = 0.55


def conduit(y=PIPE_Y):
    body = RoundedRectangle(corner_radius=0.2, width=LINK_LEN, height=0.92, stroke_color=GREEN,
                            stroke_width=2.5, fill_color=GREEN, fill_opacity=0.06).move_to([X0 + LINK_LEN / 2, y, 0])
    gloss = Line([X0 + 0.2, y + 0.3, 0], [X0 + LINK_LEN - 0.2, y + 0.3, 0], stroke_color=GREEN,
                 stroke_width=1.5, stroke_opacity=0.45)
    flanges = VGroup(*[RoundedRectangle(corner_radius=0.04, width=0.12, height=1.18, stroke_color=GREEN,
                                        stroke_width=2, fill_color=GREEN, fill_opacity=0.25).move_to([x, y, 0])
                       for x in (X0, X0 + LINK_LEN)])
    return VGroup(halo(body, GREEN, 0.9), body, gloss, flanges)


def stream_bar(bps, color, y=PIPE_Y, h=0.4):
    w = bps * UNIT
    return Rectangle(width=w, height=h, stroke_width=0, fill_color=color, fill_opacity=1) \
        .move_to([X0 + w / 2, y, 0])


# ---------------------------------------------------------------------------
# The film
# ---------------------------------------------------------------------------
class Film(MovingCameraScene):
    def setup(self):
        super().setup()
        self.camera.background_image = background_png()
        self.camera.init_background()
        self.frame = self.camera.frame
        self.cues = []
        self.fixed_mobs = []
        self._cap = self._tag = self._band = None

    # --- time and cues -------------------------------------------------------
    @property
    def now(self):
        return self.renderer.time

    def cue(self, kind, **kw):
        kw.setdefault("t", self.now)
        self.cues.append({"kind": kind, **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in kw.items()}})

    def tear_down(self):
        super().tear_down()
        os.makedirs(CUE_DIR, exist_ok=True)
        with open(os.path.join(CUE_DIR, f"{type(self).__name__}.json"), "w") as f:
            json.dump({"scene": type(self).__name__, "duration": self.now,
                       "fps": config.frame_rate, "cues": self.cues}, f, indent=1)

    # --- screen-fixed layer (captions, chapter tags, bands) ------------------
    def fixed(self, mob, alpha=0.0, z=100):
        ref = mob.copy()
        a = ValueTracker(alpha)

        def upd(m):
            s = self.frame.width / FW0
            m.become(ref.copy().scale(s, about_point=ORIGIN).shift(self.frame.get_center()))
            k = a.get_value()
            m.set_z_index(z)
            for sm in m.family_members_with_points():
                sm.set_fill(opacity=sm.get_fill_opacity() * k)
                sm.set_stroke(opacity=sm.get_stroke_opacity() * k)

        mob.alpha = a
        mob.add_updater(upd)
        upd(mob)
        self.add(mob)
        self.fixed_mobs.append(mob)
        return mob

    def drop_fixed(self, mob):
        mob.clear_updaters()
        self.remove(mob)
        if mob in self.fixed_mobs:
            self.fixed_mobs.remove(mob)

    def caption_mob(self, text, size=28, maxw=12.2):
        text = nbsp(text)
        t = words(text, size, SOFT)
        if t.width <= maxw:
            return t.move_to([0, CAP_Y1, 0])
        parts = text.split(" ")
        best = None
        for i in range(1, len(parts)):
            a, b = words(" ".join(parts[:i]), size, SOFT), words(" ".join(parts[i:]), size, SOFT)
            m = max(a.width, b.width)
            if m <= maxw and (best is None or m < best[0]):
                best = (m, a, b)
        assert best, text
        return VGroup(best[1], best[2]).arrange(DOWN, buff=0.1).move_to([0, CAP_Y2, 0])

    def cap(self, text):
        """Animations that cross-fade to a new caption (or to none)."""
        anims = []
        if self._cap is not None:
            a = self._cap.alpha.animate(rate_func=lambda t: smooth(min(1.0, t / 0.45))).set_value(0).build()
            a.keep_rate = True
            anims.append(a)
            self._stale = getattr(self, "_stale", []) + [self._cap]
        self._cap = None
        if text:
            self._cap = self.fixed(self.caption_mob(text))
            a = self._cap.alpha.animate(rate_func=lambda t: smooth(max(0.0, (t - 0.45) / 0.55))).set_value(1).build()
            a.keep_rate = True
            anims.append(a)
        return anims

    def play(self, *args, **kwargs):
        """A play-level rate_func applies to every animation except caption fades, which keep their own
        out-then-in timing so two captions never sit on top of each other."""
        rf = kwargs.pop("rate_func", None)
        anims = []
        for a in args:
            if hasattr(a, "build") and not isinstance(a, Animation):
                a = a.build()
            if rf is not None and isinstance(a, Animation) and not getattr(a, "keep_rate", False):
                a.rate_func = rf
            anims.append(a)
        super().play(*anims, **kwargs)

    def say(self, text, *extra, run_time=0.6, **kw):
        self.play(*self.cap(text), *extra, run_time=run_time, **kw)
        for m in getattr(self, "_stale", []):
            self.drop_fixed(m)
        self._stale = []

    def tag(self, text, band=0.75):
        """Chapter tag, plus a soft dark gradient under the caption area so captions stay legible."""
        n, top = 16, 2.1
        o = 1 - (1 - band) ** (1 / n)
        grad = VGroup(*[Rectangle(width=FW0 + 0.4, height=top * (i + 1) / n, stroke_width=0).set_fill(BG, o)
                        .move_to([0, -FH0 / 2 + top * (i + 1) / n / 2, 0]) for i in range(n)])
        self._band = self.fixed(grad, z=90)
        g = VGroup(Line(LEFT * 0.25, RIGHT * 0.25, stroke_color=BLUE, stroke_width=2),
                   mono(text, 19, BLUE2)).arrange(RIGHT, buff=0.18).move_to([0, TAG_Y, 0])
        self._tag = self.fixed(g)
        return [self._tag.alpha.animate.set_value(1), self._band.alpha.animate.set_value(1)]

    # --- camera ---------------------------------------------------------------
    def drift(self, rate=0.006, focus=ORIGIN, pan=ORIGIN):
        f, p = np.array(focus, float), np.array(pan, float)

        def upd(m, dt):
            m.scale(1 - rate * dt, about_point=f)
            m.shift(p * dt)

        self.frame.clear_updaters()
        self.frame.add_updater(upd)

    def hold(self):
        self.frame.clear_updaters()

    # --- chapter boundaries ---------------------------------------------------
    def content(self):
        skip = set(self.fixed_mobs) | {self.frame}
        return [m for m in self.mobjects if m not in skip]

    def clear_content(self, run_time=0.6, keep_text=True):
        c = self.content()
        anims = [FadeOut(Group(*c))] if c else []
        if not keep_text:
            anims += [m.alpha.animate.set_value(0) for m in self.fixed_mobs]
        self.play(*anims, run_time=run_time)
        self.remove(*c)

    def close(self, run_time=0.8):
        self.hold()
        c = self.content()
        anims = [FadeOut(Group(*c))] if c else []
        anims += [m.alpha.animate.set_value(0) for m in self.fixed_mobs]
        self.play(*anims, run_time=run_time)
        for m in list(self.fixed_mobs):
            self.drop_fixed(m)
        self.remove(*self.content())
        self._cap = self._tag = self._band = None
        self.frame.move_to(ORIGIN).set(width=FW0)
        self.wait(0.25)

    # =========================================================================
    # Chapters
    # =========================================================================
    def ch_title(self):
        title = Text("BDN-04", font=MONO, font_size=150, color=WHITE).move_to(UP * 1.0)
        light = bloom(title.get_center(), 3.6, BLUE, peak=0.16)
        rule = Line(LEFT * 1.5, RIGHT * 1.5, color=BLUE, stroke_width=5).next_to(title, DOWN, buff=0.35)
        rule_g = glow_path(rule, BLUE, 5)
        sub = words("An edge-compute brain–computer interface node for stroke rehabilitation", 34, SOFT)
        sub.scale_to_fit_width(min(sub.width, 12.4)).next_to(rule, DOWN, buff=0.55)
        kind = words("A software emulator: signal engine, hardware model and live dashboard", 26, MUTED) \
            .next_to(sub, DOWN, buff=0.3)
        self.frame.set(width=FW0 * 1.12)
        self.drift(rate=0.016)
        self.play(FadeIn(light), FadeIn(title, scale=1.05), run_time=1.6, rate_func=ease_out_cubic)
        self.play(Create(rule_g), run_time=0.8)
        self.play(FadeIn(sub, shift=UP * 0.1), run_time=0.9)
        self.play(FadeIn(kind), run_time=0.7)
        self.wait(3.0)
        self.close()

    def ch_idea(self):
        names = [("Imagined movement", "the patient tries\nto move the hand"),
                 ("EEG at C3", "scalp electrode over\nmotor cortex"),
                 ("BDN-04 node", "cleans, decides\non the body"),
                 ("Body link", "EQS-HBC\n10,000 b/s"),
                 ("FES", "electrical stimulation\nof the forearm"),
                 ("Hand moves", "intent and movement\nline up")]
        boxes = VGroup(*[box(n, s, accent=(i == 2)) for i, (n, s) in enumerate(names)]) \
            .arrange(RIGHT, buff=0.3).move_to(UP * 0.55)
        arrows = VGroup(*[link_arrow(boxes[i], boxes[i + 1]) for i in range(5)])
        back = CurvedArrow(boxes[-1].get_bottom() + DOWN * 0.1, boxes[0].get_bottom() + DOWN * 0.1,
                           angle=-0.5, color=DIM, stroke_width=2.5, tip_length=0.18)
        back_lab = words("seeing and feeling the movement closes the loop", 24, MUTED).next_to(back, DOWN, buff=0.1)
        node_light = bloom(boxes[2].get_center(), 1.9, BLUE, peak=0.12)

        self.frame.set(width=7.2).move_to(boxes[0].get_center() + RIGHT * 1.6 + DOWN * 0.5)
        self.play(*self.tag("01 · THE IDEA"), *self.cap(
            "After a stroke, imagining a movement still leaves a trace in the EEG over motor cortex."),
            FadeIn(boxes[0], shift=RIGHT * 0.2), run_time=0.8)
        steps = []
        for i in range(1, 6):
            steps.append(LaggedStart(Create(arrows[i - 1]), FadeIn(boxes[i], shift=RIGHT * 0.2), lag_ratio=0.4))
        self.play(LaggedStart(*steps, lag_ratio=0.75),
                  self.frame.animate.move_to(boxes[5].get_center() + LEFT * 1.6 + DOWN * 0.5),
                  run_time=4.6, rate_func=ease_in_out_sine)
        self.play(self.frame.animate.set(width=FW0).move_to(DOWN * 0.15), Create(back), FadeIn(back_lab),
                  FadeIn(node_light), *self.cap("A brain–computer interface can catch that intent and trigger "
                                                 "stimulation, so the hand moves when the patient means it to."),
                  run_time=1.8, rate_func=ease_in_out_cubic)
        self.drift(rate=0.006, focus=boxes[2].get_center())
        self.play(boxes[2][1].animate(rate_func=there_and_back).set_stroke(BLUE2, width=6), run_time=1.0)
        self.wait(5.6)
        self.close()

    def ch_constraints(self):
        cols = [("POWER", "~1 µW", f"a published 90 nm EEG node\ndraws {REF_UW} µW (Chen et al. 2010)"),
                ("BANDWIDTH", f"{LINK_BPS:,} b/s", "the body-coupled link\n(EQS-HBC) carries no more"),
                ("LATENCY", "intent → muscle", "the stimulation must follow\nthe intent it rewards")]
        groups = VGroup()
        for head, val, note in cols:
            h = mono(head, 22, MUTED)
            v = mono(val, 44, BLUE2)
            n = VGroup(*[words(line, 24, SOFT) for line in note.split("\n")]).arrange(DOWN, buff=0.1)
            groups.add(VGroup(h, v, n).arrange(DOWN, buff=0.35))
        groups.arrange(RIGHT, buff=0.9).move_to(UP * 0.45)
        if groups.width > 12.6:
            groups.scale_to_fit_width(12.6)
        rules = VGroup(*[Line(UP * 1.5, DOWN * 1.5, color=EDGE, stroke_width=2)
                         .move_to(((groups[i].get_right() + groups[i + 1].get_left()) / 2) * RIGHT + UP * 0.45)
                         for i in range(2)])
        lights = VGroup(*[bloom(g[1].get_center(), 1.5, BLUE, peak=0.08) for g in groups])

        self.frame.set(width=8.4).move_to(groups[0].get_center() + DOWN * 0.7)
        self.play(*self.tag("02 · THE CONSTRAINTS"),
                  *self.cap("BDN-04 is the node in the middle, and it has to fit inside three hard budgets."),
                  FadeIn(lights[0]), FadeIn(groups[0], shift=UP * 0.15), run_time=1.0)
        for i in (1, 2):
            self.play(self.frame.animate.move_to(groups[i].get_center() + DOWN * 0.7),
                      FadeIn(lights[i]), FadeIn(groups[i], shift=UP * 0.15), run_time=1.6,
                      rate_func=ease_in_out_cubic)
            self.wait(0.6)
        self.play(self.frame.animate.set(width=FW0).move_to(DOWN * 0.1), FadeIn(rules),
                  *self.cap("We built a software emulator that measures all three, from the signal up."),
                  run_time=1.6, rate_func=ease_in_out_cubic)
        self.drift(rate=0.007, focus=UP * 0.45)
        self.wait(5.0)
        self.close()

    def ch_bandwidth(self):
        CH, FS, BITS = CFG.n_channels, CFG.fs, CFG.adc_bits
        lane_y = [2.45 - k * 0.125 for k in range(CH)]
        lanes = VGroup(*[DashedLine([X0, y, 0], [X0 + RAW_LEN, y, 0], dash_length=RAW_LEN / FS * 0.45,
                                    dashed_ratio=0.45, stroke_color=MUTED, stroke_width=2.5) for y in lane_y])
        head = words(f"{CH} channels × {FS} samples/s × {BITS} bits", 30, MUTED).move_to([X0, 3.2, 0], aligned_edge=LEFT)
        v = ValueTracker(0)
        count = always_redraw(lambda: mono(f"{int(round(v.get_value())):,} b/s", 84, WHITE)
                              .move_to([X0, -0.75, 0], aligned_edge=LEFT))
        clock = always_redraw(lambda: mono(f"t = {v.get_value() / RAW_BPS:.2f} s", 26, MUTED)
                              .move_to([X0 + RAW_LEN, -0.75, 0], aligned_edge=RIGHT))

        self.frame.set(width=FW0 * 1.04)
        self.play(*self.tag("03 · BANDWIDTH"), *self.cap(
            "Sixteen EEG channels, 256 samples a second, 9 bits each: 36,864 bits every second."),
            FadeIn(head), FadeIn(count), FadeIn(clock), run_time=0.9)
        self.wait(0.3)
        t0 = self.now
        self.cue("ticks", anchor=t0, start=t0 - 1.0, end=t0 + 3.05)
        self.play(Create(lanes), v.animate.set_value(RAW_BPS), run_time=1.0, rate_func=linear)
        self.drift(rate=0.01, focus=[X0 + 3, 0.6, 0])
        self.wait(2.4)

        # ---- the conduit
        pipe = conduit()
        pipe_lab = VGroup(words("body link (EQS-HBC)", 26, GREEN), mono(f"{LINK_BPS:,} b/s · 415 nW", 20, GREEN)) \
            .arrange(DOWN, aligned_edge=LEFT, buff=0.06).move_to([X0, PIPE_Y + 0.62, 0], aligned_edge=DOWN + LEFT)
        raw = stream_bar(RAW_BPS, MUTED)
        raw_glow = halo(raw, MUTED, 0.6)
        end_x = X0 + LINK_LEN
        self.hold()
        self.clear_content(run_time=0.5)
        self.frame.set(width=FW0 * 1.0).move_to(ORIGIN)
        self.play(FadeIn(pipe), FadeIn(pipe_lab), *self.cap(
            "The body link carries 10,000 b/s. Streaming the raw signal would need 3.69 times that."),
            self.frame.animate.set(width=7.0).move_to([end_x - 0.6, PIPE_Y - 0.3, 0]),
            run_time=1.4, rate_func=ease_in_out_cubic)
        part = LINK_LEN / RAW_LEN
        grow = raw.copy().stretch(1e-3, 0, about_edge=LEFT)
        self.add(grow)
        inside = raw.copy().stretch(part, 0, about_edge=LEFT)
        self.play(Transform(grow, inside), run_time=0.9, rate_func=linear)
        self.cue("hit", strength=1.0)
        spill = self.overflow(end_x, PIPE_Y)
        self.play(Transform(grow, raw, rate_func=ease_out_cubic), FadeIn(raw_glow),
                  pipe[1].animate(rate_func=there_and_back).set_stroke(RED, width=5),
                  Flash([end_x + 0.05, PIPE_Y, 0], color=RED, line_length=0.35, num_lines=14, flash_radius=0.55,
                        line_stroke_width=3), spill,
                  self.frame.animate(rate_func=ease_in_out_cubic).set(width=FW0).move_to(ORIGIN),
                  run_time=2.2)
        over = words(f"raw: {RAW_BPS:,} b/s = {RAW_BPS / LINK_BPS:.2f}× the link", 26, RED) \
            .move_to([X0 + RAW_LEN, PIPE_Y + 0.75, 0], aligned_edge=RIGHT)
        mark = DashedLine([end_x, PIPE_Y - 0.9, 0], [end_x, PIPE_Y + 0.62, 0], dash_length=0.07,
                          stroke_color=RED, stroke_width=2)
        cap_lab = words("capacity", 20, GREEN).next_to(mark, DOWN, buff=0.08)
        self.play(FadeIn(over), Create(mark), FadeIn(cap_lab), run_time=0.7)
        self.drift(rate=0.008, focus=[end_x, PIPE_Y, 0])
        self.wait(3.2)
        self.close()

    def overflow(self, x, y, n=16, seed=3):
        """Red chevrons and sparks thrown off where the stream leaves the conduit."""
        rng = np.random.default_rng(seed)
        anims = []
        for k in range(n):
            ang = rng.uniform(-1.1, 1.1)
            d = np.array([np.cos(ang), np.sin(ang), 0]) * rng.uniform(0.8, 2.0)
            if k % 2:
                m = Text("›", font=MONO, font_size=34, color=RED)
            else:
                m = Circle(radius=0.035, stroke_width=0).set_fill(RED, 1)
            m.move_to([x + 0.1, y + rng.uniform(-0.25, 0.25), 0])
            anims.append(m.animate(rate_func=ease_out_cubic).shift(d).set_opacity(0))
            self.add(m)
        return LaggedStart(*anims, lag_ratio=0.06)

    def ch_noise(self):
        x0, x1, ymid, ys = -6.2, 6.2, 0.45, 1.95 / 130.0
        assert -130 < float(INP.min()) and float(INP.max()) < 130

        def trace(y, color, strength=1.0):
            pts = [[x0 + (x1 - x0) * ti, ymid + yi * ys, 0] for ti, yi in zip(T, y)]
            return glow_path(VMobject().set_points_as_corners(pts), color, 2.5, strength)

        cx = x0 + (x1 - x0) * (_bc.clench_on_s + _bc.clench_off_s) / 2
        cw = (x1 - x0) * (_bc.clench_off_s - _bc.clench_on_s)
        lit = VGroup(Rectangle(width=cw + 0.5, height=4.6, stroke_width=0).set_fill(AMBER, 0.03),
                     Rectangle(width=cw, height=4.3, stroke_width=0).set_fill(AMBER, 0.07),
                     *[Line([cx + s * cw / 2, ymid - 2.15, 0], [cx + s * cw / 2, ymid + 2.15, 0],
                            stroke_color=AMBER, stroke_width=1.5, stroke_opacity=0.55) for s in (-1, 1)]) \
            .move_to([cx, ymid, 0])
        lit_lab = words("jaw clench", 22, AMBER).move_to([cx, ymid + 1.9, 0])
        base = Line([x0, ymid - 2.3, 0], [x1, ymid - 2.3, 0], color=EDGE, stroke_width=2)
        ticks = VGroup(*[words(f"{s:g} s", 18, DIM).move_to([x0 + (x1 - x0) * s, ymid - 2.55, 0])
                         for s in (0, 0.25, 0.5, 0.75, 1.0)])
        raw, truth, dcd = trace(INP, MUTED, 0.7), trace(TRUTH, WHITE), trace(DCD, BLUE2)
        leg = VGroup(words("electrode, after the 9-bit ADC", 24, MUTED),
                     words("true brain signal (known in the simulation)", 24, WHITE),
                     words("RLS-DCD output", 24, BLUE2)).arrange(DOWN, aligned_edge=LEFT, buff=0.12) \
            .move_to([-6.6, 3.35, 0], aligned_edge=UL)
        snr = VGroup(words("SNR during the clench", 24, MUTED),
                     mono(f"{SNR_IN:+.1f} dB  →  {SNR_DCD:+.1f} dB".replace("-", "−"), 40, BLUE2)) \
            .arrange(DOWN, aligned_edge=RIGHT, buff=0.1).move_to([6.6, 3.35, 0], aligned_edge=UP + RIGHT)

        self.frame.set(width=FW0 * 1.05)
        self.drift(rate=0.006)
        self.play(*self.tag("04 · NOISE"), *self.cap(
            "The signal is also buried: a jaw clench drowns the brain signal in muscle noise (EMG)."),
            FadeIn(base), FadeIn(ticks), FadeIn(lit), FadeIn(lit_lab), run_time=0.8)
        self.play(FadeIn(leg[0]), Create(raw), run_time=2.4, rate_func=linear)
        self.wait(1.6)
        self.say("In the simulation the true brain signal is known, so every result is scored against it.")
        self.play(FadeIn(leg[1]), Create(truth), run_time=2.0, rate_func=linear)
        self.wait(1.0)
        self.hold()
        self.play(*[layer.animate.set_stroke(opacity=0) for layer in raw[:-1]],
                  raw[-1].animate.set_stroke(opacity=0.3), *self.cap(
            "BDN-04 cancels the EMG adaptively, using a reference electrode on the jaw muscle."),
            self.frame.animate.set(width=10.0).move_to([cx, ymid + 0.3, 0]),
            run_time=1.5, rate_func=ease_in_out_cubic)
        self.play(FadeIn(leg[2]), Create(dcd), run_time=2.6, rate_func=linear)
        self.play(self.frame.animate.set(width=FW0).move_to(ORIGIN), FadeIn(snr, shift=DOWN * 0.1),
                  *self.cap("Signal-to-noise during the clench rises from −12.6 dB to +4.6 dB."),
                  run_time=1.4, rate_func=ease_in_out_cubic)
        self.drift(rate=0.007, focus=snr.get_center())
        self.wait(3.6)
        self.close()

    def ch_chain(self):
        spec = [("9-bit ADC", "256 samples/s\n16 channels", False),
                ("DC blocker", "2 stages, 1.3 Hz\nshift + add only", False),
                ("Reference gate", "adapt only when\nthe reference is active", False),
                ("EMG canceller", "RLS-DCD, 8 taps\nfixed point", True),
                ("EOG canceller", "1 tap, for blinks\n(optional)", False),
                ("Clean EEG", "ready for the\nclassifier", False)]
        boxes = VGroup(*[box(a, b, accent=c) for a, b, c in spec]).arrange(RIGHT, buff=0.33).move_to(UP * 1.15)
        arrows = VGroup(*[link_arrow(boxes[i], boxes[i + 1]) for i in range(5)])
        refs = VGroup()
        for i, lab in ((3, "jaw-muscle\nreference"), (4, "frontal\nreference")):
            t = VGroup(*[words(line, 21, MUTED) for line in lab.split("\n")]).arrange(DOWN, buff=0.06) \
                .next_to(boxes[i], DOWN, buff=1.0)
            refs.add(VGroup(t, Arrow(t.get_top(), boxes[i].get_bottom(), buff=0.1, stroke_width=3, color=DIM,
                                     tip_length=0.14)))
        cost = words("drift removal: no multiplies   ·   EMG cancelling: 3M = 24 multiplies per sample", 24, SOFT) \
            .move_to(DOWN * 1.75)

        self.frame.set(width=7.0).move_to(boxes[0].get_center() + RIGHT * 1.9 + DOWN * 0.9)
        self.play(*self.tag("04 · NOISE"), *self.cap(
            "The full chain: every stage is designed for low-power, fixed-point hardware."),
            FadeIn(boxes[0], shift=RIGHT * 0.2), run_time=0.7)
        steps = [LaggedStart(Create(arrows[i - 1]), FadeIn(boxes[i], shift=RIGHT * 0.2), lag_ratio=0.4)
                 for i in range(1, 6)]
        self.play(LaggedStart(*steps, lag_ratio=0.75),
                  self.frame.animate.move_to(boxes[5].get_center() + LEFT * 1.9 + DOWN * 0.9),
                  run_time=4.2, rate_func=ease_in_out_sine)
        self.play(self.frame.animate.set(width=FW0).move_to(ORIGIN),
                  LaggedStart(*[FadeIn(r, shift=UP * 0.15) for r in refs], lag_ratio=0.4),
                  *self.cap("A gate freezes the canceller when its reference goes quiet, "
                            "so it can't distort clean EEG."),
                  run_time=1.6, rate_func=ease_in_out_cubic)
        self.play(FadeIn(cost), run_time=0.6)
        self.drift(rate=0.007, focus=boxes[3].get_center())
        self.wait(5.4)
        self.close()

    def ch_decision(self):
        pipe = conduit()
        pipe_lab = words(f"body link: {LINK_BPS:,} b/s", 24, GREEN).move_to([X0, PIPE_Y + 0.85, 0], aligned_edge=LEFT)
        raw = stream_bar(RAW_BPS, MUTED).set_opacity(0.35)
        raw_lab = words(f"raw: {RAW_BPS:,} b/s", 22, MUTED).move_to([X0 + RAW_LEN, PIPE_Y + 0.85, 0], aligned_edge=RIGHT)
        clean = stream_bar(RAW_BPS, SOFT, h=0.22)
        end_x = X0 + LINK_LEN

        self.frame.set(width=FW0 * 1.04)
        self.drift(rate=0.005)
        self.play(*self.tag("05 · THE DECISION"), *self.cap(
            "But cleaning doesn't shrink the data: a clean stream is still 36,864 b/s."),
            FadeIn(pipe), FadeIn(pipe_lab), FadeIn(raw), FadeIn(raw_lab), run_time=0.9)
        grow = clean.copy().stretch(1e-3, 0, about_edge=LEFT)
        self.add(grow)
        self.play(Transform(grow, clean.copy().stretch(LINK_LEN / RAW_LEN, 0, about_edge=LEFT)),
                  run_time=0.7, rate_func=linear)
        self.cue("hit", strength=0.7)
        self.play(Transform(grow, clean), Flash([end_x + 0.05, PIPE_Y, 0], color=RED, line_length=0.3,
                                                 num_lines=12, flash_radius=0.5, line_stroke_width=3),
                  self.overflow(end_x, PIPE_Y, n=10, seed=7), run_time=1.6, rate_func=ease_out_cubic)
        clean_lab = words(f"after cleaning: {RAW_BPS:,} b/s (unchanged)", 22, SOFT) \
            .move_to([X0 + RAW_LEN, PIPE_Y - 0.62, 0], aligned_edge=RIGHT)
        self.play(FadeIn(clean_lab), run_time=0.5)
        self.wait(2.0)

        # ---- decide on the body: the stream collapses to a sliver
        sliver = stream_bar(_bw["link_bps"], BLUE, h=0.4)
        sliver_g = VGroup(halo(sliver, BLUE, 1.4), sliver)
        sl_lab = VGroup(mono(f"{_bw['link_bps']:.0f} b/s", 15, BLUE2),
                        words(f"{100 * _bw['link_util']:.2f}% of the link", 12, SOFT)) \
            .arrange(DOWN, aligned_edge=LEFT, buff=0.04).move_to([X0 + 0.12, PIPE_Y + 0.05, 0], aligned_edge=LEFT)
        self.hold()
        self.play(*self.cap("So the node decides on the body: CSP + LDA turns each second of EEG into one label."),
                  Transform(grow, sliver), raw.animate.stretch(1e-3, 0, about_edge=LEFT).set_opacity(0),
                  FadeOut(clean_lab), FadeOut(raw_lab), run_time=2.0, rate_func=ease_in_out_cubic)
        self.remove(raw)
        self.play(FadeIn(sliver_g), FadeOut(pipe_lab), self.frame.animate.set(width=2.6).move_to([X0 + 0.75, PIPE_Y, 0]),
                  run_time=1.6, rate_func=ease_in_out_cubic)
        self.cue("hit", strength=0.35)
        self.play(FadeIn(sl_lab), run_time=0.5)
        self.wait(1.6)

        # ---- the frame that carries it
        bits = [(k, v) for k, v in FRAME.items()]
        fx0, fw, fy = -5.6, 11.2, -1.25
        cells, labels, x = VGroup(), VGroup(), fx0
        for name, nb in bits:
            w = fw * nb / FRAME_BITS
            is_label = name.lower().startswith("label")
            r = Rectangle(width=w, height=0.62, stroke_color=BG, stroke_width=3,
                          fill_color=BLUE if is_label else "#1c2230", fill_opacity=1).move_to([x + w / 2, fy, 0])
            if not is_label:
                r.set_stroke(EDGE, 2)
            nm = name.replace(" (7 octets)", "").replace("FCS (CRC-16)", "CRC-16")
            lab = VGroup(words(nm, 17, BLUE2 if is_label else MUTED), mono(f"{nb} b", 15, WHITE if is_label else DIM)) \
                .arrange(DOWN, buff=0.04).next_to(r, DOWN, buff=0.1)
            if lab.width > w + 0.6:
                lab.scale_to_fit_width(w + 0.6)
            cells.add(r)
            labels.add(lab)
            x += w
        ftitle = words(f"one {FRAME_BITS}-bit frame per second", 22, SOFT).next_to(cells, UP, buff=0.18) \
            .align_to(cells, LEFT)
        big_lab = VGroup(mono(f"{_bw['link_bps']:.0f} b/s", 30, BLUE2),
                         words(f"{100 * _bw['link_util']:.2f}% of the link", 22, SOFT)) \
            .arrange(DOWN, aligned_edge=LEFT, buff=0.06).move_to([X0 + LINK_LEN + 0.45, PIPE_Y, 0], aligned_edge=LEFT)
        self.play(self.frame.animate.set(width=FW0).move_to(ORIGIN), FadeOut(sl_lab), FadeIn(big_lab), FadeIn(pipe_lab),
                  *self.cap("One 112-bit frame a second: 112 b/s, 1.12% of the link. "
                            "Eight of those bits are the decision."),
                  run_time=1.6, rate_func=ease_in_out_cubic)
        self.play(LaggedStart(*[FadeIn(c, shift=UP * 0.1) for c in cells], lag_ratio=0.2), FadeIn(labels),
                  FadeIn(ftitle), run_time=1.2)
        self.play(Create(halo(cells[3], BLUE, 1.2)), run_time=0.6)
        self.drift(rate=0.007, focus=cells[3].get_center())
        self.wait(3.6)
        self.close()

    def ch_arith(self):
        # ---- A: the textbook square
        f1 = MarkupText("4M<sup>2</sup> + 4M + 1", font=MONO, font_size=50, color=WHITE) \
            .move_to([-6.4, 2.85, 0], aligned_edge=LEFT)
        f2 = MarkupText("(2M + 1)<sup>2</sup>", font=MONO, font_size=50, color=WHITE).move_to(f1, aligned_edge=LEFT)
        f3 = MarkupText("3M", font=MONO, font_size=50, color=BLUE2).move_to(f1, aligned_edge=LEFT)
        mlab = words("M = 8 taps", 26, MUTED).next_to(f1, DOWN, buff=0.22).align_to(f1, LEFT)
        n = ValueTracker(0)
        cnt = always_redraw(lambda: mono(f"{int(round(n.get_value()))}", 110,
                                         BLUE2 if n.get_value() < 100 and n.get_value() > 0.5 and self._collapsed
                                         else WHITE).move_to([-6.4, 0.55, 0], aligned_edge=LEFT))
        cnt_lab = words("multiplies per sample", 26, MUTED).move_to([-6.4, -0.45, 0], aligned_edge=LEFT)
        self._collapsed = False
        cell, gap = 0.24, 0.045
        pitch = cell + gap
        gc = np.array([2.6, 0.55, 0])
        cells = VGroup()
        for i in range(SIDE):
            for j in range(SIDE):
                cells.add(Square(cell, stroke_width=0).set_fill(MUTED, 0.9)
                          .move_to(gc + [(j - (SIDE - 1) / 2) * pitch, ((SIDE - 1) / 2 - i) * pitch, 0]))
        sq_frame = Square(SIDE * pitch + 0.12).move_to(gc).set_stroke(MUTED, 1.5, opacity=0.6)

        self.frame.set(width=FW0 * 1.03)
        self.drift(rate=0.008, focus=gc)
        self.play(*self.tag("06 · ARITHMETIC"), *self.cap(
            "Cleaning also costs arithmetic. Textbook RLS needs (2M + 1)² multiplies per sample: "
            "289 at 8 taps."), FadeIn(f1), FadeIn(mlab), FadeIn(cnt), FadeIn(cnt_lab), run_time=0.8)
        rows = [VGroup(*cells[i * SIDE:(i + 1) * SIDE]) for i in range(SIDE)]
        self.play(LaggedStart(*[FadeIn(r, shift=DOWN * 0.05) for r in rows], lag_ratio=0.15),
                  n.animate.set_value(RLS_MULS), run_time=2.0, rate_func=linear)
        self.play(FadeTransform(f1, f2), FadeIn(sq_frame), run_time=1.0)
        order = sorted(range(len(cells)), key=lambda k: (k // SIDE) + (k % SIDE))
        self.cue("swell", dur=1.8)
        self.play(LaggedStart(*[cells[k].animate(rate_func=there_and_back).set_fill(WHITE, 1) for k in order],
                              lag_ratio=0.006), sq_frame.animate(rate_func=there_and_back).set_stroke(WHITE, 4, 1),
                  run_time=1.8)
        self.wait(0.6)

        # ---- B: the collapse into the 3 x 8 strip
        sc, sg = 0.46, 0.08
        sp = sc + sg
        stc = np.array([1.2, 0.55, 0])
        targets = [stc + [(c - 3.5) * sp, (1 - r) * sp, 0] for r in range(3) for c in range(8)]
        keep = [cells[k] for k in np.linspace(0, len(cells) - 1, DCD_MULS).astype(int)]
        rest = [c for c in cells if all(c is not k for k in keep)]
        row_labs = VGroup(*[words(t, 24, SOFT).move_to(stc + [4 * sp + 0.3, (1 - r) * sp, 0], aligned_edge=LEFT)
                            for r, t in enumerate(["new column of R", "output  y = W·x", "update  e·x"])])
        self.hold()
        self.say("Transversal RLS-DCD solves the same problem with 3M = 24.")
        self._collapsed = True
        self.cue("collapse", dur=2.4)
        self.play(*[c.animate.move_to(t).scale(sc / cell).set_fill(BLUE, 1) for c, t in zip(keep, targets)],
                  *[c.animate.move_to(stc).scale(0.05).set_fill(opacity=0) for c in rest],
                  FadeOut(sq_frame), n.animate.set_value(DCD_MULS), FadeTransform(f2, f3),
                  self.frame.animate.set(width=FW0).move_to([0.3, 0, 0]),
                  run_time=2.4, rate_func=ease_in_out_cubic)
        self.remove(*rest)
        strip_h = halo(SurroundingRect(VGroup(*keep), 0.1), BLUE, 1.2)
        self.play(FadeIn(strip_h), LaggedStart(*[FadeIn(r, shift=LEFT * 0.1) for r in row_labs], lag_ratio=0.3),
                  run_time=1.0)
        self.drift(rate=0.008, focus=stc)
        self.wait(2.8)
        self.hold()
        self.clear_content(0.6)
        self.frame.move_to(ORIGIN).set(width=FW0)

        # ---- C: halving is a shift
        bw, bh = 0.72, 0.9
        regs = VGroup(*[RoundedRectangle(corner_radius=0.06, width=bw, height=bh, stroke_color=EDGE,
                                         stroke_width=2, fill_color=PANEL, fill_opacity=1) for _ in range(8)]) \
            .arrange(RIGHT, buff=0.1).move_to(UP * 0.9)

        def digits(v):
            return VGroup(*[mono(b, 44, WHITE).move_to(regs[i]) for i, b in enumerate(f"{v:08b}")])

        val = 104
        dg = digits(val)
        vlab = mono(f"{val}", 44, MUTED).next_to(regs, DOWN, buff=0.45)
        op = mono(">> 1", 40, BLUE2).next_to(regs, UP, buff=0.45)
        ident = MarkupText("× 2<sup>−m</sup>   ≡   &gt;&gt; m", font=MONO, font_size=52, color=WHITE) \
            .move_to(DOWN * 1.35)
        self.play(*self.cap("Its step sizes are powers of two, and multiplying by a power of two is just a bit shift."),
                  FadeIn(regs), FadeIn(dg), FadeIn(vlab), run_time=0.8)
        self.drift(rate=0.01, focus=regs.get_center())
        self.play(FadeIn(op, shift=DOWN * 0.1), run_time=0.5)
        for _ in range(2):
            new = val >> 1
            nd = digits(new)
            moved = [dg[i].animate.move_to(regs[i + 1]) for i in range(7)]
            self.cue("shift")
            self.play(*moved, dg[7].animate.shift(DOWN * 0.6).set_opacity(0), FadeIn(nd[0], shift=RIGHT * 0.3),
                      Transform(vlab, mono(f"{new}", 44, MUTED).move_to(vlab)), run_time=0.9,
                      rate_func=ease_in_out_cubic)
            self.remove(*dg, nd[0])
            dg = nd
            self.add(dg)
            val = new
            self.wait(0.5)
        self.play(FadeIn(ident, shift=UP * 0.1), run_time=0.8)
        self.wait(2.2)
        self.hold()
        self.clear_content(0.6)
        self.frame.move_to(ORIGIN).set(width=FW0)

        # ---- D: at 32 taps
        side = 4.0
        bc = np.array([-3.4, 0.75, 0])
        big = Square(side, stroke_width=0).set_fill(MUTED, 0.85).move_to(bc)
        k = 2 * M_BIG + 1
        gl = VGroup(*[Line(bc + [-side / 2 + side * i / k, -side / 2, 0], bc + [-side / 2 + side * i / k, side / 2, 0])
                      for i in range(1, k)],
                    *[Line(bc + [-side / 2, -side / 2 + side * i / k, 0], bc + [side / 2, -side / 2 + side * i / k, 0])
                      for i in range(1, k)]).set_stroke(BG, 0.8)
        big_lab = VGroup(MarkupText(f"(2·{M_BIG} + 1)<sup>2</sup> = {RLS_BIG:,}", font=MONO, font_size=34, color=WHITE),
                         words("textbook RLS", 22, MUTED)).arrange(DOWN, aligned_edge=LEFT, buff=0.08) \
            .next_to(big, DOWN, buff=0.2).align_to(big, LEFT)
        m32 = words(f"M = {M_BIG} taps", 26, MUTED).next_to(big, UP, buff=0.2).align_to(big, LEFT)
        c2, g2 = 0.1, 0.025
        stc2 = np.array([3.2, 2.0, 0])
        strip = VGroup(*[Square(c2, stroke_width=0).set_fill(BLUE, 1)
                         .move_to(stc2 + [(c - 15.5) * (c2 + g2), (1 - r) * (c2 + g2), 0])
                         for r in range(3) for c in range(M_BIG)])
        s_lab = VGroup(mono(f"3·{M_BIG} = {DCD_BIG}", 34, BLUE2), words("RLS-DCD", 22, MUTED)) \
            .arrange(DOWN, aligned_edge=LEFT, buff=0.08).next_to(strip, DOWN, buff=0.25).align_to(strip, LEFT)
        fewer = words(f"{RLS_BIG / DCD_BIG:.0f}× fewer multiplies", 38, WHITE, weight="SEMIBOLD") \
            .next_to(s_lab, DOWN, buff=0.7).align_to(strip, LEFT)
        self.play(*self.cap("At 32 taps the gap is 4,225 versus 96: 44 times fewer multiplies."),
                  FadeIn(m32), FadeIn(big, scale=0.9), FadeIn(gl), run_time=1.2, rate_func=ease_out_cubic)
        self.play(FadeIn(big_lab), LaggedStart(*[FadeIn(s) for s in strip], lag_ratio=0.01),
                  run_time=1.2)
        self.play(FadeIn(s_lab), Create(halo(SurroundingRect(strip, 0.06), BLUE, 1.2)),
                  self.frame.animate.set(width=6.2).move_to(stc2 + DOWN * 0.9), run_time=1.4,
                  rate_func=ease_in_out_cubic)
        self.wait(0.6)
        self.play(self.frame.animate.set(width=FW0).move_to(ORIGIN), FadeIn(fewer, shift=UP * 0.1),
                  run_time=1.4, rate_func=ease_in_out_cubic)
        self.drift(rate=0.007)
        self.wait(2.6)
        self.close()

    def ch_power(self):
        X, U, h = -3.6, 3.3, 0.55

        def hbar(v, y, color, x=X, op=1.0):
            return Rectangle(width=v * U, height=h, stroke_width=0, fill_color=color, fill_opacity=op) \
                .move_to([x + v * U / 2, y, 0])

        def grow(b, rt, **kw):
            s = b.copy().stretch(1e-3, 0, about_edge=LEFT)
            self.add(s)
            return s, Transform(s, b, run_time=rt, **kw)

        head1 = words("Filtering power, 16 channels (90 nm energy model)", 28, MUTED).move_to([X - 3.0, 3.25, 0], aligned_edge=LEFT)
        y1, y2 = 2.3, 1.45
        l1 = words("textbook RLS", 28, SOFT).move_to([X - 0.25, y1, 0], aligned_edge=RIGHT)
        l2 = words("RLS-DCD", 28, BLUE2, weight="SEMIBOLD").move_to([X - 0.25, y2, 0], aligned_edge=RIGHT)
        b1, b2 = hbar(P_RLS, y1, MUTED), hbar(P_DCD, y2, BLUE)
        v1 = mono(f"{P_RLS:.2f} µW", 30, WHITE).next_to(b1, RIGHT, buff=0.2)
        v2 = mono(f"{P_DCD:.2f} µW   −{(1 - P_DCD / P_RLS) * 100:.0f}%", 30, BLUE2).next_to(b2, RIGHT, buff=0.2)
        q = words(f"cleaning quality almost unchanged: {SNR_RLS:.1f} dB vs {SNR_DCD:.1f} dB SNR", 24, MUTED) \
            .move_to([X - 3.0, 0.75, 0], aligned_edge=LEFT)
        head2 = words("Whole node, RLS-DCD path", 28, MUTED).move_to([X - 3.0, -0.3, 0], aligned_edge=LEFT)
        y3 = -1.2
        parts = [("static", _node["static_uw"], MUTED), ("filtering", _node["filtering_dyn_uw"], BLUE),
                 ("CSP + LDA", _node["csp_lda_uw"], VIOLET)]
        segs, labs, x = [], VGroup(), X
        for name, v, c in parts:
            s = hbar(v, y3, c, x=x)
            segs.append(s)
            lab = words(f"{name}\n{v:.2f}", 20, BLUE2 if name == "filtering" else MUTED)
            labs.add(lab.next_to(s, UP if name == "filtering" else DOWN, buff=0.12))
            x += v * U
        link_lo = hbar(_node["link_low_uw"], y3, GREEN, x=x)
        link_rng = hbar(_node["link_high_uw"], y3, GREEN, x=x, op=0.28)
        labs.add(words("radio\n0.005–0.415", 20, GREEN).next_to(link_rng, DOWN, buff=0.12))
        tot = mono(f"{_node['total_low_uw']:.2f}–{_node['total_high_uw']:.2f} µW", 34, WHITE).next_to(link_rng, RIGHT, buff=0.25)
        refx = X + REF_UW * U
        ref = DashedLine([refx, y3 + 0.6, 0], [refx, y3 - 0.6, 0], dash_length=0.08, color=WHITE, stroke_width=2)
        reflab = words(f"published 90 nm node: {REF_UW} µW", 20, SOFT).next_to(ref, UP, buff=0.08)
        l3 = words("total", 28, SOFT).move_to([X - 0.25, y3, 0], aligned_edge=RIGHT)

        self.frame.set(width=13.4).move_to([-0.1, 1.2, 0])
        self.drift(rate=0.004, focus=[-0.6, 1.8, 0])
        self.play(*self.tag("07 · POWER"), *self.cap(
            "In a 90 nm energy model, filtering power falls from 2.28 µW to 0.74 µW, "
            "with almost the same cleaning quality."), FadeIn(head1), FadeIn(l1), FadeIn(l2), run_time=0.7)
        g1, a1 = grow(b1, 1.1)
        self.play(a1)
        self.play(FadeIn(v1), run_time=0.3)
        g2, a2 = grow(b2, 0.7)
        self.play(a2, FadeIn(halo(b2, BLUE, 1.0)))
        self.play(FadeIn(v2), run_time=0.4)
        self.play(FadeIn(q), run_time=0.5)
        self.wait(2.6)
        self.hold()
        self.play(self.frame.animate.set(width=FW0).move_to([0, -0.15, 0]), FadeIn(head2), FadeIn(l3),
                  *self.cap("The whole node, classifier and radio included, needs 1.14 to 1.55 µW: "
                            "close to the 1.41 µW of a published 90 nm design."),
                  run_time=1.4, rate_func=ease_in_out_cubic)
        for s in segs:
            _, a = grow(s, 0.45)
            self.play(a)
        self.add(link_lo)
        _, a = grow(link_rng, 0.6)
        self.play(a)
        self.play(FadeIn(labs), FadeIn(tot), run_time=0.6)
        self.play(Create(ref), FadeIn(reflab), run_time=0.6)
        self.drift(rate=0.008, focus=[refx, y3, 0])
        self.wait(4.4)
        self.close()

    def ch_latency(self):
        x0, x1, ty = -6.3, 3.9, 2.25
        k = (x1 - x0) / LAT_TOTAL
        shades = [GREEN, GREEN2, GREEN3]
        track = RoundedRectangle(corner_radius=0.08, width=x1 - x0 + 0.1, height=0.58, stroke_color=EDGE,
                                 stroke_width=2, fill_color=PANEL, fill_opacity=1).move_to([(x0 + x1) / 2, ty, 0])
        win = Rectangle(width=WINDOW_MS * k, height=0.48, stroke_width=0, fill_color=MUTED, fill_opacity=1) \
            .move_to([x0 + WINDOW_MS * k / 2, ty, 0])
        tail = VGroup(*[Rectangle(width=s["dur_ms"] * k, height=0.48, stroke_width=0, fill_color=shades[i], fill_opacity=1)
                        .move_to([x0 + (s["start_ms"] + s["dur_ms"] / 2) * k, ty, 0]) for i, s in enumerate(LAT[1:])])
        wlab = words("decoder window: one full second of brain activity", 24, SOFT) \
            .move_to([x0, ty + 0.62, 0], aligned_edge=LEFT)
        axis = VGroup(*[words(f"{ms:,} ms", 17, DIM).move_to([x0 + ms * k, ty - 0.52, 0]) for ms in (0, 250, 500, 750, 1000)])
        ms = ValueTracker(0)
        readout = always_redraw(lambda: mono(f"{ms.get_value():,.1f} ms", 34, WHITE)
                                .move_to([x1 + 0.35, ty + 0.12, 0], aligned_edge=LEFT))
        rsub = words("from intent to muscle", 20, MUTED).move_to([x1 + 0.35, ty - 0.33, 0], aligned_edge=LEFT)

        # broken axis: 0-1000 ms compressed on the left, the 52.2 ms tail stretched on the right
        by, lx0, lx1, rx0, rx1 = -0.35, -6.3, -3.7, -3.0, 5.6
        k1, k2 = (lx1 - lx0) / WINDOW_MS, (rx1 - rx0) / TAIL_MS
        stretch = k2 / k1
        ltrack = RoundedRectangle(corner_radius=0.06, width=lx1 - lx0 + 0.08, height=0.62, stroke_color=EDGE, stroke_width=2,
                                  fill_color=PANEL, fill_opacity=1).move_to([(lx0 + lx1) / 2, by, 0])
        rtrack = RoundedRectangle(corner_radius=0.06, width=rx1 - rx0 + 0.08, height=0.62, stroke_color=EDGE, stroke_width=2,
                                  fill_color=PANEL, fill_opacity=1).move_to([(rx0 + rx1) / 2, by, 0])
        brk = VGroup(*[Line([0, -0.22, 0], [0.16, 0.22, 0], stroke_color=SOFT, stroke_width=2.5).move_to([x, by, 0])
                       for x in (lx1 + 0.24, lx1 + 0.42)])
        lwin = Rectangle(width=lx1 - lx0, height=0.52, stroke_width=0, fill_color=MUTED, fill_opacity=1) \
            .move_to([(lx0 + lx1) / 2, by, 0])
        rseg, rlab = VGroup(), VGroup()
        names = {"CSP + LDA classification": "classify", "EQS-HBC frame transmission": "transmit",
                 "FES actuator delay": "stimulate"}
        for i, s in enumerate(LAT[1:]):
            st = s["start_ms"] - WINDOW_MS
            r = Rectangle(width=s["dur_ms"] * k2, height=0.52, stroke_width=4, stroke_color=PANEL,
                          fill_color=shades[i], fill_opacity=1).move_to([rx0 + (st + s["dur_ms"] / 2) * k2, by, 0])
            rseg.add(r)
            rlab.add(VGroup(words(names[s["stage"]], 22, SOFT), mono(f"{s['dur_ms']:g} ms", 22, shades[i]))
                     .arrange(DOWN, buff=0.05).next_to(r, DOWN, buff=0.16))
        btitle = words(f"broken axis: the hardware tail, stretched {stretch:.0f}×", 22, MUTED) \
            .move_to([lx0, by + 0.6, 0], aligned_edge=LEFT)
        bax = VGroup(words("0", 16, DIM).next_to(ltrack, DOWN, buff=0.1).align_to(ltrack, LEFT),
                     words("1,000 ms", 16, DIM).next_to(ltrack, DOWN, buff=0.1).align_to(ltrack, RIGHT),
                     words(f"{LAT_TOTAL:,.1f} ms", 16, DIM).move_to([rx1, by - 1.25, 0]))
        thesis = words("The wait is in the biology, not the silicon.", 44, WHITE, weight="SEMIBOLD").move_to(DOWN * 2.75)

        c0 = self.now
        self.frame.set(width=FW0 * 1.03)
        self.drift(rate=0.004)
        self.play(*self.tag("08 · LATENCY"), *self.cap(
            "From intent to muscle, the decoder first reads one full second of brain activity."),
            FadeIn(track), FadeIn(wlab), FadeIn(axis), FadeIn(readout), FadeIn(rsub), run_time=1.0)
        self.play(FadeIn(ltrack), FadeIn(rtrack), FadeIn(brk), FadeIn(btitle), FadeIn(bax), run_time=0.8)
        self.wait(1.8)
        self.say("Then the hardware adds 52.2 ms: classify, transmit, stimulate. Listen for the gap.")
        self.wait(1.6)

        # ---- the charge, in real time: tick on 0 ms, tick on 1,000 ms, click 52.2 ms later
        t0 = self.now
        g_top = win.copy().stretch(1e-3, 0, about_edge=LEFT)
        g_bot = lwin.copy().stretch(1e-3, 0, about_edge=LEFT)
        self.add(g_top, g_bot)
        self.play(Transform(g_top, win), Transform(g_bot, lwin), ms.animate.set_value(WINDOW_MS),
                  run_time=WINDOW_MS / 1000, rate_func=linear)
        self.cue("flam", t=t0 + WINDOW_MS / 1000, gap=TAIL_MS / 1000)
        self.wait(1 / config.frame_rate)
        self.hold()
        ms.set_value(LAT_TOTAL)
        self.add(tail, rseg)
        self.play(self.frame.animate(rate_func=there_and_back).scale(0.975), Flash(tail.get_right(), color=GREEN,
                  line_length=0.25, num_lines=12, flash_radius=0.35, line_stroke_width=2.5), run_time=0.45)
        self.wait(1.2)
        self.play(self.frame.animate.set(width=9.2).move_to([1.3, by - 0.55, 0]),
                  LaggedStart(*[FadeIn(lab, shift=UP * 0.1) for lab in rlab], lag_ratio=0.35),
                  run_time=1.6, rate_func=ease_in_out_cubic)
        self.wait(2.6)
        self.play(self.frame.animate.set(width=FW0).move_to(ORIGIN), *self.cap(None),
                  self._tag.alpha.animate.set_value(0), self._band.alpha.animate.set_value(0),
                  run_time=1.3, rate_func=ease_in_out_cubic)
        self.play(FadeIn(thesis, shift=UP * 0.1), run_time=0.9)
        self.drift(rate=0.006, focus=thesis.get_center())
        self.wait(3.4)
        self.cue("ticks", anchor=t0, start=c0, end=self.now - 0.05)
        self.close()

    def ch_dashboard(self):
        img = ImageMobject(DASH_IMG)
        img.set(width=12.0)
        img.move_to([0, 0, 0])
        top = img.get_top()[1]
        upx = 12.0 / img.pixel_array.shape[1]

        def at(xp, yp):
            return np.array([-6.0 + xp * upx, top - yp * upx, 0])

        bezel = RoundedRectangle(corner_radius=0.1, width=12.16, height=img.height + 0.16, stroke_color=EDGE,
                                 stroke_width=2, fill_color=PANEL, fill_opacity=1).move_to(img)
        glow = halo(bezel, BLUE, 0.6)
        stops = [((960, 560), 13.4, "All of it runs live in a dashboard. Inject noise, change the filter, "
                                    "and every number recomputes."),
                 ((1110, 520), 9.6, "Operation counts are measured by instrumented filters, "
                                    "then checked against their formulas."),
                 ((1110, 1160), 10.4, "Signal quality is scored against the simulation's ground-truth EEG."),
                 ((1110, 1830), 10.4, "Bandwidth, frame layout and node power come from the same engine."),
                 ((1110, 2520), 10.4, "And the latency waterfall plays out in real time, on the browser's clock.")]

        def frame_for(p, w):
            c = at(*p)
            return c + DOWN * 0.55 * w / FW0

        (p0, w0, c0) = stops[0]
        self.frame.set(width=w0 + 0.6).move_to(frame_for(p0, w0 + 0.6))
        self.play(*self.tag("09 · THE DASHBOARD", band=0.94), *self.cap(c0),
                  FadeIn(glow), FadeIn(bezel), FadeIn(img), run_time=1.0)
        self.drift(rate=0.008, focus=at(*p0))
        self.wait(3.6)
        for p, w, c in stops[1:]:
            self.hold()
            self.play(self.frame.animate.set(width=w).move_to(frame_for(p, w)), *self.cap(c),
                      run_time=1.7, rate_func=ease_in_out_cubic)
            self.drift(rate=0.008, focus=at(*p))
            self.wait(3.4)
        self.close()

    def ch_built(self):
        head = words("What we built", 52, WHITE, weight="SEMIBOLD").move_to([-6.2, 3.0, 0], aligned_edge=LEFT)
        rows = [("bci_core.py", "signal engine: simulated EEG, 9-bit ADC, three adaptive filters"),
                ("app.py", "live telemetry dashboard: power, signal, network, latency"),
                ("models", "90 nm energy per operation, body link, classifier, latency"),
                ("checks", "every operation count asserted against its formula"),
                ("motion/", "this film's graphics, drawn from the engine's numbers"),
                ("docs", "engineering record and documentary blueprint")]
        items = VGroup()
        y = 1.85
        for key, val in rows:
            kt = mono(key, 26, BLUE2).move_to([-3.45, y, 0], aligned_edge=RIGHT)
            vt = words(val, 26, SOFT).move_to([-3.0, y, 0], aligned_edge=LEFT)
            assert vt.get_right()[0] < 6.8
            items.add(VGroup(kt, vt))
            y -= 0.66
        rule = Line([-3.22, 2.2, 0], [-3.22, y + 0.35, 0], stroke_color=EDGE, stroke_width=2)
        self.frame.set(width=FW0 * 0.94).move_to([-1.2, 1.2, 0])
        self.play(*self.tag("10 · WHAT WE BUILT"), *self.cap(
            "A complete, checkable emulator of the node. Every number in this film comes from that code."),
            FadeIn(head, shift=UP * 0.1), run_time=0.8)
        self.play(LaggedStart(*[FadeIn(i, shift=RIGHT * 0.15) for i in items], lag_ratio=0.35), Create(rule),
                  self.frame.animate.set(width=FW0).move_to([0, 0, 0]), run_time=3.2, rate_func=ease_in_out_sine)
        self.drift(rate=0.006, focus=[-1, 0.5, 0])
        self.wait(7.6)
        self.close()

    def ch_end(self):
        title = Text("BDN-04", font=MONO, font_size=120, color=WHITE).shift(UP * 0.9)
        light = bloom(title.get_center(), 3.2, BLUE, peak=0.14)
        rule = glow_path(Line(LEFT * 1.2, RIGHT * 1.2).next_to(title, DOWN, buff=0.3), BLUE, 5)
        note = words("Simulated signals and a 90 nm energy model: model outputs, not silicon measurements.", 26, MUTED) \
            .next_to(rule, DOWN, buff=0.55)
        by = words("Varun Bajpai", 34, SOFT).next_to(note, DOWN, buff=0.6)
        self.frame.set(width=FW0 * 0.98)
        self.drift(rate=-0.01)
        self.play(FadeIn(light), FadeIn(title), Create(rule), run_time=1.2)
        self.play(FadeIn(note), FadeIn(by), run_time=0.9)
        self.wait(4.6)
        self.close(run_time=1.4)


def SurroundingRect(mob, buff):
    return RoundedRectangle(corner_radius=0.06, width=mob.width + 2 * buff, height=mob.height + 2 * buff) \
        .move_to(mob)


CHAPTERS = ["ch_title", "ch_idea", "ch_constraints", "ch_bandwidth", "ch_noise", "ch_chain", "ch_decision",
            "ch_arith", "ch_power", "ch_latency", "ch_dashboard", "ch_built", "ch_end"]


def _chapter_scene(name, method):
    return type(name, (Film,), {"construct": lambda self: getattr(self, method)()})


C00_Title = _chapter_scene("C00_Title", "ch_title")
C01_Idea = _chapter_scene("C01_Idea", "ch_idea")
C02_Constraints = _chapter_scene("C02_Constraints", "ch_constraints")
C03_Bandwidth = _chapter_scene("C03_Bandwidth", "ch_bandwidth")
C04_Noise = _chapter_scene("C04_Noise", "ch_noise")
C04b_Chain = _chapter_scene("C04b_Chain", "ch_chain")
C05_Decision = _chapter_scene("C05_Decision", "ch_decision")
C06_Arithmetic = _chapter_scene("C06_Arithmetic", "ch_arith")
C07_Power = _chapter_scene("C07_Power", "ch_power")
C08_Latency = _chapter_scene("C08_Latency", "ch_latency")
C09_Dashboard = _chapter_scene("C09_Dashboard", "ch_dashboard")
C10_Built = _chapter_scene("C10_Built", "ch_built")
C11_End = _chapter_scene("C11_End", "ch_end")
SCENE_ORDER = ["C00_Title", "C01_Idea", "C02_Constraints", "C03_Bandwidth", "C04_Noise", "C04b_Chain",
               "C05_Decision", "C06_Arithmetic", "C07_Power", "C08_Latency", "C09_Dashboard", "C10_Built",
               "C11_End"]


class MasterExplainer(Film):
    """The whole film in one render; same chapters, same order, same cues (absolute times)."""

    def construct(self):
        for method in CHAPTERS:
            self.cue("chapter", name=method)
            getattr(self, method)()
