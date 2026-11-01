"""
bdn04_motion.py - motion graphics MG-1, MG-2 and MG-3 for the BDN-04 documentary.

Every number drawn here is computed from the same formulas the engine uses
(bci_core.py), and the asserts below stop the render if any of them drift.
Formulas are set as Pango text in Source Code Pro, so no LaTeX install is needed.

Scenes (one per cut point in the master script):
    MG1_AB        2:16  4M^2+4M+1 rewrites as (2M+1)^2; a 17 x 17 square of 289 cells builds
    MG1_CD        2:52  the square collapses into a 3 x 8 strip of 24 cells; row labels
    MG1_E               optional: the same comparison at M = 32 (4,225 vs 96)
    MG2_Shift     2:39  halving is a shift: 104 -> 52 -> 26
    MG3a_Stream   0:45  16 channels x 256 samples/s x 9 bits = 36,864 b/s in one second
    MG3b_Pipe     0:59  the 10,000 b/s link; the raw stream runs off the frame (3.69x)
    MG3c_Cleaned  1:27  after cleaning: the same 36,864 b/s
    MG3d_Decision 1:37  one decision per second in a 112-bit frame: 112 b/s

Render all at 1080p, 24 fps (matches the B-roll timeline):
    manim -r 1920,1080 --fps 24 -a bdn04_motion.py
One scene, quick preview:
    manim -ql bdn04_motion.py MG1_AB

Requirements: manim (Community Edition) 0.18+, and the fonts Source Sans 3 and
Source Code Pro installed (both SIL OFL; npm packages "source-sans" and
"source-code-pro" contain the TTF files). Missing fonts fall back silently.
"""
from manim import (
    DOWN, LEFT, RIGHT, UP, UL, DL,
    Create, FadeIn, FadeOut, GrowFromEdge, LaggedStart, Line, MarkupText,
    Rectangle, Scene, Square, Text, Transform,
    TransformMatchingShapes, ValueTracker, VGroup, DashedLine, always_redraw,
    config, linear, smooth,
)

# ---------------------------------------------------------------------------
# Palette and type: the dashboard's own
# ---------------------------------------------------------------------------
BG = "#fcfcfb"       # surface
INK = "#0b0b0b"      # primary text
QUIET = "#52514e"    # secondary text
GREY = "#9a9890"     # raw signal / textbook RLS
ACCENT = "#2a78d6"   # RLS-DCD, the one accent
SANS = "Source Sans 3"
MONO = "Source Code Pro"

config.background_color = BG

# ---------------------------------------------------------------------------
# Engine numbers (bci_core.py: RLS.formula_mul, RLSDCD shift mode,
# NodeConfig, FRAME_PRESETS default)
# ---------------------------------------------------------------------------
M = 8
RLS_MULS = 4 * M * M + 4 * M + 1          # textbook RLS, multiplies per adapting sample
DCD_MULS = 3 * M                          # transversal RLS-DCD, lambda as shift-and-add
SIDE = 2 * M + 1                          # 4M^2 + 4M + 1 = (2M + 1)^2
M_BIG = 32
RLS_BIG, DCD_BIG, SIDE_BIG = 4 * M_BIG**2 + 4 * M_BIG + 1, 3 * M_BIG, 2 * M_BIG + 1

CHANNELS, FS, ADC_BITS = 16, 256, 9
RAW_BPS = CHANNELS * FS * ADC_BITS        # 36,864 b/s
LINK_BPS = 10_000                         # EQS-HBC capacity
LINK_NW = 415
FRAME = {"preamble": 24, "PHY header": 8, "MAC header": 56, "label": 8, "FCS": 16}
FRAME_BITS = sum(FRAME.values())          # 112
DECISION_BPS = FRAME_BITS / 1.0           # one decision per 1 s window

assert RLS_MULS == SIDE * SIDE == 289 and DCD_MULS == 24
assert RLS_BIG == SIDE_BIG ** 2 == 4225 and DCD_BIG == 96
assert RAW_BPS == 36_864 and FRAME_BITS == 112
assert round(RAW_BPS / LINK_BPS, 2) == 3.69 and round(100 * DECISION_BPS / LINK_BPS, 2) == 1.12


def words(s, size=30, color=INK, font=SANS, weight="NORMAL"):
    return Text(s, font=font, font_size=size, color=color, weight=weight)


def maths(markup, size=56, color=INK):
    """Pango markup in the mono face: <sup> gives true superscripts without LaTeX."""
    return MarkupText(markup, font=MONO, font_size=size, color=color)


def counter(tracker, fmt, size=56, color=INK, font=MONO):
    return always_redraw(lambda: Text(fmt(tracker.get_value()), font=font,
                                      font_size=size, color=color))


# ===========================================================================
# MG-1 . The square and the strip
# ===========================================================================
CELL, GAP = 0.27, 0.04                    # 17 cells x 0.31 = 5.27 units
PITCH = CELL + GAP
GRID_CENTER = RIGHT * 2.4 + DOWN * 0.35
STRIP_CELL, STRIP_GAP = 0.46, 0.08
STRIP_CENTER = RIGHT * 1.15 + DOWN * 0.35
ROW_LABELS = ["new column of R", "output  y = W·x", "update  e·x"]


def mg1_formula_left():
    return maths("4M<sup>2</sup> + 4M + 1", 54).to_corner(UL, buff=0.9).shift(DOWN * 0.6)


def mg1_formula_square():
    return maths("(2M + 1)<sup>2</sup>", 54).move_to(mg1_formula_left(), aligned_edge=LEFT)


def mg1_formula_dcd():
    return maths("3M", 54, ACCENT).move_to(mg1_formula_left(), aligned_edge=LEFT)


def mg1_m_label():
    return words(f"M = {M} taps", 30, QUIET).next_to(mg1_formula_left(), DOWN, buff=0.35,
                                                     aligned_edge=LEFT)


def mg1_count_anchor():
    return mg1_m_label().get_corner(DL) + DOWN * 1.0


def mg1_count_text(n, color=INK):
    t = Text(f"{int(round(n)):,}", font=MONO, font_size=96, color=color)
    return t.move_to(mg1_count_anchor(), aligned_edge=UL)


def mg1_count_caption():
    return words("multiplies per sample", 30, QUIET).next_to(
        mg1_count_text(RLS_MULS), DOWN, buff=0.25, aligned_edge=LEFT)


def mg1_grid():
    cells = VGroup()
    for r in range(SIDE):
        for c in range(SIDE):
            sq = Square(CELL, stroke_width=0, fill_color=GREY, fill_opacity=1)
            sq.move_to(GRID_CENTER + RIGHT * (c - (SIDE - 1) / 2) * PITCH
                       + UP * ((SIDE - 1) / 2 - r) * PITCH)
            cells.add(sq)
    return cells


def mg1_strip_target(i):
    r, c = divmod(i, M)
    p = STRIP_CELL + STRIP_GAP
    return (STRIP_CENTER + RIGHT * (c - (M - 1) / 2) * p + UP * (1 - r) * p)


class MG1_AB(Scene):
    """2:16 . Beat A (formula rewrite) + beat B (the 17 x 17 square builds)."""

    def construct(self):
        f1, f2 = mg1_formula_left(), mg1_formula_square()
        m_label = mg1_m_label()
        self.play(FadeIn(f1, shift=UP * 0.1), run_time=0.8)
        self.wait(0.6)
        self.play(TransformMatchingShapes(f1, f2), FadeIn(m_label), run_time=1.0)

        grid = mg1_grid()
        n = ValueTracker(0)
        count = counter(n, lambda v: f"{int(round(v)):,}", size=96)
        count.add_updater(lambda m: m.move_to(mg1_count_anchor(), aligned_edge=UL))
        cap = mg1_count_caption()
        self.add(count)
        self.play(
            LaggedStart(*[FadeIn(c, scale=0.6) for c in grid], lag_ratio=1 / len(grid) * 6),
            n.animate(rate_func=linear).set_value(RLS_MULS),
            FadeIn(cap),
            run_time=3.0,
        )
        count.clear_updaters()
        self.wait(1.5)


class MG1_CD(Scene):
    """2:52 . Beat C (collapse to a 3 x 8 strip) + beat D (row labels)."""

    def construct(self):
        # final state of MG1_AB
        f2, m_label, grid = mg1_formula_square(), mg1_m_label(), mg1_grid()
        count, cap = mg1_count_text(RLS_MULS), mg1_count_caption()
        self.add(f2, m_label, grid, count, cap)
        self.wait(0.5)

        # 24 cells spread evenly through the square survive; 265 fall away
        keep = [round(i * (len(grid) - 1) / (DCD_MULS - 1)) for i in range(DCD_MULS)]
        assert len(set(keep)) == DCD_MULS
        survivors = VGroup(*[grid[k] for k in keep])
        dropped = VGroup(*[c for i, c in enumerate(grid) if i not in set(keep)])
        assert len(dropped) == RLS_MULS - DCD_MULS

        targets = VGroup(*[
            Square(STRIP_CELL, stroke_width=0, fill_color=ACCENT, fill_opacity=1).move_to(mg1_strip_target(i))
            for i in range(DCD_MULS)
        ])
        n = ValueTracker(RLS_MULS)
        live = counter(n, lambda v: f"{int(round(v)):,}", size=96)
        live.add_updater(lambda m: m.move_to(mg1_count_anchor(), aligned_edge=UL))
        self.remove(count)
        self.add(live)

        self.play(LaggedStart(*[FadeOut(c, shift=DOWN * 0.4) for c in dropped],
                              lag_ratio=0.004), run_time=1.0)
        self.play(
            *[Transform(s, t) for s, t in zip(survivors, targets)],
            TransformMatchingShapes(f2, mg1_formula_dcd()),
            n.animate(rate_func=smooth).set_value(DCD_MULS),
            run_time=1.6,
        )
        live.clear_updaters()
        self.remove(live)
        final = mg1_count_text(DCD_MULS, ACCENT)
        self.add(final)
        self.wait(0.4)

        labels = VGroup()
        p = STRIP_CELL + STRIP_GAP
        for r, text in enumerate(ROW_LABELS):
            y = STRIP_CENTER[1] + (1 - r) * p
            lab = words(text, 28, QUIET)
            lab.move_to([STRIP_CENTER[0] + (M / 2) * p + 0.4, y, 0], aligned_edge=LEFT)
            labels.add(lab)
        self.play(LaggedStart(*[FadeIn(l, shift=RIGHT * 0.15) for l in labels], lag_ratio=0.35),
                  run_time=1.4)
        self.wait(2.0)


class MG1_E(Scene):
    """Optional . The same comparison at M = 32, cells drawn at one common scale."""

    def construct(self):
        cell = 4.9 / SIDE_BIG                          # the 65 x 65 square fits 4.9 units
        sq_origin = LEFT * 5.9 + UP * 2.45             # top-left corner of the square

        def block(rows, cols, color, origin):
            """A rows x cols block of cells: one filled rectangle cut by background lines,
            which reads as cells without drawing 4,225 separate objects."""
            w, h = cols * cell, rows * cell
            rect = Rectangle(width=w, height=h, stroke_width=0, fill_color=color, fill_opacity=1)
            rect.move_to(origin + RIGHT * w / 2 + DOWN * h / 2)
            cuts = VGroup(*[Line(origin + RIGHT * c * cell, origin + RIGHT * c * cell + DOWN * h,
                                 stroke_color=BG, stroke_width=0.6) for c in range(1, cols)],
                          *[Line(origin + DOWN * r * cell, origin + DOWN * r * cell + RIGHT * w,
                                 stroke_color=BG, stroke_width=0.6) for r in range(1, rows)])
            return VGroup(rect, cuts)

        square = block(SIDE_BIG, SIDE_BIG, GREY, sq_origin)
        strip = block(3, M_BIG, ACCENT, RIGHT * 1.2 + UP * 2.45)

        head = words(f"M = {M_BIG} taps", 32, QUIET).to_corner(UL, buff=0.5)
        rls = VGroup(maths(f"(2·{M_BIG} + 1)<sup>2</sup> = {RLS_BIG:,}", 40),
                     words("textbook RLS", 26, QUIET)).arrange(DOWN, aligned_edge=LEFT, buff=0.15)
        rls.next_to(square, DOWN, buff=0.35, aligned_edge=LEFT)
        dcd = VGroup(maths(f"3·{M_BIG} = {DCD_BIG}", 40, ACCENT),
                     words("RLS-DCD", 26, QUIET)).arrange(DOWN, aligned_edge=LEFT, buff=0.15)
        dcd.next_to(strip, DOWN, buff=0.35, aligned_edge=LEFT)
        ratio = words(f"{RLS_BIG / DCD_BIG:.0f}× fewer multiplies", 34, INK, weight="SEMIBOLD")
        ratio.next_to(dcd, DOWN, buff=0.6, aligned_edge=LEFT)

        self.play(FadeIn(head), run_time=0.5)
        self.play(GrowFromEdge(square, UP), FadeIn(rls), run_time=1.6)
        self.play(GrowFromEdge(strip, LEFT), FadeIn(dcd), run_time=1.0)
        self.play(FadeIn(ratio), run_time=0.6)
        self.wait(2.0)


# ===========================================================================
# MG-2 . Halving is a shift
# ===========================================================================
REG_CELL, REG_GAP = 0.95, 0.12
REG_PITCH = REG_CELL + REG_GAP
REG_Y = 0.4


def reg_x(i):
    return (i - 3.5) * REG_PITCH


def bits_of(v):
    s = format(v, "08b")
    assert int(s, 2) == v
    return s


class MG2_Shift(Scene):
    """2:39 . 0110 1000 (104) >> 1 = 0011 0100 (52) >> 1 = 0001 1010 (26)."""

    def construct(self):
        values = [104, 52, 26]
        assert values[1] == values[0] >> 1 and values[2] == values[1] >> 1

        boxes = VGroup(*[Square(REG_CELL, stroke_color=INK, stroke_width=2.5)
                         .move_to([reg_x(i), REG_Y, 0]) for i in range(8)])
        nibble_gap = Line([reg_x(3.5), REG_Y - 0.55, 0], [reg_x(3.5), REG_Y + 0.55, 0],
                          stroke_color=QUIET, stroke_width=1)

        def bit_text(ch, i):
            return Text(ch, font=MONO, font_size=60, color=INK).move_to([reg_x(i), REG_Y, 0])

        bits = VGroup(*[bit_text(ch, i) for i, ch in enumerate(bits_of(values[0]))])
        value = Text(f"{values[0]}", font=MONO, font_size=60, color=QUIET).move_to([0, REG_Y - 1.4, 0])

        self.play(Create(boxes), FadeIn(nibble_gap), run_time=0.8)
        self.play(FadeIn(bits), FadeIn(value), run_time=0.5)

        half = maths("× ½", 52).move_to([0, REG_Y + 1.5, 0])
        shift = maths("&gt;&gt; 1", 52, ACCENT).move_to(half)
        self.play(FadeIn(half, shift=DOWN * 0.1), run_time=0.5)
        self.play(TransformMatchingShapes(half, shift), run_time=0.6)

        for v_new in values[1:]:
            new_bits = bits_of(v_new)
            leaving = bits[7]
            moving = VGroup(*bits[:7])
            entering = bit_text(new_bits[0], 0).shift(LEFT * REG_PITCH).set_opacity(0)
            new_value = Text(f"{v_new}", font=MONO, font_size=60, color=QUIET).move_to(value)
            self.play(
                moving.animate.shift(RIGHT * REG_PITCH),
                leaving.animate.shift(RIGHT * REG_PITCH + DOWN * 0.6).set_opacity(0),
                entering.animate.shift(RIGHT * REG_PITCH).set_opacity(1),
                Transform(value, new_value),
                run_time=0.8,
            )
            self.remove(leaving)
            bits = VGroup(entering, *moving)
            assert "".join(b.text for b in bits) == new_bits
            self.wait(0.5)

        rule = maths("× 2<sup>−m</sup>  ≡  &gt;&gt; m", 52).move_to([0, -2.6, 0])
        self.play(FadeIn(rule, shift=UP * 0.1), run_time=0.8)
        self.wait(1.8)


# ===========================================================================
# MG-3 . The bit budget (one honest horizontal scale for all four beats)
# ===========================================================================
X0 = -6.4                                   # left edge of every bar
LINK_LEN = 0.60 * config.frame_width        # 10,000 b/s spans 60% of the frame width
UNIT = LINK_LEN / LINK_BPS                  # scene units per b/s
ROW_Y = {"raw": 1.55, "clean": -0.05, "decision": -1.65}
BAR_H = 0.34
LANE_TOP, LANE_STEP, LANE_LEN = 2.35, 0.24, 11.6


def mg3_lanes():
    return VGroup(*[
        DashedLine([X0, LANE_TOP - k * LANE_STEP, 0], [X0 + LANE_LEN, LANE_TOP - k * LANE_STEP, 0],
                   dash_length=LANE_LEN / FS * 0.45, dashed_ratio=0.45,
                   stroke_color=GREY, stroke_width=3)
        for k in range(CHANNELS)
    ])


def mg3_header():
    return words(f"{CHANNELS} channels × {FS} samples/s × {ADC_BITS} bits", 32, QUIET) \
        .move_to([X0, 3.25, 0], aligned_edge=LEFT)


def mg3_total(v):
    return Text(f"{int(round(v)):,} b/s", font=MONO, font_size=84, color=INK) \
        .move_to([X0, -2.55, 0], aligned_edge=LEFT)


def bar(kind, bps, color):
    w = bps * UNIT
    return Rectangle(width=w, height=BAR_H, stroke_width=0, fill_color=color, fill_opacity=1) \
        .move_to([X0 + w / 2, ROW_Y[kind], 0])


def grow_width(scene, mob, run_time, **kw):
    """Grow a bar from its left end in width only, at full height from the first frame."""
    start = mob.copy().stretch(1e-3, 0, about_edge=LEFT)
    scene.add(start)
    scene.play(Transform(start, mob), run_time=run_time, **kw)


def mg3_pipe():
    return Rectangle(width=LINK_LEN, height=BAR_H + 0.36, stroke_color=INK, stroke_width=3) \
        .move_to([X0 + LINK_LEN / 2, ROW_Y["raw"], 0])


def mg3_cap_mark(kind):
    """Short dashed mark at x = 10,000 b/s on one bar row (the link's capacity), clear of labels."""
    x, y = X0 + LINK_LEN, ROW_Y[kind]
    return DashedLine([x, y + 0.32, 0], [x, y - 0.32, 0], dash_length=0.06,
                      stroke_color=QUIET, stroke_width=1.5)


def label_above(kind, text, color=INK):
    return words(text, 28, color).move_to([X0, ROW_Y[kind] + 0.52, 0], aligned_edge=LEFT)


def mg3_b_labels():
    pipe = label_above("raw", f"body link (EQS-HBC): {LINK_BPS:,} b/s · {LINK_NW} nW")
    raw = words(f"raw: {RAW_BPS:,} b/s = {RAW_BPS / LINK_BPS:.2f}×  →", 28, INK) \
        .move_to([config.frame_width / 2 - 0.35, ROW_Y["raw"] + 0.52, 0], aligned_edge=RIGHT)
    assert raw.get_left()[0] > X0 + LINK_LEN + 0.2, "raw label would cross the end of the pipe"
    return pipe, raw


def mg3_c_label():
    return label_above("clean", f"after cleaning: {RAW_BPS:,} b/s (unchanged)")


def mg3_d_label():
    return label_above("decision",
                       f"one decision per second, {FRAME_BITS}-bit frame: {DECISION_BPS:.0f} b/s "
                       f"= {100 * DECISION_BPS / LINK_BPS:.2f}% of the link", ACCENT)


class MG3a_Stream(Scene):
    """0:45 . 16 lanes fill at 256 samples/s; the counter reaches 36,864 in one second."""

    def construct(self):
        lanes, head = mg3_lanes(), mg3_header()
        bits = ValueTracker(0)
        total = always_redraw(lambda: mg3_total(bits.get_value()))
        self.play(FadeIn(head), run_time=0.6)
        self.add(total)
        self.play(*[Create(l) for l in lanes], bits.animate.set_value(RAW_BPS),
                  run_time=1.0, rate_func=linear)       # exactly one second of screen time
        total.clear_updaters()
        self.wait(2.5)


class MG3b_Pipe(Scene):
    """0:59 . The link as a pipe; the raw stream runs through it and off the frame."""

    def construct(self):
        lanes, head, total = mg3_lanes(), mg3_header(), mg3_total(RAW_BPS)
        self.add(lanes, head, total)
        self.wait(0.3)
        pipe = mg3_pipe()
        pipe_label, raw_label = mg3_b_labels()
        raw = bar("raw", RAW_BPS, GREY)                 # 31.5 units long: leaves the frame
        self.play(FadeOut(lanes), FadeOut(head), FadeOut(total), run_time=0.6)
        self.play(Create(pipe), FadeIn(pipe_label), run_time=1.2)
        self.wait(0.6)
        self.add_foreground_mobject(pipe)
        grow_width(self, raw, 1.6, rate_func=linear)
        self.play(FadeIn(raw_label), run_time=0.5)
        self.wait(2.0)


def mg3_b_final(scene):
    pipe = mg3_pipe()
    pipe_label, raw_label = mg3_b_labels()
    scene.add(bar("raw", RAW_BPS, GREY), pipe_label, raw_label)
    scene.add_foreground_mobject(pipe)


class MG3c_Cleaned(Scene):
    """1:27 . After cleaning: exactly as long as the raw bar."""

    def construct(self):
        mg3_b_final(self)
        self.wait(0.3)
        self.play(FadeIn(mg3_c_label()), FadeIn(mg3_cap_mark("clean")), run_time=0.5)
        grow_width(self, bar("clean", RAW_BPS, GREY), 1.6, rate_func=linear)
        self.wait(2.0)


class MG3d_Decision(Scene):
    """1:37 . One decision per second in a 112-bit frame: a 1.12% sliver."""

    def construct(self):
        mg3_b_final(self)
        self.add(mg3_c_label(), bar("clean", RAW_BPS, GREY), mg3_cap_mark("clean"))
        self.wait(0.3)
        sliver = bar("decision", DECISION_BPS, ACCENT)   # ~0.096 units = ~13 px at 1920 wide
        self.play(FadeIn(mg3_d_label()), FadeIn(mg3_cap_mark("decision")), run_time=0.6)
        grow_width(self, sliver, 0.5)
        self.wait(2.5)
