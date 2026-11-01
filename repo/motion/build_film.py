"""
build_film.py - assemble BDN-04_cinematic_explainer.mp4 from the rendered chapters.

  1. concatenates the chapter MP4s (rendered by render_chapters.sh) without re-encoding
  2. synthesises the score from the chapters' cue sheets (score.py)
  3. normalises it to -14 LUFS integrated, true peak <= -1 dBTP (ffmpeg loudnorm, two passes)
  4. muxes video + AAC audio, then measures the result and fails loudly if the mix is off spec

Usage:  python build_film.py [scene_dir] [out.mp4]
"""
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import score  # noqa: E402

SCENES = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "media", "videos", "explainer", "1080p24")
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "media", "BDN-04_cinematic_explainer.mp4")
CUES = os.path.join(HERE, "media", "cues")
WORK = os.path.join(HERE, "media", "build")
os.makedirs(WORK, exist_ok=True)
TARGET_I, TARGET_TP = -14.0, -1.5          # -1.5 dBTP before AAC leaves headroom for the -1 dBTP spec


def run(cmd, capture=False):
    r = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return r.stderr if capture else None


def loudness(path):
    err = run(["ffmpeg", "-hide_banner", "-nostats", "-i", path, "-map", "0:a", "-af", "ebur128=peak=true",
               "-f", "null", "-"], capture=True)
    summ = err[err.rfind("Summary:"):]
    i = float(re.search(r"I:\s+(-?[\d.]+) LUFS", summ).group(1))
    tp = float(re.search(r"Peak:\s+(-?[\d.]+) dBFS", summ).group(1))
    return i, tp


def main():
    # 1. video
    lst = os.path.join(WORK, "chapters.txt")
    with open(lst, "w") as f:
        for name in score.ORDER:
            f.write(f"file '{os.path.join(SCENES, name + '.mp4')}'\n")
    video = os.path.join(WORK, "video.mp4")
    run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", lst, "-c", "copy", video])

    # 2. score
    raw = os.path.join(WORK, "score_raw.wav")
    chapters, cues, total = score.timeline(SCENES, CUES)
    score.write_wav(raw, score.compose(chapters, cues, total))

    # 3. loudness: two-pass loudnorm
    ln = f"loudnorm=I={TARGET_I}:TP={TARGET_TP}:LRA=20"
    err = run(["ffmpeg", "-hide_banner", "-nostats", "-i", raw, "-af", ln + ":print_format=json", "-f", "null", "-"],
              capture=True)
    m = json.loads(err[err.rfind("{"):err.rfind("}") + 1])
    second = (f"{ln}:measured_I={m['input_i']}:measured_TP={m['input_tp']}:measured_LRA={m['input_lra']}:"
              f"measured_thresh={m['input_thresh']}:offset={m['target_offset']}:linear=true")
    norm = os.path.join(WORK, "score.wav")
    run(["ffmpeg", "-v", "error", "-y", "-i", raw, "-af", second, "-ar", "48000", "-c:a", "pcm_s24le", norm])

    # 4. mux
    run(["ffmpeg", "-v", "error", "-y", "-i", video, "-i", norm, "-map", "0:v", "-map", "1:a", "-c:v", "copy",
         "-c:a", "aac", "-b:a", "256k", "-shortest", "-movflags", "+faststart", OUT])
    i, tp = loudness(OUT)
    print(f"{OUT}\n  duration {total:.2f} s   integrated {i:.1f} LUFS   true peak {tp:.1f} dBTP")
    assert abs(i - TARGET_I) <= 0.5, f"integrated loudness {i} LUFS is off target"
    assert tp <= -1.0, f"true peak {tp} dBTP exceeds -1 dBTP"
    flams = [c for c in cues if c["kind"] == "flam"]
    for c in flams:
        print(f"  flam: tick at {c['t']:.4f} s, click at {c['t'] + c['gap']:.4f} s (gap {c['gap'] * 1000:.1f} ms)")


if __name__ == "__main__":
    main()
