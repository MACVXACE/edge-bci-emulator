#!/usr/bin/env bash
# Render the film's chapters two at a time.
#   ./render_chapters.sh -qh -r 1920,1080 --fps 24            (all chapters)
#   CHAPTERS="C03_Bandwidth C08_Latency" ./render_chapters.sh -qh -r 1920,1080 --fps 24
cd "$(dirname "$0")"
mkdir -p media
ORDER=${CHAPTERS:-"C00_Title C01_Idea C02_Constraints C03_Bandwidth C04_Noise C04b_Chain C05_Decision C06_Arithmetic C07_Power C08_Latency C09_Dashboard C10_Built C11_End"}
printf "%s\n" $ORDER | xargs -P 2 -I{} sh -c 'manim "$@" explainer.py {} > media/log_{}.txt 2>&1 && echo "done {}" || echo "FAILED {}"' _ "$@"
