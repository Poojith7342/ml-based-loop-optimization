#!/bin/sh
# Reproduces the whole experiment (about 10-15 minutes on one core).
set -e
cd "$(dirname "$0")"
python3 src/kernel_gen.py kernels          # 1. generate the loop-kernel corpus
python3 src/build_dataset.py               # 2. features + timings + labels  (slow step, resumable)
python3 src/train_model.py                 # 3. train / cross-validate the decision model
python3 src/make_report_assets.py          # 4. LaTeX tables, plots and macros
cd report && pdflatex -interaction=nonstopmode main.tex && pdflatex -interaction=nonstopmode main.tex
