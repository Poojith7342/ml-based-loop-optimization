# ML-Based Loop Optimization for High-Performance Compiler Design
Compiler Design Lab (CSE3024), VIT-AP University

A small compiler front end extracts static features from C loop nests; an ML model
then chooses one of 7 GCC optimisation configurations. Everything is measured on the
machine it runs on (compile + execute every kernel under every configuration).

## Layout
| Path | Purpose |
|------|---------|
| `src/lexer.py` | Phase 1 - lexical analysis (regex scanner) |
| `src/loop_parser.py` | Phase 2 - recursive-descent parser, AST |
| `src/loop_analyzer.py` | Phase 3 - semantic + loop analysis, 23 features |
| `src/tac.py` | Phase 4 - three-address code |
| `src/kernel_gen.py` | Generates the 182-kernel corpus (9 families) + timing harness |
| `src/benchmark.py` | Compiles/times a kernel under all configurations |
| `src/build_dataset.py` | Features + timings + safety filter + labels |
| `src/train_model.py` | Trains/evaluates DT, RF, GB, k-NN (5-fold CV and leave-one-family-out) |
| `src/predict.py` | CLI advisor for a new loop |
| `src/make_report_assets.py` | Generates LaTeX tables/plots/macros from the results |
| `report/main.tex` | The report (CDL template format); `main.pdf` is the compiled version |
| `data/`, `results/` | Measured timings, dataset, model outputs |

## Run
```
pip install -r requirements.txt
./run_all.sh                       # whole experiment (~10-15 min on 1 core), needs gcc + pdflatex
python3 src/predict.py kernels/matmul_double_96_jki.c --run --tac
```
Your own kernel: a C file with `#define` sizes, `typedef double DT;`, `static DT a[N];`
declarations and a function `kernel(...)` containing a `for` nest (see `kernels/*.c`).

## Notes
* Results depend on the CPU (this run: Xeon with AVX2/AVX-512). Re-run `run_all.sh` on your
  own machine and the report numbers regenerate automatically.
* Timing on shared VMs is noisy (see report, Section "Measurement Noise").
* Title page: check which register number belongs to which student.
