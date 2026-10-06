# Reproduce the paper's numerical results

This directory contains anonymous experimental observations and the code to
regenerate detector, label-recovery, and annotation-time results.

## Run

From the repository root:

```bash
python -m pip install -r reproduction/requirements.txt
python tools/reproduce_paper_tables.py \
  --data reproduction/data \
  --output outputs/paper_tables
```

Choose a new output directory. The analysis runs on CPU with NumPy and SciPy.
Add `--check-reference` to compare the results with the supplied reference tables
and displayed paper values.

To build the table PDF, install LaTeX and run:

```bash
cd outputs/paper_tables
pdflatex result_tables.tex
```

## Results

A generated copy is available in [generated/](generated/), including the
[result-table PDF](generated/result_tables.pdf).

| Output | Contents |
| --- | --- |
| Result CSVs | Detector metrics, paired comparisons, label recovery, annotation times, and density summaries |
| Interval and sensitivity CSVs | Label-recovery and density confidence intervals; annotation regression coefficients |
| Statistical JSON files | Label-recovery tests, annotation analyses, and the AdaBN selection interval |
| `result_tables.tex` | Standalone LaTeX tables |
| `verification.json` | Run summary and optional reference comparisons |

## Input data

| File in [data/](data/) | Contents |
| --- | --- |
| `detector_runs.csv` | 600 detector evaluations with training seeds, thresholds, counts, and AP |
| `subset_resampling_runs.csv` | 180 evaluations across five data-subset draws and three training seeds |
| `test_tiles.csv` | 22,552 tiles, source-frame IDs, and ground-truth box counts |
| `deployment_presence.jsonl.gz` | Predicted-positive tile IDs for 210 deployment evaluations |
| `label_recovery_images.csv` | Per-image TP/FP/FN for 300 reference images across the label-recovery methods |
| `annotation_images.csv`, `annotation_rounds.csv` | Image-level timing records and ten counterbalanced annotation rounds |
| `automatic_processing.json` | Automatic processing times for 100-image batches |
| `adabn_selection_runs.csv` | 300 AP observations for two adaptation budgets, 15 training seeds, and ten background draws |
| `protocol.json` | Experimental and statistical settings |
| `table_layouts.json` | Output columns and row ordering |
| `reference/` | Comparison values for `--check-reference` |

AP aggregation uses the recorded per-run COCOeval values. The `analysis_order`
field fixes the input order for label-recovery resampling. Automatic timings
cover the 100-image processing loop from image loading through per-image export.

The auxiliary subset-resampling dataset has one background tile in common with
deployment calibration. This overlap concerns the auxiliary experiment;
the fixed-main datasets are unaffected.

## Statistical methods

| Analysis | Method |
| --- | --- |
| Detector summaries | Means, sample SDs (`ddof=1`), and ranges across 15 matched seeds |
| Source-frame metrics | Logical OR over eight tiles per frame; recall and false-alert fractions |
| Detector comparisons | 200,000 paired bootstrap resamples, exact sign-flip tests, and Holm adjustment |
| Subset-resampling comparisons | 200,000 two-way bootstrap resamples over subset draws and training seeds; exact tests on five subset means |
| Label recovery | Micro precision/recall/F1, 50,000 image-cluster bootstrap resamples, and 500,000 paired label swaps |
| Annotation time | Whole-block totals, paired round and block-bootstrap intervals, and OLS adjustment for workflow, round, and presentation position |
| AdaBN selection | 128-versus-512-image AP differences with 10,000 crossed seed/background-draw bootstrap resamples |

Statistical resampling uses NumPy's `default_rng(42)`, with separate generators
for each procedure. Full settings are in [protocol.json](data/protocol.json).

## Statistical tests

From the repository root:

```bash
python -m unittest discover -s tests -v
```
