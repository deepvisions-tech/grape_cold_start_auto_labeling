# Grape Cold-Start Automatic Labeling

Automatic labeling and numerical reproduction code for:

> **A cold-start synthetic-to-real framework for grapevine visual symptom
> detection using background-preserving generative editing**  
> Taewoo Kim, Gyuwon Lee, Wonhee Cho, Bongsu Kang, and Junsang Cho.

This repository provides:

- SAM3-based seed-label recovery, with optional image exemplars and pairwise
  DINOv2 filtering of original–generated image pairs.
- Bounding-box export in YOLO format and label-recovery evaluation.
- Anonymous experimental observations, statistical analysis, and result-table
  generation.

## Paper results

[Result tables (PDF)](reproduction/generated/result_tables.pdf) ·
[CSVs and statistics](reproduction/generated/) ·
[Reproduction guide](reproduction/README.md)

Regenerate the tables from the supplied observations on CPU:

```bash
python -m pip install -r reproduction/requirements.txt
python tools/reproduce_paper_tables.py \
  --data reproduction/data \
  --output outputs/paper_tables
```

Choose a new output directory. This CPU workflow recomputes tables and
statistics from the supplied observations.
Add `--check-reference` to compare the results with the supplied reference values.

## Installation

For image labeling, install CUDA-enabled PyTorch, torchvision, and
[SAM 3](https://github.com/facebookresearch/sam3) following the upstream
installation instructions. Then, from this repository:

```bash
python -m pip install -e .
```

Obtain access to the SAM 3 checkpoint, or configure a local checkpoint.
DINOv2 weights are downloaded when its filter is enabled. The model versions
used in the paper are listed in [MODEL_VERSIONS.md](MODEL_VERSIONS.md).

## Usage

Copy the [example configuration](configs/auto_labeling.example.yaml):

```bash
cp configs/auto_labeling.example.yaml configs/auto_labeling.yaml
```

Set the input/output paths and text prompts. For DINOv2 filtering, provide
original images with filenames matching the generated images. Exemplar mode
also requires an exemplar image and its YOLO labels. Then run:

```bash
grape-auto-label --config configs/auto_labeling.yaml --check-only
grape-auto-label --config configs/auto_labeling.yaml
```

Outputs include YOLO labels, evaluation records, and optional visualizations.
Generated labels require human review before training. Use a new or empty
output directory.

See [paper configurations](configs/reproduction/README.md) for the three
label-recovery variants. For analysis of your own per-image results, run
`grape-label-analysis --help`.

## Detector code

The companion repository, `grape_cold_start_detection_yolo`, provides YOLO
training, evaluation, relabeling, and seed sweeps.

## License

Original source code, including numerical reproduction code, is licensed under
[PolyForm Noncommercial 1.0.0](LICENSE). See [NOTICE](NOTICE) for the copyright
notice. Third-party software and models retain their own licenses; see
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
