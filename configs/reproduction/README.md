# Reported method configurations

These configurations record the settings of the three training-free seed-label
recovery arms reported in the paper:

- `sam3_only.example.yaml`
- `sam3_exemplar.example.yaml`
- `sam3_exemplar_dinov2.example.yaml`

Copy the required file to a local, untracked YAML file and replace every
`/path/to/...` placeholder before running it. The directory name
`reproduction` refers to method and configuration reproduction on
user-provided data, not end-to-end reproduction of the paper metrics.

The controlled evaluation data and the exact paper exemplar are not
distributed. The paper used one exemplar image with seven human-drawn boxes;
users must provide their own valid exemplar image and YOLO label file. The
runtime validates the general input format but does not require the paper
exemplar, its hash, or exactly seven boxes.

The model identifiers and artifact hashes used in the reported run are listed
in `../../MODEL_VERSIONS.md` for provenance. Checkpoint hashes and SAM 3/DINOv2
revisions are not enforced at runtime. The runtime recognizes the historical
SAM 3 source used in the study and applies its two dtype fixes in memory only
when needed. A newer or otherwise unrecognized SAM 3 source is left unmodified
and is allowed to load normally. Configured local repository and checkpoint
paths must exist and model loading must succeed.

## Comparison-arm settings

| Configuration | Prompt mode | Exemplar | SAM3 threshold | DINOv2 gate |
|---|---|---:|---:|---|
| `sam3_only.example.yaml` | text | no | 0.7 | disabled |
| `sam3_exemplar.example.yaml` | text + exemplar | yes | 0.7 | disabled |
| `sam3_exemplar_dinov2.example.yaml` | text + exemplar | yes | 0.5 | candidate-adaptive |

## Manuscript and reported-run audit

The three files were audited on 2026-08-03 against the manuscript method,
experiment, and result sections; the supplementary prompt and pairwise-gate
descriptions; and the canonical `summary.json` metadata from each reported
run. The reported method settings match as follows:

| Reported setting | Paper / run value | Reference YAML value | Result |
|---|---|---|---|
| comparison arms | SAM3-only; SAM3 + exemplar; SAM3 + exemplar + DINOv2 | one file per arm; the DINOv2 arm enables pairwise filtering | match |
| prompt strings and order | the three strings listed below | identical strings in identical order | match |
| prompt mode / exemplar | text only; then text + one seven-box exemplar for both exemplar arms | `text`; then `text_and_exemplar` with one user-supplied exemplar entry | match |
| SAM3 confidence | 0.7 / 0.7 / 0.5 | 0.7 / 0.7 / 0.5 | match |
| proposal NMS IoU | 0.6 | 0.6 | match |
| mixed precision | fp16 / AMP enabled | `AMP: true` | match |
| output and evaluation | class ID 0, YOLO TXT, matching IoU 0.5 | class ID 0, YOLO TXT output, evaluation IoU 0.5 | match |
| DINOv2 backbone and input | `dinov2_vitb14`, 518, 37 x 37 patch grid | `dinov2_vitb14`, input size 518 | match |
| relative tiling | width / 3, height / 2, stride ratio 0.67 | divisors 3 / 2, stride ratio 0.67 | match |
| candidate support and score | fully-inside patches, cosine difference p75 | fully-inside enabled; candidate-adaptive p75 implementation | match |
| adaptive gate | candidate Otsu at >= 4 scores; otherwise image-wide fallback; degenerate/near-zero p80 fallback | `candidate_adaptive`, Otsu, offset 0.06; the same fallback rules documented below | match |

The canonical DINOv2 run metadata additionally confirms edge suppression 2,
tile batch size 16, and fallback offset 0.06. The supplementary draft still contains confirmation notes for the prompt
strings and cutoff offset; the run metadata resolves both notes without a
configuration change.

Paths, GPU indices, and local repository/checkpoint locations are deployment
fields rather than paper method parameters. They intentionally remain
placeholders or user-selectable values and are excluded from the equality
claim above. `OUTPUT_EXISTING_POLICY: error` is a public-runtime safety setting,
not a paper method parameter. Batch size 4, four data-loading workers, disabled
visual output, and no detection cap are retained from the reported run metadata
but do not affect the manuscript's method definition.

All three configurations use NMS IoU 0.6, class ID 0, YOLO-compatible TXT
output, and evaluation IoU 0.5. Text prompts are evaluated in this order:

1. `single chlorotic grape leaf with visible green veins`
2. `individual grape leaf with interveinal chlorosis`
3. `single grape leaf showing yellow-green interveinal chlorosis`

The exemplar configurations require a readable image and a YOLO box label file
with exactly five fields per non-empty line: `class cx cy width height`. Values
must be finite, class IDs must be non-negative integers, centers must be in
`[0, 1]`, widths and heights must be in `(0, 1]`, and the resulting box must lie
fully inside the normalized image bounds. At least one valid box must remain
for the configured exemplar class.

## DINOv2 candidate-adaptive gate

The reported DINOv2 arm uses `dinov2_vitb14`, input size 518, relative source
tiles with width divisor 3 and height divisor 2, stride ratio 0.67, edge
suppression of two patch rows/columns, and patches fully inside the SAM3 mask.
Pair alignment follows the original SIFT/FLANN/RANSAC procedure, with ORB and
resize fallbacks. Image-level concurrency is controlled by
`AUTO_LABELING_NUM_WORKERS`.

The gate is applied to the SAM3 candidate pool produced at confidence threshold
0.5:

1. Each candidate receives the 75th percentile (`p75`) of cosine-difference
   patch scores fully inside its mask.
2. With at least four valid candidate scores, a per-image Otsu cutoff over the
   candidate p75 scores is used directly.
3. With fewer than four valid candidate scores, the cutoff falls back to an
   image-wide adaptive threshold over all patch scores. The configured offset
   0.06 is subtracted from this fallback cutoff and the result is clamped at
   zero.
4. A degenerate/uniform Otsu distribution, or a resulting threshold below
   0.05, falls back to the 80th percentile.
5. A candidate is retained when its p75 score is greater than or equal to the
   selected cutoff.

Gate-before and gate-after comparisons must reuse this same SAM3 candidate pool
rather than run SAM3 again.

Example commands:

Run a model-free path and setting check first:

```bash
grape-auto-label --config configs/reproduction/sam3_only.example.yaml --check-only
```

```bash
grape-auto-label --config configs/reproduction/sam3_only.example.yaml
```

```bash
grape-auto-label --config configs/reproduction/sam3_exemplar.example.yaml
```

```bash
grape-auto-label --config configs/reproduction/sam3_exemplar_dinov2.example.yaml
```

All generated labels are seed proposals requiring human review. Images with
zero proposals receive an empty TXT file and must also be reviewed.
