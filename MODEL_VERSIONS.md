# Model versions

This file records the exact external model artifacts used for the reported
SAM 3--DINOv2 label-recovery experiments. The manuscript retains only the
model identity and result-relevant variant distinction.

These identifiers are provenance records. Checkpoint hashes and SAM 3/DINOv2
revisions are not enforced at runtime. The public installation may use a current
SAM 3 clone. The runtime applies the historical dtype compatibility only when
it recognizes the reported source; other source revisions are left unmodified.

## SAM 3

- Distribution: `facebook/sam3`
- Checkpoint: `sam3.pt`
- Checkpoint SHA-256:
  `9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e`
- Resolved Hugging Face revision:
  `3c879f39826c281e95690f02c7821c4de09afae7`
- Code commit: `8e451d5eb43c817b64ae7577fb7b9ae223db88a9`

The pristine source at this commit fails in this pipeline's fp16 and fp32
paths. The public runtime leaves that checkout clean and applies in-memory
equivalents of the two fixes used by the reported runs. It also recognizes the
historical locally patched source without wrapping it again. A newer or
otherwise unrecognized source is allowed and left unmodified. Compatibility
status and the two inspected source hashes are written to `summary.json`.

The checkpoint was loaded through
`build_sam3_image_model(..., load_from_HF=True)` without a pinned
`revision=` argument.

## DINOv2

- Variant: `dinov2_vitb14` (ViT-B/14 without registers)
- Checkpoint: `dinov2_vitb14_pretrain.pth`
- Checkpoint SHA-256:
  `0b8b82f85de91b424aded121c7e1dcc2b7bc6d0adeea651bf73a13307fad8c73`
- Code commit: `7764ea0f912e53c92e82eb78a2a1631e92725fc8`

The model was loaded through `torch.hub.load` from the upstream default branch
without a pinned ref.

## Reported experiment environment

All reported YOLO training and inference and all SAM 3--DINOv2 automatic
labeling were executed in one Conda environment. The full historical snapshot
is retained in the source study repository as
`runtime_environment_packages.md`. The public package pins only the packages it
needs and leaves the platform-specific PyTorch installation to the user.

| Component | Reported version or observation |
|---|---|
| Python | CPython 3.13.13 |
| Platform | Linux 5.15.0-140-generic, x86_64, glibc 2.35 |
| PyTorch | 2.10.0+cu128 |
| torchvision | 0.25.0+cu128 |
| PyTorch CUDA build | 12.8 |
| Packaged CUDA runtime | `nvidia-cuda-runtime-cu12` 12.8.90 |
| cuDNN package | `nvidia-cudnn-cu12` 9.10.2.21 |
| SAM 3 Python distribution | 0.1.0 |
| NumPy | 1.26.4 |
| OpenCV | `opencv-python` 4.10.0.82 |
| Pillow | 12.2.0 |
| PyYAML | 6.0.3 |
| tqdm | 4.67.3 |
| timm | 1.0.27 |
| einops | 0.8.2 |
| huggingface-hub | 1.16.1 |

The historical snapshot did not contain Albumentations or xFormers. The exact
PyTorch and torchvision values above are execution provenance, not package
constraints imposed on public users.

## Reproducibility note

The revision and commit identifiers above record the artifacts resolved for
the reported execution. The original experiment loading paths did not pin the
upstream revisions; that historical detail is preserved in the notes above.

The public configurations intentionally do not pin SAM 3/DINOv2 revisions or
enforce the recorded checkpoint hashes. When a local repository or checkpoint
path is configured, the path must exist and the model must load successfully.
The SAM 3 compatibility check applies wrappers only to the recognized historical
source and leaves other revisions unchanged.
