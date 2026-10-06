# Third-party software and models

This repository does not vendor third-party model code or checkpoints.

For the SAM 3 revision identified in `MODEL_VERSIONS.md`, the runtime verifies
two affected source files and installs a narrow dtype compatibility shim in
memory before model construction. It does not rewrite or redistribute the
upstream checkout. The shim reproduces the two local dtype fixes used by the
reported experiments.

- SAM 3 is obtained separately from
  <https://github.com/facebookresearch/sam3> and is governed by Meta's
  [SAM License](https://github.com/facebookresearch/sam3/blob/main/LICENSE).
- DINOv2 is loaded from
  <https://github.com/facebookresearch/dinov2>. Its code and standard model
  weights are distributed under the Apache License 2.0.

Review the upstream terms before downloading, using, or redistributing model
materials. This repository's original source code is licensed under the
[PolyForm Noncommercial License 1.0.0](LICENSE), with its copyright notice in
[NOTICE](NOTICE). That license does not replace the terms governing third-party
software, model materials, or code derived from them.
