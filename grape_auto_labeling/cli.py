"""Console entry point that configures GPU visibility before model imports."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Optional, Sequence

from .config import load_config
from .utils.devices import single_gpu_mask_from_device


DEFAULT_CONFIG_PATH = "configs/auto_labeling.yaml"


def _bootstrap_cuda_visible_devices(config_path: str) -> None:
    path = Path(config_path)
    if not path.is_file() or "CUDA_VISIBLE_DEVICES" in os.environ:
        return
    mask = single_gpu_mask_from_device(load_config(path).get("DEVICE"))
    if mask is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = mask


def main(argv: Optional[Sequence[str]] = None):
    parser = argparse.ArgumentParser(description="Run SAM3-DINOv2 automatic label recovery.")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH, help="Path to a YAML config file.")
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate config, inputs, and output safety without loading models or writing files.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=None,
        help="Allow writing into a non-empty LABEL_OUTPUT_DIR without deleting stale files.",
    )
    args = parser.parse_args(argv)
    cfg = load_config(args.config)

    if args.check_only:
        from .labeling.preflight import check_auto_labeling_config

        return check_auto_labeling_config(cfg, overwrite=args.overwrite)

    _bootstrap_cuda_visible_devices(args.config)

    from .labeling.pipeline import run_auto_labeling

    return run_auto_labeling(cfg, overwrite=args.overwrite)


def console_main(argv: Optional[Sequence[str]] = None) -> int:
    """Console entry point.

    ``main`` returns the summary payload for library callers. The installed
    command must return a process exit status instead, because the generated
    console script passes this return value straight to ``sys.exit``.
    """
    main(argv)
    return 0


if __name__ == "__main__":
    raise SystemExit(console_main())
