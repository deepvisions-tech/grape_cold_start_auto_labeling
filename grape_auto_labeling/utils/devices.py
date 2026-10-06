"""CUDA device selection, single-GPU masking, and mixed-precision context.

Device strings are validated without importing a model runtime so that the
preflight stays model free."""

from __future__ import annotations

import os
from typing import Any, Dict, Optional


def validate_device_config(cfg: Dict[str, Any]) -> str:
    """Check DEVICE without importing a model runtime, so --check-only can fail fast."""
    raw_device = cfg.get("DEVICE", "auto")
    text = str(raw_device if raw_device is not None else "auto").strip().lower()
    if text in {"", "auto"}:
        return "auto"
    if text.isdigit():
        return text
    if text == "cpu":
        raise RuntimeError(
            "DEVICE=cpu is not supported because the supported SAM3 image model "
            "constructs CUDA tensors while it is built. Use auto or a GPU index."
        )
    if text.startswith("cuda"):
        index = text.split(":", 1)[1].strip() if ":" in text else ""
        if index == "" or index.isdigit():
            return text
    raise ValueError(
        'DEVICE must be auto, a GPU index such as "0", or cuda:N; '
        f"got {raw_device!r}."
    )


def single_gpu_mask_from_device(raw_device: Any) -> Optional[str]:
    if raw_device is None:
        return None
    text = str(raw_device).strip().lower()
    if text in {"", "auto", "cpu", "cuda"}:
        return None
    if text.isdigit():
        return text
    if text.startswith("cuda:"):
        index = text.split(":", 1)[1].strip()
        if index.isdigit():
            return index
    return None


def _device_uses_visible_single_gpu(raw_device: Any) -> bool:
    mask = single_gpu_mask_from_device(raw_device)
    if mask is None:
        return False
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    return visible_devices is not None and visible_devices.strip() == mask


def configure_auto_label_runtime(cfg: Dict[str, Any]) -> Dict[str, Any]:
    raw_device = cfg.get("DEVICE")
    requested_device = str(raw_device).strip() if raw_device is not None else "auto"
    info = {
        "requested_device": requested_device,
        "limit_visible_device": False,
        "cuda_visible_devices_before": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cuda_visible_devices_after": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    mask = single_gpu_mask_from_device(raw_device)
    if mask is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = mask
        info["limit_visible_device"] = True
        info["cuda_visible_devices_after"] = os.environ.get("CUDA_VISIBLE_DEVICES")
    return info


def resolve_torch_device(raw_device: Any, name: str, require_cuda: bool = False) -> str:
    text = str(raw_device if raw_device is not None else "auto").strip().lower()
    import torch

    if text in {"", "auto"}:
        if torch.cuda.is_available():
            return "cuda"
        if require_cuda:
            raise RuntimeError(f"{name}=auto requires CUDA, but CUDA is not available.")
        return "cpu"
    if text == "cpu":
        if require_cuda:
            raise RuntimeError(f"{name}=cpu is not supported for this runtime.")
        return "cpu"
    if text.isdigit():
        if not torch.cuda.is_available():
            raise RuntimeError(f"{name}={text} requested CUDA, but CUDA is not available.")
        cuda_index = 0 if _device_uses_visible_single_gpu(raw_device) else int(text)
        torch.cuda.set_device(cuda_index)
        return f"cuda:{cuda_index}"
    if text.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(f"{name}={text} requested CUDA, but CUDA is not available.")
        if ":" in text:
            requested_index = int(text.split(":", 1)[1])
            cuda_index = 0 if _device_uses_visible_single_gpu(raw_device) else requested_index
            torch.cuda.set_device(cuda_index)
            return f"cuda:{cuda_index}"
        return "cuda"
    return str(raw_device)


def resolve_sam3_device(raw_device: Any, name: str = "DEVICE") -> str:
    text = str(raw_device if raw_device is not None else "auto").strip().lower()
    import torch

    if text in {"", "auto"}:
        if torch.cuda.is_available():
            return "cuda"
        raise RuntimeError(
            "SAM3 auto labeling requires CUDA with the supported upstream SAM3 code. "
            "CUDA is not available in the current environment."
        )
    if text == "cpu":
        raise RuntimeError(
            f"{name}=cpu is not supported by the supported upstream SAM3 image model "
            "because it constructs CUDA tensors during model build."
        )
    if text.isdigit():
        if not torch.cuda.is_available():
            raise RuntimeError(f"{name}={text} requested CUDA, but CUDA is not available.")
        cuda_index = 0 if _device_uses_visible_single_gpu(raw_device) else int(text)
        torch.cuda.set_device(cuda_index)
        return "cuda"
    if text.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(f"{name}={text} requested CUDA, but CUDA is not available.")
        if ":" in text:
            requested_index = int(text.split(":", 1)[1])
            cuda_index = 0 if _device_uses_visible_single_gpu(raw_device) else requested_index
            torch.cuda.set_device(cuda_index)
            return "cuda"
        return "cuda"
    return str(raw_device)


def convert_module_floating_dtype(module: Any, dtype: Any) -> None:
    import torch

    for parameter in module.parameters(recurse=False):
        if torch.is_floating_point(parameter):
            parameter.data = parameter.data.to(dtype=dtype)
            if parameter.grad is not None:
                parameter.grad.data = parameter.grad.data.to(dtype=dtype)
    for name, buffer in module.named_buffers(recurse=False):
        if buffer is not None and torch.is_floating_point(buffer):
            setattr(module, name, buffer.to(dtype=dtype))
    for child in module.children():
        convert_module_floating_dtype(child, dtype)


def amp_inference_context(enabled: bool, device: str = "cuda", dtype: str = "float16"):
    import torch

    normalized_dtype = str(dtype).strip().lower()
    if normalized_dtype in {"float16", "fp16"}:
        torch_dtype = torch.float16
    elif normalized_dtype in {"bfloat16", "bf16"}:
        torch_dtype = torch.bfloat16
    else:
        raise ValueError(f"Unsupported AMP dtype: {dtype}")

    is_cuda = str(device).startswith("cuda") and torch.cuda.is_available()
    return torch.amp.autocast(
        device_type="cuda",
        dtype=torch_dtype,
        enabled=bool(enabled) and is_cuda,
    )

