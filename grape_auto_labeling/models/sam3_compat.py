"""Runtime compatibility for the SAM 3 revision used by this project.

The reported experiments used upstream SAM 3 commit ``8e451d5e`` with two
local dtype fixes.  Public installations keep the upstream checkout pristine,
so this module reproduces those fixes in memory before a model is built.
Unrecognized upstream sources are left unchanged so a current SAM 3 clone can
use its own implementation without a project-specific source pin.
"""

from __future__ import annotations

from functools import wraps
import hashlib
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


SAM3_COMPAT_UPSTREAM_COMMIT = "8e451d5eb43c817b64ae7577fb7b9ae223db88a9"

_DECODER_CLEAN_SHA256 = "2bc0ad4ab894f09306e6225c116119f9a696e5b5b3dfa5aafde35081ad68d1e0"
_DECODER_PATCHED_SHA256 = "ff8795daf2298d97aaaca765fea2ec16503bf559211aa356bb38df1f38e383f8"
_FUSED_CLEAN_SHA256 = "f528109e043331da4c240741d1403812b2020801d8f01af0b5fbf0c6e93c1af6"
_FUSED_PATCHED_SHA256 = "2a1148cef6b5b17b1d705b801347c18506edc9dd53820c7febe25e1995eaf4da"

_COMPAT_MARKER = "__grape_sam3_dtype_compat__"


class Sam3CompatibilityError(RuntimeError):
    """Raised when the loaded SAM 3 source cannot be inspected."""


def _source_sha256(module: Any) -> Tuple[str, str]:
    source_file = getattr(module, "__file__", None)
    if not source_file:
        raise Sam3CompatibilityError(
            f"Cannot identify the source file for SAM 3 module {module.__name__!r}."
        )
    path = Path(source_file).resolve(strict=False)
    try:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise Sam3CompatibilityError(f"Cannot read SAM 3 source file: {path}") from exc
    return str(path), digest


def _classify_source(
    module: Any,
    *,
    clean_sha256: str,
    patched_sha256: str,
) -> Dict[str, str]:
    path, digest = _source_sha256(module)
    if digest == clean_sha256:
        state = "clean_supported"
    elif digest == patched_sha256:
        state = "source_patch_present"
    else:
        state = "unrecognized"
    return {"source_path": path, "source_sha256": digest, "source_state": state}


def _install_decoder_wrapper(decoder_module: Any) -> str:
    decoder_layer = decoder_module.TransformerDecoderLayer
    current = decoder_layer.forward_ffn
    if getattr(current, _COMPAT_MARKER, False):
        return "already_applied"

    original = current

    @wraps(original)
    def compatible_forward_ffn(self, tgt):
        # The upstream method disables autocast before its Linear layers.  Pass
        # the cast tensor into the original method so its residual path uses the
        # same dtype as the locally patched source used by the reported runs.
        return original(self, tgt.to(dtype=self.linear1.weight.dtype))

    setattr(compatible_forward_ffn, _COMPAT_MARKER, True)
    decoder_layer.forward_ffn = compatible_forward_ffn
    return "applied"


def _install_fused_wrapper(fused_module: Any, vitdet_module: Any) -> str:
    current = fused_module.addmm_act
    if getattr(current, _COMPAT_MARKER, False):
        # vitdet imports addmm_act directly, so repair that binding even when a
        # previous model load already installed the wrapper.
        vitdet_module.addmm_act = current
        return "already_applied"

    import torch

    original = current

    @wraps(original)
    def compatible_addmm_act(activation, linear, mat1):
        if torch.is_grad_enabled():
            # Preserve upstream's exact error behavior for training calls.
            return original(activation, linear, mat1)

        if (
            not torch.is_autocast_enabled("cuda")
            and mat1.dtype == torch.float32
            and linear.weight.dtype == torch.float32
        ):
            y = linear(mat1)
            if activation in [torch.nn.functional.relu, torch.nn.ReLU]:
                return torch.nn.functional.relu(y)
            if activation in [torch.nn.functional.gelu, torch.nn.GELU]:
                return torch.nn.functional.gelu(y)
            raise ValueError(f"Unexpected activation {activation}")

        return original(activation, linear, mat1).to(linear.weight.dtype)

    setattr(compatible_addmm_act, _COMPAT_MARKER, True)
    fused_module.addmm_act = compatible_addmm_act
    # sam3.model.vitdet uses ``from sam3.perflib.fused import addmm_act``.
    vitdet_module.addmm_act = compatible_addmm_act
    return "applied"


def apply_sam3_dtype_compatibility(
    *,
    decoder_module: Optional[Any] = None,
    fused_module: Optional[Any] = None,
    vitdet_module: Optional[Any] = None,
) -> Dict[str, Any]:
    """Apply the two known dtype fixes without changing the SAM 3 checkout.

    Optional module arguments exist for isolated tests.  Production callers
    omit them and patch the modules selected by ``SAM3_REPO_DIR`` or the active
    installed package.
    """

    if decoder_module is None or fused_module is None or vitdet_module is None:
        from sam3.model import decoder as loaded_decoder
        from sam3.model import vitdet as loaded_vitdet
        from sam3.perflib import fused as loaded_fused

        decoder_module = loaded_decoder
        fused_module = loaded_fused
        vitdet_module = loaded_vitdet

    decoder_info = _classify_source(
        decoder_module,
        clean_sha256=_DECODER_CLEAN_SHA256,
        patched_sha256=_DECODER_PATCHED_SHA256,
    )
    fused_info = _classify_source(
        fused_module,
        clean_sha256=_FUSED_CLEAN_SHA256,
        patched_sha256=_FUSED_PATCHED_SHA256,
    )

    if decoder_info["source_state"] == "clean_supported":
        decoder_info["runtime_action"] = _install_decoder_wrapper(decoder_module)
    elif decoder_info["source_state"] == "source_patch_present":
        decoder_info["runtime_action"] = "not_needed"
    else:
        decoder_info["runtime_action"] = "left_unmodified"

    if fused_info["source_state"] == "clean_supported":
        fused_info["runtime_action"] = _install_fused_wrapper(fused_module, vitdet_module)
    elif fused_info["source_state"] == "source_patch_present":
        # Keep the directly imported binding aligned even for the source-patched
        # checkout, which also makes the invariant explicit in provenance tests.
        vitdet_module.addmm_act = fused_module.addmm_act
        fused_info["runtime_action"] = "not_needed"
    else:
        fused_info["runtime_action"] = "left_unmodified"

    actions = {decoder_info["runtime_action"], fused_info["runtime_action"]}
    if "left_unmodified" in actions and ("applied" in actions or "already_applied" in actions):
        status = "partial_known_compatibility"
    elif "left_unmodified" in actions:
        status = "unrecognized_source_unmodified"
    elif "applied" in actions:
        status = "applied"
    elif "already_applied" in actions:
        status = "already_applied"
    else:
        status = "source_patch_present"

    return {
        "status": status,
        "target_upstream_commit": SAM3_COMPAT_UPSTREAM_COMMIT,
        "checkout_files_modified": False,
        "decoder": decoder_info,
        "fused": fused_info,
        "vitdet_binding_matches_fused": vitdet_module.addmm_act is fused_module.addmm_act,
    }
