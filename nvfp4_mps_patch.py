"""
MPS compatibility patch for Comfy Kitchen NVFP4 tensors.

Krea 2 NVFP4 can fall back from its CUDA-only scaled_mm path into
comfy_kitchen.dequantize_nvfp4. On Apple Silicon MPS, that path fails when it
touches FP8 block scales:

    RuntimeError: Undefined type Float8_e4m3fn

This module installs a narrow fallback for MPS tensors. It dequantizes on CPU
using Comfy Kitchen's eager implementation, then moves the result back to MPS.
That is slower than a Metal kernel, but it is correct and gives us a stable
integration point for profiling and future acceleration.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any

import torch

_original_dequantize_nvfp4 = None
_eager_dequantize_nvfp4 = None
_installed = False


@dataclass
class Nvfp4FallbackStats:
    calls: int = 0
    metal_calls: int = 0
    metal_seconds: float = 0.0
    fallback_calls: int = 0
    fallback_seconds: float = 0.0
    elements: int = 0
    last_shape: tuple[int, ...] | None = None
    last_output_type: str | None = None


_stats = Nvfp4FallbackStats()


def _verbose_enabled() -> bool:
    return os.environ.get("FP8_MPS_METAL_NVFP4_VERBOSE", "").lower() in {"1", "true", "yes", "on"}


def _backend_mode() -> str:
    return os.environ.get("FP8_MPS_METAL_NVFP4_BACKEND", "auto").lower()


def _tensor_device(tensor: Any) -> torch.device | None:
    if isinstance(tensor, torch.Tensor):
        return tensor.device
    return None


def _find_mps_device(*values: Any) -> torch.device | None:
    for value in values:
        device = _tensor_device(value)
        if device is not None and device.type == "mps":
            return device
    return None


def _is_fp8_dtype(dtype: torch.dtype) -> bool:
    return dtype in {
        getattr(torch, "float8_e4m3fn", None),
        getattr(torch, "float8_e5m2", None),
    }


def _to_cpu_preserving_fp8_bytes(tensor: torch.Tensor) -> torch.Tensor:
    if _is_fp8_dtype(tensor.dtype):
        return tensor.view(torch.uint8).cpu().view(tensor.dtype)
    return tensor.cpu()


def _dequantize_nvfp4_mps_safe(
    qx: torch.Tensor,
    per_tensor_scale: torch.Tensor,
    block_scales: torch.Tensor,
    output_type: torch.dtype = torch.bfloat16,
    hi_first: bool = True,
) -> torch.Tensor:
    global _stats

    _stats.calls += 1
    mps_device = _find_mps_device(qx, per_tensor_scale, block_scales)
    if mps_device is None:
        return _original_dequantize_nvfp4(
            qx,
            per_tensor_scale,
            block_scales,
            output_type,
            hi_first,
        )

    mode = _backend_mode()
    if mode not in {"auto", "metal", "cpu"}:
        mode = "auto"

    if mode in {"auto", "metal"}:
        start = time.perf_counter()
        try:
            import fp8_mps_native

            result = fp8_mps_native.nvfp4_dequantize(
                qx,
                per_tensor_scale,
                block_scales,
                hi_first=hi_first,
            )
            if output_type != torch.float16:
                result = result.to(output_type)
            elapsed = time.perf_counter() - start

            _stats.metal_calls += 1
            _stats.metal_seconds += elapsed
            _stats.elements += int(result.numel())
            _stats.last_shape = tuple(result.shape)
            _stats.last_output_type = str(output_type)

            if _verbose_enabled():
                print(
                    "[fp8-nvfp4-mps-metal] NVFP4 Metal dequant "
                    f"shape={tuple(result.shape)} dtype={output_type} time={elapsed:.4f}s"
                )

            return result
        except Exception as exc:
            if mode == "metal":
                raise
            if _verbose_enabled():
                print(f"[fp8-nvfp4-mps-metal] NVFP4 Metal path failed, using CPU fallback: {exc}")

    start = time.perf_counter()
    result = _eager_dequantize_nvfp4(
        _to_cpu_preserving_fp8_bytes(qx),
        _to_cpu_preserving_fp8_bytes(per_tensor_scale),
        _to_cpu_preserving_fp8_bytes(block_scales),
        output_type,
        hi_first,
    ).to(mps_device)
    elapsed = time.perf_counter() - start

    _stats.fallback_calls += 1
    _stats.fallback_seconds += elapsed
    _stats.elements += int(result.numel())
    _stats.last_shape = tuple(result.shape)
    _stats.last_output_type = str(output_type)

    if _verbose_enabled():
        print(
            "[fp8-nvfp4-mps-metal] NVFP4 CPU fallback "
            f"shape={tuple(result.shape)} dtype={output_type} time={elapsed:.4f}s"
        )

    return result


def install() -> bool:
    """Install the NVFP4 MPS CPU fallback.

    Returns True when the patch is active. Returns False when Comfy Kitchen is
    not importable, which is expected outside ComfyUI.
    """
    global _original_dequantize_nvfp4, _eager_dequantize_nvfp4, _installed
    if _installed:
        return True

    try:
        import comfy_kitchen as ck
        from comfy_kitchen.backends.eager.quantization import (
            dequantize_nvfp4 as eager_dequantize_nvfp4,
        )
    except Exception as exc:
        print(f"[fp8-nvfp4-mps-metal] Comfy Kitchen unavailable, skipping NVFP4 patch: {exc}")
        return False

    _original_dequantize_nvfp4 = ck.dequantize_nvfp4
    _eager_dequantize_nvfp4 = eager_dequantize_nvfp4
    ck.dequantize_nvfp4 = _dequantize_nvfp4_mps_safe
    _installed = True
    print("[fp8-nvfp4-mps-metal] Installed NVFP4 MPS CPU fallback")
    return True


def uninstall() -> None:
    """Restore the original Comfy Kitchen NVFP4 function."""
    global _original_dequantize_nvfp4, _eager_dequantize_nvfp4, _installed
    if not _installed:
        return

    try:
        import comfy_kitchen as ck

        ck.dequantize_nvfp4 = _original_dequantize_nvfp4
    finally:
        _original_dequantize_nvfp4 = None
        _eager_dequantize_nvfp4 = None
        _installed = False


def is_installed() -> bool:
    return _installed


def get_stats() -> dict[str, int | float | str | tuple[int, ...] | None]:
    return {
        "calls": _stats.calls,
        "metal_calls": _stats.metal_calls,
        "metal_seconds": _stats.metal_seconds,
        "fallback_calls": _stats.fallback_calls,
        "fallback_seconds": _stats.fallback_seconds,
        "elements": _stats.elements,
        "last_shape": _stats.last_shape,
        "last_output_type": _stats.last_output_type,
    }


def reset_stats() -> None:
    global _stats
    _stats = Nvfp4FallbackStats()
