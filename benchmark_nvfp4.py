from __future__ import annotations

import argparse
import os
import time

import torch

import fp8_mps_patch
import nvfp4_mps_patch


def _time_call(fn, warmup: int, runs: int) -> float:
    for _ in range(warmup):
        fn()
    if torch.backends.mps.is_available():
        torch.mps.synchronize()
    start = time.perf_counter()
    for _ in range(runs):
        fn()
    if torch.backends.mps.is_available():
        torch.mps.synchronize()
    return (time.perf_counter() - start) / max(runs, 1)


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark NVFP4 Metal vs CPU fallback.")
    parser.add_argument("--rows", type=int, default=1024)
    parser.add_argument("--cols", type=int, default=1024)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--runs", type=int, default=10)
    args = parser.parse_args()

    if not torch.backends.mps.is_available():
        raise SystemExit("MPS is not available")

    fp8_mps_patch.install()
    nvfp4_mps_patch.install()

    import comfy_kitchen as ck

    x = torch.randn(args.rows, args.cols, dtype=torch.float32).clamp(-2, 2)
    scale = torch.amax(x.abs()) / (448.0 * 6.0)
    qx, block_scales = ck.quantize_nvfp4(x, scale, pad_16x=False)

    qx_mps = qx.to("mps")
    scale_mps = scale.to("mps")
    block_scales_mps = block_scales.to("mps")

    expected = ck.dequantize_nvfp4(qx, scale, block_scales, torch.float16)

    def run_metal():
        os.environ["FP8_MPS_METAL_NVFP4_BACKEND"] = "metal"
        return ck.dequantize_nvfp4(qx_mps, scale_mps, block_scales_mps, torch.float16)

    def run_cpu():
        os.environ["FP8_MPS_METAL_NVFP4_BACKEND"] = "cpu"
        return ck.dequantize_nvfp4(qx_mps, scale_mps, block_scales_mps, torch.float16)

    metal_s = _time_call(run_metal, args.warmup, args.runs)
    cpu_s = _time_call(run_cpu, args.warmup, args.runs)
    metal_out = run_metal().cpu()
    diff = (metal_out - expected).abs()

    print(
        {
            "shape": (args.rows, args.cols),
            "metal_seconds": metal_s,
            "cpu_seconds": cpu_s,
            "speedup": cpu_s / metal_s if metal_s else None,
            "max_diff": diff.max().item(),
            "mean_diff": diff.float().mean().item(),
            "stats": nvfp4_mps_patch.get_stats(),
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
