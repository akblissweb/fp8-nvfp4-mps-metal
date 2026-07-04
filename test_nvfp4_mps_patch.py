import importlib.util
import unittest

try:
    import torch
except ImportError:  # pragma: no cover - exercised only outside ML envs
    torch = None

if torch is not None:
    import fp8_mps_patch
    import nvfp4_mps_patch
else:
    fp8_mps_patch = None
    nvfp4_mps_patch = None

HAS_COMFY_KITCHEN = importlib.util.find_spec("comfy_kitchen") is not None
HAS_TORCH = torch is not None


@unittest.skipUnless(HAS_TORCH, "torch is not installed")
@unittest.skipUnless(HAS_COMFY_KITCHEN, "comfy_kitchen is not installed")
class Nvfp4MpsPatchTest(unittest.TestCase):
    def tearDown(self):
        nvfp4_mps_patch.uninstall()
        fp8_mps_patch.uninstall()
        nvfp4_mps_patch.reset_stats()

    def test_cpu_path_still_uses_original_function(self):
        import comfy_kitchen as ck

        self.assertTrue(nvfp4_mps_patch.install())
        x = torch.randn(16, 16, dtype=torch.float32)
        scale = torch.amax(x.abs()) / (448.0 * 6.0)
        qx, block_scales = ck.quantize_nvfp4(x, scale, pad_16x=False)

        out = ck.dequantize_nvfp4(qx, scale, block_scales, torch.float16)

        self.assertEqual(out.device.type, "cpu")
        self.assertEqual(out.dtype, torch.float16)
        self.assertEqual(tuple(out.shape), (16, 16))
        self.assertTrue(torch.isfinite(out).all().item())
        self.assertEqual(nvfp4_mps_patch.get_stats()["fallback_calls"], 0)

    @unittest.skipUnless(HAS_TORCH and torch.backends.mps.is_available(), "MPS is not available")
    def test_mps_path_falls_back_to_cpu_and_returns_mps(self):
        import comfy_kitchen as ck

        fp8_mps_patch.install()
        self.assertTrue(nvfp4_mps_patch.install())
        x = torch.linspace(-1.0, 1.0, 256, dtype=torch.float32).reshape(16, 16)
        scale = torch.amax(x.abs()) / (448.0 * 6.0)
        qx, block_scales = ck.quantize_nvfp4(x, scale, pad_16x=False)
        expected = ck.dequantize_nvfp4(qx, scale, block_scales, torch.float16)

        out = ck.dequantize_nvfp4(
            qx.to("mps"),
            scale.to("mps"),
            block_scales.to("mps"),
            torch.float16,
        )

        self.assertEqual(out.device.type, "mps")
        self.assertEqual(out.dtype, torch.float16)
        self.assertEqual(tuple(out.shape), (16, 16))
        actual = out.cpu()
        self.assertTrue(torch.isfinite(actual).all().item())
        self.assertTrue(torch.allclose(actual, expected, atol=5e-4, rtol=5e-4))

        stats = nvfp4_mps_patch.get_stats()
        self.assertEqual(stats["metal_calls"], 1)
        self.assertEqual(stats["fallback_calls"], 0)
        self.assertEqual(stats["last_shape"], (16, 16))
        self.assertEqual(stats["last_output_type"], "torch.float16")


if __name__ == "__main__":
    unittest.main()
