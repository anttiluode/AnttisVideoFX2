import unittest

import cv2
import numpy as np

from anchor_gather import (
    compose_backward_map_np,
    gather_from_anchor_np,
    identity_map,
)


class AnchorGatherMathTests(unittest.TestCase):
    @staticmethod
    def texture(n=128):
        yy, xx = np.mgrid[0:n, 0:n].astype(np.float32)
        x = 0.5 + 0.12 * np.sin(xx * 0.9) + 0.10 * np.sin(yy * 1.1)
        x += 0.08 * np.sin((xx + yy) * 1.3)
        x += (((xx.astype(np.int32) // 2 + yy.astype(np.int32) // 2) % 2) * 0.16 - 0.08)
        return np.clip(x, 0, 1).astype(np.float32)

    def test_composed_map_carries_address_not_pixels(self):
        n = 64
        mx, my = identity_map(n, n)
        dx = np.full((n, n), -0.35, np.float32)
        dy = np.full((n, n), 0.20, np.float32)
        for _ in range(20):
            mx, my = compose_backward_map_np(mx, my, dx, dy)
        # OpenCV's remap interpolation tables quantize subpixel coordinates, so
        # this is deliberately a loose geometry check, not an exact arithmetic claim.
        self.assertAlmostEqual(float(np.median(mx[16:-16, 16:-16] - np.arange(n, dtype=np.float32)[None, 16:-16])),
                               -7.0, delta=0.8)
        yy = np.arange(n, dtype=np.float32)[:, None]
        self.assertAlmostEqual(float(np.median(my[16:-16, 16:-16] - yy[16:-16])),
                               4.0, delta=0.8)

    def test_anchor_gather_keeps_texture_where_recursive_raster_blurs(self):
        """Core kill gate: repeated subpixel raster resampling loses detail.

        The source-anchored path may accumulate coordinate error, but every
        displayed image is still one sample from the pristine source, so it
        should retain far more high-frequency energy.
        """
        img = self.texture(128)
        step_x, step_y, frames = 0.37, -0.23, 50

        iterative = img.copy()
        M = np.float32([[1, 0, step_x], [0, 1, step_y]])
        for _ in range(frames):
            iterative = cv2.warpAffine(
                iterative, M, (128, 128), flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT,
            )

        mx, my = identity_map(128, 128)
        dx = np.full((128, 128), -step_x, np.float32)
        dy = np.full((128, 128), -step_y, np.float32)
        for _ in range(frames):
            mx, my = compose_backward_map_np(mx, my, dx, dy)
        anchored = gather_from_anchor_np(img, mx, my)

        crop = (slice(24, -24), slice(24, -24))
        e_iter = float(cv2.Laplacian(iterative[crop], cv2.CV_32F).var())
        e_anchor = float(cv2.Laplacian(anchored[crop], cv2.CV_32F).var())
        self.assertGreater(e_anchor, 50.0 * max(e_iter, 1e-12))

    def test_effect_registers(self):
        import fx_anchor_gather  # noqa: F401
        from fx_core import EFFECTS_BY_NAME, PRESETS
        self.assertIn("AnchorGatherLayer", EFFECTS_BY_NAME)
        self.assertIn("Antti Anchor Gather", PRESETS)


if __name__ == "__main__":
    unittest.main()
