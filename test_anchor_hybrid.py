import unittest

import numpy as np

from anchor_hybrid import (
    address_quality_map,
    apply_mask_agreement,
    band_correlation,
    confidence_fuse,
)


class AnchorHybridMathTests(unittest.TestCase):
    def test_identity_and_rigid_translation_keep_address_quality(self):
        n = 64
        yy, xx = np.mgrid[0:n, 0:n].astype(np.float32)
        q0, _, _ = address_quality_map(xx, yy)
        qt, _, _ = address_quality_map(xx + 5.0, yy - 3.0)
        self.assertGreater(float(np.mean(q0[4:-4, 4:-4])), 0.99)
        # Ignore newly exposed borders: a rigid translation is not deformation.
        self.assertGreater(float(np.mean(qt[10:-10, 10:-10])), 0.95)

    def test_shear_is_detected_as_address_deformation(self):
        n = 64
        yy, xx = np.mgrid[0:n, 0:n].astype(np.float32)
        q0, _, _ = address_quality_map(xx, yy)
        qs, _, _ = address_quality_map(xx + 0.35 * yy, yy)
        self.assertLess(float(np.mean(qs[8:-8, 8:-8])),
                        0.65 * float(np.mean(q0[8:-8, 8:-8])))

    def test_mask_disagreement_marks_new_or_stale_body_regions_untrusted(self):
        n = 64
        q = np.ones((n, n), np.float32)
        current = np.zeros((n, n), np.float32)
        old = np.zeros((n, n), np.float32)
        current[20:40, 20:40] = 1.0
        old[20:40, 26:46] = 1.0
        out = apply_mask_agreement(q, current, old, disagreement_gain=3.0)
        self.assertLess(float(np.mean(out[24:36, 20:25])), 0.10)
        self.assertGreater(float(np.mean(out[24:36, 27:39])), 0.95)

    def test_band_correlation_cannot_be_won_by_unrelated_sharp_noise(self):
        rng = np.random.default_rng(3)
        n = 96
        yy, xx = np.mgrid[0:n, 0:n].astype(np.float32)
        base = 0.5 + 0.22 * np.sin(xx * 0.8) + 0.18 * np.cos(yy * 0.65)
        img = np.repeat(np.clip(base, 0, 1)[..., None], 3, axis=2).astype(np.float32)
        noisy = rng.random(img.shape, dtype=np.float32)
        same = band_correlation(img, img, band="fine")
        unrelated = band_correlation(img, noisy, band="fine")
        self.assertGreater(same, 0.99)
        self.assertLess(abs(unrelated), 0.20)
        self.assertLessEqual(abs(same), 1.0)

    def test_low_trust_suppresses_stale_detail(self):
        n = 96
        yy, xx = np.mgrid[0:n, 0:n].astype(np.float32)
        checker = (((xx.astype(int) + yy.astype(int)) % 2) * 0.5).astype(np.float32)
        carried = np.repeat(checker[..., None], 3, axis=2)
        live = np.full_like(carried, 0.35)
        high = confidence_fuse(carried, live, np.ones((n, n), np.float32))
        low = confidence_fuse(carried, live, np.zeros((n, n), np.float32), untrusted_detail=0.05)
        e_high = float(np.var(high - high.mean(axis=(0, 1), keepdims=True)))
        e_low = float(np.var(low - low.mean(axis=(0, 1), keepdims=True)))
        self.assertLess(e_low, 0.15 * e_high)

    def test_effect_registers(self):
        import fx_anchor_hybrid  # noqa: F401
        from fx_core import EFFECTS_BY_NAME, PRESETS
        self.assertIn("AnchorHybridLayer", EFFECTS_BY_NAME)
        self.assertIn("Antti Anchor Hybrid", PRESETS)


if __name__ == "__main__":
    unittest.main()
