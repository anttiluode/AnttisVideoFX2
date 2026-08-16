import unittest

import cv2
import numpy as np

from detail_metabolism import (
    DetailDebtMonitor,
    align_donor_to_current,
    band_energy,
    evaluate_donor,
    split_laplacian,
    transplant_detail,
)


def synthetic(size=128):
    y, x = np.mgrid[0:size, 0:size].astype(np.float32)
    base = 0.22 + 0.28 * np.exp(-((x - 64) ** 2 + (y - 62) ** 2) / 1200.0)
    stripes = 0.10 * np.sin(x * 0.9) * np.exp(-((x - 64) ** 2 + (y - 58) ** 2) / 650.0)
    eyes = 0.20 * np.exp(-((x - 48) ** 2 + (y - 52) ** 2) / 15.0)
    eyes += 0.20 * np.exp(-((x - 80) ** 2 + (y - 52) ** 2) / 15.0)
    g = np.clip(base + stripes + eyes, 0, 1)
    img = np.stack([g * 0.75, g * 0.9, g], -1).astype(np.float32)
    mask = (((x - 64) / 44) ** 2 + ((y - 64) / 54) ** 2 < 1).astype(np.float32)
    return img, mask


class DetailMetabolismMathTests(unittest.TestCase):
    def test_blur_creates_detail_debt(self):
        sharp, mask = synthetic()
        blurred = cv2.GaussianBlur(sharp, (0, 0), 2.8)
        mon = DetailDebtMonitor()
        mon.anchor(sharp, mask)
        r = mon.update(blurred, mask)
        self.assertGreater(r.debt, 0.25)
        self.assertLess(r.health, 0.75)

    def test_global_gain_does_not_fake_large_debt(self):
        sharp, mask = synthetic()
        changed = np.clip(sharp * 0.62, 0, 1)
        mon = DetailDebtMonitor()
        mon.anchor(sharp, mask)
        r = mon.update(changed, mask)
        self.assertLess(r.debt, 0.08)

    def test_high_frequency_transplant_preserves_low_structure(self):
        donor, mask = synthetic()
        current = cv2.GaussianBlur(donor, (0, 0), 2.4)
        repaired = transplant_detail(current, donor, mask, fine_mix=0.95, mid_mix=0.35)
        cl, _, cf = split_laplacian(current)
        rl, _, rf = split_laplacian(repaired)
        self.assertLess(float(np.mean(np.abs(cl - rl))), 0.02)
        self.assertGreater(band_energy(rf, mask), band_energy(cf, mask) * 1.25)

    def test_flow_alignment_reduces_geometry_error_for_small_shift(self):
        current, mask = synthetic()
        M = np.float32([[1, 0, 5.0], [0, 1, -3.0]])
        donor = cv2.warpAffine(current, M, (128, 128), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REFLECT)
        before = evaluate_donor(current, donor, mask, 0.0).geometry_error
        aligned, flow = align_donor_to_current(donor, current, mask)
        after = evaluate_donor(current, aligned, mask, flow).geometry_error
        self.assertLess(after, before)


class DetailMailboxTests(unittest.TestCase):
    def test_mailbox_only_runs_after_request(self):
        import detail_metabolism_patch  # noqa: F401
        import fx_ai
        from fx_core import MapStore

        store = MapStore()
        cfg = fx_ai.AIConfig()
        worker = fx_ai.DiffusionWorker(store, cfg, lambda _: None)
        channel = worker._channel("detail_donor")
        self.assertFalse(worker._due("detail_donor", channel))

        image, _ = synthetic(64)
        store.put("detail_request_image", image)
        store.put("detail_request", np.asarray([1.0], np.float32))
        channel = worker._channel("detail_donor")
        self.assertTrue(worker._due("detail_donor", channel))

        worker._detail_request_done_stamp = store.stamp("detail_request")
        self.assertFalse(worker._due("detail_donor", channel))

    def test_effect_and_preset_register(self):
        import fx_detail_metabolism
        from fx_core import EFFECTS_BY_NAME, PRESETS

        self.assertIn("DetailMetabolismLayer", EFFECTS_BY_NAME)
        self.assertIn("Antti Detail Metabolism", PRESETS)


if __name__ == "__main__":
    unittest.main()
