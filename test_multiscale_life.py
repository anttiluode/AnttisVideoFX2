import sys
import types
import unittest

import cv2
import numpy as np

from multiscale_life import BandHealthMonitor, ScaleLifeScheduler, band_energies
from fx_core import FXContext, MapStore


def textured(size=128):
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    coarse = 0.35 + 0.18 * np.sin(xx * 0.035) + 0.14 * np.cos(yy * 0.028)
    fine = 0.12 * np.sin(xx * 1.35) * np.cos(yy * 1.10)
    x = np.clip(coarse + fine, 0.0, 1.0)
    return np.repeat(x[..., None], 3, axis=2).astype(np.float32)


class MultiscaleLifeTests(unittest.TestCase):
    def test_blur_kills_fine_before_coarse(self):
        a = textured()
        b = cv2.GaussianBlur(a, (0, 0), 3.0)
        ea = band_energies(a)
        eb = band_energies(b)
        fine_ret = eb.fine / ea.fine
        coarse_ret = eb.coarse / ea.coarse
        self.assertLess(fine_ret, 0.35)
        self.assertGreater(coarse_ret, 0.90)
        self.assertLess(fine_ret, coarse_ret)

    def test_health_monitor_anchors_at_one(self):
        a = textured()
        mon = BandHealthMonitor()
        h0 = mon.anchor(a)
        self.assertEqual(h0.as_dict(), {
            "coarse": 1.0, "low": 1.0, "medium": 1.0, "fine": 1.0
        })
        h1 = mon.update(cv2.GaussianBlur(a, (0, 0), 2.5))
        self.assertLess(h1.fine, h1.coarse)

    def test_default_lifetime_order_is_exact(self):
        s = ScaleLifeScheduler(fine=8, medium=25, low=60, coarse=100)
        seen = {0: None, 1: None, 2: None, 3: None, 4: None}
        for frame in range(1, 101):
            s.advance()
            for i in s.due():
                if seen[i] is None:
                    seen[i] = frame
        self.assertEqual(seen[0], 8)
        self.assertEqual(seen[1], 25)
        self.assertEqual(seen[2], 25)
        self.assertEqual(seen[3], 60)
        self.assertEqual(seen[4], 100)

    def test_repair_resets_only_due_scale_age(self):
        s = ScaleLifeScheduler(fine=3, medium=8, low=9, coarse=10)
        for _ in range(3):
            s.advance()
        self.assertEqual(s.due(), [0])
        s.repaired([0])
        self.assertEqual(s.ages.tolist(), [0, 3, 3, 3, 3])

    def test_effect_runs_with_fake_life_rail(self):
        class FakeLifeRail:
            def __init__(self, size=128, device="cpu"):
                self.size = size
                self.target = None
                self.n = 0

            def configure_lives(self, **kwargs):
                self.cfg = kwargs

            def set_target(self, target):
                self.target = target.copy()
                self.n = 0

            def process(self, source, **kwargs):
                self.n += 1
                out = self.target.copy() if self.target is not None else source.copy()
                coh = np.ones(source.shape[:2], np.float32)
                metrics = {
                    "confidence": 1.0,
                    "motion": 0.0,
                    "coherence": 1.0,
                    "removed": 0.0,
                    "life_relocked": (0,) if self.n == 9 else (),
                    "life_scale_health": (0.95, 0.97, 0.98, 0.99, 1.0),
                    "life_age_fine": self.n % 8,
                    "life_age_medium": self.n % 25,
                    "life_age_low": self.n % 60,
                    "life_age_coarse": self.n % 100,
                }
                return out.astype(np.float32), coh, metrics

        fake = types.ModuleType("fx_phase_rail_life")
        fake.LayerPhaseRailLife = FakeLifeRail
        old = sys.modules.get("fx_phase_rail_life")
        sys.modules["fx_phase_rail_life"] = fake
        try:
            from fx_multiscale_life import MultiscaleLifeLayer

            h, w = 96, 128
            store = MapStore()
            mask = np.ones((h, w), np.float32)
            person = textured(128)[:h, :w]
            background = np.zeros((h, w, 3), np.float32)
            background[..., 0] = 0.25
            store.put("mask", mask)
            store.put("person_style", person)
            store.put("background_style", background)

            fx = MultiscaleLifeLayer({
                "device": "cpu",
                "rail_size": "64",
                "show_life": True,
                "life_repair": True,
            })
            frame = np.zeros((h, w, 3), np.float32)
            ctx = FXContext(store, 1.0, 1, (h, w))
            out = fx.apply(frame, ctx)
            self.assertEqual(out.shape, frame.shape)
            self.assertEqual(out.dtype, np.float32)
            self.assertTrue(np.isfinite(out).all())
            self.assertGreater(float(out.mean()), 0.01)
        finally:
            if old is None:
                sys.modules.pop("fx_phase_rail_life", None)
            else:
                sys.modules["fx_phase_rail_life"] = old


if __name__ == "__main__":
    unittest.main()
