import sys
import types
import unittest

import cv2
import numpy as np

from causal_refresh import CausalRefreshController, normalized_structure_mismatch
from fx_core import FXContext, MapStore


def bars(horizontal=False, size=96):
    img = np.zeros((size, size, 3), np.float32)
    if horizontal:
        img[size // 2 - 5:size // 2 + 5, 12:size - 12] = 1.0
    else:
        img[12:size - 12, size // 2 - 5:size // 2 + 5] = 1.0
    return cv2.GaussianBlur(img, (0, 0), 0.8)


class FakeRail:
    """Tiny PhaseRail contract for effect integration tests."""
    def __init__(self, size=128, device="cpu"):
        self.size = size
        self.target = None

    def set_target(self, target):
        self.target = target.copy()

    def process(self, source, **kwargs):
        out = self.target.copy() if self.target is not None else source.copy()
        coherence = np.ones(source.shape[:2], np.float32)
        metrics = {"confidence": 1.0, "motion": 0.0, "coherence": 1.0, "removed": 0.0}
        return out.astype(np.float32), coherence, metrics


class CausalRefreshTests(unittest.TestCase):
    def test_identical_structure_is_dark(self):
        a = bars(False)
        m = np.ones(a.shape[:2], np.float32)
        self.assertLess(normalized_structure_mismatch(a, a.copy(), m), 1e-7)

    def test_global_contrast_is_mostly_nuisance(self):
        a = bars(False)
        b = np.clip(0.35 + 0.45 * a, 0, 1)
        m = np.ones(a.shape[:2], np.float32)
        self.assertLess(normalized_structure_mismatch(a, b, m), 0.03)

    def test_changed_geometry_is_visible(self):
        a = bars(False)
        b = bars(True)
        m = np.ones(a.shape[:2], np.float32)
        self.assertGreater(normalized_structure_mismatch(a, b, m), 0.05)

    def test_constant_nonzero_floor_does_not_eventually_trigger(self):
        """A fixed reference mismatch is nuisance after calibration, not a clock."""
        reference = bars(False)
        current = bars(True)
        m = np.ones(reference.shape[:2], np.float32)
        ctrl = CausalRefreshController(
            decay=0.95, threshold=0.08, min_keyframe_age=0.0,
            warmup_samples=8,
        )
        readings = []
        for _ in range(160):
            readings.append(ctrl.update(
                current, reference, mask=m, phase_confidence=0.65,
                motion=0.3, max_motion=3.5, keyframe_age=30.0,
            ))
        self.assertTrue(readings[-1].calibrated)
        self.assertGreater(readings[-1].baseline, 0.0)
        self.assertFalse(any(r.triggered for r in readings))
        self.assertLess(readings[-1].evidence, 0.01)

    def test_later_geometry_departure_triggers(self):
        reference = bars(False)
        moved = bars(True)
        m = np.ones(reference.shape[:2], np.float32)
        ctrl = CausalRefreshController(
            decay=0.95, threshold=0.08, min_keyframe_age=0.0,
            warmup_samples=8,
        )

        # Stay near the accepted keyframe pose first.
        for _ in range(20):
            r = ctrl.update(
                reference, reference, mask=m, phase_confidence=0.65,
                motion=0.1, max_motion=3.5, keyframe_age=10.0,
            )
            self.assertFalse(r.triggered)
        learned = r.baseline

        triggered = False
        for _ in range(40):
            r = ctrl.update(
                moved, reference, mask=m, phase_confidence=0.65,
                motion=0.2, max_motion=3.5, keyframe_age=10.0,
            )
            if r.triggered:
                triggered = True
                break
        self.assertTrue(triggered)
        self.assertGreater(r.structural, 0.05)
        self.assertGreater(r.instant, learned)
        self.assertGreater(r.excess, 0.0)

    def test_keyframe_age_gate_blocks_early_refresh(self):
        reference = bars(False)
        moved = bars(True)
        m = np.ones(reference.shape[:2], np.float32)
        ctrl = CausalRefreshController(
            decay=0.95, threshold=0.02, min_keyframe_age=2.0,
            warmup_samples=6,
        )
        for _ in range(10):
            ctrl.update(
                reference, reference, mask=m, phase_confidence=0.8,
                motion=0.0, max_motion=3.5, keyframe_age=0.5,
            )

        r = None
        for _ in range(20):
            r = ctrl.update(
                moved, reference, mask=m, phase_confidence=0.8,
                motion=1.0, max_motion=3.5, keyframe_age=0.5,
            )
        self.assertIsNotNone(r)
        self.assertGreater(r.evidence, r.threshold)
        self.assertFalse(r.triggered)

        r = ctrl.update(
            moved, reference, mask=m, phase_confidence=0.8,
            motion=1.0, max_motion=3.5, keyframe_age=2.5,
        )
        self.assertTrue(r.triggered)

    def test_reset_for_new_keyframe_recalibrates(self):
        a = bars(False)
        ctrl = CausalRefreshController(warmup_samples=5)
        for _ in range(8):
            r = ctrl.update(a, a, keyframe_age=10.0, phase_confidence=0.7)
        self.assertTrue(r.calibrated)
        self.assertIsNotNone(ctrl.baseline)
        ctrl.reset()
        self.assertIsNone(ctrl.baseline)
        self.assertEqual(ctrl.evidence, 0.0)
        r = ctrl.update(a, a, keyframe_age=10.0, phase_confidence=0.7)
        self.assertFalse(r.calibrated)
        self.assertFalse(r.triggered)

    def _with_fake_rail(self):
        fake = types.ModuleType("fx_phase_rail")
        fake.LayerPhaseRail = FakeRail
        old = sys.modules.get("fx_phase_rail")
        sys.modules["fx_phase_rail"] = fake
        return old

    @staticmethod
    def _restore_fake_rail(old):
        if old is None:
            sys.modules.pop("fx_phase_rail", None)
        else:
            sys.modules["fx_phase_rail"] = old

    def test_effect_runs_and_captures_live_keyframe_reference(self):
        old = self._with_fake_rail()
        try:
            from fx_causal_refresh import CausalPhaseRailLayer

            h, w = 96, 128
            store = MapStore()
            mask = np.ones((h, w), np.float32)
            person = np.zeros((h, w, 3), np.float32)
            person[..., 1] = 0.9
            background = np.zeros((h, w, 3), np.float32)
            background[..., 0] = 0.8
            store.put("mask", mask)
            store.put("person_style", person)
            store.put("background_style", background)

            effect = CausalPhaseRailLayer({
                "device": "cpu",
                "rail_size": "64",
                "show_refresh": True,
                "auto_refresh": True,
                "refresh_threshold": 10.0,
                "refresh_min_age": 0.0,
            })
            frame = np.zeros((h, w, 3), np.float32)
            ctx = FXContext(store, 1.0, 1, (h, w))
            out = effect.apply(frame, ctx)
            st = ctx.st(effect)
            self.assertEqual(out.shape, frame.shape)
            self.assertEqual(out.dtype, np.float32)
            self.assertTrue(np.isfinite(out).all())
            self.assertGreater(float(out.mean()), 0.01)
            self.assertIsNotNone(st.get("refresh_reference"))
            self.assertLess(st["refresh_reading"].structural, 1e-7)
        finally:
            self._restore_fake_rail(old)

    def test_effect_requests_new_keyframe_after_live_geometry_leaves_anchor(self):
        """Integration gate for the failure seen on webcam: old style stays frozen."""
        old = self._with_fake_rail()
        try:
            from fx_causal_refresh import CausalPhaseRailLayer

            h, w = 96, 96
            store = MapStore()
            store.put("mask", np.ones((h, w), np.float32))
            person = np.zeros((h, w, 3), np.float32)
            person[..., 1] = 0.8
            store.put("person_style", person)

            effect = CausalPhaseRailLayer({
                "device": "cpu",
                "rail_size": "64",
                "show_refresh": False,
                "auto_refresh": True,
                "refresh_threshold": 0.03,
                "refresh_min_age": 0.0,
                "refresh_decay": 0.95,
            })

            stable = bars(False, h)
            moved = bars(True, h)
            for i in range(16):
                effect.apply(stable, FXContext(store, float(i), i, (h, w)))
                self.assertIsNotNone(store.get("person_style"))

            fired = False
            for i in range(16, 60):
                effect.apply(moved, FXContext(store, float(i), i, (h, w)))
                if store.get("person_style") is None:
                    fired = True
                    break
            self.assertTrue(fired)
            st = store.state[effect.uid]
            self.assertEqual(st.get("refresh_count"), 1)
            self.assertTrue(st.get("refresh_pending"))
        finally:
            self._restore_fake_rail(old)


if __name__ == "__main__":
    unittest.main()
