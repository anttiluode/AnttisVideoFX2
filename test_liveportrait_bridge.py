import unittest

import numpy as np


class LivePortraitBridgeTests(unittest.TestCase):
    def test_runtime_patch_installs_bidirectional_bridge(self):
        import fx_ai
        import liveportrait_bridge_patch  # noqa: F401
        self.assertEqual(fx_ai.PortraitBridgeWorker.__name__, "WiredPortraitBridgeWorker")

    def test_assist_effect_registers(self):
        import fx_liveportrait_assist  # noqa: F401
        from fx_core import EFFECTS_BY_NAME, PRESETS
        self.assertIn("LivePortraitAssistLayer", EFFECTS_BY_NAME)
        self.assertIn("Antti LivePortrait Assist", PRESETS)

    def test_alpha_shaper_is_bounded(self):
        from fx_liveportrait_assist import LivePortraitAssistLayer
        a = np.zeros((64, 64), np.float32)
        a[20:40, 20:40] = 1.0
        out = LivePortraitAssistLayer._shape_alpha(a, expand=2, feather=2.0)
        self.assertGreaterEqual(float(out.min()), 0.0)
        self.assertLessEqual(float(out.max()), 1.0)
        self.assertGreater(float(out.sum()), float(a.sum()))


if __name__ == "__main__":
    unittest.main()
