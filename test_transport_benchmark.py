import unittest

import cv2
import numpy as np

from transport_benchmark import (
    duplicate_distance,
    frame_metrics,
    keyframe_budget,
)


class TransportBenchmarkTests(unittest.TestCase):
    def test_exact_duplicate_is_zero(self):
        x = np.zeros((80, 120, 3), np.uint8)
        x[20:50, 30:70] = 180
        self.assertAlmostEqual(duplicate_distance(x, x.copy()), 0.0, places=6)

    def test_motion_is_not_duplicate(self):
        a = np.zeros((80, 120, 3), np.uint8)
        b = a.copy()
        a[20:50, 30:70] = 200
        b[20:50, 42:82] = 200
        self.assertGreater(duplicate_distance(a, b), 1.0)

    def test_budget_requires_persistent_failure(self):
        vals = [32, 31, 20, 31, 19, 18, 17]
        # isolated dip at index 2 is forgiven; the 3-frame run begins at 4.
        self.assertEqual(keyframe_budget(vals, 25, higher_is_better=True, patience=3), 4)

    def test_identical_frames_score_as_identical(self):
        x = np.full((64, 64, 3), 127, np.uint8)
        x[10:30, 15:45] = 220
        m = frame_metrics(x, x.copy())
        self.assertLess(m["l1"], 1e-9)
        self.assertGreater(m["psnr"], 90.0)
        self.assertGreater(m["ssim"], 0.999)


if __name__ == "__main__":
    unittest.main()
