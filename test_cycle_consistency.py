import unittest

import numpy as np

from cycle_consistency import cycle_error_from_maps, cycle_trust


class CycleConsistencyTests(unittest.TestCase):
    def test_inverse_translation_closes_cycle(self):
        n = 80
        yy, xx = np.mgrid[0:n, 0:n].astype(np.float32)
        # current -> anchor is -4,+3; anchor -> current is +4,-3.
        bx, by = xx - 4.0, yy + 3.0
        fx, fy = xx + 4.0, yy - 3.0
        e = cycle_error_from_maps(bx, by, fx, fy)
        self.assertLess(float(np.mean(e[10:-10, 10:-10])), 1e-4)

    def test_wrong_smooth_map_is_detected(self):
        n = 80
        yy, xx = np.mgrid[0:n, 0:n].astype(np.float32)
        bx, by = xx - 4.0, yy
        # A perfectly smooth but wrong forward history. Jacobian strain alone
        # would call both maps healthy; the loop does not close.
        fx, fy = xx + 1.0, yy
        e = cycle_error_from_maps(bx, by, fx, fy)
        self.assertGreater(float(np.mean(e[10:-10, 10:-10])), 2.5)
        q = cycle_trust(e, good_px=0.4, bad_px=2.0)
        self.assertLess(float(np.mean(q[10:-10, 10:-10])), 0.1)


if __name__ == "__main__":
    unittest.main()
