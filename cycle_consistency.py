"""Long-range forward/backward flow consistency receiver.

Jacobian strain can say a beautifully smooth *wrong* address map is healthy.
This receiver asks a different question using an independent optical-flow path:

    current --backward history--> anchor --forward history--> current_hat

and measures ``|current_hat-current|``.  Incremental forward and backward
Farneback flows are composed through time, so the check is long-range rather
than merely t<->t-1 consistency.  It is a diagnostic/gate: it can tell us that
correspondence is lost, not synthesize unseen anatomy.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from anchor_gather import compose_backward_map_np, identity_map

EPS = 1e-7


def _gray8(img: np.ndarray) -> np.ndarray:
    x = np.asarray(img)
    if x.ndim == 3:
        if x.dtype != np.uint8:
            x = np.clip(x.astype(np.float32) * 255.0, 0, 255).astype(np.uint8)
        return cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
    if x.dtype != np.uint8:
        x = np.clip(x.astype(np.float32) * 255.0, 0, 255).astype(np.uint8)
    return x


def _sample(field: np.ndarray, map_x: np.ndarray, map_y: np.ndarray) -> np.ndarray:
    return cv2.remap(
        np.asarray(field, np.float32),
        np.asarray(map_x, np.float32),
        np.asarray(map_y, np.float32),
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    ).astype(np.float32)


def cycle_error_from_maps(
    backward_x: np.ndarray,
    backward_y: np.ndarray,
    forward_x: np.ndarray,
    forward_y: np.ndarray,
) -> np.ndarray:
    """current->anchor->current loop error in current-grid pixels."""
    h, w = backward_x.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    x_hat = _sample(forward_x, backward_x, backward_y)
    y_hat = _sample(forward_y, backward_x, backward_y)
    return np.sqrt((x_hat - xx) ** 2 + (y_hat - yy) ** 2).astype(np.float32)


def project_rail_scalar(field: np.ndarray, meta, out_shape: tuple[int, int]) -> np.ndarray:
    """Crop letterbox padding and resize a rail scalar map to full-frame shape."""
    x0, y0, nw, nh, _, _ = meta
    x = np.asarray(field, np.float32)[y0:y0 + nh, x0:x0 + nw]
    oh, ow = map(int, out_shape)
    return cv2.resize(x, (ow, oh), interpolation=cv2.INTER_LINEAR).astype(np.float32)


def cycle_trust(error: np.ndarray, good_px: float = 0.45, bad_px: float = 2.5) -> np.ndarray:
    e = np.asarray(error, np.float32)
    lo, hi = float(good_px), max(float(good_px) + EPS, float(bad_px))
    t = np.clip((e - lo) / (hi - lo), 0.0, 1.0)
    smooth = t * t * (3.0 - 2.0 * t)
    return (1.0 - smooth).astype(np.float32)


@dataclass
class CycleReading:
    mean_error: float = 0.0
    p90_error: float = 0.0
    bad_fraction: float = 0.0


class LongRangeCycleMonitor:
    def __init__(self) -> None:
        self.prev_gray: np.ndarray | None = None
        self.backward_x: np.ndarray | None = None
        self.backward_y: np.ndarray | None = None
        self.forward_x: np.ndarray | None = None
        self.forward_y: np.ndarray | None = None
        self.last_error: np.ndarray | None = None
        self.last_reading = CycleReading()

    def reset(self) -> None:
        self.prev_gray = None
        self.backward_x = self.backward_y = None
        self.forward_x = self.forward_y = None
        self.last_error = None
        self.last_reading = CycleReading()

    @staticmethod
    def _flow(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return cv2.calcOpticalFlowFarneback(
            a, b, None,
            pyr_scale=0.5, levels=3, winsize=21, iterations=3,
            poly_n=5, poly_sigma=1.2, flags=0,
        ).astype(np.float32)

    def update(self, current_bgr: np.ndarray, *, bad_px: float = 2.5) -> tuple[np.ndarray, CycleReading]:
        cur = _gray8(current_bgr)
        h, w = cur.shape
        if self.prev_gray is None or self.prev_gray.shape != cur.shape:
            ix, iy = identity_map(h, w)
            self.backward_x, self.backward_y = ix.copy(), iy.copy()
            self.forward_x, self.forward_y = ix.copy(), iy.copy()
            self.prev_gray = cur
            self.last_error = np.zeros((h, w), np.float32)
            self.last_reading = CycleReading()
            return self.last_error, self.last_reading

        # Flow convention: flow_ab(x_a) is the displacement from coordinates in
        # frame a to the corresponding coordinates in frame b.
        fwd = self._flow(self.prev_gray, cur)   # previous -> current
        back = self._flow(cur, self.prev_gray) # current -> previous

        self.backward_x, self.backward_y = compose_backward_map_np(
            self.backward_x, self.backward_y, back[..., 0], back[..., 1]
        )

        # forward_map stores, for each anchor coordinate, where it currently
        # lives.  Sample the new prev->current displacement at those previous
        # coordinates and advance the values without resampling any appearance.
        px, py = self.forward_x, self.forward_y
        dfx = _sample(fwd[..., 0], px, py)
        dfy = _sample(fwd[..., 1], px, py)
        self.forward_x = (px + dfx).astype(np.float32)
        self.forward_y = (py + dfy).astype(np.float32)

        err = cycle_error_from_maps(
            self.backward_x, self.backward_y, self.forward_x, self.forward_y
        )
        self.last_error = err
        self.prev_gray = cur
        self.last_reading = CycleReading(
            mean_error=float(np.mean(err)),
            p90_error=float(np.percentile(err, 90)),
            bad_fraction=float(np.mean(err >= float(bad_px))),
        )
        return err, self.last_reading
