"""Detail-metabolism primitives for AnttisVideoFX2.

The question is deliberately narrower than generic motion error:

    how much fine prompt-defined visual detail has evaporated from a carried
    generated person since its last accepted detail state?

A fresh img2img result is treated as a *donor*, not as a replacement frame.
The donor is first aligned to the current carried image, then only its
medium/fine Laplacian content is transplanted. Coarse/low structure remains
owned by the living carrier.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

EPS = 1e-7


def _gray(img: np.ndarray) -> np.ndarray:
    x = np.asarray(img, np.float32)
    if x.ndim == 2:
        return x
    return cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)


def _gauss(img: np.ndarray, sigma: float) -> np.ndarray:
    return cv2.GaussianBlur(np.asarray(img, np.float32), (0, 0), float(sigma))


def _masked_mean(x: np.ndarray, mask: np.ndarray | None) -> float:
    a = np.asarray(x, np.float32)
    if mask is None:
        return float(np.mean(a))
    m = np.clip(np.asarray(mask, np.float32), 0.0, 1.0)
    if m.shape[:2] != a.shape[:2]:
        m = cv2.resize(m, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_LINEAR)
    if a.ndim == 3:
        m = m[..., None]
    den = float(np.sum(m))
    if a.ndim == 3:
        den *= a.shape[2]
    if den < EPS:
        return float(np.mean(a))
    return float(np.sum(a * m) / den)


def split_laplacian(img: np.ndarray, sigma_fine: float = 1.0, sigma_mid: float = 3.2):
    """Return low, mid, fine components whose sum reconstructs img."""
    x = np.asarray(img, np.float32)
    b1 = _gauss(x, sigma_fine)
    low = _gauss(x, sigma_mid)
    mid = b1 - low
    fine = x - b1
    return low, mid, fine


def band_energy(band: np.ndarray, mask: np.ndarray | None = None) -> float:
    return _masked_mean(np.asarray(band, np.float32) ** 2, mask)


def detail_ratio(img: np.ndarray, mask: np.ndarray | None = None) -> float:
    """Fine energy relative to medium energy; exposure mostly cancels."""
    _, mid, fine = split_laplacian(img)
    ef = band_energy(fine, mask)
    em = band_energy(mid, mask)
    return float(np.sqrt((ef + EPS) / (em + EPS)))


@dataclass
class DetailReading:
    ratio: float
    anchor_ratio: float
    health: float
    debt: float


class DetailDebtMonitor:
    """Relative fine-detail survival since the last accepted detail state."""

    def __init__(self) -> None:
        self.anchor_ratio: float | None = None
        self.last: DetailReading | None = None

    def reset(self) -> None:
        self.anchor_ratio = None
        self.last = None

    def anchor(self, image: np.ndarray, mask: np.ndarray | None = None) -> DetailReading:
        r = max(EPS, detail_ratio(image, mask))
        self.anchor_ratio = r
        self.last = DetailReading(ratio=r, anchor_ratio=r, health=1.0, debt=0.0)
        return self.last

    def update(self, image: np.ndarray, mask: np.ndarray | None = None) -> DetailReading:
        if self.anchor_ratio is None:
            return self.anchor(image, mask)
        r = detail_ratio(image, mask)
        health_raw = float(r / max(EPS, self.anchor_ratio))
        # Health may exceed 1 because motion/noise/artifacts can create energy.
        # Debt is one-sided: only loss relative to the accepted state counts.
        debt = float(np.clip(1.0 - health_raw, 0.0, 1.0))
        self.last = DetailReading(
            ratio=float(r), anchor_ratio=float(self.anchor_ratio),
            health=health_raw, debt=debt,
        )
        return self.last


def _normalize_masked(gray: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    g = np.asarray(gray, np.float32)
    if mask is None:
        vals = g.ravel()
    else:
        m = np.asarray(mask, np.float32)
        if m.shape != g.shape:
            m = cv2.resize(m, (g.shape[1], g.shape[0]), interpolation=cv2.INTER_LINEAR)
        vals = g[m > 0.15]
        if vals.size < 32:
            vals = g.ravel()
    mean = float(np.mean(vals)) if vals.size else 0.0
    std = float(np.std(vals)) if vals.size else 1.0
    return (g - mean) / max(0.03, std)


def _masked_for_alignment(
    gray: np.ndarray,
    mask: np.ndarray | None,
) -> np.ndarray:
    g = np.asarray(gray, np.float32)
    if mask is None:
        return g
    m = np.clip(np.asarray(mask, np.float32), 0.0, 1.0)
    if m.shape != g.shape:
        m = cv2.resize(m, (g.shape[1], g.shape[0]), interpolation=cv2.INTER_LINEAR)
    m = _gauss(m, 2.0)
    vals = g[m > 0.15]
    neutral = float(np.mean(vals)) if vals.size else float(np.mean(g))
    return g * m + neutral * (1.0 - m)


def align_donor_to_current(
    donor: np.ndarray,
    current: np.ndarray,
    mask: np.ndarray | None = None,
    max_flow: float = 18.0,
) -> tuple[np.ndarray, float]:
    """Warp donor into the current carrier's coordinate frame.

    Two stages are used because the img2img donor can finish several video
    frames after it was requested:

    1. phase correlation absorbs the dominant global translation;
    2. Farneback estimates a smaller dense residual on 8-bit normalized luma.

    This is still deliberately cheap. It is not a face correspondence model.
    """
    cur = np.asarray(current, np.float32)
    don = np.asarray(donor, np.float32)
    if don.shape[:2] != cur.shape[:2]:
        don = cv2.resize(don, (cur.shape[1], cur.shape[0]), interpolation=cv2.INTER_LINEAR)

    cg = _masked_for_alignment(_gauss(_gray(cur), 0.9), mask)
    dg = _masked_for_alignment(_gauss(_gray(don), 0.9), mask)
    h, w = cg.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)

    # cv2.phaseCorrelate(current, donor) returns the donor displacement.  To
    # express donor in current coordinates, sample donor at x+dx, y+dy.
    (dx0, dy0), response = cv2.phaseCorrelate(cg.astype(np.float32), dg.astype(np.float32))
    max_global = max(2.0, float(max_flow) * 1.5)
    dx0 = float(np.clip(dx0, -max_global, max_global))
    dy0 = float(np.clip(dy0, -max_global, max_global))
    global_aligned = cv2.remap(
        don,
        xx + dx0, yy + dy0,
        cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT,
    ).astype(np.float32)

    gg = _masked_for_alignment(_gauss(_gray(global_aligned), 0.9), mask)
    # Farneback on [0,1] images can underflow into effectively zero motion on
    # OpenCV 5. Normalize to 8-bit before the residual flow solve.
    c8 = np.clip((_normalize_masked(cg, mask) * 36.0) + 128.0, 0, 255).astype(np.uint8)
    g8 = np.clip((_normalize_masked(gg, mask) * 36.0) + 128.0, 0, 255).astype(np.uint8)
    flow = cv2.calcOpticalFlowFarneback(
        c8, g8, None, 0.5, 3, 21, 3, 5, 1.2, 0
    ).astype(np.float32)
    mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
    limiter = np.maximum(1.0, mag / max(1.0, float(max_flow)))
    flow[..., 0] /= limiter
    flow[..., 1] /= limiter
    warped = cv2.remap(
        global_aligned,
        xx + flow[..., 0], yy + flow[..., 1],
        cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT,
    )
    mean_motion = float(np.hypot(dx0, dy0) + np.mean(np.minimum(mag, max_flow)))
    return warped.astype(np.float32), mean_motion


def geometry_error(
    current: np.ndarray,
    donor_aligned: np.ndarray,
    mask: np.ndarray | None = None,
) -> float:
    """Low-frequency, contrast-normalized disagreement after alignment."""
    a = _normalize_masked(_gauss(_gray(current), 3.5), mask)
    b = _normalize_masked(_gauss(_gray(donor_aligned), 3.5), mask)
    return _masked_mean(np.abs(a - b), mask)


@dataclass
class DonorStats:
    fine_gain: float
    geometry_error: float
    mean_flow: float


def transplant_detail(
    current: np.ndarray,
    donor_aligned: np.ndarray,
    mask: np.ndarray | None,
    *,
    fine_mix: float = 0.90,
    mid_mix: float = 0.30,
) -> np.ndarray:
    """Keep carrier low/coarse state; replace selected donor frequencies."""
    cur = np.asarray(current, np.float32)
    don = np.asarray(donor_aligned, np.float32)
    cl, cm, cf = split_laplacian(cur)
    _, dm, df = split_laplacian(don)
    fm = float(np.clip(fine_mix, 0.0, 1.0))
    mm = float(np.clip(mid_mix, 0.0, 1.0))
    repaired = cl + (1.0 - mm) * cm + mm * dm + (1.0 - fm) * cf + fm * df
    repaired = np.clip(repaired, 0.0, 1.0)
    if mask is None:
        return repaired.astype(np.float32)
    m = np.clip(np.asarray(mask, np.float32), 0.0, 1.0)
    if m.shape[:2] != cur.shape[:2]:
        m = cv2.resize(m, (cur.shape[1], cur.shape[0]), interpolation=cv2.INTER_LINEAR)
    m = _gauss(m, 1.5)[..., None]
    return np.clip(cur + (repaired - cur) * m, 0.0, 1.0).astype(np.float32)


def evaluate_donor(
    current: np.ndarray,
    donor_aligned: np.ndarray,
    mask: np.ndarray | None,
    mean_flow: float = 0.0,
) -> DonorStats:
    _, _, cf = split_laplacian(current)
    _, _, df = split_laplacian(donor_aligned)
    gain = float(np.sqrt((band_energy(df, mask) + EPS) / (band_energy(cf, mask) + EPS)))
    return DonorStats(
        fine_gain=gain,
        geometry_error=geometry_error(current, donor_aligned, mask),
        mean_flow=float(mean_flow),
    )
