"""Detail-metabolism primitives for AnttisVideoFX2.

The question is deliberately narrower than generic motion error:

    how much fine prompt-defined visual detail has evaporated from a carried
    generated person since its last accepted detail state?

A fresh img2img result is treated as a *donor*, not as a replacement frame.
The donor is first flow-aligned to the current carried image, then only its
medium/fine Laplacian content is transplanted.  Coarse/low structure remains
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


def align_donor_to_current(
    donor: np.ndarray,
    current: np.ndarray,
    mask: np.ndarray | None = None,
    max_flow: float = 18.0,
) -> tuple[np.ndarray, float]:
    """Dense-flow warp donor into the current carrier's coordinate frame.

    Farneback is intentionally a cheap first attempt.  The donor was generated
    from a recent carried frame, so this only needs to absorb the motion that
    happened while diffusion was running, not solve arbitrary correspondence.
    """
    cur = np.asarray(current, np.float32)
    don = np.asarray(donor, np.float32)
    if don.shape[:2] != cur.shape[:2]:
        don = cv2.resize(don, (cur.shape[1], cur.shape[0]), interpolation=cv2.INTER_LINEAR)
    cg = _gauss(_gray(cur), 0.9)
    dg = _gauss(_gray(don), 0.9)
    if mask is not None:
        m = np.clip(np.asarray(mask, np.float32), 0.0, 1.0)
        if m.shape != cg.shape:
            m = cv2.resize(m, (cg.shape[1], cg.shape[0]), interpolation=cv2.INTER_LINEAR)
        m = _gauss(m, 2.0)
        neutral_c = float(np.mean(cg[m > 0.15])) if np.any(m > 0.15) else float(np.mean(cg))
        neutral_d = float(np.mean(dg[m > 0.15])) if np.any(m > 0.15) else float(np.mean(dg))
        cg = cg * m + neutral_c * (1.0 - m)
        dg = dg * m + neutral_d * (1.0 - m)

    # Flow from current -> donor gives donor sampling coordinates for each
    # current pixel: donor(x + flow_x, y + flow_y).
    flow = cv2.calcOpticalFlowFarneback(
        cg, dg, None, 0.5, 3, 21, 3, 5, 1.2, 0
    ).astype(np.float32)
    mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
    limiter = np.maximum(1.0, mag / max(1.0, float(max_flow)))
    flow[..., 0] /= limiter
    flow[..., 1] /= limiter
    h, w = cg.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    warped = cv2.remap(
        don,
        xx + flow[..., 0], yy + flow[..., 1],
        cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT,
    )
    return warped.astype(np.float32), float(np.mean(np.minimum(mag, max_flow)))


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
