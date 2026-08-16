"""Pure helpers for confidence-gated Anchor Gather.

Anchor Gather solved recursive texture blur by keeping a pristine appearance
source and transporting only an address field.  Its remaining failure is
correspondence: a stale/folded address map can keep sharp detail attached to the
wrong body part.  These helpers make that failure measurable and degrade it
into a soft live-geometry fallback instead of a Picasso-like sharp warp.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

EPS = 1e-7


def _gauss(img: np.ndarray, sigma: float) -> np.ndarray:
    return cv2.GaussianBlur(np.asarray(img, np.float32), (0, 0), float(sigma))


def _smoothstep(e0: float, e1: float, x: np.ndarray) -> np.ndarray:
    t = np.clip((np.asarray(x, np.float32) - float(e0)) / max(EPS, float(e1 - e0)), 0.0, 1.0)
    return (t * t * (3.0 - 2.0 * t)).astype(np.float32)


@dataclass
class AddressQualityReading:
    health: float
    bad_fraction: float
    strain_mean: float
    fold_fraction: float


def address_quality_map(
    map_x: np.ndarray,
    map_y: np.ndarray,
    *,
    meta=None,
    out_shape: tuple[int, int] | None = None,
    strain_scale: float = 3.5,
    det_floor: float = 0.05,
    det_good: float = 0.65,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return local address trust, Jacobian strain and determinant.

    A rigid translation should remain trustworthy: its Jacobian is identity.
    Shear/stretch/folding lowers trust.  Sampling outside the original anchor
    also lowers trust, because BORDER_REFLECT would otherwise manufacture a
    plausible-looking but false correspondence at newly exposed edges.
    """
    mx = np.asarray(map_x, np.float32)
    my = np.asarray(map_y, np.float32)
    if mx.shape != my.shape or mx.ndim != 2:
        raise ValueError("map_x and map_y must be same-shape 2-D arrays")

    dmx_dy, dmx_dx = np.gradient(mx)
    dmy_dy, dmy_dx = np.gradient(my)
    strain = np.sqrt(
        ((dmx_dx - 1.0) ** 2 + dmx_dy ** 2 + dmy_dx ** 2 + (dmy_dy - 1.0) ** 2)
        * 0.5
    ).astype(np.float32)
    det = (dmx_dx * dmy_dy - dmx_dy * dmy_dx).astype(np.float32)

    q = np.exp(-float(strain_scale) * strain).astype(np.float32)
    q *= _smoothstep(det_floor, det_good, det)

    h, w = mx.shape
    border = np.minimum.reduce([mx, my, (w - 1.0) - mx, (h - 1.0) - my])
    q *= _smoothstep(-0.5, 1.5, border)
    q = np.clip(q, 0.0, 1.0).astype(np.float32)

    if meta is not None:
        x0, y0, nw, nh, _, _ = meta
        q = q[y0:y0 + nh, x0:x0 + nw]
        strain = strain[y0:y0 + nh, x0:x0 + nw]
        det = det[y0:y0 + nh, x0:x0 + nw]
    if out_shape is not None:
        oh, ow = map(int, out_shape)
        q = cv2.resize(q, (ow, oh), interpolation=cv2.INTER_LINEAR)
        strain = cv2.resize(strain, (ow, oh), interpolation=cv2.INTER_LINEAR)
        det = cv2.resize(det, (ow, oh), interpolation=cv2.INTER_LINEAR)
    return q.astype(np.float32), strain.astype(np.float32), det.astype(np.float32)


def apply_mask_agreement(
    quality: np.ndarray,
    current_mask: np.ndarray,
    warped_anchor_mask: np.ndarray,
    *,
    disagreement_gain: float = 3.0,
) -> np.ndarray:
    """Penalize pixels where current ownership disagrees with transported ownership."""
    q = np.asarray(quality, np.float32)
    cur = np.clip(np.asarray(current_mask, np.float32), 0.0, 1.0)
    old = np.clip(np.asarray(warped_anchor_mask, np.float32), 0.0, 1.0)
    if cur.shape != q.shape:
        cur = cv2.resize(cur, (q.shape[1], q.shape[0]), interpolation=cv2.INTER_LINEAR)
    if old.shape != q.shape:
        old = cv2.resize(old, (q.shape[1], q.shape[0]), interpolation=cv2.INTER_LINEAR)
    agreement = np.exp(-float(disagreement_gain) * np.abs(cur - old)).astype(np.float32)
    return np.clip(q * agreement, 0.0, 1.0).astype(np.float32)


def summarize_quality(
    quality: np.ndarray,
    mask: np.ndarray | None = None,
    *,
    bad_threshold: float = 0.35,
    strain: np.ndarray | None = None,
    determinant: np.ndarray | None = None,
) -> AddressQualityReading:
    q = np.clip(np.asarray(quality, np.float32), 0.0, 1.0)
    if mask is None:
        w = np.ones_like(q, np.float32)
    else:
        w = np.clip(np.asarray(mask, np.float32), 0.0, 1.0)
        if w.shape != q.shape:
            w = cv2.resize(w, (q.shape[1], q.shape[0]), interpolation=cv2.INTER_LINEAR)
    den = float(np.sum(w))
    if den < EPS:
        den = float(q.size)
        w = np.ones_like(q, np.float32)
    health = float(np.sum(w * q) / den)
    bad_fraction = float(np.sum(w * (q < float(bad_threshold))) / den)
    strain_mean = 0.0
    if strain is not None:
        s = np.asarray(strain, np.float32)
        if s.shape != q.shape:
            s = cv2.resize(s, (q.shape[1], q.shape[0]), interpolation=cv2.INTER_LINEAR)
        strain_mean = float(np.sum(w * s) / den)
    fold_fraction = 0.0
    if determinant is not None:
        d = np.asarray(determinant, np.float32)
        if d.shape != q.shape:
            d = cv2.resize(d, (q.shape[1], q.shape[0]), interpolation=cv2.INTER_LINEAR)
        fold_fraction = float(np.sum(w * (d <= 0.0)) / den)
    return AddressQualityReading(health, bad_fraction, strain_mean, fold_fraction)


def confidence_fuse(
    carried: np.ndarray,
    live: np.ndarray,
    quality: np.ndarray,
    *,
    low_sigma: float = 5.0,
    live_geometry: float = 0.72,
    untrusted_detail: float = 0.08,
) -> np.ndarray:
    """Keep sharp anchor detail only where its address remains trustworthy.

    Low-trust pixels borrow only low-frequency structure from the live frame;
    stale high-frequency detail is attenuated.  Thus failure becomes soft/blurred
    rather than sharp-but-wrong.
    """
    car = np.asarray(carried, np.float32)
    liv = np.asarray(live, np.float32)
    q = np.clip(np.asarray(quality, np.float32), 0.0, 1.0)
    if q.shape != car.shape[:2]:
        q = cv2.resize(q, (car.shape[1], car.shape[0]), interpolation=cv2.INTER_LINEAR)
    q = _gauss(q, 1.2)
    q3 = q[..., None]

    carry_low = _gauss(car, low_sigma)
    live_low = _gauss(liv, low_sigma)
    detail = car - carry_low

    geom_w = float(np.clip(live_geometry, 0.0, 1.0)) * (1.0 - q3)
    detail_keep = float(np.clip(untrusted_detail, 0.0, 1.0)) + (
        1.0 - float(np.clip(untrusted_detail, 0.0, 1.0))
    ) * q3
    base = carry_low + geom_w * (live_low - carry_low)
    return np.clip(base + detail_keep * detail, 0.0, 1.0).astype(np.float32)


def _band(img: np.ndarray, band: str) -> np.ndarray:
    x = np.asarray(img, np.float32)
    b1 = _gauss(x, 1.0)
    low = _gauss(x, 3.2)
    if band == "fine":
        return x - b1
    if band == "mid":
        return b1 - low
    if band == "low":
        return low
    raise ValueError("band must be 'low', 'mid' or 'fine'")


def band_correlation(
    a: np.ndarray,
    b: np.ndarray,
    mask: np.ndarray | None = None,
    *,
    band: str = "mid",
) -> float:
    """Bounded, energy-normalized structural agreement in one spatial band."""
    aa = _band(a, band)
    bb = _band(b, band)
    if aa.shape != bb.shape:
        bb = cv2.resize(bb, (aa.shape[1], aa.shape[0]), interpolation=cv2.INTER_LINEAR)
    if mask is None:
        w = np.ones(aa.shape[:2], np.float32)
    else:
        w = np.clip(np.asarray(mask, np.float32), 0.0, 1.0)
        if w.shape != aa.shape[:2]:
            w = cv2.resize(w, (aa.shape[1], aa.shape[0]), interpolation=cv2.INTER_LINEAR)
    w3 = w[..., None] if aa.ndim == 3 else w
    den_w = float(np.sum(w))
    if den_w < EPS:
        return 0.0
    if aa.ndim == 3:
        den_w_ch = max(EPS, den_w)
        ma = np.sum(aa * w3, axis=(0, 1), keepdims=True) / den_w_ch
        mb = np.sum(bb * w3, axis=(0, 1), keepdims=True) / den_w_ch
    else:
        ma = float(np.sum(aa * w) / den_w)
        mb = float(np.sum(bb * w) / den_w)
    aa0 = aa - ma
    bb0 = bb - mb
    dot = float(np.sum(w3 * aa0 * bb0))
    ea = float(np.sum(w3 * aa0 * aa0))
    eb = float(np.sum(w3 * bb0 * bb0))
    if ea < EPS or eb < EPS:
        return 0.0
    return float(np.clip(dot / np.sqrt(ea * eb + EPS), -1.0, 1.0))
