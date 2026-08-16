"""Multiscale representation-health instrumentation for AnttisVideoFX2.

The working hypothesis is deliberately simple and falsifiable: a transported
synthetic appearance may remain geometrically useful after its fine visual
information has begun to die.  This module does not decide when to regenerate.
It only exposes four cheap, separately observable bands so the live experiment
can tell us whether they really have different lifetimes.

The initial labels are:

    coarse / silhouette-ish structure
    low-frequency appearance
    medium structure
    fine texture

These are image-space Gaussian/Laplacian bands, not claims about semantic
features.  The default 100/60/25/8 frame lifetimes live in ScaleLifeScheduler;
they are hypotheses to test, not measured constants.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

EPS = 1e-7


@dataclass(frozen=True)
class BandEnergies:
    coarse: float
    low: float
    medium: float
    fine: float

    def as_dict(self) -> dict[str, float]:
        return {
            "coarse": float(self.coarse),
            "low": float(self.low),
            "medium": float(self.medium),
            "fine": float(self.fine),
        }


@dataclass(frozen=True)
class BandHealth:
    coarse: float
    low: float
    medium: float
    fine: float

    def as_dict(self) -> dict[str, float]:
        return {
            "coarse": float(self.coarse),
            "low": float(self.low),
            "medium": float(self.medium),
            "fine": float(self.fine),
        }


def _mask3(mask: np.ndarray | None, shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    if mask is None:
        return np.ones((h, w, 1), np.float32)
    m = np.asarray(mask, dtype=np.float32)
    if m.shape[:2] != (h, w):
        m = cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR)
    return np.clip(m, 0.0, 1.0)[..., None]


def _rms(x: np.ndarray, mask3: np.ndarray) -> float:
    # The same ownership mask is used for every band so their *change over time*
    # is meaningful.  We intentionally do not compare absolute values across
    # bands; a high-frequency residual naturally has less energy than a lowpass.
    w = np.broadcast_to(mask3, x.shape)
    denom = float(np.sum(w))
    if denom < EPS:
        return 0.0
    return float(np.sqrt(np.sum((x * x) * w) / denom + EPS))


def laplacian_bands(image: np.ndarray) -> dict[str, np.ndarray]:
    """Four cheap, overlapping visual scales from a BGR float image.

    The sigmas are in pixels at the effect's working resolution.  They are not
    tied to PhaseRail's Gabor bank; this sensor is intentionally independent so
    it can catch failure modes that the internal phasor state misses.
    """
    x = np.asarray(image, dtype=np.float32)
    if x.ndim != 3 or x.shape[2] != 3:
        raise ValueError("expected HxWx3 BGR image")
    g1 = cv2.GaussianBlur(x, (0, 0), 1.15)
    g4 = cv2.GaussianBlur(x, (0, 0), 3.5)
    g10 = cv2.GaussianBlur(x, (0, 0), 9.0)
    return {
        "fine": x - g1,
        "medium": g1 - g4,
        "low": g4 - g10,
        "coarse": g10,
    }


def band_energies(image: np.ndarray, mask: np.ndarray | None = None) -> BandEnergies:
    bands = laplacian_bands(image)
    m3 = _mask3(mask, image.shape[:2])
    return BandEnergies(
        coarse=_rms(bands["coarse"], m3),
        low=_rms(bands["low"], m3),
        medium=_rms(bands["medium"], m3),
        fine=_rms(bands["fine"], m3),
    )


class BandHealthMonitor:
    """Track visible band-energy retention relative to a fresh carried frame.

    This is a sensor, not a controller.  A ratio below 1 means that band has
    less RMS energy than it had at anchor time; values above 1 are retained in
    the raw reading because motion/noise can genuinely add energy.  The HUD may
    clip them for display.
    """

    def __init__(self) -> None:
        self.reference: BandEnergies | None = None
        self.last_energy: BandEnergies | None = None
        self.last_health: BandHealth | None = None

    def reset(self) -> None:
        self.reference = None
        self.last_energy = None
        self.last_health = None

    def anchor(self, image: np.ndarray, mask: np.ndarray | None = None) -> BandHealth:
        e = band_energies(image, mask)
        self.reference = e
        self.last_energy = e
        self.last_health = BandHealth(1.0, 1.0, 1.0, 1.0)
        return self.last_health

    def update(self, image: np.ndarray, mask: np.ndarray | None = None) -> BandHealth:
        if self.reference is None:
            return self.anchor(image, mask)
        e = band_energies(image, mask)
        r = self.reference

        def ratio(cur: float, ref: float) -> float:
            return float(cur / max(ref, EPS))

        h = BandHealth(
            coarse=ratio(e.coarse, r.coarse),
            low=ratio(e.low, r.low),
            medium=ratio(e.medium, r.medium),
            fine=ratio(e.fine, r.fine),
        )
        self.last_energy = e
        self.last_health = h
        return h


class ScaleLifeScheduler:
    """Independent ages for PhaseRail's five Gabor scales.

    PhaseRail frequencies are ordered fine -> coarse:
        0.32, 0.18, 0.10, 0.055, 0.03 cycles/pixel.

    We map them to the four working lifetimes:
        scale 0        fine      8 frames
        scales 1, 2    medium   25 frames
        scale 3        low      60 frames
        scale 4        coarse  100 frames
    """

    def __init__(
        self,
        *,
        fine: int = 8,
        medium: int = 25,
        low: int = 60,
        coarse: int = 100,
        num_scales: int = 5,
    ) -> None:
        if num_scales != 5:
            raise ValueError("v0.1 scheduler expects the current 5-scale PhaseRail bank")
        self.num_scales = int(num_scales)
        self.configure(fine=fine, medium=medium, low=low, coarse=coarse)
        self.ages = np.zeros(self.num_scales, dtype=np.int32)

    def configure(self, *, fine: int, medium: int, low: int, coarse: int) -> None:
        vals = [fine, medium, medium, low, coarse]
        self.lifetimes = np.asarray([max(1, int(v)) for v in vals], dtype=np.int32)

    def reset(self) -> None:
        self.ages[:] = 0

    def due(self) -> list[int]:
        return [int(i) for i in np.flatnonzero(self.ages >= self.lifetimes)]

    def repaired(self, scales: list[int]) -> None:
        for i in scales:
            self.ages[int(i)] = 0

    def advance(self) -> None:
        self.ages += 1

    def group_ages(self) -> dict[str, int]:
        return {
            "fine": int(self.ages[0]),
            "medium": int(max(self.ages[1], self.ages[2])),
            "low": int(self.ages[3]),
            "coarse": int(self.ages[4]),
        }
