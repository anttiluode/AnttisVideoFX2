"""Keyframe-relative causal refresh for the Layered PhaseRail video path.

The expensive person image is generated occasionally and transported cheaply by
PhaseRail between keyframes.  The useful control question is not whether a
stylised output looks like the webcam -- a marble/robot/etc person never will.
It is:

    has the live person geometry moved far enough away from the geometry that
    was accepted when this generated keyframe was anchored that buying another
    diffusion keyframe is justified?

Each fresh generated keyframe therefore gets a *live geometry reference* at the
moment it is accepted by PhaseRail.  The detector compares the current live
person with that fixed reference using contrast-normalised blurred edge
structure inside the ownership mask.  PhaseRail confidence and per-frame motion
are only small secondary terms.

A short per-keyframe warm-up estimates the ordinary segmentation/noise floor and
a one-sided leaky CUSUM-like detector accumulates only later upward change.  A
constant non-zero floor cannot mathematically force refreshes forever.

This remains a heuristic controller, not a calibrated probability or proof of
optimality.  Its purpose is directly testable: remain quiet around one accepted
pose, then fire when live geometry persistently leaves that pose.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

EPS = 1e-7


def _gray(image: np.ndarray) -> np.ndarray:
    x = np.asarray(image, dtype=np.float32)
    if x.ndim == 2:
        return x
    if x.shape[-1] == 1:
        return x[..., 0]
    # Inputs in AIvideoFX are BGR float images.
    return cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)


def _edge_signature(image: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """Low-cost contrast-normalised structure signature in [0,1]."""
    g = _gray(image)
    g = cv2.GaussianBlur(g, (0, 0), 1.25)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy).astype(np.float32)

    if mask is None:
        values = mag.ravel()
    else:
        m = np.asarray(mask, np.float32)
        values = mag[m > 0.08]
        if values.size < 16:
            values = mag.ravel()
    scale = float(np.percentile(values, 90)) if values.size else 0.0
    if scale < EPS:
        return np.zeros_like(mag, np.float32)
    return np.clip(mag / scale, 0.0, 1.0).astype(np.float32)


def normalized_structure_mismatch(
    current: np.ndarray,
    reference: np.ndarray,
    mask: np.ndarray | None = None,
) -> float:
    """Style/contrast-tolerant structural distance between two live geometries."""
    if current.shape[:2] != reference.shape[:2]:
        reference = cv2.resize(
            reference, (current.shape[1], current.shape[0]), interpolation=cv2.INTER_LINEAR
        )
    if mask is not None and mask.shape[:2] != current.shape[:2]:
        mask = cv2.resize(mask, (current.shape[1], current.shape[0]), interpolation=cv2.INTER_LINEAR)

    a = _edge_signature(current, mask)
    b = _edge_signature(reference, mask)
    diff = np.abs(a - b).astype(np.float32)
    if mask is None:
        return float(np.mean(diff))
    m = np.clip(np.asarray(mask, np.float32), 0.0, 1.0)
    denom = float(np.sum(m))
    if denom < EPS:
        return float(np.mean(diff))
    return float(np.sum(diff * m) / denom)


@dataclass
class RefreshReading:
    # ``structural`` is now live geometry drift from the accepted keyframe pose,
    # not live-vs-generated appearance mismatch.
    structural: float
    confidence_penalty: float
    motion_penalty: float
    instant: float
    baseline: float
    excess: float
    evidence: float
    threshold: float
    calibrated: bool
    triggered: bool


class CausalRefreshController:
    """Per-keyframe geometry reference + one-sided leaky change detector.

    The effect owns/captures the fixed live reference image.  This controller
    receives ``current_live`` and ``keyframe_live`` and turns their structural
    drift into a refresh event.

    A short warm-up estimates residual mask jitter / low PhaseRail confidence.
    After calibration::

        excess   = score - baseline * (1 + baseline_margin)
        evidence = max(0, decay * evidence + excess)

    The baseline follows slowly only while evidence is convincingly quiet, so
    gradual nuisance drift can be ignored without chasing away a real event.
    """

    def __init__(
        self,
        *,
        decay: float = 0.94,
        threshold: float = 0.20,
        min_keyframe_age: float = 0.80,
        baseline_alpha: float = 0.02,
        baseline_margin: float = 0.12,
        warmup_samples: int = 8,
    ) -> None:
        self.decay = float(decay)
        self.threshold = float(threshold)
        self.min_keyframe_age = float(min_keyframe_age)
        self.baseline_alpha = float(baseline_alpha)
        self.baseline_margin = float(baseline_margin)
        self.warmup_samples = max(3, int(warmup_samples))
        self.evidence = 0.0
        self.baseline: float | None = None
        self._warmup: list[float] = []
        self.last: RefreshReading | None = None

    def configure(self, *, decay: float, threshold: float, min_keyframe_age: float) -> None:
        self.decay = float(np.clip(decay, 0.0, 0.9995))
        self.threshold = max(1e-6, float(threshold))
        self.min_keyframe_age = max(0.0, float(min_keyframe_age))

    def reset(self) -> None:
        self.evidence = 0.0
        self.baseline = None
        self._warmup = []
        self.last = None

    def _change_update(self, instant: float) -> tuple[float, float, bool]:
        if len(self._warmup) < self.warmup_samples:
            self._warmup.append(float(instant))
            self.baseline = float(np.median(self._warmup))
            self.evidence = 0.0
            return float(self.baseline), 0.0, False

        assert self.baseline is not None
        baseline_before = float(self.baseline)
        excess = float(instant - baseline_before * (1.0 + self.baseline_margin))
        self.evidence = max(0.0, self.decay * self.evidence + excess)

        if self.evidence < 0.20 * self.threshold:
            a = float(np.clip(self.baseline_alpha, 0.0, 1.0))
            self.baseline = (1.0 - a) * baseline_before + a * float(instant)

        return float(self.baseline), excess, True

    def update(
        self,
        current_live: np.ndarray,
        keyframe_live: np.ndarray,
        *,
        mask: np.ndarray | None = None,
        phase_confidence: float = 1.0,
        motion: float = 0.0,
        max_motion: float = 3.5,
        keyframe_age: float = 0.0,
    ) -> RefreshReading:
        structural = normalized_structure_mismatch(current_live, keyframe_live, mask)
        confidence_penalty = float(np.clip(1.0 - float(phase_confidence), 0.0, 1.0))
        ratio = float(motion) / max(0.25, float(max_motion))
        motion_penalty = float(np.clip((ratio - 0.60) / 0.40, 0.0, 1.0))

        # Keyframe-relative geometry is the receiver.  Solver confidence/motion
        # are only tie-breakers; they must not recreate an absolute style floor.
        instant = float(np.clip(
            0.86 * structural + 0.10 * confidence_penalty + 0.04 * motion_penalty,
            0.0, 1.0,
        ))

        baseline, excess, calibrated = self._change_update(instant)
        triggered = bool(
            calibrated
            and float(keyframe_age) >= self.min_keyframe_age
            and self.evidence >= self.threshold
        )
        self.last = RefreshReading(
            structural=structural,
            confidence_penalty=confidence_penalty,
            motion_penalty=motion_penalty,
            instant=instant,
            baseline=baseline,
            excess=excess,
            evidence=float(self.evidence),
            threshold=float(self.threshold),
            calibrated=calibrated,
            triggered=triggered,
        )
        return self.last
