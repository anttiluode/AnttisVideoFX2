"""Source-anchored gather transport for AnttisVideoFX2.

Instead of recursively evolving appearance, keep one immutable sharp keyframe K0
and evolve only a backward address field Phi_t:

    Phi_t(x) = Phi_{t-1}(x + d_t(x))
    I_t(x)   = K0(Phi_t(x))

where d_t is the current->previous displacement estimated by the existing
PhaseRail Gabor motion solver.

The address field may drift or fold, but texture is always sampled from the
pristine source once per displayed frame.  This deliberately trades appearance
blur for geometric/address error so the two failure modes can be measured
separately.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from fx_phase_rail import EPS, TORCH_OK, TorchGaborPyramid

if TORCH_OK:  # pragma: no branch
    import torch
    import torch.nn.functional as F
else:  # pragma: no cover
    torch = None
    F = None


def identity_map(h: int, w: int) -> tuple[np.ndarray, np.ndarray]:
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    return xx, yy


def compose_backward_map_np(
    map_x: np.ndarray,
    map_y: np.ndarray,
    dx: np.ndarray,
    dy: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compose current->previous displacement with previous->anchor map."""
    h, w = map_x.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    sx = xx + np.asarray(dx, np.float32)
    sy = yy + np.asarray(dy, np.float32)
    mx = cv2.remap(map_x.astype(np.float32), sx, sy, cv2.INTER_LINEAR,
                   borderMode=cv2.BORDER_REPLICATE)
    my = cv2.remap(map_y.astype(np.float32), sx, sy, cv2.INTER_LINEAR,
                   borderMode=cv2.BORDER_REPLICATE)
    return mx.astype(np.float32), my.astype(np.float32)


def gather_from_anchor_np(anchor: np.ndarray, map_x: np.ndarray, map_y: np.ndarray) -> np.ndarray:
    return cv2.remap(
        np.asarray(anchor, np.float32), map_x.astype(np.float32), map_y.astype(np.float32),
        cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT,
    ).astype(np.float32)


def fullres_gather_from_rail_map(
    anchor: np.ndarray,
    rail_map_x: np.ndarray,
    rail_map_y: np.ndarray,
    meta,
) -> np.ndarray:
    """Gather the original full-resolution anchor using a low-res address map.

    The PhaseRail motion solver runs on a square letterboxed rail.  We convert
    the accumulated rail displacement (map - identity) back to full-frame pixel
    units, then remap the pristine full-resolution anchor exactly once.
    """
    x0, y0, nw, nh, w, h = meta
    H, W = rail_map_x.shape
    id_x, id_y = identity_map(H, W)
    ux = rail_map_x - id_x
    uy = rail_map_y - id_y
    ux = ux[y0:y0 + nh, x0:x0 + nw]
    uy = uy[y0:y0 + nh, x0:x0 + nw]
    ux_full = cv2.resize(ux, (w, h), interpolation=cv2.INTER_LINEAR) * (w / max(nw, 1))
    uy_full = cv2.resize(uy, (w, h), interpolation=cv2.INTER_LINEAR) * (h / max(nh, 1))
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    return cv2.remap(
        np.asarray(anchor, np.float32),
        xx + ux_full.astype(np.float32),
        yy + uy_full.astype(np.float32),
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    ).astype(np.float32)


@dataclass
class AddressMetrics:
    motion: float = 0.0
    confidence: float = 0.0
    displacement: float = 0.0
    stretch: float = 0.0


class AnchorAddressRail:
    """Use PhaseRail only as a motion sensor; evolve an address field instead."""

    def __init__(self, size: int = 128, device: str = "cuda") -> None:
        if not TORCH_OK:
            raise RuntimeError("AnchorGather requires torch")
        self.p = TorchGaborPyramid(size=size, orientations=6, device=device)
        self.previous_source_z: Optional[object] = None
        self.map_field: Optional[object] = None  # [2,H,W], pixel coordinates in anchor rail
        self._identity: Optional[object] = None

    @property
    def device(self) -> str:
        return str(self.p.device)

    def reset(self) -> None:
        self.previous_source_z = None
        self.map_field = None
        self._identity = None

    def _ensure_map(self):
        if self.map_field is not None:
            return
        H = self.p.size
        yy, xx = torch.meshgrid(
            torch.arange(H, device=self.p.device, dtype=torch.float32),
            torch.arange(H, device=self.p.device, dtype=torch.float32),
            indexing="ij",
        )
        self._identity = torch.stack([xx, yy], dim=0)
        self.map_field = self._identity.clone()

    def _sample_map(self, sx, sy):
        H = self.p.size
        gx = 2.0 * sx / max(H - 1, 1) - 1.0
        gy = 2.0 * sy / max(H - 1, 1) - 1.0
        grid = torch.stack([gx, gy], dim=-1)[None]
        return F.grid_sample(
            self.map_field[None], grid, mode="bilinear",
            padding_mode="border", align_corners=True,
        )[0]

    @torch.no_grad()
    def process(self, source_bgr: np.ndarray, max_displacement: float = 3.5):
        self._ensure_map()
        p = self.p
        src = p.to_device(np.ascontiguousarray(source_bgr.astype(np.float32)))
        gray = 0.0722 * src[..., 0] + 0.7152 * src[..., 1] + 0.2126 * src[..., 2]
        source_z, _, _ = p.analyze_gray(gray)

        if self.previous_source_z is None:
            dx = torch.zeros_like(gray)
            dy = torch.zeros_like(gray)
            confidence = torch.zeros_like(gray)
        else:
            dx, dy, confidence, _ = p.estimate_bound_phase_flow(
                source_z, self.previous_source_z, max_displacement
            )
            # Mirror the confidence gating used for PhaseRail's predicted phase.
            gate = (confidence * 1.75).clamp(0.0, 1.0)
            dx = dx * gate
            dy = dy * gate
            yy, xx = torch.meshgrid(
                torch.arange(p.size, device=p.device, dtype=torch.float32),
                torch.arange(p.size, device=p.device, dtype=torch.float32),
                indexing="ij",
            )
            self.map_field = self._sample_map(xx + dx, yy + dy)

        disp = self.map_field - self._identity
        # A cheap map-distortion diagnostic: local finite-difference departure
        # from identity.  This is geometry/address health, not texture health.
        dux_dx = disp[0, :, 1:] - disp[0, :, :-1]
        duy_dy = disp[1, 1:, :] - disp[1, :-1, :]
        stretch = 0.5 * (dux_dx.abs().mean() + duy_dy.abs().mean())
        metrics = AddressMetrics(
            motion=float(torch.sqrt(dx * dx + dy * dy).mean().item()),
            confidence=float(confidence.mean().item()),
            displacement=float(torch.sqrt(disp[0] ** 2 + disp[1] ** 2).mean().item()),
            stretch=float(stretch.item()),
        )
        self.previous_source_z = source_z
        field = self.map_field.detach().cpu().numpy().astype(np.float32)
        return field[0], field[1], metrics
