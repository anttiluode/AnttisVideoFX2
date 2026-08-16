"""Multiscale-life wrapper around the existing LayerPhaseRail.

This is deliberately a small experiment rather than a new renderer.  The
ordinary PhaseRail keeps two useful complex states:

    target_z  - the generated appearance after phase transport
    output_z  - the visible transported state

The working hypothesis is that output_z can age at different rates by scale.
This wrapper periodically re-locks selected output scales toward target_z while
keeping the current transported phase frame.  No new diffusion call is made.

Default scale lives (frames):
    fine=8, medium=25, low=60, coarse=100

Those numbers are hypotheses exposed to the UI; they are not measured facts.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from fx_phase_rail import LayerPhaseRail
from multiscale_life import ScaleLifeScheduler


class LayerPhaseRailLife:
    """PhaseRail plus independent per-scale repair clocks."""

    def __init__(self, size: int = 128, device: str = "cuda") -> None:
        self.base = LayerPhaseRail(size=size, device=device)
        self.scheduler = ScaleLifeScheduler()
        self.repair_enabled = True
        self.relock_strength = 1.0
        self.last_relocked: list[int] = []

    @property
    def size(self) -> int:
        return int(self.base.p.size)

    @property
    def device(self) -> str:
        return self.base.device

    @property
    def state(self):
        return self.base.state

    def reset(self) -> None:
        self.base.reset()
        self.scheduler.reset()
        self.last_relocked = []

    def set_target(self, target_bgr: np.ndarray) -> None:
        self.base.set_target(target_bgr)
        self.scheduler.reset()
        self.last_relocked = []

    def configure_lives(
        self,
        *,
        fine: int,
        medium: int,
        low: int,
        coarse: int,
        repair_enabled: bool = True,
        relock_strength: float = 1.0,
    ) -> None:
        self.scheduler.configure(
            fine=fine, medium=medium, low=low, coarse=coarse
        )
        self.repair_enabled = bool(repair_enabled)
        self.relock_strength = float(np.clip(relock_strength, 0.0, 1.0))

    def _relock_scale(self, scale_index: int) -> None:
        """Move one visible Gabor scale toward the transported reference.

        Blend amplitude and unit phasor separately.  A direct complex lerp can
        collapse amplitude when phases oppose, which would manufacture the very
        blur we are trying to measure.
        """
        st = self.base.state
        if st.output_z is None or st.target_z is None:
            return
        p = self.base.p
        C = int(st.output_z.shape[0])
        S, O, H = p.num_scales, p.num_orient, p.size
        out = st.output_z.reshape(C, S, O, H, H)
        tgt = st.target_z.reshape(C, S, O, H, H)
        si = int(scale_index)
        r = float(self.relock_strength)
        if r <= 0.0:
            return
        if r >= 0.999999:
            out[:, si] = tgt[:, si]
            return

        import torch

        o = out[:, si]
        t = tgt[:, si]
        amp = (1.0 - r) * o.abs() + r * t.abs()
        unit = (1.0 - r) * torch.exp(1j * torch.angle(o)) + r * torch.exp(
            1j * torch.angle(t)
        )
        unit = unit / (unit.abs() + 1e-7)
        out[:, si] = amp * unit

    def _internal_scale_health(self) -> list[float]:
        """1/(1+relative squared error) between visible and reference state."""
        st = self.base.state
        if st.output_z is None or st.target_z is None:
            return [1.0] * 5
        import torch

        p = self.base.p
        C = int(st.output_z.shape[0])
        S, O, H = p.num_scales, p.num_orient, p.size
        out = st.output_z.reshape(C, S, O, H, H)
        tgt = st.target_z.reshape(C, S, O, H, H)
        health: list[float] = []
        for si in range(S):
            d = out[:, si] - tgt[:, si]
            num = (d.abs() ** 2).sum()
            den = (tgt[:, si].abs() ** 2).sum() + 1e-7
            rel = float((num / den).detach().clamp(min=0.0, max=1e6).item())
            health.append(float(1.0 / (1.0 + rel)))
        return health

    def process(self, source_bgr: np.ndarray, **kwargs: Any):
        # Repair from the previous frame's already-transported reference before
        # applying this frame's motion.  Both repaired visible state and target
        # then receive the same new phase increment in LayerPhaseRail.process().
        due = self.scheduler.due() if self.repair_enabled else []
        repaired: list[int] = []
        if self.base.state.output_z is not None and self.base.state.target_z is not None:
            for si in due:
                self._relock_scale(si)
                repaired.append(int(si))
            if repaired:
                self.scheduler.repaired(repaired)

        out, coherence, metrics = self.base.process(source_bgr, **kwargs)
        self.scheduler.advance()
        self.last_relocked = repaired

        scale_health = self._internal_scale_health()
        ages = self.scheduler.group_ages()
        metrics = dict(metrics)
        metrics.update(
            life_relocked=tuple(repaired),
            life_scale_health=tuple(float(x) for x in scale_health),
            life_age_fine=int(ages["fine"]),
            life_age_medium=int(ages["medium"]),
            life_age_low=int(ages["low"]),
            life_age_coarse=int(ages["coarse"]),
        )
        return out, coherence, metrics
