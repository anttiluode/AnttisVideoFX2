"""Experimental multiscale-lifetime video effect for AnttisVideoFX2.

This sibling of Antti Layered PhaseRail asks a different question from causal
refresh: can one generated person persist as a *living representation* if its
frequency bands are allowed to have different ages and are repaired from the
transported phasor reference instead of buying a whole new diffusion frame?

Default hypotheses:
    fine texture              8 frames
    medium structure         25 frames
    low-frequency appearance 60 frames
    coarse geometry         100 frames

The effect also measures visible image-space band-energy retention.  Repair can
be disabled live, giving an immediate A/B without changing the generated
keyframe or the motion path.
"""
from __future__ import annotations

import cv2
import numpy as np

from fx_core import (
    EFFECT_CLASSES,
    EFFECTS_BY_NAME,
    PRESETS,
    AnttisDeepfakeLayer,
    Bloom,
    ColorGrade,
    Param,
    blur,
)
from multiscale_life import BandHealthMonitor


class MultiscaleLifeLayer(AnttisDeepfakeLayer):
    name = "Antti Multiscale Life"
    blurb = (
        "A generated person with independently aging spectral bands. Fine, "
        "medium, low and coarse Gabor scales periodically re-lock to the "
        "separately transported PhaseRail reference. No extra diffusion call is "
        "made. The four HUD bars measure visible band-energy retention so the "
        "8/25/60/100-frame lifetime hypothesis can fail honestly."
    )
    params = AnttisDeepfakeLayer.params + [
        Param("life_repair", "Multiscale repair", "bool", True),
        Param("life_fine", "Fine life (frames)", "int", 8, 1, 240),
        Param("life_medium", "Medium life (frames)", "int", 25, 1, 300),
        Param("life_low", "Low life (frames)", "int", 60, 1, 600),
        Param("life_coarse", "Coarse life (frames)", "int", 100, 1, 1200),
        Param("life_relock", "Repair strength", "float", 1.0, 0.0, 1.0),
        Param("show_life", "Show life monitor", "bool", True),
    ]

    @staticmethod
    def _health_monitor(st) -> BandHealthMonitor:
        monitor = st.get("life_health_monitor")
        if monitor is None:
            monitor = BandHealthMonitor()
            st["life_health_monitor"] = monitor
        return monitor

    @staticmethod
    def _group_internal_health(metrics: dict) -> dict[str, float]:
        z = metrics.get("life_scale_health", (1.0, 1.0, 1.0, 1.0, 1.0))
        if len(z) != 5:
            z = (1.0, 1.0, 1.0, 1.0, 1.0)
        return {
            "fine": float(z[0]),
            "medium": float(0.5 * (z[1] + z[2])),
            "low": float(z[3]),
            "coarse": float(z[4]),
        }

    @staticmethod
    def _draw_life_hud(
        out: np.ndarray,
        health,
        metrics: dict,
        lives: dict[str, int],
        repair_enabled: bool,
    ) -> np.ndarray:
        if health is None:
            return out
        hud = np.clip(out * 255.0, 0, 255).astype(np.uint8)
        h, w = hud.shape[:2]
        visible = health.as_dict()
        internal = MultiscaleLifeLayer._group_internal_health(metrics)
        ages = {
            "fine": int(metrics.get("life_age_fine", 0)),
            "medium": int(metrics.get("life_age_medium", 0)),
            "low": int(metrics.get("life_age_low", 0)),
            "coarse": int(metrics.get("life_age_coarse", 0)),
        }
        relocked = set(int(x) for x in metrics.get("life_relocked", ()))
        relock_groups = set()
        if 0 in relocked:
            relock_groups.add("fine")
        if 1 in relocked or 2 in relocked:
            relock_groups.add("medium")
        if 3 in relocked:
            relock_groups.add("low")
        if 4 in relocked:
            relock_groups.add("coarse")

        labels = ["coarse", "low", "medium", "fine"]
        x0 = 12
        width = min(330, max(150, w - 24))
        bar_h = 8
        gap = 22
        total_h = gap * len(labels) + 18
        y0 = max(18, h - total_h - 8)
        mode = "REPAIR" if repair_enabled else "OBSERVE"
        cv2.putText(
            hud,
            f"MULTISCALE LIFE  {mode}",
            (x0, y0),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        y = y0 + 14
        for name in labels:
            v = float(visible.get(name, 1.0))
            zi = float(internal.get(name, 1.0))
            age = ages[name]
            life = max(1, int(lives[name]))
            fill = int(width * np.clip(v, 0.0, 1.0))
            cv2.rectangle(hud, (x0, y), (x0 + width, y + bar_h), (18, 18, 18), -1)
            # Colour is intentionally fixed and semantically boring; the number
            # is the measurement.  White flash marks an actual internal relock.
            cv2.rectangle(hud, (x0, y), (x0 + fill, y + bar_h), (80, 210, 80), -1)
            if name in relock_groups:
                cv2.rectangle(hud, (x0, y), (x0 + width, y + bar_h), (255, 255, 255), 1)
            text = f"{name:6s} vis {v:4.2f}  z {zi:4.2f}  age {age:3d}/{life}"
            cv2.putText(
                hud,
                text,
                (x0, y - 3),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.36,
                (5, 5, 5),
                3,
                cv2.LINE_AA,
            )
            cv2.putText(
                hud,
                text,
                (x0, y - 3),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.36,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            y += gap
        return hud.astype(np.float32) * np.float32(1.0 / 255.0)

    def apply(self, img, ctx):
        mask = ctx.mask()
        if mask is None:
            return img
        person_ai = ctx.map("person_style")
        background_ai = ctx.map("background_style")
        if person_ai is None and background_ai is None:
            return img

        m = self._ownership(
            mask, int(self.p("mask_expand")), float(self.p("mask_feather"))
        )
        if self.p("show_mask"):
            return np.repeat(m[..., None], 3, axis=2)
        m3 = m[..., None]

        person_target = person_ai if person_ai is not None else img
        background = background_ai if background_ai is not None else img
        background = img + (background - img) * float(self.p("background_mix"))
        background = self._move_background(
            background, self.p("back_x"), self.p("back_y"), self.p("back_zoom")
        )

        neutral = np.full_like(img, 0.12, dtype=np.float32)
        source_owned = img * m3 + neutral * (1.0 - m3)
        target_owned = person_target * m3 + neutral * (1.0 - m3)

        size = int(self.p("rail_size"))
        device = str(self.p("device"))
        st = ctx.st(self)
        rail = st.get("life_rail")
        if rail is None or st.get("life_rail_size") != size or st.get("life_device") != device:
            try:
                from fx_phase_rail_life import LayerPhaseRailLife

                rail = LayerPhaseRailLife(size=size, device=device)
                st.update(
                    life_rail=rail,
                    life_rail_size=size,
                    life_device=device,
                    life_person_stamp=-1.0,
                    life_error="",
                    life_needs_anchor=True,
                )
                self._health_monitor(st).reset()
            except Exception as exc:
                st["life_error"] = str(exc)
                front = person_target
                return np.clip(front * m3 + background * (1.0 - m3), 0.0, 1.0)

        rail.configure_lives(
            fine=int(self.p("life_fine")),
            medium=int(self.p("life_medium")),
            low=int(self.p("life_low")),
            coarse=int(self.p("life_coarse")),
            repair_enabled=bool(self.p("life_repair")),
            relock_strength=float(self.p("life_relock")),
        )

        source_small, meta = self._letterbox(source_owned, size)
        target_small, _ = self._letterbox(target_owned, size)
        stamp = ctx.stamp("person_style")
        if stamp != st.get("life_person_stamp"):
            rail.set_target(target_small)
            st["life_person_stamp"] = stamp
            st["life_needs_anchor"] = True
            self._health_monitor(st).reset()

        try:
            carried_small, coherence, metrics = rail.process(
                source_small,
                phase_lock=float(self.p("phase_lock")),
                style_strength=float(self.p("style_strength")),
                nullspace_strength=float(self.p("nullspace")),
                structure_follow=float(self.p("structure")),
                detail_follow=float(self.p("detail")),
                max_displacement=float(self.p("max_motion")),
            )
            st["life_metrics"] = metrics
            st["life_coherence"] = coherence
            carried = self._unletterbox(carried_small, meta)
        except Exception as exc:
            st["life_error"] = str(exc)
            carried = person_target
            metrics = {}

        monitor = self._health_monitor(st)
        if st.get("life_needs_anchor", False):
            health = monitor.anchor(carried, m)
            st["life_needs_anchor"] = False
        else:
            health = monitor.update(carried, m)
        st["life_health"] = health

        person_mix = float(self.p("person_mix"))
        front = img + (carried - img) * person_mix
        rescue = float(self.p("edge_live"))
        if rescue > 0 and float(self.p("mask_feather")) > 0:
            inner = blur(m, max(1.0, float(self.p("mask_feather")) * 1.8))
            edge = np.clip(np.abs(m - inner) * 4.0, 0.0, 1.0)[..., None]
            front = front * (1.0 - edge * rescue) + img * (edge * rescue)

        out = np.clip(front * m3 + background * (1.0 - m3), 0.0, 1.0)
        if bool(self.p("show_life")):
            lives = {
                "fine": int(self.p("life_fine")),
                "medium": int(self.p("life_medium")),
                "low": int(self.p("life_low")),
                "coarse": int(self.p("life_coarse")),
            }
            out = self._draw_life_hud(
                out, health, metrics, lives, bool(self.p("life_repair"))
            )
        return out


def register() -> None:
    if MultiscaleLifeLayer not in EFFECT_CLASSES:
        try:
            idx = EFFECT_CLASSES.index(AnttisDeepfakeLayer) + 1
        except ValueError:
            idx = len(EFFECT_CLASSES)
        EFFECT_CLASSES.insert(idx, MultiscaleLifeLayer)
    EFFECTS_BY_NAME[MultiscaleLifeLayer.__name__] = MultiscaleLifeLayer
    PRESETS.setdefault(
        "Antti Multiscale Life",
        [
            {"type": "MultiscaleLifeLayer", "values": {
                "phase_lock": 0.92,
                "style_strength": 1.0,
                "structure": 0.92,
                "detail": 0.835,
                "mask_expand": 2,
                "mask_feather": 4.0,
                "life_repair": True,
                "life_fine": 8,
                "life_medium": 25,
                "life_low": 60,
                "life_coarse": 100,
                "life_relock": 1.0,
                "show_life": True,
            }},
            {"type": "Bloom", "values": {"threshold": 0.72, "intensity": 0.25}},
            {"type": "ColorGrade", "values": {"contrast": 1.04, "saturation": 1.02}},
        ],
    )


register()
