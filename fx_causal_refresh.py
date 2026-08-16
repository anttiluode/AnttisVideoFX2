"""Causal-refresh sibling for the existing Antti Layered PhaseRail effect.

Importing this module registers one extra effect/preset without changing the
shipped AnttisDeepfakeLayer.  The ordinary PhaseRail still carries a frozen
generated person between expensive diffusion calls.

The refresh receiver is deliberately *not* live-vs-generated appearance.  When
a fresh generated person keyframe is accepted, this effect snapshots the live
actor geometry at that same moment.  Later frames are compared with that fixed
live reference.  A persistent rise in keyframe-relative geometry drift asks the
existing diffusion worker for another person keyframe.
"""
from __future__ import annotations

import cv2
import numpy as np

from causal_refresh import CausalRefreshController
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


class CausalPhaseRailLayer(AnttisDeepfakeLayer):
    name = "Antti Causal Refresh"
    blurb = (
        "Layered PhaseRail with a keyframe-relative automatic spend rule. "
        "Each generated person is anchored to the live geometry present when "
        "that keyframe is accepted. The old generated appearance is transported "
        "until live geometry persistently leaves that anchor, then diffusion is "
        "asked for a replacement."
    )
    params = AnttisDeepfakeLayer.params + [
        Param("auto_refresh", "Auto refresh", "bool", True),
        Param("refresh_threshold", "Geometry-change threshold", "float", 0.20, 0.02, 2.0),
        Param("refresh_decay", "Change memory", "float", 0.94, 0.0, 0.995),
        Param("refresh_min_age", "Min key age (s)", "float", 0.80, 0.0, 10.0),
        Param("show_refresh", "Show refresh meter", "bool", True),
    ]

    def _controller(self, st) -> CausalRefreshController:
        ctrl = st.get("refresh_controller")
        if ctrl is None:
            ctrl = CausalRefreshController(
                decay=float(self.p("refresh_decay")),
                threshold=float(self.p("refresh_threshold")),
                min_keyframe_age=float(self.p("refresh_min_age")),
            )
            st["refresh_controller"] = ctrl
        ctrl.configure(
            decay=float(self.p("refresh_decay")),
            threshold=float(self.p("refresh_threshold")),
            min_keyframe_age=float(self.p("refresh_min_age")),
        )
        return ctrl

    @staticmethod
    def _draw_meter(out: np.ndarray, reading, pending: bool, count: int) -> np.ndarray:
        if reading is None:
            return out

        # OpenCV 5's text renderer requires uint8; the effect bus is float32.
        hud = np.clip(out * 255.0, 0, 255).astype(np.uint8)
        h, w = hud.shape[:2]
        x0, y0 = 12, max(8, h - 34)
        width = min(350, max(80, w - 24))
        height = 10
        ratio = float(np.clip(reading.evidence / max(1e-6, reading.threshold), 0.0, 1.0))
        cv2.rectangle(hud, (x0, y0), (x0 + width, y0 + height), (13, 13, 13), -1)
        colour = (38, 242, 115) if reading.calibrated else (35, 190, 245)
        cv2.rectangle(hud, (x0, y0), (x0 + int(width * ratio), y0 + height), colour, -1)
        if pending:
            state = "WAITING"
        elif not reading.calibrated:
            state = "CAL"
        else:
            state = "armed"
        label = (
            f"refresh {reading.evidence:.2f}/{reading.threshold:.2f} "
            f"geom {reading.structural:.2f} base {reading.baseline:.2f} "
            f"d {reading.excess:+.2f} {state} n={count}"
        )
        cv2.putText(hud, label, (x0, y0 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.38, (5, 5, 5), 3, cv2.LINE_AA)
        cv2.putText(hud, label, (x0, y0 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.38, (255, 255, 255), 1, cv2.LINE_AA)
        return hud.astype(np.float32) * np.float32(1.0 / 255.0)

    def apply(self, img, ctx):
        mask = ctx.mask()
        if mask is None:
            return img

        st = ctx.st(self)
        live_person = ctx.map("person_style")
        background_ai = ctx.map("background_style")
        live_stamp = ctx.stamp("person_style") if live_person is not None else 0.0

        # While a replacement is generating, keep rendering the previous
        # keyframe rather than exposing raw webcam or blanking the subject.
        pending = bool(st.get("refresh_pending", False))
        held = st.get("held_person_ai")
        held_stamp = float(st.get("held_person_stamp", -1.0))
        if pending and live_person is not None and live_stamp != held_stamp:
            pending = False
            st["refresh_pending"] = False
            st["held_person_ai"] = None
            st["held_person_stamp"] = -1.0
            self._controller(st).reset()
            held = None

        person_ai = live_person if live_person is not None else held
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
        rail = st.get("rail")
        if rail is None or st.get("rail_size") != size or st.get("device") != device:
            try:
                from fx_phase_rail import LayerPhaseRail
                rail = LayerPhaseRail(size=size, device=device)
                st.update(
                    rail=rail,
                    rail_size=size,
                    device=device,
                    person_stamp=-1.0,
                    error="",
                    refresh_pending=False,
                    held_person_ai=None,
                    held_person_stamp=-1.0,
                    refresh_count=0,
                    refresh_reference=None,
                )
                pending = False
                self._controller(st).reset()
            except Exception as exc:
                st["error"] = str(exc)
                front = person_target
                return np.clip(front * m3 + background * (1.0 - m3), 0.0, 1.0)

        source_small, meta = self._letterbox(source_owned, size)
        target_small, _ = self._letterbox(target_owned, size)
        mask_small, _ = self._letterbox(m, size, fill=0.0)

        # The key point: accept generated style and live geometry together.
        # PhaseRail's new target is defined relative to *this* source frame, so
        # this is the natural zero of geometry age for the generated keyframe.
        target_stamp = live_stamp if live_person is not None else held_stamp
        if target_stamp != st.get("person_stamp"):
            rail.set_target(target_small)
            st["person_stamp"] = target_stamp
            if live_person is not None:
                st["refresh_reference"] = source_small.copy()
                self._controller(st).reset()

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
            st["metrics"] = metrics
            st["coherence"] = coherence
            carried = self._unletterbox(carried_small, meta)
        except Exception as exc:
            st["error"] = str(exc)
            carried_small = target_small
            carried = person_target
            metrics = {"confidence": 0.0, "motion": float(self.p("max_motion"))}

        reading = st.get("refresh_reading")
        if bool(self.p("auto_refresh")) and person_ai is not None and not pending:
            reference = st.get("refresh_reference")
            if reference is None or reference.shape != source_small.shape:
                reference = source_small.copy()
                st["refresh_reference"] = reference
                self._controller(st).reset()

            ctrl = self._controller(st)
            reading = ctrl.update(
                source_small,
                reference,
                mask=mask_small,
                phase_confidence=float(metrics.get("confidence", 1.0)),
                motion=float(metrics.get("motion", 0.0)),
                max_motion=float(self.p("max_motion")),
                keyframe_age=(ctx.age("person_style") if live_person is not None else 0.0),
            )
            st["refresh_reading"] = reading
            if reading.triggered and live_person is not None:
                # Hold the visible old keyframe, then clear only the worker's
                # publication slot. DiffusionWorker sees the missing map and
                # generates a replacement from the current camera stream.
                st["held_person_ai"] = live_person.copy()
                st["held_person_stamp"] = live_stamp
                st["refresh_pending"] = True
                st["refresh_count"] = int(st.get("refresh_count", 0)) + 1
                pending = True
                ctx.store.drop("person_style")
        elif not bool(self.p("auto_refresh")):
            ctrl = st.get("refresh_controller")
            if ctrl is not None:
                ctrl.reset()
            st["refresh_reading"] = None
            reading = None

        person_mix = float(self.p("person_mix"))
        front = img + (carried - img) * person_mix

        rescue = float(self.p("edge_live"))
        if rescue > 0 and float(self.p("mask_feather")) > 0:
            inner = blur(m, max(1.0, float(self.p("mask_feather")) * 1.8))
            edge = np.clip(np.abs(m - inner) * 4.0, 0.0, 1.0)[..., None]
            front = front * (1.0 - edge * rescue) + img * (edge * rescue)

        out = np.clip(front * m3 + background * (1.0 - m3), 0.0, 1.0)
        if bool(self.p("show_refresh")):
            out = self._draw_meter(
                out,
                reading,
                bool(st.get("refresh_pending", False)),
                int(st.get("refresh_count", 0)),
            )
        return out


def register() -> None:
    """Install the sibling effect/preset into fx_core's live registries."""
    if CausalPhaseRailLayer not in EFFECT_CLASSES:
        try:
            idx = EFFECT_CLASSES.index(AnttisDeepfakeLayer) + 1
        except ValueError:
            idx = len(EFFECT_CLASSES)
        EFFECT_CLASSES.insert(idx, CausalPhaseRailLayer)
    EFFECTS_BY_NAME[CausalPhaseRailLayer.__name__] = CausalPhaseRailLayer
    PRESETS.setdefault(
        "Antti Causal Refresh",
        [
            {"type": "CausalPhaseRailLayer", "values": {
                "phase_lock": 0.92,
                "style_strength": 1.0,
                "structure": 0.92,
                "detail": 0.82,
                "mask_expand": 2,
                "mask_feather": 4.0,
                "auto_refresh": True,
                "refresh_threshold": 0.20,
                "refresh_decay": 0.94,
                "refresh_min_age": 0.80,
                "show_refresh": True,
            }},
            {"type": "Bloom", "values": {"threshold": 0.72, "intensity": 0.35}},
            {"type": "ColorGrade", "values": {"contrast": 1.06, "saturation": 1.05}},
        ],
    )


register()
