"""LivePortrait face assist for the layered AnttisVideoFX2 experiment.

This is deliberately an unfair control rather than another home-grown motion
model: the ordinary PhaseRail layer still carries the generated whole person,
but the face region is replaced by a LivePortrait result driven by the same
camera frame.  If the face survives motion while the body still deforms, we
have isolated learned/canonical facial correspondence as the missing ingredient.
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


class LivePortraitAssistLayer(AnttisDeepfakeLayer):
    name = "Antti LivePortrait Assist"
    group = "layer"
    needs = set(AnttisDeepfakeLayer.needs) | {"live_portrait", "live_portrait_alpha"}
    blurb = (
        "Control experiment: keep the existing PhaseRail whole-person transport, "
        "but let the official LivePortrait model own the face correspondence. "
        "The generated person keyframe remains the appearance source. If the face "
        "stays coherent while the body fails, the missing ingredient is the "
        "learned portrait prior/canonical motion representation, not another "
        "confidence threshold."
    )
    params = AnttisDeepfakeLayer.params + [
        Param("lp_face_mix", "LivePortrait face mix", "float", 1.0, 0.0, 1.0),
        Param("lp_alpha_expand", "Face alpha expand", "int", 0, -12, 24),
        Param("lp_alpha_feather", "Face alpha feather", "float", 2.0, 0.0, 18.0),
        Param("lp_intersect_person", "Keep face in person mask", "bool", True),
        Param("lp_show_alpha", "Show LivePortrait alpha", "bool", False),
        Param("lp_show_monitor", "Show LivePortrait monitor", "bool", True),
    ]

    @staticmethod
    def _shape_alpha(alpha: np.ndarray, expand: int, feather: float) -> np.ndarray:
        a = np.asarray(alpha, np.float32)
        if a.ndim == 3:
            a = a[..., 0]
        a = np.clip(a, 0.0, 1.0)
        n = abs(int(expand))
        if n:
            ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (n * 2 + 1, n * 2 + 1))
            a = cv2.dilate(a, ker) if expand > 0 else cv2.erode(a, ker)
        if feather > 0.1:
            a = blur(a, feather)
        return np.clip(a, 0.0, 1.0).astype(np.float32)

    @staticmethod
    def _draw_hud(out: np.ndarray, ctx, alpha: np.ndarray | None) -> np.ndarray:
        hud = np.clip(out * 255.0, 0, 255).astype(np.uint8)
        h = hud.shape[0]
        age = float(ctx.age("live_portrait"))
        coverage = 0.0 if alpha is None else float(np.mean(alpha > 0.10))
        txt = f"LIVEPORTRAIT ASSIST  age {age:.2f}s  face {coverage * 100:.1f}%"
        y = max(24, h - 28)
        cv2.putText(hud, txt, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.40,
                    (5, 5, 5), 3, cv2.LINE_AA)
        cv2.putText(hud, txt, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.40,
                    (255, 255, 255), 1, cv2.LINE_AA)
        return hud.astype(np.float32) / 255.0

    def apply(self, img, ctx):
        # First produce the exact same whole-person PhaseRail baseline. The only
        # changed variable below is who owns facial correspondence.
        base = super().apply(img, ctx)
        portrait = ctx.map("live_portrait")
        alpha = ctx.map("live_portrait_alpha")
        if portrait is None or alpha is None:
            if bool(self.p("lp_show_monitor")):
                return self._draw_hud(base, ctx, None)
            return base

        a = self._shape_alpha(
            alpha,
            int(self.p("lp_alpha_expand")),
            float(self.p("lp_alpha_feather")),
        )
        if bool(self.p("lp_intersect_person")):
            m = ctx.mask()
            if m is not None:
                if m.shape != a.shape:
                    m = cv2.resize(m, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_LINEAR)
                # A soft intersection prevents a delayed portrait patch from
                # floating into the generated background.
                a *= np.clip(blur(np.asarray(m, np.float32), 2.0), 0.0, 1.0)

        if bool(self.p("lp_show_alpha")):
            return np.repeat(a[..., None], 3, axis=2)

        mix = float(self.p("lp_face_mix"))
        a3 = np.clip(a * mix, 0.0, 1.0)[..., None]
        out = np.clip(base * (1.0 - a3) + portrait * a3, 0.0, 1.0)
        if bool(self.p("lp_show_monitor")):
            out = self._draw_hud(out, ctx, a)
        return out


def register() -> None:
    if LivePortraitAssistLayer not in EFFECT_CLASSES:
        try:
            idx = EFFECT_CLASSES.index(AnttisDeepfakeLayer) + 1
        except ValueError:
            idx = len(EFFECT_CLASSES)
        EFFECT_CLASSES.insert(idx, LivePortraitAssistLayer)
    EFFECTS_BY_NAME[LivePortraitAssistLayer.__name__] = LivePortraitAssistLayer
    PRESETS.setdefault(
        "Antti LivePortrait Assist",
        [
            {"type": "LivePortraitAssistLayer", "values": {
                "phase_lock": 0.92,
                "style_strength": 1.0,
                "structure": 0.92,
                "detail": 0.82,
                "max_motion": 3.5,
                "person_mix": 1.0,
                "background_mix": 1.0,
                "mask_expand": 2,
                "mask_feather": 4.0,
                "edge_live": 0.10,
                "lp_face_mix": 1.0,
                "lp_alpha_expand": 0,
                "lp_alpha_feather": 2.0,
                "lp_intersect_person": True,
                "lp_show_monitor": True,
            }},
            {"type": "Bloom", "values": {"threshold": 0.74, "intensity": 0.14}},
            {"type": "ColorGrade", "values": {"contrast": 1.02, "saturation": 1.01}},
        ],
    )


register()
