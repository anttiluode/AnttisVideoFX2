"""Source-anchored gather sibling effect for AnttisVideoFX2.

The generated person is immutable appearance memory.  PhaseRail's Gabor bank is
used only to estimate incremental motion.  Those increments are composed into a
current->anchor address field, and every displayed frame samples the original
sharp generated person exactly once.

This is deliberately the opposite of recursive appearance transport: if it
fails, it should fail geometrically (drift/folding/wrong correspondence), not by
turning the source texture into a photocopy-of-a-photocopy.
"""
from __future__ import annotations

import cv2
import numpy as np

from anchor_gather import AnchorAddressRail, fullres_gather_from_rail_map
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


class AnchorGatherLayer(AnttisDeepfakeLayer):
    name = "Antti Anchor Gather"
    group = "layer"
    blurb = (
        "Do not carry appearance through time. Carry only a backward address "
        "field and gather every frame from the pristine generated person. "
        "Texture gets one resampling per frame no matter how old the keyframe is; "
        "the expected failure moves from blur into address/geometry drift."
    )
    params = AnttisDeepfakeLayer.params + [
        Param("anchor_live_low", "Borrow live low structure", "float", 0.0, 0.0, 1.0),
        Param("anchor_low_sigma", "Live low radius", "float", 7.0, 2.0, 20.0),
        Param("show_address", "Show address monitor", "bool", True),
        Param("show_address_map", "Show displacement map", "bool", False),
    ]

    @staticmethod
    def _draw_hud(out: np.ndarray, metrics) -> np.ndarray:
        hud = np.clip(out * 255.0, 0, 255).astype(np.uint8)
        h = hud.shape[0]
        x, y = 12, max(24, h - 35)
        txt = (
            f"ANCHOR GATHER  motion {metrics.motion:.2f}  conf {metrics.confidence:.2f}  "
            f"addr {metrics.displacement:.1f}px  stretch {metrics.stretch:.3f}"
        )
        cv2.putText(hud, txt, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.40,
                    (5, 5, 5), 3, cv2.LINE_AA)
        cv2.putText(hud, txt, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.40,
                    (255, 255, 255), 1, cv2.LINE_AA)
        return hud.astype(np.float32) * np.float32(1.0 / 255.0)

    def apply(self, img, ctx):
        mask = ctx.mask()
        if mask is None:
            return img
        person_ai = ctx.map("person_style")
        background_ai = ctx.map("background_style")
        if person_ai is None and background_ai is None:
            return img

        m = self._ownership(mask, int(self.p("mask_expand")), float(self.p("mask_feather")))
        if self.p("show_mask"):
            return np.repeat(m[..., None], 3, axis=2)
        m3 = m[..., None]

        person_target = person_ai if person_ai is not None else img
        background = background_ai if background_ai is not None else img
        background = img + (background - img) * float(self.p("background_mix"))
        background = self._move_background(
            background, self.p("back_x"), self.p("back_y"), self.p("back_zoom")
        )

        # The motion sensor sees only actor-owned evidence, as in PhaseRail.
        neutral = np.full_like(img, 0.12, dtype=np.float32)
        source_owned = img * m3 + neutral * (1.0 - m3)

        size = int(self.p("rail_size"))
        device = str(self.p("device"))
        st = ctx.st(self)
        rail = st.get("anchor_rail")
        if rail is None or st.get("anchor_size") != size or st.get("anchor_device") != device:
            try:
                rail = AnchorAddressRail(size=size, device=device)
                st.update(
                    anchor_rail=rail,
                    anchor_size=size,
                    anchor_device=device,
                    anchor_person_stamp=-1.0,
                    anchor_image=None,
                    anchor_error="",
                )
            except Exception as exc:
                st["anchor_error"] = str(exc)
                front = person_target
                return np.clip(front * m3 + background * (1.0 - m3), 0.0, 1.0)

        stamp = float(ctx.stamp("person_style"))
        if stamp != float(st.get("anchor_person_stamp", -1.0)):
            rail.reset()
            # Immutable full-resolution appearance source.  It is never replaced
            # by a carried/rendered frame in this effect.
            st["anchor_image"] = np.asarray(person_target, np.float32).copy()
            st["anchor_person_stamp"] = stamp

        source_small, meta = self._letterbox(source_owned, size)
        try:
            map_x, map_y, metrics = rail.process(
                source_small, max_displacement=float(self.p("max_motion"))
            )
            st["anchor_metrics"] = metrics
            carried = fullres_gather_from_rail_map(st["anchor_image"], map_x, map_y, meta)
        except Exception as exc:
            st["anchor_error"] = str(exc)
            carried = person_target
            metrics = None

        # Optional very-low-frequency borrowing from the camera.  Default is
        # zero so the first test isolates source-anchored gathering itself.
        low_mix = float(self.p("anchor_live_low"))
        if low_mix > 0.0:
            sigma = float(self.p("anchor_low_sigma"))
            live_low = blur(img, sigma)
            carry_low = blur(carried, sigma)
            carried = np.clip(carried + low_mix * (live_low - carry_low), 0.0, 1.0)

        if bool(self.p("show_address_map")) and metrics is not None:
            # Visualize displacement magnitude from the low-resolution field.
            H, W = map_x.shape
            yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
            mag = np.sqrt((map_x - xx) ** 2 + (map_y - yy) ** 2)
            scale = max(1.0, float(np.percentile(mag, 95)))
            heat = cv2.applyColorMap(
                np.clip(mag / scale * 255.0, 0, 255).astype(np.uint8), cv2.COLORMAP_TURBO
            ).astype(np.float32) / 255.0
            carried = cv2.resize(heat, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

        person_mix = float(self.p("person_mix"))
        front = img + (carried - img) * person_mix
        rescue = float(self.p("edge_live"))
        if rescue > 0 and float(self.p("mask_feather")) > 0:
            inner = blur(m, max(1.0, float(self.p("mask_feather")) * 1.8))
            edge = np.clip(np.abs(m - inner) * 4.0, 0.0, 1.0)[..., None]
            front = front * (1.0 - edge * rescue) + img * (edge * rescue)

        out = np.clip(front * m3 + background * (1.0 - m3), 0.0, 1.0)
        if bool(self.p("show_address")) and metrics is not None and not bool(self.p("show_address_map")):
            out = self._draw_hud(out, metrics)
        return out


def register() -> None:
    if AnchorGatherLayer not in EFFECT_CLASSES:
        try:
            idx = EFFECT_CLASSES.index(AnttisDeepfakeLayer) + 1
        except ValueError:
            idx = len(EFFECT_CLASSES)
        EFFECT_CLASSES.insert(idx, AnchorGatherLayer)
    EFFECTS_BY_NAME[AnchorGatherLayer.__name__] = AnchorGatherLayer
    PRESETS.setdefault(
        "Antti Anchor Gather",
        [
            {"type": "AnchorGatherLayer", "values": {
                "max_motion": 3.5,
                "person_mix": 1.0,
                "background_mix": 1.0,
                "mask_expand": 2,
                "mask_feather": 4.0,
                "edge_live": 0.10,
                "anchor_live_low": 0.0,
                "show_address": True,
                "show_address_map": False,
            }},
            {"type": "Bloom", "values": {"threshold": 0.74, "intensity": 0.16}},
            {"type": "ColorGrade", "values": {"contrast": 1.02, "saturation": 1.01}},
        ],
    )


register()
