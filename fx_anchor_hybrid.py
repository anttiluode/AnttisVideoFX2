"""Confidence-gated Anchor Gather with donor soft re-anchoring.

The first Anchor Gather experiment removed recursive blur but exposed the next
failure cleanly: address drift.  A pristine generated face can remain sharp
while its features, hands and silhouette are sampled from the wrong places,
producing the characteristic Picasso-like failure.

This sibling keeps the useful part of Anchor Gather and adds two controls:

1. Measure whether the accumulated current->anchor address field is still
   locally believable (Jacobian strain/folding + transported ownership mask).
2. Where it is not believable, suppress stale high-frequency anchor detail and
   borrow only low-frequency structure from the live frame.

When too much of the subject becomes untrustworthy, request one img2img donor
through detail_metabolism_patch.  The donor is aligned and quality-gated, only
its mid/fine detail is transplanted, and the accepted result becomes a new
sharp anchor with an identity address map.  It is a soft re-anchor, not a full
frame replacement.
"""
from __future__ import annotations

import cv2
import numpy as np

from anchor_gather import AnchorAddressRail, fullres_gather_from_rail_map
from anchor_hybrid import (
    AddressQualityReading,
    address_quality_map,
    apply_mask_agreement,
    band_correlation,
    confidence_fuse,
    summarize_quality,
)
from detail_metabolism import align_donor_to_current, evaluate_donor, transplant_detail
from fx_core import (
    EFFECT_CLASSES,
    EFFECTS_BY_NAME,
    PRESETS,
    AnttisDeepfakeLayer,
    Param,
    blur,
)


class AnchorHybridLayer(AnttisDeepfakeLayer):
    name = "Antti Anchor Hybrid"
    group = "layer"
    needs = set(AnttisDeepfakeLayer.needs) | {"detail_donor"}
    blurb = (
        "Keep a pristine generated anchor, but stop trusting its detail where "
        "the accumulated address field folds, stretches or disagrees with the "
        "current person mask. Untrusted regions fall back to soft live geometry. "
        "If correspondence debt grows too large, request one img2img donor, "
        "transplant only its mid/fine detail, and soft re-anchor the address map."
    )
    params = AnttisDeepfakeLayer.params + [
        Param("hybrid_strain_scale", "Address strain penalty", "float", 3.5, 0.5, 10.0),
        Param("hybrid_mask_gain", "Mask disagreement penalty", "float", 3.0, 0.0, 8.0),
        Param("hybrid_live_geometry", "Live low geometry", "float", 0.72, 0.0, 1.0),
        Param("hybrid_low_sigma", "Live low radius", "float", 5.0, 1.5, 16.0),
        Param("hybrid_untrusted_detail", "Detail kept when lost", "float", 0.08, 0.0, 0.5),
        Param("hybrid_auto_reanchor", "Auto soft re-anchor", "bool", True),
        Param("hybrid_health_trigger", "Address health trigger", "float", 0.68, 0.10, 0.98),
        Param("hybrid_bad_trigger", "Bad area trigger", "float", 0.22, 0.01, 0.90),
        Param("hybrid_bad_threshold", "Bad pixel threshold", "float", 0.35, 0.05, 0.80),
        Param("hybrid_min_frames", "Min donor interval", "int", 24, 1, 300),
        Param("hybrid_donor_strength", "Donor img2img strength", "float", 0.30, 0.08, 0.75),
        Param("hybrid_donor_steps", "Donor img2img steps", "int", 4, 1, 12),
        Param("hybrid_donor_guidance", "Donor guidance", "float", 0.0, 0.0, 8.0),
        Param("hybrid_fine_mix", "Donor fine detail", "float", 0.88, 0.0, 1.0),
        Param("hybrid_mid_mix", "Donor medium detail", "float", 0.34, 0.0, 1.0),
        Param("hybrid_geom_max", "Donor geometry error", "float", 0.68, 0.05, 2.0),
        Param("hybrid_mid_corr_min", "Donor mid correlation", "float", 0.10, -0.5, 0.95),
        Param("hybrid_gain_min", "Donor fine gain min", "float", 0.78, 0.30, 2.0),
        Param("hybrid_gain_max", "Donor fine gain max", "float", 1.85, 0.7, 4.0),
        Param("hybrid_show_monitor", "Show hybrid monitor", "bool", True),
        Param("hybrid_show_trust", "Show address trust map", "bool", False),
    ]

    @staticmethod
    def _draw_stage(out: np.ndarray, text: str) -> np.ndarray:
        hud = np.clip(out * 255.0, 0, 255).astype(np.uint8)
        h, w = hud.shape[:2]
        x, y = 12, max(30, h - 38)
        label = f"ANCHOR HYBRID — {text}"
        if len(label) > 105:
            label = label[:102] + "..."
        cv2.rectangle(hud, (x - 4, y - 19), (min(w - 5, x + 720), y + 8), (0, 0, 0), -1)
        cv2.putText(hud, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    (0, 210, 255), 1, cv2.LINE_AA)
        return hud.astype(np.float32) * np.float32(1.0 / 255.0)

    @staticmethod
    def _draw_hud(out: np.ndarray, st: dict, reading: AddressQualityReading, metrics) -> np.ndarray:
        hud = np.clip(out * 255.0, 0, 255).astype(np.uint8)
        h, w = hud.shape[:2]
        x0, y0 = 12, max(28, h - 58)
        width = min(360, max(150, w - 24))
        health = float(np.clip(reading.health, 0.0, 1.0))
        cv2.rectangle(hud, (x0, y0), (x0 + width, y0 + 9), (18, 18, 18), -1)
        cv2.rectangle(hud, (x0, y0), (x0 + int(width * health), y0 + 9), (80, 220, 120), -1)
        pending = bool(st.get("hybrid_pending", False))
        state = "DONOR" if pending else "gather"
        ok = int(st.get("hybrid_accept_count", 0))
        reject = int(st.get("hybrid_reject_count", 0))
        d = st.get("hybrid_last_donor")
        if d:
            tail = (
                f" mid {d['mid_corr']:.2f} fine {d['fine_corr']:.2f} "
                f"gain {d['fine_gain']:.2f} geom {d['geometry_error']:.2f}"
            )
        else:
            tail = ""
        txt = (
            f"ANCHOR HYBRID  health {reading.health:.2f} bad {reading.bad_fraction:.2f} "
            f"strain {reading.strain_mean:.3f} conf {metrics.confidence:.2f} {state} "
            f"ok={ok} reject={reject}{tail}"
        )
        cv2.putText(hud, txt, (x0, y0 + 27), cv2.FONT_HERSHEY_SIMPLEX, 0.36,
                    (5, 5, 5), 3, cv2.LINE_AA)
        cv2.putText(hud, txt, (x0, y0 + 27), cv2.FONT_HERSHEY_SIMPLEX, 0.36,
                    (255, 255, 255), 1, cv2.LINE_AA)
        return hud.astype(np.float32) * np.float32(1.0 / 255.0)

    def _request_donor(self, ctx, person: np.ndarray, mask: np.ndarray, st: dict) -> None:
        neutral = np.full_like(person, 0.12, dtype=np.float32)
        m3 = mask[..., None]
        request = person * m3 + neutral * (1.0 - m3)
        ctx.store.state["_detail_request_cfg"] = {
            "strength": float(self.p("hybrid_donor_strength")),
            "steps": int(self.p("hybrid_donor_steps")),
            "guidance": float(self.p("hybrid_donor_guidance")),
        }
        ctx.store.put("detail_request_image", request.astype(np.float32))
        ctx.store.put("detail_request", np.asarray([float(ctx.idx)], np.float32))
        st["hybrid_pending"] = True
        st["hybrid_request_idx"] = int(ctx.idx)

    def apply(self, img, ctx):
        mask = ctx.mask()
        person_ai = ctx.map("person_style")
        background_ai = ctx.map("background_style")
        if mask is None:
            return self._draw_stage(img, "waiting for segmentation mask")
        if person_ai is None or background_ai is None:
            missing = []
            if person_ai is None:
                missing.append("person_style")
            if background_ai is None:
                missing.append("background_style")
            return self._draw_stage(img, "waiting for " + " + ".join(missing))

        m = self._ownership(mask, int(self.p("mask_expand")), float(self.p("mask_feather")))
        if self.p("show_mask"):
            return np.repeat(m[..., None], 3, axis=2)
        m3 = m[..., None]

        background = img + (background_ai - img) * float(self.p("background_mix"))
        background = self._move_background(
            background, self.p("back_x"), self.p("back_y"), self.p("back_zoom")
        )

        neutral = np.full_like(img, 0.12, dtype=np.float32)
        source_owned = img * m3 + neutral * (1.0 - m3)
        size = int(self.p("rail_size"))
        device = str(self.p("device"))
        st = ctx.st(self)
        rail = st.get("hybrid_rail")
        if rail is None or st.get("hybrid_size") != size or st.get("hybrid_device") != device:
            try:
                rail = AnchorAddressRail(size=size, device=device)
                st.update(
                    hybrid_rail=rail,
                    hybrid_size=size,
                    hybrid_device=device,
                    hybrid_person_stamp=-1.0,
                    hybrid_anchor_image=None,
                    hybrid_anchor_mask=None,
                    hybrid_pending=False,
                    hybrid_request_idx=-10**9,
                    hybrid_accept_count=0,
                    hybrid_reject_count=0,
                    hybrid_last_donor=None,
                    hybrid_donor_seen=float(ctx.stamp("detail_donor")),
                    hybrid_error="",
                )
            except Exception as exc:
                composite = np.clip(person_ai * m3 + background * (1.0 - m3), 0.0, 1.0)
                return self._draw_stage(composite, f"motion rail init failed: {exc}")

        person_stamp = float(ctx.stamp("person_style"))
        if person_stamp != float(st.get("hybrid_person_stamp", -1.0)):
            rail.reset()
            st["hybrid_anchor_image"] = np.asarray(person_ai, np.float32).copy()
            st["hybrid_anchor_mask"] = m.copy()
            st["hybrid_person_stamp"] = person_stamp
            st["hybrid_pending"] = False
            st["hybrid_request_idx"] = int(ctx.idx)
            st["hybrid_last_donor"] = None
            st["hybrid_donor_seen"] = float(ctx.stamp("detail_donor"))

        source_small, meta = self._letterbox(source_owned, size)
        try:
            map_x, map_y, metrics = rail.process(
                source_small, max_displacement=float(self.p("max_motion"))
            )
            carried = fullres_gather_from_rail_map(
                st["hybrid_anchor_image"], map_x, map_y, meta
            )
            warped_anchor_mask = fullres_gather_from_rail_map(
                st["hybrid_anchor_mask"], map_x, map_y, meta
            )
        except Exception as exc:
            st["hybrid_error"] = str(exc)
            composite = np.clip(person_ai * m3 + background * (1.0 - m3), 0.0, 1.0)
            return self._draw_stage(composite, f"motion rail process failed: {exc}")

        q, strain, det = address_quality_map(
            map_x, map_y,
            meta=meta,
            out_shape=img.shape[:2],
            strain_scale=float(self.p("hybrid_strain_scale")),
        )
        q = apply_mask_agreement(
            q, m, warped_anchor_mask,
            disagreement_gain=float(self.p("hybrid_mask_gain")),
        )
        q = np.clip(blur(q, 1.5), 0.0, 1.0).astype(np.float32)
        reading = summarize_quality(
            q, m,
            bad_threshold=float(self.p("hybrid_bad_threshold")),
            strain=strain,
            determinant=det,
        )

        person = confidence_fuse(
            carried, img, q,
            low_sigma=float(self.p("hybrid_low_sigma")),
            live_geometry=float(self.p("hybrid_live_geometry")),
            untrusted_detail=float(self.p("hybrid_untrusted_detail")),
        )
        st["hybrid_detail_corr"] = band_correlation(person, carried, m, band="fine")
        st["hybrid_reading"] = reading
        st["hybrid_metrics"] = metrics

        donor = ctx.map("detail_donor")
        donor_stamp = float(ctx.stamp("detail_donor"))
        if donor is not None and donor_stamp != float(st.get("hybrid_donor_seen", 0.0)):
            st["hybrid_donor_seen"] = donor_stamp
            if bool(st.get("hybrid_pending", False)):
                aligned, mean_flow = align_donor_to_current(donor, person, m)
                stats = evaluate_donor(person, aligned, m, mean_flow)
                mid_corr = band_correlation(person, aligned, m, band="mid")
                fine_corr = band_correlation(person, aligned, m, band="fine")
                st["hybrid_last_donor"] = {
                    "fine_gain": float(stats.fine_gain),
                    "geometry_error": float(stats.geometry_error),
                    "mean_flow": float(stats.mean_flow),
                    "mid_corr": float(mid_corr),
                    "fine_corr": float(fine_corr),
                }
                accept = (
                    float(self.p("hybrid_gain_min")) <= stats.fine_gain <= float(self.p("hybrid_gain_max"))
                    and stats.geometry_error <= float(self.p("hybrid_geom_max"))
                    and mid_corr >= float(self.p("hybrid_mid_corr_min"))
                )
                if accept:
                    repair_mask = np.clip(m * (0.25 + 0.75 * (1.0 - q)), 0.0, 1.0)
                    repaired = transplant_detail(
                        person, aligned, repair_mask,
                        fine_mix=float(self.p("hybrid_fine_mix")),
                        mid_mix=float(self.p("hybrid_mid_mix")),
                    )
                    st["hybrid_anchor_image"] = repaired.copy()
                    st["hybrid_anchor_mask"] = m.copy()
                    rail.reset()
                    person = repaired
                    q = np.ones_like(m, np.float32)
                    reading = AddressQualityReading(1.0, 0.0, 0.0, 0.0)
                    st["hybrid_accept_count"] = int(st.get("hybrid_accept_count", 0)) + 1
                    st["hybrid_request_idx"] = int(ctx.idx)
                else:
                    st["hybrid_reject_count"] = int(st.get("hybrid_reject_count", 0)) + 1
                st["hybrid_pending"] = False

        enough_age = (
            int(ctx.idx) - int(st.get("hybrid_request_idx", -10**9))
            >= int(self.p("hybrid_min_frames"))
        )
        wants_reanchor = (
            reading.health <= float(self.p("hybrid_health_trigger"))
            or reading.bad_fraction >= float(self.p("hybrid_bad_trigger"))
        )
        if (
            bool(self.p("hybrid_auto_reanchor"))
            and not bool(st.get("hybrid_pending", False))
            and enough_age
            and wants_reanchor
        ):
            self._request_donor(ctx, person, m, st)

        if bool(st.get("hybrid_pending", False)):
            request = person * m3 + neutral * (1.0 - m3)
            ctx.store.put("detail_request_image", request.astype(np.float32))

        if bool(self.p("hybrid_show_trust")):
            heat = cv2.applyColorMap(
                np.clip(q * 255.0, 0, 255).astype(np.uint8), cv2.COLORMAP_TURBO
            ).astype(np.float32) / 255.0
            return heat

        person_mix = float(self.p("person_mix"))
        front = img + (person - img) * person_mix
        rescue = float(self.p("edge_live"))
        if rescue > 0 and float(self.p("mask_feather")) > 0:
            inner = blur(m, max(1.0, float(self.p("mask_feather")) * 1.8))
            edge = np.clip(np.abs(m - inner) * 4.0, 0.0, 1.0)[..., None]
            front = front * (1.0 - edge * rescue) + img * (edge * rescue)

        out = np.clip(front * m3 + background * (1.0 - m3), 0.0, 1.0)
        if bool(self.p("hybrid_show_monitor")):
            out = self._draw_hud(out, st, reading, metrics)
        return out


def register() -> None:
    if AnchorHybridLayer not in EFFECT_CLASSES:
        try:
            idx = EFFECT_CLASSES.index(AnttisDeepfakeLayer) + 1
        except ValueError:
            idx = len(EFFECT_CLASSES)
        EFFECT_CLASSES.insert(idx, AnchorHybridLayer)
    EFFECTS_BY_NAME[AnchorHybridLayer.__name__] = AnchorHybridLayer
    PRESETS.setdefault(
        "Antti Anchor Hybrid",
        [
            {"type": "AnchorHybridLayer", "values": {
                "max_motion": 3.5,
                "person_mix": 1.0,
                "background_mix": 1.0,
                "mask_expand": 2,
                "mask_feather": 4.0,
                "edge_live": 0.10,
                "hybrid_strain_scale": 3.5,
                "hybrid_mask_gain": 3.0,
                "hybrid_live_geometry": 0.72,
                "hybrid_low_sigma": 5.0,
                "hybrid_untrusted_detail": 0.08,
                "hybrid_auto_reanchor": True,
                "hybrid_health_trigger": 0.68,
                "hybrid_bad_trigger": 0.22,
                "hybrid_bad_threshold": 0.35,
                "hybrid_min_frames": 24,
                "hybrid_donor_strength": 0.30,
                "hybrid_donor_steps": 4,
                "hybrid_donor_guidance": 0.0,
                "hybrid_fine_mix": 0.88,
                "hybrid_mid_mix": 0.34,
                "hybrid_geom_max": 0.68,
                "hybrid_mid_corr_min": 0.10,
                "hybrid_gain_min": 0.78,
                "hybrid_gain_max": 1.85,
                "hybrid_show_monitor": True,
                "hybrid_show_trust": False,
            }},
            {"type": "Bloom", "values": {"threshold": 0.74, "intensity": 0.14}},
            {"type": "ColorGrade", "values": {"contrast": 1.02, "saturation": 1.01}},
        ],
    )


register()
