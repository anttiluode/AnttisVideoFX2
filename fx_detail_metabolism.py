"""Living detail-metabolism effect for AnttisVideoFX2.

A generated person remains a PhaseRail-carried dynamical state.  When its
fine-detail debt rises, the effect requests one img2img donor from the current
carried image.  The donor is flow-aligned, quality-gated, and used only as a
medium/fine-frequency transplant.  The repaired image is then installed back
into PhaseRail as the new living memory.

The original ``person_style`` publication remains untouched and acts as the
identity/style origin.  Donor images never replace it wholesale.
"""
from __future__ import annotations

import cv2
import numpy as np

from detail_metabolism import (
    DetailDebtMonitor,
    align_donor_to_current,
    evaluate_donor,
    transplant_detail,
)
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


class DetailMetabolismLayer(AnttisDeepfakeLayer):
    name = "Antti Detail Metabolism"
    group = "layer"
    needs = set(AnttisDeepfakeLayer.needs) | {"detail_donor"}
    blurb = (
        "Keep the generated person alive with PhaseRail, but spend img2img only "
        "when fine detail has decayed. A fresh donor is generated from the "
        "current carried person, flow-aligned, rejected if it changes coarse "
        "geometry or fails to add detail, then only its mid/fine frequencies "
        "are transplanted. The repaired image becomes the next living memory."
    )
    params = AnttisDeepfakeLayer.params + [
        Param("detail_auto", "Detail metabolism", "bool", True),
        Param("detail_threshold", "Detail debt trigger", "float", 0.20, 0.02, 0.90),
        Param("detail_min_frames", "Min donor interval", "int", 18, 1, 300),
        Param("donor_strength", "Img2img donor strength", "float", 0.32, 0.08, 0.80),
        Param("donor_steps", "Img2img donor steps", "int", 4, 1, 12),
        Param("donor_guidance", "Img2img donor guidance", "float", 0.0, 0.0, 8.0),
        Param("fine_mix", "Fresh fine detail", "float", 0.90, 0.0, 1.0),
        Param("mid_mix", "Fresh medium detail", "float", 0.30, 0.0, 1.0),
        Param("donor_min_gain", "Minimum detail gain", "float", 1.03, 0.80, 2.50),
        Param("donor_geom_max", "Maximum geometry error", "float", 0.70, 0.05, 2.0),
        Param("show_metabolism", "Show metabolism monitor", "bool", True),
    ]

    @staticmethod
    def _monitor(st) -> DetailDebtMonitor:
        mon = st.get("detail_monitor")
        if mon is None:
            mon = DetailDebtMonitor()
            st["detail_monitor"] = mon
        return mon

    @staticmethod
    def _draw_hud(out: np.ndarray, st: dict, reading) -> np.ndarray:
        if reading is None:
            return out
        hud = np.clip(out * 255.0, 0, 255).astype(np.uint8)
        h, w = hud.shape[:2]
        x0, y0 = 12, max(22, h - 54)
        width = min(360, max(150, w - 24))
        debt = float(np.clip(reading.debt, 0.0, 1.0))
        cv2.putText(
            hud, "DETAIL METABOLISM", (x0, y0 - 18), cv2.FONT_HERSHEY_SIMPLEX,
            0.44, (255, 255, 255), 1, cv2.LINE_AA,
        )
        cv2.rectangle(hud, (x0, y0), (x0 + width, y0 + 9), (18, 18, 18), -1)
        cv2.rectangle(hud, (x0, y0), (x0 + int(width * debt), y0 + 9), (40, 195, 245), -1)
        pending = bool(st.get("detail_pending", False))
        state = "WAITING DONOR" if pending else "living"
        accepted = int(st.get("detail_accept_count", 0))
        rejected = int(st.get("detail_reject_count", 0))
        stats = st.get("detail_last_stats")
        if stats is None:
            tail = ""
        else:
            tail = f" gain {stats.fine_gain:.2f} geom {stats.geometry_error:.2f} flow {stats.mean_flow:.1f}"
        txt = (
            f"health {reading.health:.2f} debt {reading.debt:.2f} "
            f"{state}  ok={accepted} reject={rejected}{tail}"
        )
        cv2.putText(hud, txt, (x0, y0 + 25), cv2.FONT_HERSHEY_SIMPLEX,
                    0.37, (5, 5, 5), 3, cv2.LINE_AA)
        cv2.putText(hud, txt, (x0, y0 + 25), cv2.FONT_HERSHEY_SIMPLEX,
                    0.37, (255, 255, 255), 1, cv2.LINE_AA)
        return hud.astype(np.float32) * np.float32(1.0 / 255.0)

    def _request_donor(self, ctx, carried: np.ndarray, m: np.ndarray, st: dict) -> None:
        neutral = np.full_like(carried, 0.12, dtype=np.float32)
        m3 = m[..., None]
        request = carried * m3 + neutral * (1.0 - m3)
        ctx.store.state["_detail_request_cfg"] = {
            "prompt": getattr(ctx.store.state.get("_detail_prompt_holder", {}), "prompt", None)
                      if False else None,
            "strength": float(self.p("donor_strength")),
            "steps": int(self.p("donor_steps")),
            "guidance": float(self.p("donor_guidance")),
        }
        # None prompt is removed so the worker falls back to cfg.person_prompt.
        ctx.store.state["_detail_request_cfg"] = {
            k: v for k, v in ctx.store.state["_detail_request_cfg"].items() if v is not None
        }
        ctx.store.put("detail_request_image", request.astype(np.float32))
        ctx.store.put("detail_request", np.asarray([float(ctx.idx)], np.float32))
        st["detail_pending"] = True
        st["detail_request_idx"] = int(ctx.idx)

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
        rail = st.get("detail_rail")
        if rail is None or st.get("detail_rail_size") != size or st.get("detail_device") != device:
            try:
                from fx_phase_rail import LayerPhaseRail
                rail = LayerPhaseRail(size=size, device=device)
                st.update(
                    detail_rail=rail,
                    detail_rail_size=size,
                    detail_device=device,
                    detail_person_stamp=-1.0,
                    detail_pending=False,
                    detail_request_idx=-10**9,
                    detail_accept_count=0,
                    detail_reject_count=0,
                    detail_last_stats=None,
                    detail_error="",
                    detail_needs_anchor=True,
                    detail_donor_seen=float(ctx.stamp("detail_donor")),
                )
                self._monitor(st).reset()
            except Exception as exc:
                st["detail_error"] = str(exc)
                front = person_target
                return np.clip(front * m3 + background * (1.0 - m3), 0.0, 1.0)

        source_small, meta = self._letterbox(source_owned, size)
        target_small, _ = self._letterbox(target_owned, size)
        person_stamp = float(ctx.stamp("person_style"))
        if person_stamp != st.get("detail_person_stamp"):
            rail.set_target(target_small)
            st["detail_person_stamp"] = person_stamp
            st["detail_pending"] = False
            st["detail_request_idx"] = int(ctx.idx)
            st["detail_needs_anchor"] = True
            st["detail_identity_anchor"] = None if person_ai is None else person_ai.copy()
            st["detail_donor_seen"] = float(ctx.stamp("detail_donor"))
            self._monitor(st).reset()

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
            st["detail_metrics"] = metrics
            st["detail_coherence"] = coherence
            carried = self._unletterbox(carried_small, meta)
        except Exception as exc:
            st["detail_error"] = str(exc)
            carried = person_target

        mon = self._monitor(st)
        if st.get("detail_needs_anchor", False):
            reading = mon.anchor(carried, m)
            st["detail_needs_anchor"] = False
        else:
            reading = mon.update(carried, m)

        # A donor is a proposal, never an authority. Consume each publication
        # once.  It may have been generated from a pose a few frames old, hence
        # the dense-flow alignment before any frequency transplantation.
        donor = ctx.map("detail_donor")
        donor_stamp = float(ctx.stamp("detail_donor"))
        if donor is not None and donor_stamp != float(st.get("detail_donor_seen", 0.0)):
            st["detail_donor_seen"] = donor_stamp
            if bool(st.get("detail_pending", False)):
                aligned, mean_flow = align_donor_to_current(donor, carried, m)
                stats = evaluate_donor(carried, aligned, m, mean_flow)
                st["detail_last_stats"] = stats
                accept = (
                    stats.fine_gain >= float(self.p("donor_min_gain"))
                    and stats.geometry_error <= float(self.p("donor_geom_max"))
                )
                if accept:
                    repaired = transplant_detail(
                        carried, aligned, m,
                        fine_mix=float(self.p("fine_mix")),
                        mid_mix=float(self.p("mid_mix")),
                    )
                    repaired_owned = repaired * m3 + neutral * (1.0 - m3)
                    repaired_small, _ = self._letterbox(repaired_owned, size)
                    # This is the important step: the donor-repaired image is
                    # installed as the next persistent PhaseRail memory. It is
                    # not merely overlaid for one video frame.
                    rail.set_target(repaired_small)
                    carried = repaired
                    carried_small, _ = self._letterbox(repaired_owned, size)
                    reading = mon.anchor(carried, m)
                    st["detail_accept_count"] = int(st.get("detail_accept_count", 0)) + 1
                    st["detail_request_idx"] = int(ctx.idx)
                else:
                    st["detail_reject_count"] = int(st.get("detail_reject_count", 0)) + 1
                st["detail_pending"] = False

        auto = bool(self.p("detail_auto"))
        enough_age = (
            int(ctx.idx) - int(st.get("detail_request_idx", -10**9))
            >= int(self.p("detail_min_frames"))
        )
        if auto and not bool(st.get("detail_pending", False)) and enough_age:
            if float(reading.debt) >= float(self.p("detail_threshold")):
                self._request_donor(ctx, carried, m, st)

        # While pending, keep the request image current until the diffusion
        # worker actually begins its pass. This reduces pose staleness if the
        # worker was busy with another channel first.
        if bool(st.get("detail_pending", False)):
            request = carried * m3 + neutral * (1.0 - m3)
            ctx.store.put("detail_request_image", request.astype(np.float32))

        person_mix = float(self.p("person_mix"))
        front = img + (carried - img) * person_mix
        rescue = float(self.p("edge_live"))
        if rescue > 0 and float(self.p("mask_feather")) > 0:
            inner = blur(m, max(1.0, float(self.p("mask_feather")) * 1.8))
            edge = np.clip(np.abs(m - inner) * 4.0, 0.0, 1.0)[..., None]
            front = front * (1.0 - edge * rescue) + img * (edge * rescue)

        out = np.clip(front * m3 + background * (1.0 - m3), 0.0, 1.0)
        if bool(self.p("show_metabolism")):
            out = self._draw_hud(out, st, reading)
        return out


def register() -> None:
    if DetailMetabolismLayer not in EFFECT_CLASSES:
        try:
            idx = EFFECT_CLASSES.index(AnttisDeepfakeLayer) + 1
        except ValueError:
            idx = len(EFFECT_CLASSES)
        EFFECT_CLASSES.insert(idx, DetailMetabolismLayer)
    EFFECTS_BY_NAME[DetailMetabolismLayer.__name__] = DetailMetabolismLayer
    PRESETS.setdefault(
        "Antti Detail Metabolism",
        [
            {"type": "DetailMetabolismLayer", "values": {
                "phase_lock": 0.92,
                "style_strength": 1.0,
                "structure": 0.92,
                "detail": 0.835,
                "mask_expand": 2,
                "mask_feather": 4.0,
                "detail_auto": True,
                "detail_threshold": 0.20,
                "detail_min_frames": 18,
                "donor_strength": 0.32,
                "donor_steps": 4,
                "donor_guidance": 0.0,
                "fine_mix": 0.90,
                "mid_mix": 0.30,
                "donor_min_gain": 1.03,
                "donor_geom_max": 0.70,
                "show_metabolism": True,
            }},
            {"type": "Bloom", "values": {"threshold": 0.74, "intensity": 0.18}},
            {"type": "ColorGrade", "values": {"contrast": 1.03, "saturation": 1.02}},
        ],
    )


register()
