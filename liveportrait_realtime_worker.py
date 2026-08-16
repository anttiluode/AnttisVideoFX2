#!/usr/bin/env python
"""Realtime file worker using the official LivePortrait model.

Run this in LivePortrait's own Python 3.10 / conda environment while
AnttisVideoFX2 runs in its normal environment.  The two processes communicate
through three images:

    layer_person_keyframe.png   generated/stylised source appearance
    live_portrait_drive.png     latest webcam frame from AnttisVideoFX2
    live_portrait_latest.png    LivePortrait face placed in current coordinates
    live_portrait_alpha.png     companion face alpha matte

This is intentionally a *face correspondence control*, not a replacement for
our whole-person transport.  ``Antti LivePortrait Assist`` uses the normal
PhaseRail body and lets LivePortrait own only the face region.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


def atomic_imwrite(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp" + path.suffix)
    if not cv2.imwrite(str(tmp), image):
        raise IOError(f"could not write {tmp}")
    os.replace(tmp, path)


def read_image_retry(path: Path) -> Optional[np.ndarray]:
    try:
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    except Exception:
        return None
    return img


class RealtimeLivePortrait:
    def __init__(self, home: Path, *, device_id: int = 0, force_cpu: bool = False,
                 multiplier: float = 1.0) -> None:
        home = home.resolve()
        if not (home / "src").exists():
            raise FileNotFoundError(f"LivePortrait checkout not found at {home}")
        sys.path.insert(0, str(home))

        # Import only after the external checkout is on sys.path.  Keeping these
        # imports out of AnttisVideoFX2 is the whole point of the process bridge.
        from src.config.inference_config import InferenceConfig
        from src.config.crop_config import CropConfig
        from src.live_portrait_wrapper import LivePortraitWrapper
        from src.utils.cropper import Cropper
        from src.utils.camera import get_rotation_matrix
        from src.utils.crop import prepare_paste_back, paste_back

        self.get_rotation_matrix = get_rotation_matrix
        self.prepare_paste_back = prepare_paste_back
        self.paste_back = paste_back

        self.inf_cfg = InferenceConfig(
            device_id=int(device_id),
            flag_force_cpu=bool(force_cpu),
            flag_stitching=True,
            flag_relative_motion=True,
            flag_pasteback=False,
            flag_do_crop=True,
            driving_multiplier=float(multiplier),
        )
        self.crop_cfg = CropConfig(device_id=int(device_id), flag_force_cpu=bool(force_cpu))
        self.wrapper = LivePortraitWrapper(inference_cfg=self.inf_cfg)
        self.cropper = Cropper(
            crop_cfg=self.crop_cfg,
            device_id=int(device_id),
            flag_force_cpu=bool(force_cpu),
        )

        self.source_bgr: Optional[np.ndarray] = None
        self.x_s_info = None
        self.x_c_s = None
        self.R_s = None
        self.f_s = None
        self.x_s = None
        self.x_d0_info = None
        self.R_d0 = None

    @staticmethod
    def _clone_info(info: dict) -> dict:
        out = {}
        for k, v in info.items():
            try:
                out[k] = v.clone()
            except AttributeError:
                out[k] = v
        return out

    def _source_crop(self, source_rgb: np.ndarray, driver_rgb: Optional[np.ndarray]):
        crop = self.cropper.crop_source_image(source_rgb, self.crop_cfg)
        if crop is not None:
            return crop

        # Stylised sources can occasionally defeat the face detector even though
        # they were generated from a face-aligned webcam frame.  In that case use
        # a real driving face only to obtain the crop transform; appearance still
        # comes entirely from the generated source.
        if driver_rgb is None:
            return None
        drive_crop = self.cropper.crop_source_image(driver_rgb, self.crop_cfg)
        if drive_crop is None:
            return None
        M_o2c = np.asarray(drive_crop["M_o2c"], np.float32)
        stylised_crop = cv2.warpPerspective(
            source_rgb,
            M_o2c,
            (int(self.crop_cfg.dsize), int(self.crop_cfg.dsize)),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT,
        )
        drive_crop = dict(drive_crop)
        drive_crop["img_crop"] = stylised_crop
        drive_crop["img_crop_256x256"] = cv2.resize(
            stylised_crop, (256, 256), interpolation=cv2.INTER_AREA
        )
        return drive_crop

    def set_source(self, source_bgr: np.ndarray, driver_bgr: Optional[np.ndarray] = None) -> bool:
        source_rgb = cv2.cvtColor(source_bgr, cv2.COLOR_BGR2RGB)
        driver_rgb = None if driver_bgr is None else cv2.cvtColor(driver_bgr, cv2.COLOR_BGR2RGB)
        crop = self._source_crop(source_rgb, driver_rgb)
        if crop is None:
            return False

        I_s = self.wrapper.prepare_source(crop["img_crop_256x256"])
        x_s_info = self.wrapper.get_kp_info(I_s)
        self.x_s_info = x_s_info
        self.x_c_s = x_s_info["kp"]
        self.R_s = self.get_rotation_matrix(
            x_s_info["pitch"], x_s_info["yaw"], x_s_info["roll"]
        )
        self.f_s = self.wrapper.extract_feature_3d(I_s)
        self.x_s = self.wrapper.transform_keypoint(x_s_info)
        self.x_d0_info = None
        self.R_d0 = None
        self.source_bgr = source_bgr.copy()
        return True

    def animate(self, driver_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
        if self.x_s_info is None:
            return None
        driver_rgb = cv2.cvtColor(driver_bgr, cv2.COLOR_BGR2RGB)
        crop = self.cropper.crop_source_image(driver_rgb, self.crop_cfg)
        if crop is None:
            return None

        I_d = self.wrapper.prepare_source(crop["img_crop_256x256"])
        x_d_info = self.wrapper.get_kp_info(I_d)
        R_d = self.get_rotation_matrix(
            x_d_info["pitch"], x_d_info["yaw"], x_d_info["roll"]
        )
        if self.x_d0_info is None:
            self.x_d0_info = self._clone_info(x_d_info)
            self.R_d0 = R_d.clone()

        # This is the image-source / relative-motion branch of the official
        # LivePortrait pipeline, kept deliberately close to upstream Algorithm 1.
        R_new = (R_d @ self.R_d0.permute(0, 2, 1)) @ self.R_s
        delta_new = self.x_s_info["exp"] + (x_d_info["exp"] - self.x_d0_info["exp"])
        scale_new = self.x_s_info["scale"] * (
            x_d_info["scale"] / self.x_d0_info["scale"]
        )
        t_new = self.x_s_info["t"] + (x_d_info["t"] - self.x_d0_info["t"])
        t_new = t_new.clone()
        t_new[..., 2].fill_(0)
        x_d_new = scale_new * (self.x_c_s @ R_new + delta_new) + t_new
        if self.inf_cfg.flag_stitching:
            x_d_new = self.wrapper.stitching(self.x_s, x_d_new)
        x_d_new = self.x_s + (x_d_new - self.x_s) * float(self.inf_cfg.driving_multiplier)

        out = self.wrapper.warp_decode(self.f_s, self.x_s, x_d_new)
        face_rgb = self.wrapper.parse_output(out["out"])[0]

        h, w = driver_rgb.shape[:2]
        M_c2o = crop["M_c2o"]
        alpha = self.prepare_paste_back(
            self.inf_cfg.mask_crop, M_c2o, dsize=(w, h)
        ).astype(np.float32)
        if alpha.ndim == 2:
            alpha3 = np.repeat(alpha[..., None], 3, axis=2)
        else:
            alpha3 = alpha
        zeros = np.zeros((h, w, 3), np.uint8)
        ones = np.ones((h, w, 3), np.float32)
        # mask=1 asks upstream paste_back only for the transformed portrait;
        # the real alpha is exported separately and composited in our FX layer.
        face_full_rgb = self.paste_back(face_rgb, M_c2o, zeros, ones)
        alpha_gray = np.clip(np.mean(alpha3, axis=2) * 255.0, 0, 255).astype(np.uint8)
        face_full_bgr = cv2.cvtColor(face_full_rgb, cv2.COLOR_RGB2BGR)
        return face_full_bgr, alpha_gray


def main() -> None:
    ap = argparse.ArgumentParser(description="Realtime LivePortrait file bridge for AnttisVideoFX2")
    ap.add_argument("--liveportrait-dir", default=os.environ.get("LIVEPORTRAIT_HOME", ""))
    ap.add_argument("--source", default="layer_person_keyframe.png")
    ap.add_argument("--driving", default=os.environ.get("LIVEPORTRAIT_DRIVE_PATH", "live_portrait_drive.png"))
    ap.add_argument("--output", default="live_portrait_latest.png")
    ap.add_argument("--alpha", default=os.environ.get("LIVEPORTRAIT_ALPHA_PATH", "live_portrait_alpha.png"))
    ap.add_argument("--device-id", type=int, default=0)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--multiplier", type=float, default=1.0)
    ap.add_argument("--poll", type=float, default=0.01)
    args = ap.parse_args()

    if not args.liveportrait_dir:
        raise SystemExit(
            "Set LIVEPORTRAIT_HOME or pass --liveportrait-dir pointing to the official LivePortrait checkout."
        )

    engine = RealtimeLivePortrait(
        Path(args.liveportrait_dir),
        device_id=args.device_id,
        force_cpu=args.cpu,
        multiplier=args.multiplier,
    )
    src_path = Path(args.source)
    drv_path = Path(args.driving)
    out_path = Path(args.output)
    alpha_path = Path(args.alpha)
    src_mtime = -1.0
    drv_mtime = -1.0
    last_driver: Optional[np.ndarray] = None
    count = 0
    t0 = time.time()

    print("LivePortrait bridge ready")
    print(f"  source : {src_path}")
    print(f"  driving: {drv_path}")
    print(f"  output : {out_path}")

    while True:
        try:
            try:
                sm = src_path.stat().st_mtime
            except OSError:
                sm = -1.0
            try:
                dm = drv_path.stat().st_mtime
            except OSError:
                dm = -1.0

            if sm >= 0 and sm != src_mtime:
                source = read_image_retry(src_path)
                if source is not None and engine.set_source(source, last_driver):
                    src_mtime = sm
                    print("accepted new generated source; driving origin reset")
                elif source is not None:
                    print("source face not found yet; waiting for a driving face to borrow crop geometry")

            if dm >= 0 and dm != drv_mtime:
                driver = read_image_retry(drv_path)
                if driver is not None:
                    last_driver = driver
                    # A stylised source that could not be detected gets one more
                    # chance now that a real driving crop is available.
                    if sm >= 0 and sm != src_mtime:
                        source = read_image_retry(src_path)
                        if source is not None and engine.set_source(source, driver):
                            src_mtime = sm
                            print("accepted generated source using driving-face crop")
                    result = engine.animate(driver)
                    if result is not None:
                        face, alpha = result
                        atomic_imwrite(out_path, face)
                        atomic_imwrite(alpha_path, alpha)
                        count += 1
                        if count % 30 == 0:
                            dt = max(1e-6, time.time() - t0)
                            print(f"{count / dt:.1f} fps  frames={count}")
                    drv_mtime = dm
            time.sleep(max(0.001, float(args.poll)))
        except KeyboardInterrupt:
            break
        except Exception as exc:
            print(f"LivePortrait bridge error: {type(exc).__name__}: {exc}", flush=True)
            time.sleep(0.25)


if __name__ == "__main__":
    main()
