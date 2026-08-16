"""Runtime patch that completes the existing LivePortrait file bridge.

AnttisVideoFX2 already had both halves of a LivePortrait integration:

* diffusion exports ``layer_person_keyframe.png``;
* ``PortraitBridgeWorker`` watches ``live_portrait_latest.png`` and publishes it
  as the ``live_portrait`` map.

What was missing was the wire in the other direction.  This patch exports the
latest webcam/source frame to ``live_portrait_drive.png`` whenever an effect
asks for LivePortrait, and also ingests a companion alpha matte written by the
realtime worker.

The external LivePortrait process deliberately stays in its own Python/conda
environment.  The main app therefore remains replaceable and does not inherit
LivePortrait's dependency stack.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import cv2
import numpy as np

import fx_ai


BasePortraitBridgeWorker = fx_ai.PortraitBridgeWorker


def _atomic_imwrite(path: Path, image: np.ndarray) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp" + path.suffix)
    ok = cv2.imwrite(str(tmp), image)
    if ok:
        os.replace(tmp, path)
    return bool(ok)


class WiredPortraitBridgeWorker(BasePortraitBridgeWorker):
    """Bidirectional file bridge for a separate LivePortrait process."""

    def __init__(self, store, cfg, log):
        super().__init__(store, cfg, log)
        self.drive_path = Path(os.environ.get("LIVEPORTRAIT_DRIVE_PATH", "live_portrait_drive.png"))
        self.alpha_path = Path(os.environ.get("LIVEPORTRAIT_ALPHA_PATH", "live_portrait_alpha.png"))
        self.drive_fps = max(0.5, float(os.environ.get("LIVEPORTRAIT_DRIVE_FPS", "15")))
        self._last_drive_write = 0.0
        self._alpha_mtime = -1.0
        self.rates.setdefault("live_portrait_alpha", 0.0)
        self.rates.setdefault("live_portrait_drive", 0.0)

    def _publish_output(self) -> bool:
        changed = False
        raw = self.cfg.live_portrait_path.strip()
        if raw:
            path = Path(raw)
            try:
                mtime = path.stat().st_mtime
            except OSError:
                mtime = -1.0
            if mtime >= 0 and mtime != self._mtime:
                image = cv2.imread(str(path), cv2.IMREAD_COLOR)
                if image is not None:
                    self.store.put("live_portrait", image.astype(np.float32) / 255.0)
                    self._mtime = mtime
                    self._tick("live_portrait")
                    changed = True

        try:
            alpha_mtime = self.alpha_path.stat().st_mtime
        except OSError:
            alpha_mtime = -1.0
        if alpha_mtime >= 0 and alpha_mtime != self._alpha_mtime:
            alpha = cv2.imread(str(self.alpha_path), cv2.IMREAD_GRAYSCALE)
            if alpha is not None:
                self.store.put("live_portrait_alpha", alpha.astype(np.float32) / 255.0)
                self._alpha_mtime = alpha_mtime
                self._tick("live_portrait_alpha")
                changed = True
        return changed

    def _publish_drive(self, frame: np.ndarray) -> bool:
        now = time.time()
        if now - self._last_drive_write < 1.0 / self.drive_fps:
            return False
        if _atomic_imwrite(self.drive_path, frame):
            self._last_drive_write = now
            self._tick("live_portrait_drive")
            return True
        return False

    def run(self) -> None:
        while not self._stop_event.is_set():
            frame, needs = self._grab()
            requested = bool({"live_portrait", "live_portrait_alpha"} & needs)
            if not requested:
                self.status = "idle"
                time.sleep(0.06)
                continue

            if frame is not None:
                try:
                    self._publish_drive(frame)
                except Exception as exc:
                    self.log(f"LivePortrait drive export failed: {exc}")

            changed = False
            try:
                changed = self._publish_output()
            except Exception as exc:
                self.log(f"LivePortrait output ingest failed: {exc}")

            source_ok = bool(self.cfg.person_keyframe_path.strip()) and Path(
                self.cfg.person_keyframe_path.strip()
            ).exists()
            output_ok = bool(self.cfg.live_portrait_path.strip()) and Path(
                self.cfg.live_portrait_path.strip()
            ).exists()
            if changed or output_ok:
                self.status = "receiving LivePortrait"
            elif not source_ok:
                self.status = "waiting for generated person keyframe"
            else:
                self.status = f"driving -> {self.drive_path.name}; waiting for LivePortrait"
            time.sleep(0.015)


# AIWorker resolves this module global when it is constructed.  Import this
# patch before ai_video_fx, exactly like detail_metabolism_patch.
fx_ai.PortraitBridgeWorker = WiredPortraitBridgeWorker
