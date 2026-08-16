"""Runtime patch adding a dedicated img2img detail-donor channel.

Import this module *before* ai_video_fx. It replaces fx_ai.DiffusionWorker with
one compatible subclass that understands a fourth key, ``detail_donor``.

The effect publishes:
    detail_request        - tiny nonce array; its store stamp is the request id
    detail_request_image  - latest carried BGR float image to use as img2img init

Settings for that one request live in store.state['_detail_request_cfg'] so the
ordinary GUI/AIConfig does not need a second permanent diffusion panel.

When a normal ``person_style`` keyframe is generated, its prompt/negative prompt
are frozen into store.state as the donor anchor prompt. Later text edits do not
silently alter metabolism donors; pressing Generate person intentionally creates
a new anchor and therefore a new frozen donor prompt.
"""
from __future__ import annotations

import time
import traceback

import cv2
import numpy as np

import fx_ai


BaseDiffusionWorker = fx_ai.DiffusionWorker


class MetabolismDiffusionWorker(BaseDiffusionWorker):
    KEYS = set(BaseDiffusionWorker.KEYS) | {"detail_donor"}

    def __init__(self, store, cfg, log):
        super().__init__(store, cfg, log)
        self._detail_request_done_stamp = 0.0

    def _detail_cfg(self) -> dict:
        raw = self.store.state.get("_detail_request_cfg", {})
        return dict(raw) if isinstance(raw, dict) else {}

    def _channel(self, key: str):
        if key != "detail_donor":
            return super()._channel(key)
        d = self._detail_cfg()
        anchor_prompt = str(
            self.store.state.get("_detail_anchor_prompt", self.cfg.person_prompt)
        )
        anchor_negative = str(
            self.store.state.get("_detail_anchor_negative", self.cfg.person_negative)
        )
        return dict(
            prompt=str(d.get("prompt", anchor_prompt)),
            negative=str(d.get("negative", anchor_negative)),
            strength=float(d.get("strength", 0.32)),
            steps=int(d.get("steps", max(4, int(self.cfg.layer_steps)))),
            guidance=float(d.get("guidance", self.cfg.layer_guidance)),
            res=int(d.get("res", self.cfg.layer_res)),
            seed=int(d.get("seed", self.cfg.layer_seed)),
            live=False,
            revision=0,
            mode="detail",
        )

    def _due(self, key: str, channel: dict) -> bool:
        if key != "detail_donor":
            return super()._due(key, channel)
        req_stamp = float(self.store.stamp("detail_request"))
        return (
            req_stamp > self._detail_request_done_stamp
            and self.store.get("detail_request_image") is not None
        )

    def _run_detail(self, key: str, channel: dict) -> None:
        pipe = self.hub.style(self.cfg.style_model, self.cfg.device)
        if pipe is None:
            return
        from PIL import Image

        request = self.store.get("detail_request_image")
        if request is None:
            return
        x = np.asarray(request)
        if x.dtype == np.uint8:
            prepared = x
        else:
            prepared = np.clip(x.astype(np.float32) * 255.0, 0, 255).astype(np.uint8)

        res = int(channel["res"])
        small = cv2.resize(prepared, (res, res), interpolation=cv2.INTER_AREA)
        pil = Image.fromarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))

        generator = None
        if int(channel["seed"]) >= 0:
            generator = fx_ai.torch.Generator(device="cpu").manual_seed(int(channel["seed"]))
        steps = max(1, int(channel["steps"]))
        strength = float(np.clip(channel["strength"], 0.05, 0.95))
        if steps * strength < 1.0:
            steps = int(np.ceil(1.0 / strength))
        kwargs = dict(
            prompt=channel["prompt"], image=pil, strength=strength,
            num_inference_steps=steps, guidance_scale=float(channel["guidance"]),
        )
        if channel["guidance"] > 0 and channel["negative"]:
            kwargs["negative_prompt"] = channel["negative"]
        if generator is not None:
            kwargs["generator"] = generator

        # Capture request id immediately before the expensive call. If the
        # effect keeps updating detail_request_image while we generate, this
        # donor still belongs to the request that caused this pass.
        req_stamp = float(self.store.stamp("detail_request"))
        with fx_ai.torch.inference_mode():
            image = pipe(**kwargs).images[0]
        array = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR).astype(np.float32) / 255.0
        h, w = prepared.shape[:2]
        array = cv2.resize(array, (w, h), interpolation=cv2.INTER_LINEAR)
        self.store.put(key, array)
        self.store.put("detail_donor_request_stamp", np.asarray([req_stamp], np.float64))
        self._detail_request_done_stamp = req_stamp
        self._last_run[key] = time.time()
        self._tick(key)

    def _run_channel(self, frame: np.ndarray, key: str, channel: dict) -> None:
        if key == "detail_donor":
            self._run_detail(key, channel)
            return

        super()._run_channel(frame, key, channel)
        if key == "person_style" and self.store.get("person_style") is not None:
            # Freeze the actual prompt which produced this accepted identity
            # anchor. Donors reuse it until the user deliberately makes a new
            # person keyframe.
            self.store.state["_detail_anchor_prompt"] = str(channel["prompt"])
            self.store.state["_detail_anchor_negative"] = str(channel["negative"])

    def run(self) -> None:
        order = ("style", "person_style", "background_style", "detail_donor")
        while not self._stop_event.is_set():
            frame, needs = self._grab()
            requested = [key for key in order if key in needs]
            if frame is None or not requested:
                self.status = "idle"
                time.sleep(0.04)
                continue
            did_work = False
            try:
                self.status = "running"
                for key in requested:
                    channel = self._channel(key)
                    if self._due(key, channel):
                        self.status = f"generating {key}"
                        self._run_channel(frame, key, channel)
                        did_work = True
                if not did_work:
                    self.status = "frozen keyframes"
                    time.sleep(0.04)
            except Exception as exc:
                self.status = f"error: {exc}"
                self.log(f"diffusion AI error: {exc}")
                self.log(traceback.format_exc(limit=3))
                time.sleep(1.0)


# AIWorker.__init__ resolves this module global at runtime, so replacing it
# before App constructs AIWorker is sufficient; no fork of ai_video_fx.py.
fx_ai.DiffusionWorker = MetabolismDiffusionWorker
