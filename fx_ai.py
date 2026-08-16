"""AI model hosting for AI Video FX.

The app now uses two independent workers:

* geometry worker: depth + semantic segmentation
* diffusion worker: legacy whole-frame Dream plus frozen/live person and
  background keyframes for the layered PhaseRail effect

A slow diffusion pass therefore cannot make the ownership mask several seconds
stale.  All results are published into fx_core.MapStore.
"""
from __future__ import annotations

import threading
import time
import traceback
from pathlib import Path
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np

TORCH_OK = TRANSFORMERS_OK = DIFFUSERS_OK = False
try:
    import torch
    TORCH_OK = True
except Exception:  # pragma: no cover
    torch = None
try:
    from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation, pipeline
    TRANSFORMERS_OK = TORCH_OK
except Exception:  # pragma: no cover
    pass
try:
    from diffusers import AutoPipelineForImage2Image
    DIFFUSERS_OK = TORCH_OK
except Exception:  # pragma: no cover
    pass

DEPTH_MODELS = [
    "depth-anything/Depth-Anything-V2-Small-hf",
    "depth-anything/Depth-Anything-V2-Base-hf",
    "Intel/dpt-swinv2-tiny-256",
    "Intel/dpt-hybrid-midas",
    "vinvino02/glpn-nyu",
]
SEG_MODELS = [
    "nvidia/segformer-b0-finetuned-ade-512-512",
    "nvidia/segformer-b2-finetuned-ade-512-512",
    "mattmdjaga/segformer_b2_clothes",
    "jonathandinu/face-parsing",
]
STYLE_MODELS = [
    "stabilityai/sd-turbo",
    "stabilityai/sdxl-turbo",
    "Lykon/dreamshaper-8",
]


def best_device() -> str:
    if not TORCH_OK:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def deps_report() -> str:
    bits = [
        f"torch={'y' if TORCH_OK else 'n'}",
        f"transformers={'y' if TRANSFORMERS_OK else 'n'}",
        f"diffusers={'y' if DIFFUSERS_OK else 'n'}",
    ]
    if TORCH_OK:
        bits.append(f"device={best_device()}")
    return "  ".join(bits)


@dataclass
class AIConfig:
    device: str = field(default_factory=best_device)
    fps: float = 10.0

    depth_model: str = DEPTH_MODELS[0]
    depth_res: int = 384
    depth_invert: bool = False
    depth_smooth: float = 0.5

    seg_model: str = SEG_MODELS[0]
    seg_res: int = 512
    seg_targets: str = "person"
    seg_smooth: float = 0.4

    # Legacy full-frame AIDream settings.
    style_model: str = STYLE_MODELS[0]
    style_res: int = 512
    style_prompt: str = "a dreamlike oil painting, thick brush strokes, glowing colours"
    style_negative: str = "blurry, low quality, text, watermark"
    style_strength: float = 0.45
    style_steps: int = 2
    style_guidance: float = 0.0
    style_seed: int = -1
    style_live: bool = True
    style_revision: int = 0

    # Layered world settings. They deliberately share one diffusion model so a
    # 12 GB card does not try to keep two SDXL pipelines resident.
    person_prompt: str = "marble statue person, studio lighting, sharp facial detail"
    person_negative: str = "blurry, distorted face, extra limbs, text, watermark"
    person_strength: float = 0.58
    person_live: bool = False
    person_revision: int = 0

    background_prompt: str = "cinematic moonlit cathedral interior, no people"
    background_negative: str = "person, people, face, blurry, text, watermark"
    background_strength: float = 0.72
    background_live: bool = False
    background_revision: int = 0

    layer_res: int = 512
    layer_steps: int = 2
    layer_guidance: float = 0.0
    layer_seed: int = -1

    # Optional file bridge for an external LivePortrait/FasterLivePortrait
    # process. The generated person keyframe is exported; the external system
    # may write its latest animated frame to live_portrait_path.
    person_keyframe_path: str = "layer_person_keyframe.png"
    live_portrait_path: str = "live_portrait_latest.png"


class ModelHub:
    """Lazy model loader used by one worker thread."""

    def __init__(self, log: Callable[[str], None]) -> None:
        self.log = log
        self._depth = None
        self._depth_key = None
        self._seg = None
        self._seg_key = None
        self._style = None
        self._style_key = None
        self.seg_labels: List[str] = []

    def depth(self, model_id: str, device: str):
        if not TRANSFORMERS_OK:
            return None
        key = (model_id, device)
        if key != self._depth_key:
            self.log(f"loading depth model {model_id} on {device} …")
            dev = 0 if device == "cuda" else (device if device != "cpu" else -1)
            self._depth = pipeline("depth-estimation", model=model_id, device=dev)
            self._depth_key = key
            self.log(f"depth ready: {model_id}")
        return self._depth

    def seg(self, model_id: str, device: str):
        if not TRANSFORMERS_OK:
            return None
        key = (model_id, device)
        if key != self._seg_key:
            self.log(f"loading segmentation model {model_id} on {device} …")
            proc = AutoImageProcessor.from_pretrained(model_id)
            model = AutoModelForSemanticSegmentation.from_pretrained(model_id).to(device).eval()
            self._seg = (proc, model)
            self._seg_key = key
            id2label = getattr(model.config, "id2label", {}) or {}
            self.seg_labels = [
                str(v) for _, v in sorted(id2label.items(), key=lambda kv: int(kv[0]))
            ]
            self.log(f"segmentation ready: {len(self.seg_labels)} classes")
        return self._seg

    def target_ids(self, targets: str) -> List[int]:
        wanted = [t.strip().lower() for t in targets.split(",") if t.strip()]
        ids: List[int] = []
        for wanted_label in wanted:
            if wanted_label.isdigit():
                ids.append(int(wanted_label))
                continue
            for i, label in enumerate(self.seg_labels):
                parts = [p.strip() for p in label.lower().split(",")]
                if (
                    wanted_label == label.lower()
                    or wanted_label in parts
                    or wanted_label in label.lower()
                ):
                    ids.append(i)
        return sorted(set(ids))

    def style(self, model_id: str, device: str):
        if not DIFFUSERS_OK:
            return None
        key = (model_id, device)
        if key != self._style_key:
            # Drop a previous pipeline before loading a different one. This is
            # kinder to 8–12 GB cards than caching several large pipelines.
            if self._style is not None:
                del self._style
                self._style = None
                if TORCH_OK and torch.cuda.is_available():
                    torch.cuda.empty_cache()
            self.log(f"loading diffusion model {model_id} on {device} … (first run downloads GBs)")
            dtype = torch.float16 if device == "cuda" else torch.float32
            pipe = AutoPipelineForImage2Image.from_pretrained(
                model_id, torch_dtype=dtype, safety_checker=None, variant=None
            )
            pipe = pipe.to(device)
            pipe.set_progress_bar_config(disable=True)
            try:
                pipe.enable_attention_slicing()
            except Exception:
                pass
            self._style = pipe
            self._style_key = key
            self.log(f"diffusion ready: {model_id}")
        return self._style


class _BaseWorker(threading.Thread):
    daemon = True

    def __init__(self, name: str, store, cfg: AIConfig, log: Callable[[str], None]) -> None:
        super().__init__(name=name)
        self.store = store
        self.cfg = cfg
        self.log = log
        self.hub = ModelHub(log)
        self.needs: set = set()
        self._frame: Optional[np.ndarray] = None
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self.status = "idle"
        self.rates: dict[str, float] = {}
        self._last_t: dict[str, float] = {}

    def submit(self, frame_u8: np.ndarray, needs: set) -> None:
        with self._lock:
            self._frame = frame_u8
            self.needs = set(needs)

    def stop(self) -> None:
        self._stop_event.set()

    def _grab(self) -> Tuple[Optional[np.ndarray], set]:
        with self._lock:
            return (None if self._frame is None else self._frame.copy()), set(self.needs)

    def _tick(self, key: str) -> None:
        now = time.time()
        old = self._last_t.get(key, 0.0)
        dt = now - old
        current = self.rates.get(key, 0.0)
        if 0 < dt < 30:
            self.rates[key] = 0.7 * current + 0.3 * (1.0 / dt)
        self._last_t[key] = now

    @staticmethod
    def _fit(frame: np.ndarray, longest: int) -> np.ndarray:
        h, w = frame.shape[:2]
        scale = longest / float(max(h, w))
        if scale >= 1.0:
            return frame
        return cv2.resize(
            frame,
            (max(8, int(w * scale)), max(8, int(h * scale))),
            interpolation=cv2.INTER_AREA,
        )

    @staticmethod
    def _ema(store, key: str, new: np.ndarray, k: float) -> np.ndarray:
        old = store.get(key)
        if old is not None and old.shape == new.shape and k > 0:
            return (old * k + new * (1 - k)).astype(np.float32)
        return new


class GeometryWorker(_BaseWorker):
    def __init__(self, store, cfg: AIConfig, log: Callable[[str], None]) -> None:
        super().__init__("ai-geometry", store, cfg, log)
        self.rates = {"depth": 0.0, "mask": 0.0}

    def _run_depth(self, frame: np.ndarray) -> None:
        pipe = self.hub.depth(self.cfg.depth_model, self.cfg.device)
        if pipe is None:
            return
        from PIL import Image

        small = self._fit(frame, self.cfg.depth_res)
        pil = Image.fromarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
        out = pipe(pil)
        pred = out.get("predicted_depth", None)
        if pred is not None:
            depth = pred.squeeze().detach().float().cpu().numpy()
        else:
            depth = np.asarray(out["depth"], np.float32)
        lo, hi = np.percentile(depth, [2, 98])
        depth = np.clip((depth - lo) / max(1e-6, hi - lo), 0, 1).astype(np.float32)
        if self.cfg.depth_invert:
            depth = 1.0 - depth
        depth = cv2.resize(depth, (small.shape[1], small.shape[0]), interpolation=cv2.INTER_LINEAR)
        self.store.put("depth", self._ema(self.store, "depth", depth, self.cfg.depth_smooth))
        self._tick("depth")

    def _run_seg(self, frame: np.ndarray) -> None:
        pair = self.hub.seg(self.cfg.seg_model, self.cfg.device)
        if pair is None:
            return
        proc, model = pair
        small = self._fit(frame, self.cfg.seg_res)
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        inputs = proc(images=rgb, return_tensors="pt").to(self.cfg.device)
        with torch.no_grad():
            logits = model(**inputs).logits
        logits = torch.nn.functional.interpolate(
            logits, size=small.shape[:2], mode="bilinear", align_corners=False
        )
        ids = self.hub.target_ids(self.cfg.seg_targets)
        if ids:
            prob = logits.softmax(1)[0]
            mask = prob[ids].sum(0).detach().float().cpu().numpy()
        else:
            labels = logits.argmax(1)[0].detach().cpu().numpy()
            mask = (labels != 0).astype(np.float32)
        mask = np.clip(mask, 0, 1).astype(np.float32)
        self.store.put("mask", self._ema(self.store, "mask", mask, self.cfg.seg_smooth))
        self._tick("mask")

    def run(self) -> None:
        while not self._stop_event.is_set():
            frame, needs = self._grab()
            requested = needs & {"depth", "mask"}
            if frame is None or not requested:
                self.status = "idle"
                time.sleep(0.03)
                continue
            started = time.time()
            try:
                self.status = "running"
                if "depth" in requested:
                    self._run_depth(frame)
                if "mask" in requested:
                    self._run_seg(frame)
            except Exception as exc:
                self.status = f"error: {exc}"
                self.log(f"geometry AI error: {exc}")
                self.log(traceback.format_exc(limit=3))
                time.sleep(1.0)
            budget = 1.0 / max(0.5, self.cfg.fps)
            time.sleep(max(0.0, budget - (time.time() - started)))


class DiffusionWorker(_BaseWorker):
    KEYS = {"style", "person_style", "background_style"}

    def __init__(self, store, cfg: AIConfig, log: Callable[[str], None]) -> None:
        super().__init__("ai-diffusion", store, cfg, log)
        self.rates = {key: 0.0 for key in self.KEYS}
        self._done_signature: dict[str, tuple] = {}
        self._last_run: dict[str, float] = {key: 0.0 for key in self.KEYS}

    def _channel(self, key: str):
        if key == "style":
            return dict(
                prompt=self.cfg.style_prompt,
                negative=self.cfg.style_negative,
                strength=self.cfg.style_strength,
                steps=self.cfg.style_steps,
                guidance=self.cfg.style_guidance,
                res=self.cfg.style_res,
                seed=self.cfg.style_seed,
                live=self.cfg.style_live,
                revision=self.cfg.style_revision,
                mode="whole",
            )
        if key == "person_style":
            return dict(
                prompt=self.cfg.person_prompt,
                negative=self.cfg.person_negative,
                strength=self.cfg.person_strength,
                steps=self.cfg.layer_steps,
                guidance=self.cfg.layer_guidance,
                res=self.cfg.layer_res,
                seed=self.cfg.layer_seed,
                live=self.cfg.person_live,
                revision=self.cfg.person_revision,
                mode="person",
            )
        return dict(
            prompt=self.cfg.background_prompt,
            negative=self.cfg.background_negative,
            strength=self.cfg.background_strength,
            steps=self.cfg.layer_steps,
            guidance=self.cfg.layer_guidance,
            res=self.cfg.layer_res,
            seed=self.cfg.layer_seed,
            live=self.cfg.background_live,
            revision=self.cfg.background_revision,
            mode="background",
        )

    def _signature(self, key: str, channel: dict) -> tuple:
        # Prompt text is intentionally committed by the Generate button when
        # frozen. In Live mode it is read on every pass.
        return (
            int(channel["revision"]),
            self.cfg.style_model,
            float(channel["strength"]),
            int(channel["steps"]),
            float(channel["guidance"]),
            int(channel["res"]),
            int(channel["seed"]),
            channel["mode"],
        )

    def _due(self, key: str, channel: dict) -> bool:
        signature = self._signature(key, channel)
        if self.store.get(key) is None or self._done_signature.get(key) != signature:
            return True
        if not channel["live"]:
            return False
        return time.time() - self._last_run[key] >= 1.0 / max(0.25, self.cfg.fps)

    def _prepare_input(self, frame: np.ndarray, mode: str) -> np.ndarray:
        if mode == "whole":
            return frame
        mask = self.store.get("mask")
        if mask is None:
            return frame
        h, w = frame.shape[:2]
        if mask.shape[:2] != (h, w):
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)
        mask = np.clip(mask, 0, 1)[..., None].astype(np.float32)
        source = frame.astype(np.float32)
        # The AI sees the correct subject outline but not a sharp accidental
        # room cutout. Conversely the background generator sees the room with a
        # soft hole where the person was and is free to dream that hole closed.
        blurred = cv2.GaussianBlur(frame, (0, 0), 28).astype(np.float32)
        if mode == "person":
            prepared = source * mask + blurred * (1.0 - mask)
        else:
            prepared = source * (1.0 - mask) + blurred * mask
        return np.clip(prepared, 0, 255).astype(np.uint8)

    def _run_channel(self, frame: np.ndarray, key: str, channel: dict) -> None:
        pipe = self.hub.style(self.cfg.style_model, self.cfg.device)
        if pipe is None:
            return
        from PIL import Image

        prepared = self._prepare_input(frame, channel["mode"])
        res = int(channel["res"])
        small = cv2.resize(prepared, (res, res), interpolation=cv2.INTER_AREA)
        pil = Image.fromarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))

        generator = None
        if int(channel["seed"]) >= 0:
            generator = torch.Generator(device="cpu").manual_seed(int(channel["seed"]))
        steps = max(1, int(channel["steps"]))
        strength = float(np.clip(channel["strength"], 0.05, 0.99))
        if steps * strength < 1.0:
            steps = int(np.ceil(1.0 / strength))
        kwargs = dict(
            prompt=channel["prompt"],
            image=pil,
            strength=strength,
            num_inference_steps=steps,
            guidance_scale=float(channel["guidance"]),
        )
        if channel["guidance"] > 0 and channel["negative"]:
            kwargs["negative_prompt"] = channel["negative"]
        if generator is not None:
            kwargs["generator"] = generator
        with torch.inference_mode():
            image = pipe(**kwargs).images[0]
        array = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR).astype(np.float32) / 255.0
        h, w = frame.shape[:2]
        array = cv2.resize(array, (w, h), interpolation=cv2.INTER_LINEAR)
        self.store.put(key, array)
        if key == "person_style" and self.cfg.person_keyframe_path.strip():
            try:
                cv2.imwrite(self.cfg.person_keyframe_path.strip(),
                            np.clip(array * 255.0, 0, 255).astype(np.uint8))
            except Exception as exc:
                self.log(f"could not export person keyframe: {exc}")
        self._done_signature[key] = self._signature(key, channel)
        self._last_run[key] = time.time()
        self._tick(key)

    def run(self) -> None:
        while not self._stop_event.is_set():
            frame, needs = self._grab()
            requested = [key for key in ("style", "person_style", "background_style") if key in needs]
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


class PortraitBridgeWorker(_BaseWorker):
    """Watch a file written by an external real-time portrait animator."""

    def __init__(self, store, cfg: AIConfig, log: Callable[[str], None]) -> None:
        super().__init__("portrait-bridge", store, cfg, log)
        self.rates = {"live_portrait": 0.0}
        self._mtime = -1.0

    def run(self) -> None:
        while not self._stop_event.is_set():
            _, needs = self._grab()
            if "live_portrait" not in needs:
                self.status = "idle"
                time.sleep(0.08)
                continue
            raw = self.cfg.live_portrait_path.strip()
            if not raw:
                self.status = "no bridge path"
                time.sleep(0.2)
                continue
            path = Path(raw)
            try:
                mtime = path.stat().st_mtime
            except OSError:
                self.status = f"waiting for {path.name}"
                time.sleep(0.08)
                continue
            if mtime != self._mtime:
                image = cv2.imread(str(path), cv2.IMREAD_COLOR)
                if image is not None:
                    self.store.put("live_portrait", image.astype(np.float32) / 255.0)
                    self._mtime = mtime
                    self._tick("live_portrait")
                    self.status = "receiving"
            time.sleep(0.02)


class AIWorker:
    """Facade preserving the original App-facing API."""

    def __init__(self, store, cfg: AIConfig, log: Callable[[str], None]) -> None:
        self.geometry = GeometryWorker(store, cfg, log)
        self.diffusion = DiffusionWorker(store, cfg, log)
        self.portrait = PortraitBridgeWorker(store, cfg, log)
        self.hub = self.geometry.hub  # segmentation class list compatibility

    def start(self) -> None:
        self.geometry.start()
        self.diffusion.start()
        self.portrait.start()

    def stop(self) -> None:
        self.geometry.stop()
        self.diffusion.stop()
        self.portrait.stop()

    def submit(self, frame_u8: np.ndarray, needs: set) -> None:
        self.geometry.submit(frame_u8, needs)
        self.diffusion.submit(frame_u8, needs)
        self.portrait.submit(frame_u8, needs)

    @property
    def rates(self) -> dict[str, float]:
        out = {"depth": 0.0, "mask": 0.0, "style": 0.0,
               "person_style": 0.0, "background_style": 0.0,
               "live_portrait": 0.0}
        out.update(self.geometry.rates)
        out.update(self.diffusion.rates)
        out.update(self.portrait.rates)
        return out

    @property
    def status(self) -> str:
        g = self.geometry.status
        d = self.diffusion.status
        p = self.portrait.status
        if d not in ("idle", "frozen keyframes"):
            return d
        if p not in ("idle",):
            return f"portrait {p}; diffusion {d}"
        if g != "idle":
            return f"geometry {g}; diffusion {d}"
        return d if d != "idle" else "idle"
