"""
fx_core.py — effect engine for AI Video FX.

Design notes
------------
* The pixel bus is float32 BGR in [0, 1], shape (H, W, 3). Effects get and
  return that; conversion to uint8 happens once, at the very end.
* Slow AI models never run inline. They publish maps (depth / mask / stylized
  frame) into a MapStore from a worker thread; effects read whatever is
  currently there through FXContext, which resizes and caches per frame.
  That is what keeps the preview at camera framerate while a depth model
  chugs along at 8 fps.
* Per-effect persistent state (feedback buffers, ring buffers) lives in
  ctx.state[effect.uid], so an effect instance can be duplicated safely.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Set

import cv2
import numpy as np

EPS = 1e-6


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------
def to_f32(frame_u8: np.ndarray) -> np.ndarray:
    return frame_u8.astype(np.float32) * (1.0 / 255.0)


def to_u8(img: np.ndarray) -> np.ndarray:
    return np.clip(img * 255.0, 0, 255).astype(np.uint8)


_LUMA_W = np.array([0.0722, 0.7152, 0.2126], np.float32)   # BGR order


def luma(img: np.ndarray) -> np.ndarray:
    """Rec.709 luminance of a BGR float image -> (H, W) float32. matmul beats
    three broadcast multiplies by ~4x, and this runs on every frame."""
    return img @ _LUMA_W


def smoothstep(e0: float, e1: float, x: np.ndarray) -> np.ndarray:
    t = np.clip((x - e0) / (e1 - e0 + EPS), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def hex_to_bgr(h: str) -> np.ndarray:
    h = h.lstrip("#")
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return np.array([b, g, r], np.float32) / 255.0


def screen(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return 1.0 - (1.0 - a) * (1.0 - b)


def blur(img: np.ndarray, radius: float) -> np.ndarray:
    """Odd-kernel gaussian that tolerates radius 0 and large radii cheaply."""
    r = int(max(0, radius))
    if r < 1:
        return img
    k = r * 2 + 1
    if k > 31:  # downsample-blur-upsample: much faster, visually identical
        h, w = img.shape[:2]
        s = min(4, max(2, k // 15))
        small = cv2.resize(img, (max(2, w // s), max(2, h // s)), interpolation=cv2.INTER_AREA)
        k2 = max(3, (k // s) | 1)
        small = cv2.GaussianBlur(small, (k2, k2), 0)
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    return cv2.GaussianBlur(img, (k, k), 0)


def normalize01(a: np.ndarray, lo_pct: float = 2.0, hi_pct: float = 98.0) -> np.ndarray:
    lo, hi = np.percentile(a, [lo_pct, hi_pct])
    if hi - lo < EPS:
        return np.zeros_like(a, np.float32)
    return np.clip((a - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


_LUT_CACHE: Dict[str, np.ndarray] = {}


def colormap_lut(name: str) -> np.ndarray:
    """256x3 float32 BGR LUT. Indexing a LUT is ~5x faster than applyColorMap."""
    lut = _LUT_CACHE.get(name)
    if lut is None:
        ramp = np.arange(256, dtype=np.uint8).reshape(1, 256)
        lut = cv2.applyColorMap(ramp, COLORMAPS[name])[0].astype(np.float32) / 255.0
        _LUT_CACHE[name] = lut
    return lut


COLORMAPS = {
    "turbo": cv2.COLORMAP_TURBO,
    "inferno": cv2.COLORMAP_INFERNO,
    "magma": cv2.COLORMAP_MAGMA,
    "viridis": cv2.COLORMAP_VIRIDIS,
    "ocean": cv2.COLORMAP_OCEAN,
    "bone": cv2.COLORMAP_BONE,
    "hsv": cv2.COLORMAP_HSV,
}


# --------------------------------------------------------------------------
# parameters
# --------------------------------------------------------------------------
@dataclass
class Param:
    key: str
    label: str
    kind: str = "float"          # float | int | bool | choice | color
    default: Any = 0.0
    lo: float = 0.0
    hi: float = 1.0
    choices: Sequence[str] = ()
    hint: str = ""


# --------------------------------------------------------------------------
# shared map store (written by the AI worker, read by the processor)
# --------------------------------------------------------------------------
class MapStore:
    """Thread-safe slot for the latest AI outputs plus effect state."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._maps: Dict[str, np.ndarray] = {}
        self._stamps: Dict[str, float] = {}
        self.state: Dict[str, dict] = {}

    def put(self, key: str, arr: np.ndarray) -> None:
        with self._lock:
            self._maps[key] = arr
            self._stamps[key] = time.time()

    def get(self, key: str) -> Optional[np.ndarray]:
        with self._lock:
            return self._maps.get(key)

    def age(self, key: str) -> float:
        with self._lock:
            t = self._stamps.get(key)
        return 1e9 if t is None else time.time() - t

    def stamp(self, key: str) -> float:
        """Monotonic-enough publication stamp for detecting a new AI keyframe."""
        with self._lock:
            return float(self._stamps.get(key, 0.0))

    def drop(self, key: str) -> None:
        with self._lock:
            self._maps.pop(key, None)
            self._stamps.pop(key, None)

    def clear_state(self) -> None:
        with self._lock:
            self.state.clear()


class FXContext:
    """Per-frame view onto the store: resizes maps to the working resolution once."""

    def __init__(self, store: MapStore, t: float, idx: int, shape) -> None:
        self.store = store
        self.t = t
        self.idx = idx
        self.shape = shape           # (H, W)
        self.state = store.state
        self._cache: Dict[str, Optional[np.ndarray]] = {}

    def _fit(self, key: str) -> Optional[np.ndarray]:
        if key in self._cache:
            return self._cache[key]
        arr = self.store.get(key)
        if arr is not None:
            h, w = self.shape
            if arr.shape[0] != h or arr.shape[1] != w:
                arr = cv2.resize(arr, (w, h), interpolation=cv2.INTER_LINEAR)
        self._cache[key] = arr
        return arr

    def depth(self) -> Optional[np.ndarray]:
        return self._fit("depth")

    def mask(self) -> Optional[np.ndarray]:
        return self._fit("mask")

    def style(self) -> Optional[np.ndarray]:
        return self._fit("style")

    def map(self, key: str) -> Optional[np.ndarray]:
        """Read any named AI map through the same resize/cache path."""
        return self._fit(key)

    def stamp(self, key: str) -> float:
        return self.store.stamp(key)

    def age(self, key: str) -> float:
        return self.store.age(key)

    def st(self, effect: "Effect") -> dict:
        return self.state.setdefault(effect.uid, {})


# --------------------------------------------------------------------------
# effect base
# --------------------------------------------------------------------------
_UID = [0]


class Effect:
    name = "effect"
    group = "classic"
    needs: Set[str] = set()      # 'depth' | 'mask' | 'style'
    params: List[Param] = []
    blurb = ""

    def __init__(self, values: Optional[dict] = None) -> None:
        _UID[0] += 1
        self.uid = f"{self.__class__.__name__}#{_UID[0]}"
        self.enabled = True
        self.values = {p.key: p.default for p in self.params}
        if values:
            for k, v in values.items():
                if k in self.values:
                    self.values[k] = v

    # convenience accessor
    def p(self, key: str) -> Any:
        return self.values[key]

    def requires(self) -> Set[str]:
        return set(self.needs)

    def apply(self, img: np.ndarray, ctx: FXContext) -> np.ndarray:
        return img

    def to_dict(self) -> dict:
        return {"type": self.__class__.__name__, "enabled": self.enabled, "values": dict(self.values)}


# ==========================================================================
# DEPTH-DRIVEN EFFECTS
# ==========================================================================
class DepthFog(Effect):
    name = "Depth Fog"
    group = "depth"
    needs = {"depth"}
    blurb = "Atmospheric falloff — far pixels dissolve into the fog colour."
    params = [
        Param("density", "Density", "float", 0.8, 0.0, 1.0),
        Param("falloff", "Falloff", "float", 1.6, 0.2, 5.0),
        Param("color", "Fog colour", "color", "#102a3a"),
        Param("glow", "Near glow", "float", 0.0, 0.0, 1.0),
    ]

    def apply(self, img, ctx):
        d = ctx.depth()
        if d is None:
            return img
        far = np.clip(1.0 - d, 0, 1) ** float(self.p("falloff"))
        a = (far * float(self.p("density")))[..., None]
        out = img * (1 - a) + hex_to_bgr(self.p("color")) * a
        g = float(self.p("glow"))
        if g > 0:
            out = screen(out, blur(img, 12) * (d ** 2)[..., None] * g)
        return out


class DepthOfField(Effect):
    name = "Depth of Field"
    group = "depth"
    needs = {"depth"}
    blurb = "Real bokeh from a monocular depth map. Two blur tiers, blended by circle of confusion."
    params = [
        Param("focus", "Focus plane", "float", 0.65, 0.0, 1.0),
        Param("range", "In-focus range", "float", 0.18, 0.01, 0.8),
        Param("strength", "Blur strength", "float", 0.7, 0.0, 1.0),
        Param("bokeh", "Highlight bokeh", "float", 0.3, 0.0, 1.0),
    ]

    def apply(self, img, ctx):
        d = ctx.depth()
        if d is None:
            return img
        s = float(self.p("strength"))
        if s <= 0:
            return img
        coc = np.clip(np.abs(d - float(self.p("focus"))) / float(self.p("range")), 0, 1)
        src = img
        b = float(self.p("bokeh"))
        if b > 0:  # push highlights before blurring so they bloom into discs
            hi = np.clip(src - 0.7, 0, 1) * (b * 3.0)
            src = src + hi
        b1 = blur(src, 5 + 10 * s)
        b2 = blur(src, 14 + 40 * s)
        coc2 = coc * 2.0
        a1 = np.clip(coc2, 0, 1)[..., None]
        a2 = np.clip(coc2 - 1.0, 0, 1)[..., None]
        out = img + (b1 - img) * a1          # lerp with 2 temporaries, not 4
        out += (b2 - out) * a2
        return np.clip(out, 0, 4, out=out)


class DepthParallax(Effect):
    name = "Depth Parallax"
    group = "depth"
    needs = {"depth"}
    blurb = "Fake 3D sway — displaces pixels by depth on an orbiting camera path."
    params = [
        Param("amount", "Amount (px)", "float", 18.0, 0.0, 80.0),
        Param("speed", "Orbit speed", "float", 0.7, 0.0, 4.0),
        Param("tilt", "Vertical ratio", "float", 0.35, 0.0, 1.0),
        Param("centre", "Pivot depth", "float", 0.5, 0.0, 1.0),
    ]

    def apply(self, img, ctx):
        d = ctx.depth()
        if d is None:
            return img
        h, w = img.shape[:2]
        st = ctx.st(self)
        if st.get("shape") != (h, w):
            gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
            st.update(shape=(h, w), gx=gx, gy=gy)
        disp = (d - float(self.p("centre"))).astype(np.float32)
        ang = ctx.t * float(self.p("speed")) * 2 * np.pi * 0.25
        amp = float(self.p("amount"))
        dx = disp * (amp * np.cos(ang))
        dy = disp * (amp * float(self.p("tilt")) * np.sin(ang))
        mx = (st["gx"] + dx).astype(np.float32)
        my = (st["gy"] + dy).astype(np.float32)
        return cv2.remap(img, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


class DepthColorize(Effect):
    name = "Depth Colorize"
    group = "depth"
    needs = {"depth"}
    blurb = "Paints the depth field over the image. Keeps luminance detail if asked."
    params = [
        Param("map", "Colormap", "choice", "turbo", choices=tuple(COLORMAPS)),
        Param("mix", "Mix", "float", 0.75, 0.0, 1.0),
        Param("keep_luma", "Keep detail", "bool", True),
    ]

    def apply(self, img, ctx):
        d = ctx.depth()
        if d is None:
            return img
        cm = colormap_lut(self.p("map"))[to_u8(d)]
        if self.p("keep_luma"):
            cm *= (0.35 + 1.3 * luma(img))[..., None]
        m = float(self.p("mix"))
        return np.clip(img + (cm - img) * m, 0, 1)


class DepthScanline(Effect):
    name = "Depth Scan (lidar)"
    group = "depth"
    needs = {"depth"}
    blurb = "A plane of light sweeping through the depth volume."
    params = [
        Param("speed", "Sweep speed", "float", 0.35, 0.0, 2.0),
        Param("width", "Slice width", "float", 0.05, 0.005, 0.4),
        Param("intensity", "Intensity", "float", 1.0, 0.0, 3.0),
        Param("color", "Slice colour", "color", "#39ffcf"),
        Param("darken", "Darken rest", "float", 0.6, 0.0, 1.0),
        Param("bounce", "Ping-pong", "bool", True),
    ]

    def apply(self, img, ctx):
        d = ctx.depth()
        if d is None:
            return img
        ph = (ctx.t * float(self.p("speed"))) % 2.0
        z = ph if not self.p("bounce") else (ph if ph < 1 else 2 - ph)
        w = float(self.p("width"))
        band = np.exp(-(((d - z) / w) ** 2))
        out = img * (1.0 - float(self.p("darken")))
        out = screen(out, band[..., None] * hex_to_bgr(self.p("color")) * float(self.p("intensity")))
        return np.clip(out, 0, 1)


# ==========================================================================
# SEGMENTATION-DRIVEN EFFECTS
# ==========================================================================
class BackgroundBlur(Effect):
    name = "Background Blur"
    group = "segment"
    needs = {"mask"}
    blurb = "Portrait mode. Feathered matte from any HF semantic-segmentation model."
    params = [
        Param("radius", "Blur radius", "float", 22.0, 1.0, 80.0),
        Param("feather", "Feather", "float", 3.0, 0.0, 20.0),
        Param("dim", "Dim background", "float", 0.15, 0.0, 1.0),
        Param("invert", "Invert (blur subject)", "bool", False),
    ]

    def apply(self, img, ctx):
        m = ctx.mask()
        if m is None:
            return img
        if self.p("invert"):
            m = 1.0 - m
        m = blur(m, float(self.p("feather")))[..., None]
        bg = blur(img, float(self.p("radius"))) * (1.0 - float(self.p("dim")))
        return img * m + bg * (1 - m)


class BackgroundMatte(Effect):
    name = "Background Matte"
    group = "segment"
    needs = {"mask"}
    blurb = "Replace the background with flat colour, a gradient, or the depth field."
    params = [
        Param("mode", "Mode", "choice", "colour", choices=("colour", "gradient", "depth", "black")),
        Param("color", "Colour", "color", "#0b1020"),
        Param("color2", "Colour 2", "color", "#4a1d7a"),
        Param("feather", "Feather", "float", 2.0, 0.0, 20.0),
        Param("invert", "Invert", "bool", False),
    ]

    def apply(self, img, ctx):
        m = ctx.mask()
        if m is None:
            return img
        if self.p("invert"):
            m = 1.0 - m
        h, w = img.shape[:2]
        mode = self.p("mode")
        if mode == "black":
            bg = np.zeros_like(img)
        elif mode == "gradient":
            g = np.linspace(0, 1, h, dtype=np.float32)[:, None, None]
            bg = hex_to_bgr(self.p("color")) * (1 - g) + hex_to_bgr(self.p("color2")) * g
            bg = np.broadcast_to(bg, img.shape).copy()
        elif mode == "depth":
            d = ctx.depth()
            bg = colormap_lut("turbo")[to_u8(d)] if d is not None else np.zeros_like(img)
        else:
            bg = np.broadcast_to(hex_to_bgr(self.p("color")), img.shape).copy()
        mm = blur(m, float(self.p("feather")))[..., None]
        return img * mm + bg * (1 - mm)


class NeonOutline(Effect):
    name = "Neon Outline"
    group = "segment"
    needs = {"mask"}
    blurb = "Rim-light traced from the segmentation contour."
    params = [
        Param("thickness", "Thickness", "int", 3, 1, 20),
        Param("glow", "Glow radius", "float", 12.0, 0.0, 60.0),
        Param("intensity", "Intensity", "float", 1.4, 0.0, 4.0),
        Param("color", "Colour", "color", "#ff2bd6"),
        Param("inner", "Inner fill", "float", 0.0, 0.0, 1.0),
    ]

    def apply(self, img, ctx):
        m = ctx.mask()
        if m is None:
            return img
        k = int(self.p("thickness")) * 2 + 1
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        edge = cv2.dilate(m, ker) - cv2.erode(m, ker)
        glow = blur(edge, float(self.p("glow"))) + edge
        col = hex_to_bgr(self.p("color"))
        out = screen(img, np.clip(glow, 0, 1)[..., None] * col * float(self.p("intensity")))
        f = float(self.p("inner"))
        if f > 0:
            out = screen(out, m[..., None] * col * f * 0.6)
        return np.clip(out, 0, 1)


# ==========================================================================
# DIFFUSION-DRIVEN EFFECT
# ==========================================================================
class AIDream(Effect):
    name = "AI Dream (img2img)"
    group = "diffusion"
    needs = {"style"}
    blurb = "Blends the latest diffusion re-imagining of the frame back over live video."
    params = [
        Param("mix", "Dream mix", "float", 0.75, 0.0, 1.0),
        Param("detail", "Re-add live detail", "float", 0.35, 0.0, 1.0),
        Param("keep_luma", "Lock exposure", "bool", True),
        Param("smear", "Temporal smear", "float", 0.35, 0.0, 0.95),
    ]

    def apply(self, img, ctx):
        s = ctx.style()
        if s is None:
            return img
        st = ctx.st(self)
        prev = st.get("prev")
        k = float(self.p("smear"))
        if prev is not None and prev.shape == s.shape:
            s = prev * k + s * (1 - k)
        st["prev"] = s
        if self.p("keep_luma"):
            s = s * ((luma(img) + 0.05) / (luma(s) + 0.05))[..., None]
        out = img * (1 - float(self.p("mix"))) + s * float(self.p("mix"))
        d = float(self.p("detail"))
        if d > 0:
            out = out + (img - blur(img, 3)) * (d * 2.0)
        return np.clip(out, 0, 1)



# ==========================================================================
# LAYERED GENERATIVE EFFECTS
# ==========================================================================
class AnttisDeepfakeLayer(Effect):
    name = "Antti Layered PhaseRail"
    group = "layer"
    needs = {"mask", "person_style", "background_style"}
    blurb = (
        "Two separate worlds. The person prompt becomes a persistent foreground "
        "carried by the CUDA PhaseRail; the background prompt is a separate stable "
        "plate. The segmentation mask is motion ownership, not a face detector."
    )
    params = [
        Param("rail_size", "Rail resolution", "choice", "128",
              choices=("64", "96", "128", "160", "192")),
        Param("device", "Rail device", "choice", "cuda", choices=("cuda", "cpu")),
        Param("phase_lock", "Phase lock", "float", 0.92, 0.0, 1.0),
        Param("style_strength", "Person style", "float", 1.0, 0.0, 1.25),
        Param("structure", "Live structure", "float", 0.92, 0.0, 1.0),
        Param("detail", "Generated detail", "float", 0.82, 0.0, 1.0),
        Param("nullspace", "Geometry protect", "float", 0.0, 0.0, 1.0),
        Param("max_motion", "Motion radius", "float", 3.5, 0.5, 8.0),
        Param("person_mix", "Person mix", "float", 1.0, 0.0, 1.0),
        Param("background_mix", "Background mix", "float", 1.0, 0.0, 1.0),
        Param("mask_expand", "Mask expand", "int", 2, -12, 24),
        Param("mask_feather", "Mask feather", "float", 4.0, 0.0, 24.0),
        Param("edge_live", "Live edge rescue", "float", 0.15, 0.0, 1.0),
        Param("back_x", "Background X", "float", 0.0, -160.0, 160.0),
        Param("back_y", "Background Y", "float", 0.0, -100.0, 100.0),
        Param("back_zoom", "Background zoom", "float", 1.0, 0.70, 1.60),
        Param("show_mask", "Show ownership", "bool", False),
    ]

    @staticmethod
    def _ownership(mask: np.ndarray, expand: int, feather: float) -> np.ndarray:
        m = np.clip(mask.astype(np.float32), 0.0, 1.0)
        n = abs(int(expand))
        if n:
            k = n * 2 + 1
            ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
            m = cv2.dilate(m, ker) if expand > 0 else cv2.erode(m, ker)
        if feather > 0.1:
            m = blur(m, feather)
        return np.clip(m, 0.0, 1.0).astype(np.float32)

    @staticmethod
    def _letterbox(img: np.ndarray, size: int, fill: float = 0.12):
        h, w = img.shape[:2]
        scale = min(size / max(w, 1), size / max(h, 1))
        nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
        resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
        if img.ndim == 2:
            canvas = np.full((size, size), fill, np.float32)
        else:
            canvas = np.full((size, size, img.shape[2]), fill, np.float32)
        x0, y0 = (size - nw) // 2, (size - nh) // 2
        canvas[y0:y0 + nh, x0:x0 + nw] = resized
        return canvas, (x0, y0, nw, nh, w, h)

    @staticmethod
    def _unletterbox(canvas: np.ndarray, meta) -> np.ndarray:
        x0, y0, nw, nh, w, h = meta
        crop = canvas[y0:y0 + nh, x0:x0 + nw]
        return cv2.resize(crop, (w, h), interpolation=cv2.INTER_LINEAR)

    @staticmethod
    def _move_background(bg: np.ndarray, dx: float, dy: float, zoom: float) -> np.ndarray:
        h, w = bg.shape[:2]
        matrix = cv2.getRotationMatrix2D((w * 0.5, h * 0.5), 0.0, float(zoom))
        matrix[0, 2] += float(dx)
        matrix[1, 2] += float(dy)
        return cv2.warpAffine(
            bg, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT
        )

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

        # Give the phase solver only actor-owned evidence. A neutral exterior
        # preserves the silhouette while preventing the room from owning motion.
        neutral = np.full_like(img, 0.12, dtype=np.float32)
        source_owned = img * m3 + neutral * (1.0 - m3)
        target_owned = person_target * m3 + neutral * (1.0 - m3)

        size = int(self.p("rail_size"))
        device = str(self.p("device"))
        st = ctx.st(self)
        rail = st.get("rail")
        if rail is None or st.get("rail_size") != size or st.get("device") != device:
            try:
                from fx_phase_rail import LayerPhaseRail
                rail = LayerPhaseRail(size=size, device=device)
                st.update(rail=rail, rail_size=size, device=device, person_stamp=-1.0, error="")
            except Exception as exc:
                st["error"] = str(exc)
                # The layer still works as a clean two-world composite without rail.
                front = person_target
                return np.clip(front * m3 + background * (1.0 - m3), 0.0, 1.0)

        source_small, meta = self._letterbox(source_owned, size)
        target_small, _ = self._letterbox(target_owned, size)
        stamp = ctx.stamp("person_style")
        if stamp != st.get("person_stamp"):
            rail.set_target(target_small)
            st["person_stamp"] = stamp

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
            carried = person_target

        person_mix = float(self.p("person_mix"))
        front = img + (carried - img) * person_mix

        # A thin boundary band can borrow real camera colour/detail. It is not
        # used for ownership; it merely removes blue/grey AI matte contamination.
        rescue = float(self.p("edge_live"))
        if rescue > 0 and float(self.p("mask_feather")) > 0:
            inner = blur(m, max(1.0, float(self.p("mask_feather")) * 1.8))
            edge = np.clip(np.abs(m - inner) * 4.0, 0.0, 1.0)[..., None]
            front = front * (1.0 - edge * rescue) + img * (edge * rescue)

        return np.clip(front * m3 + background * (1.0 - m3), 0.0, 1.0)


class ExternalPortraitLayer(Effect):
    name = "LivePortrait Layer (bridge)"
    group = "layer"
    needs = {"mask", "live_portrait", "background_style"}
    blurb = (
        "Composite hook for an external LivePortrait/FasterLivePortrait worker. "
        "It expects that worker to publish a BGR float frame named 'live_portrait'. "
        "The rest of this app — prompt background, mask, effects and recording — stays unchanged."
    )
    params = [
        Param("person_mix", "Portrait mix", "float", 1.0, 0.0, 1.0),
        Param("background_mix", "Background mix", "float", 1.0, 0.0, 1.0),
        Param("mask_expand", "Mask expand", "int", 2, -12, 24),
        Param("mask_feather", "Mask feather", "float", 4.0, 0.0, 24.0),
    ]

    def apply(self, img, ctx):
        m = ctx.mask()
        portrait = ctx.map("live_portrait")
        if m is None or portrait is None:
            return img
        m = AnttisDeepfakeLayer._ownership(
            m, int(self.p("mask_expand")), float(self.p("mask_feather"))
        )[..., None]
        bg = ctx.map("background_style")
        if bg is None:
            bg = img
        bg = img + (bg - img) * float(self.p("background_mix"))
        front = img + (portrait - img) * float(self.p("person_mix"))
        return np.clip(front * m + bg * (1.0 - m), 0.0, 1.0)


# ==========================================================================
# CLASSIC / PROCEDURAL EFFECTS
# ==========================================================================
class GhostTrails(Effect):
    name = "Ghost Trails"
    group = "classic"
    blurb = "Max-decay feedback. Gate it with the segmentation mask for spirit-photography."
    params = [
        Param("decay", "Decay", "float", 0.93, 0.5, 0.995),
        Param("intensity", "Intensity", "float", 0.8, 0.0, 2.0),
        Param("use_mask", "Only subject", "bool", False),
        Param("tint", "Tint", "color", "#7fd4ff"),
        Param("tint_amt", "Tint amount", "float", 0.4, 0.0, 1.0),
    ]

    def requires(self):
        return {"mask"} if self.p("use_mask") else set()

    def apply(self, img, ctx):
        st = ctx.st(self)
        buf = st.get("buf")
        if buf is None or buf.shape != img.shape:
            buf = np.zeros_like(img)
        src = img
        if self.p("use_mask"):
            m = ctx.mask()
            if m is not None:
                src = img * m[..., None]
        buf = np.maximum(buf * float(self.p("decay")), src)
        st["buf"] = buf
        ghost = buf
        ta = float(self.p("tint_amt"))
        if ta > 0:
            ghost = ghost * (1 - ta) + luma(ghost)[..., None] * hex_to_bgr(self.p("tint")) * ta * 2.0
        return np.clip(screen(img, ghost * float(self.p("intensity"))), 0, 1)


class FeedbackTunnel(Effect):
    name = "Feedback Tunnel"
    group = "classic"
    blurb = "Frame fed back through a zoom/rotate warp — the classic video-feedback attractor."
    params = [
        Param("zoom", "Zoom", "float", 1.02, 0.94, 1.10),
        Param("rotate", "Rotate (deg/frame)", "float", 0.4, -6.0, 6.0),
        Param("persist", "Persistence", "float", 0.86, 0.0, 0.99),
        Param("mix", "Mix", "float", 0.6, 0.0, 1.0),
        Param("hue_roll", "Channel roll", "float", 0.06, 0.0, 0.5),
        Param("drift", "Drift (px)", "float", 0.0, -8.0, 8.0),
    ]

    def apply(self, img, ctx):
        st = ctx.st(self)
        buf = st.get("buf")
        if buf is None or buf.shape != img.shape:
            buf = img.copy()
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), float(self.p("rotate")), float(self.p("zoom")))
        M[0, 2] += float(self.p("drift"))
        warped = cv2.warpAffine(buf, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        r = float(self.p("hue_roll"))
        if r > 0:
            rolled = warped[..., [1, 2, 0]]
            warped = warped * (1 - r) + rolled * r
        k = float(self.p("persist"))
        buf = warped * k + img * (1 - k)
        st["buf"] = buf
        m = float(self.p("mix"))
        return np.clip(img * (1 - m) + buf * m, 0, 1)


class SlitScan(Effect):
    name = "Slit Scan"
    group = "classic"
    blurb = "Each pixel sampled from a different moment. In depth mode, distance becomes time."
    params = [
        Param("mode", "Mode", "choice", "rows", choices=("rows", "cols", "depth", "radial")),
        Param("frames", "Time depth", "int", 24, 2, 90),
        Param("stretch", "Range", "float", 1.0, 0.05, 1.0),
        Param("invert", "Invert", "bool", False),
    ]

    def requires(self):
        return {"depth"} if self.p("mode") == "depth" else set()

    def apply(self, img, ctx):
        st = ctx.st(self)
        n = int(self.p("frames"))
        h, w = img.shape[:2]
        buf = st.get("buf")
        if buf is None or buf.shape[0] != n or buf.shape[1:3] != (h, w):
            buf = np.repeat(to_u8(img)[None], n, axis=0)
            st["ptr"] = 0
        ptr = st.get("ptr", 0)
        buf[ptr] = to_u8(img)
        st["buf"], st["ptr"] = buf, (ptr + 1) % n

        mode = self.p("mode")
        if mode == "rows":
            k = np.linspace(0, 1, h, dtype=np.float32)[:, None]
            k = np.broadcast_to(k, (h, w))
        elif mode == "cols":
            k = np.linspace(0, 1, w, dtype=np.float32)[None, :]
            k = np.broadcast_to(k, (h, w))
        elif mode == "radial":
            yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
            k = np.sqrt(((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2) / 1.414
        else:
            d = ctx.depth()
            if d is None:
                return img
            k = d.astype(np.float32)
        if self.p("invert"):
            k = 1.0 - k
        k = np.clip(k * float(self.p("stretch")), 0, 1)
        back = (k * (n - 1)).astype(np.int32)
        idx = ((ptr - back) % n).ravel()
        if st.get("ar_n") != h * w:
            st["ar"], st["ar_n"] = np.arange(h * w), h * w
        out = buf.reshape(n, h * w, 3)[idx, st["ar"]].reshape(h, w, 3)
        return out.astype(np.float32) * np.float32(1.0 / 255.0)


class ChromaticAberration(Effect):
    name = "Chromatic Aberration"
    group = "classic"
    blurb = "Radial RGB separation; can be driven by depth so only far things fringe."
    params = [
        Param("amount", "Amount", "float", 4.0, 0.0, 30.0),
        Param("by_depth", "Scale by depth", "bool", False),
        Param("edge_bias", "Edge bias", "float", 1.0, 0.0, 3.0),
    ]

    def requires(self):
        return {"depth"} if self.p("by_depth") else set()

    def apply(self, img, ctx):
        a = float(self.p("amount"))
        if a <= 0:
            return img
        h, w = img.shape[:2]
        st = ctx.st(self)
        if st.get("shape") != (h, w):
            gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
            nx = (gx - w / 2) / (w / 2)
            ny = (gy - h / 2) / (h / 2)
            st.update(shape=(h, w), gx=gx, gy=gy, nx=nx, ny=ny)
        gx, gy, nx, ny = st["gx"], st["gy"], st["nx"], st["ny"]
        r2 = (nx * nx + ny * ny) ** (0.5 * float(self.p("edge_bias")))
        scale = r2 * a
        if self.p("by_depth"):
            d = ctx.depth()
            if d is not None:
                scale = scale * (1.0 - d)
        out = np.empty_like(img)
        out[..., 1] = img[..., 1]
        for ch, s in ((0, -1.0), (2, 1.0)):
            mx = (gx + nx * scale * s).astype(np.float32)
            my = (gy + ny * scale * s).astype(np.float32)
            out[..., ch] = cv2.remap(img[..., ch], mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        return out


class Bloom(Effect):
    name = "Bloom"
    group = "classic"
    blurb = "Threshold, blur wide, screen back. The glue that makes everything look filmic."
    params = [
        Param("threshold", "Threshold", "float", 0.65, 0.0, 1.0),
        Param("radius", "Radius", "float", 28.0, 1.0, 120.0),
        Param("intensity", "Intensity", "float", 0.8, 0.0, 3.0),
    ]

    def apply(self, img, ctx):
        hi = np.clip(img - float(self.p("threshold")), 0, 1) / max(EPS, 1 - float(self.p("threshold")))
        g = blur(hi, float(self.p("radius"))) * float(self.p("intensity"))
        return np.clip(screen(img, g), 0, 1)


class EdgeNeon(Effect):
    name = "Edge Neon"
    group = "classic"
    blurb = "Sobel magnitude, colour-mapped by direction, screened over a darkened plate."
    params = [
        Param("gain", "Edge gain", "float", 2.2, 0.1, 10.0),
        Param("thickness", "Thickness", "float", 1.0, 0.0, 8.0),
        Param("darken", "Darken source", "float", 0.55, 0.0, 1.0),
        Param("angle_hue", "Hue from angle", "bool", True),
        Param("color", "Colour", "color", "#67f9ff"),
    ]

    def apply(self, img, ctx):
        g = luma(img)
        gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
        mag = np.clip(np.sqrt(gx * gx + gy * gy) * float(self.p("gain")), 0, 1)
        mag = blur(mag, float(self.p("thickness")))
        if self.p("angle_hue"):
            ang = ((np.arctan2(gy, gx) + np.pi) / (2 * np.pi) * 179).astype(np.uint8)
            hsv = np.stack([ang, np.full_like(ang, 235), to_u8(mag)], -1)
            neon = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR).astype(np.float32) / 255.0
        else:
            neon = mag[..., None] * hex_to_bgr(self.p("color"))
        return np.clip(screen(img * (1 - float(self.p("darken"))), neon), 0, 1)


class PixelSort(Effect):
    name = "Pixel Sort"
    group = "classic"
    blurb = "Rows above a brightness threshold get sorted by luminance. Glitch-art staple."
    params = [
        Param("threshold", "Row threshold", "float", 0.35, 0.0, 1.0),
        Param("vertical", "Vertical", "bool", False),
        Param("mix", "Mix", "float", 1.0, 0.0, 1.0),
        Param("reverse", "Reverse", "bool", False),
    ]

    def apply(self, img, ctx):
        work = np.transpose(img, (1, 0, 2)) if self.p("vertical") else img
        lum = work.mean(2)
        order = np.argsort(lum, axis=1)
        if self.p("reverse"):
            order = order[:, ::-1]
        srt = np.take_along_axis(work, order[:, :, None], axis=1)
        rows = (lum.mean(1) > float(self.p("threshold")))[:, None, None]
        m = float(self.p("mix"))
        out = np.where(rows, work * (1 - m) + srt * m, work)
        return np.transpose(out, (1, 0, 2)) if self.p("vertical") else out


class PosterizeDither(Effect):
    name = "Posterize + Dither"
    group = "classic"
    blurb = "Bayer-ordered dithering into N levels. Good over depth colormaps."
    params = [
        Param("levels", "Levels", "int", 6, 2, 32),
        Param("dither", "Dither", "float", 0.8, 0.0, 2.0),
        Param("scale", "Matrix scale", "int", 1, 1, 6),
    ]

    _BAYER = np.array([[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]], np.float32) / 16.0 - 0.5

    def apply(self, img, ctx):
        h, w = img.shape[:2]
        s = int(self.p("scale"))
        b = np.kron(self._BAYER, np.ones((s, s), np.float32))
        tile = np.tile(b, (h // b.shape[0] + 1, w // b.shape[1] + 1))[:h, :w]
        n = int(self.p("levels"))
        x = img + tile[..., None] * (float(self.p("dither")) / n)
        return np.clip(np.round(x * (n - 1)) / (n - 1), 0, 1)


class Kaleidoscope(Effect):
    name = "Kaleidoscope"
    group = "classic"
    blurb = "Angular mirror fold around a rotating centre."
    params = [
        Param("segments", "Segments", "int", 6, 2, 24),
        Param("spin", "Spin speed", "float", 0.15, -2.0, 2.0),
        Param("zoom", "Zoom", "float", 1.0, 0.3, 3.0),
        Param("mix", "Mix", "float", 1.0, 0.0, 1.0),
    ]

    def apply(self, img, ctx):
        h, w = img.shape[:2]
        st = ctx.st(self)
        if st.get("shape") != (h, w):
            yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
            st.update(shape=(h, w),
                      r=np.sqrt((xx - w / 2) ** 2 + (yy - h / 2) ** 2),
                      th=np.arctan2(yy - h / 2, xx - w / 2))
        n = max(2, int(self.p("segments")))
        wedge = 2 * np.pi / n
        th = st["th"] + ctx.t * float(self.p("spin"))
        f = np.mod(th, wedge)
        f = np.minimum(f, wedge - f)          # mirror inside the wedge
        r = st["r"] / float(self.p("zoom"))
        mx = (w / 2 + r * np.cos(f)).astype(np.float32)
        my = (h / 2 + r * np.sin(f)).astype(np.float32)
        k = cv2.remap(img, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        m = float(self.p("mix"))
        return img * (1 - m) + k * m


class Halftone(Effect):
    name = "Halftone"
    group = "classic"
    blurb = "Print-style dot screen; dot size tracks luminance."
    params = [
        Param("size", "Dot pitch", "float", 6.0, 2.0, 30.0),
        Param("angle", "Screen angle", "float", 25.0, 0.0, 90.0),
        Param("sharp", "Hardness", "float", 0.35, 0.02, 1.0),
        Param("mix", "Mix", "float", 1.0, 0.0, 1.0),
        Param("colored", "Keep colour", "bool", True),
    ]

    def apply(self, img, ctx):
        h, w = img.shape[:2]
        st = ctx.st(self)
        key = (h, w, round(float(self.p("size")), 2), round(float(self.p("angle")), 1))
        if st.get("key") != key:
            yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
            a = np.deg2rad(float(self.p("angle")))
            f = np.float32(np.pi / float(self.p("size")))
            ca, sa = np.float32(np.cos(a)), np.float32(np.sin(a))
            u = (xx * ca - yy * sa) * f
            v = (xx * sa + yy * ca) * f
            st.update(key=key, grid=(np.sin(u) * np.sin(v)).astype(np.float32))
        g = st["grid"]
        l = luma(img)
        sh = float(self.p("sharp"))
        dots = smoothstep(-sh, sh, l * 2.0 - 1.0 - g * 0.9)[..., None]
        out = dots * (img / (l[..., None] + 0.15)) if self.p("colored") else np.repeat(dots, 3, 2)
        m = float(self.p("mix"))
        return np.clip(img + (out - img) * m, 0, 1)


class ColorGrade(Effect):
    name = "Colour Grade"
    group = "classic"
    blurb = "Finishing pass: contrast, saturation, temperature, lift, vignette, grain."
    params = [
        Param("exposure", "Exposure", "float", 0.0, -1.5, 1.5),
        Param("contrast", "Contrast", "float", 1.1, 0.2, 2.5),
        Param("saturation", "Saturation", "float", 1.15, 0.0, 3.0),
        Param("temp", "Temperature", "float", 0.0, -0.5, 0.5),
        Param("lift", "Shadow lift", "float", 0.0, -0.2, 0.3),
        Param("vignette", "Vignette", "float", 0.3, 0.0, 1.5),
        Param("grain", "Grain", "float", 0.02, 0.0, 0.2),
    ]

    def apply(self, img, ctx):
        out = img * np.float32(2.0 ** float(self.p("exposure")))
        c = np.float32(self.p("contrast"))
        out -= 0.5
        out *= c
        out += np.float32(0.5 + float(self.p("lift")))
        l = luma(out)[..., None]
        sat = np.float32(self.p("saturation"))
        out = l + (out - l) * sat
        t = float(self.p("temp"))
        if abs(t) > 1e-3:
            out = out * np.array([1 - t, 1.0, 1 + t], np.float32)
        v = float(self.p("vignette"))
        if v > 0:
            h, w = img.shape[:2]
            st = ctx.st(self)
            if st.get("shape") != (h, w):
                yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
                r = np.sqrt(((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2)
                st.update(shape=(h, w), r=r)
            out = out * (1 - np.clip(st["r"] - 0.45, 0, 2)[..., None] * v)
        g = float(self.p("grain"))
        if g > 0:
            out += (np.random.random_sample(img.shape[:2]).astype(np.float32) - 0.5)[..., None] * g
        return np.clip(out, 0, 1, out=out)


# --------------------------------------------------------------------------
# registry + presets
# --------------------------------------------------------------------------
EFFECT_CLASSES: List[type] = [
    DepthFog, DepthOfField, DepthParallax, DepthColorize, DepthScanline,
    BackgroundBlur, BackgroundMatte, NeonOutline,
    AIDream,
    AnttisDeepfakeLayer, ExternalPortraitLayer,
    GhostTrails, FeedbackTunnel, SlitScan, ChromaticAberration, Bloom,
    EdgeNeon, PixelSort, PosterizeDither, Kaleidoscope, Halftone, ColorGrade,
]
EFFECTS_BY_NAME = {c.__name__: c for c in EFFECT_CLASSES}

GROUP_LABEL = {
    "depth": "Depth (AI)",
    "segment": "Segmentation (AI)",
    "diffusion": "Diffusion (AI)",
    "layer": "AI Layers",
    "classic": "Procedural",
}

PRESETS: Dict[str, List[dict]] = {
    "Antti Layered Dream": [
        {"type": "AnttisDeepfakeLayer", "values": {
            "phase_lock": 0.92, "style_strength": 1.0, "structure": 0.92,
            "detail": 0.82, "mask_expand": 2, "mask_feather": 4.0
        }},
        {"type": "Bloom", "values": {"threshold": 0.72, "intensity": 0.35}},
        {"type": "ColorGrade", "values": {"contrast": 1.06, "saturation": 1.05}},
    ],
    "Hologram": [
        {"type": "DepthColorize", "values": {"map": "ocean", "mix": 0.55}},
        {"type": "NeonOutline", "values": {"color": "#39ffcf", "intensity": 1.6}},
        {"type": "ChromaticAberration", "values": {"amount": 6.0, "by_depth": True}},
        {"type": "Bloom", "values": {"threshold": 0.5, "intensity": 1.1}},
        {"type": "ColorGrade", "values": {"contrast": 1.2, "vignette": 0.6, "grain": 0.04}},
    ],
    "Lidar Sweep": [
        {"type": "DepthFog", "values": {"density": 0.95, "color": "#04060c", "falloff": 1.2}},
        {"type": "DepthScanline", "values": {"speed": 0.3, "intensity": 1.6}},
        {"type": "EdgeNeon", "values": {"darken": 0.8, "angle_hue": False, "color": "#39ffcf"}},
        {"type": "Bloom", "values": {"radius": 40, "intensity": 1.0}},
    ],
    "Portrait": [
        {"type": "DepthOfField", "values": {"focus": 0.8, "range": 0.16, "strength": 0.75}},
        {"type": "BackgroundBlur", "values": {"radius": 18, "dim": 0.2}},
        {"type": "ColorGrade", "values": {"saturation": 1.1, "contrast": 1.08, "vignette": 0.35}},
    ],
    "Spirit Photography": [
        {"type": "GhostTrails", "values": {"use_mask": True, "decay": 0.96, "intensity": 1.0}},
        {"type": "DepthFog", "values": {"density": 0.7, "color": "#0d1b2a"}},
        {"type": "Bloom", "values": {"intensity": 1.2}},
        {"type": "ColorGrade", "values": {"saturation": 0.7, "grain": 0.06, "vignette": 0.7}},
    ],
    "Time Sculpture": [
        {"type": "SlitScan", "values": {"mode": "depth", "frames": 40}},
        {"type": "ChromaticAberration", "values": {"amount": 5.0}},
        {"type": "ColorGrade", "values": {"contrast": 1.15, "saturation": 1.3}},
    ],
    "Dream Machine": [
        {"type": "AIDream", "values": {"mix": 0.8, "smear": 0.4}},
        {"type": "FeedbackTunnel", "values": {"mix": 0.35, "persist": 0.8, "zoom": 1.01}},
        {"type": "Bloom", "values": {"intensity": 0.7}},
        {"type": "ColorGrade", "values": {"saturation": 1.25, "vignette": 0.4}},
    ],
    "Glitch Print": [
        {"type": "PixelSort", "values": {"threshold": 0.4}},
        {"type": "Halftone", "values": {"size": 5, "mix": 0.7}},
        {"type": "PosterizeDither", "values": {"levels": 5}},
        {"type": "ColorGrade", "values": {"contrast": 1.2, "grain": 0.05}},
    ],
}


def build_chain(spec: List[dict]) -> List[Effect]:
    chain: List[Effect] = []
    for item in spec:
        cls = EFFECTS_BY_NAME.get(item.get("type", ""))
        if cls is None:
            continue
        fx = cls(item.get("values"))
        fx.enabled = bool(item.get("enabled", True))
        chain.append(fx)
    return chain


def chain_requirements(chain: List[Effect]) -> Set[str]:
    need: Set[str] = set()
    for fx in chain:
        if fx.enabled:
            need |= fx.requires()
    return need
