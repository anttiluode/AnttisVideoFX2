#!/usr/bin/env python3
"""
AI Video FX — real-time AI video filter studio.

    python ai_video_fx.py                 # webcam 0
    python ai_video_fx.py --source clip.mp4
    python ai_video_fx.py --width 480 --preset "Lidar Sweep"

Threading model
---------------
    capture thread  ──►  latest frame slot
                              │
    processing thread ────────┴──► effect chain ──► display slot ──► Tk (after loop)
                              └──► AI worker (own framerate) ──► MapStore

The AI worker publishes depth / mask / stylised maps whenever it finishes;
the effect chain always uses the most recent ones. So a 4 fps diffusion model
and a 30 fps preview coexist without either blocking the other.
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import threading
import time
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import tkinter as tk
from tkinter import colorchooser, filedialog, messagebox, ttk
from PIL import Image, ImageTk

import fx_ai
from fx_ai import AIConfig, AIWorker, DEPTH_MODELS, SEG_MODELS, STYLE_MODELS, deps_report
from fx_core import (EFFECT_CLASSES, EFFECTS_BY_NAME, GROUP_LABEL, PRESETS, Effect,
                     FXContext, MapStore, Param, build_chain, chain_requirements, to_f32, to_u8)

APP = "AI Video FX"
OUT_DIR = Path("captures")

# dark palette
BG = "#12151c"
BG2 = "#1a1f2a"
BG3 = "#232a38"
FG = "#d7dde8"
MUT = "#8b95a7"
ACC = "#39d0c8"
WARN = "#ffb454"


# ==========================================================================
# capture
# ==========================================================================
class Camera(threading.Thread):
    daemon = True

    def __init__(self, source, req_w: int = 1280, req_h: int = 720) -> None:
        super().__init__(name="camera")
        self.source = source
        self.req = (req_w, req_h)
        self.cap: Optional[cv2.VideoCapture] = None
        self.flip = True
        self._frame: Optional[np.ndarray] = None
        self._count = 0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self.error: Optional[str] = None
        self.is_file = isinstance(source, str)
        self.fps_hint = 30.0

    def open(self) -> bool:
        if isinstance(self.source, int) and os.name == "nt":
            cap = cv2.VideoCapture(self.source, cv2.CAP_DSHOW)
        else:
            cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            self.error = f"cannot open source {self.source!r}"
            return False
        if not self.is_file:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.req[0])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.req[1])
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        f = cap.get(cv2.CAP_PROP_FPS)
        self.fps_hint = f if 1 < f < 240 else 30.0
        self.cap = cap
        return True

    def run(self) -> None:
        if not self.open():
            return
        period = 1.0 / self.fps_hint if self.is_file else 0.0
        while not self._stop.is_set():
            t0 = time.time()
            ok, frame = self.cap.read()
            if not ok:
                if self.is_file:                       # loop the file
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                self.error = "capture read failed"
                break
            if self.flip and not self.is_file:
                frame = cv2.flip(frame, 1)
            with self._lock:
                self._frame = frame
                self._count += 1
            if period:
                time.sleep(max(0.0, period - (time.time() - t0)))
        if self.cap:
            self.cap.release()

    def latest(self):
        """Returns (frame, sequence number) so the processor can skip duplicates."""
        with self._lock:
            return self._frame, self._count

    def stop(self) -> None:
        self._stop.set()


class Runtime:
    """Plain mirror of the Tk control variables.

    Tcl is not thread-safe: reading a tk.Variable from the processing thread
    can wedge the interpreter. The GUI writes here via trace callbacks (main
    thread only) and the worker threads read these plain attributes.
    """
    __slots__ = ("process_width", "bypass", "split_view", "show_hud")

    def __init__(self, width: int) -> None:
        self.process_width = width
        self.bypass = False
        self.split_view = False
        self.show_hud = True


# ==========================================================================
# processing
# ==========================================================================
class Processor(threading.Thread):
    daemon = True

    def __init__(self, app: "App") -> None:
        super().__init__(name="processor")
        self.app = app
        self._stop = threading.Event()
        self.out: Optional[np.ndarray] = None
        self._lock = threading.Lock()
        self.fps = 0.0
        self.ms = 0.0
        self.t0 = time.time()
        self.idx = 0
        self.writer: Optional[cv2.VideoWriter] = None
        self.writer_size = None
        self.record_path: Optional[Path] = None
        self.snapshot_req = False
        self.last_snapshot: Optional[Path] = None

    def stop(self) -> None:
        self._stop.set()

    def display(self) -> Optional[np.ndarray]:
        with self._lock:
            return self.out

    def run(self) -> None:
        last_seq = -1
        while not self._stop.is_set():
            cam = self.app.camera
            frame, seq = cam.latest() if cam else (None, -1)
            if frame is None:
                time.sleep(0.02)
                continue
            if seq == last_seq:
                time.sleep(0.003)
                continue
            last_seq = seq
            t_start = time.time()

            pw = self.app.rt.process_width
            h, w = frame.shape[:2]
            if w != pw:
                frame = cv2.resize(frame, (pw, max(2, int(h * pw / w))),
                                   interpolation=cv2.INTER_AREA if w > pw else cv2.INTER_LINEAR)

            chain = self.app.get_chain()
            needs = chain_requirements(chain)
            if self.app.ai_worker and needs:
                self.app.ai_worker.submit(frame, needs)

            ctx = FXContext(self.app.store, time.time() - self.t0, self.idx, frame.shape[:2])
            img = to_f32(frame)
            if not self.app.rt.bypass:
                for fx in chain:
                    if not fx.enabled:
                        continue
                    try:
                        img = fx.apply(img, ctx)
                    except Exception as exc:
                        self.app.log(f"{fx.name} failed: {exc}")
                        fx.enabled = False
            out = to_u8(img)
            self.idx += 1

            if self.app.rt.split_view:
                half = out.shape[1] // 2
                out[:, :half] = frame[:, :half]
                cv2.line(out, (half, 0), (half, out.shape[0]), (255, 255, 255), 1)

            if self.snapshot_req:
                self.snapshot_req = False
                OUT_DIR.mkdir(exist_ok=True)
                p = OUT_DIR / f"shot_{time.strftime('%Y%m%d-%H%M%S')}.png"
                cv2.imwrite(str(p), out)
                self.last_snapshot = p
                self.app.log(f"saved {p}")

            if self.writer is not None:
                if self.writer_size != (out.shape[1], out.shape[0]):
                    self.app.log("frame size changed — recording stopped")
                    self.stop_record()
                else:
                    self.writer.write(out)

            if self.app.rt.show_hud:
                self._hud(out, chain)

            with self._lock:
                self.out = out
            dt = time.time() - t_start
            self.ms = 0.85 * self.ms + 0.15 * dt * 1000
            self.fps = 0.9 * self.fps + 0.1 * (1.0 / max(1e-3, dt))

    def _hud(self, out: np.ndarray, chain: List[Effect]) -> None:
        on = sum(1 for f in chain if f.enabled)
        txt = f"{self.fps:4.1f} fps | {self.ms:4.1f} ms | {on} fx"
        w = self.app.ai_worker
        if w:
            r = w.rates
            live = [f"{k} {r.get(k, 0.0):.1f}" for k in
                    ("depth", "mask", "style", "person_style", "background_style", "live_portrait")
                    if r.get(k, 0.0) > 0.05]
            if live:
                txt += " | AI " + " ".join(live)
        if self.writer is not None:
            txt += "  ● REC"
        cv2.putText(out, txt, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(out, txt, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    # -- recording ---------------------------------------------------------
    def start_record(self, fps: float) -> Optional[Path]:
        frame = self.display()
        if frame is None:
            return None
        OUT_DIR.mkdir(exist_ok=True)
        p = OUT_DIR / f"fx_{time.strftime('%Y%m%d-%H%M%S')}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        h, w = frame.shape[:2]
        self.writer_size = (w, h)
        self.writer = cv2.VideoWriter(str(p), fourcc, max(5.0, min(60.0, fps)), (w, h))
        self.record_path = p
        return p

    def stop_record(self) -> None:
        if self.writer is not None:
            self.writer.release()
            self.writer = None


# ==========================================================================
# small Tk helpers
# ==========================================================================
class ScrollFrame(ttk.Frame):
    def __init__(self, parent, height=260, **kw):
        super().__init__(parent, **kw)
        self.canvas = tk.Canvas(self, bg=BG2, highlightthickness=0, height=height)
        self.vsb = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.inner = ttk.Frame(self.canvas)
        self.inner.bind("<Configure>",
                        lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self._win = self.canvas.create_window((0, 0), window=self.inner, anchor="nw")
        self.canvas.bind("<Configure>", lambda e: self.canvas.itemconfig(self._win, width=e.width))
        self.canvas.configure(yscrollcommand=self.vsb.set)
        self.canvas.pack(side="left", fill="both", expand=True)
        self.vsb.pack(side="right", fill="y")
        for w in (self.canvas, self.inner):
            w.bind("<MouseWheel>", self._wheel)
            w.bind("<Button-4>", self._wheel)
            w.bind("<Button-5>", self._wheel)

    def _wheel(self, ev):
        d = -1 if getattr(ev, "num", 0) == 4 or getattr(ev, "delta", 0) > 0 else 1
        self.canvas.yview_scroll(d, "units")

    def clear(self):
        for c in self.inner.winfo_children():
            c.destroy()


# ==========================================================================
# application
# ==========================================================================
class App(tk.Tk):
    def __init__(self, args) -> None:
        super().__init__()
        self.title(APP)
        self.geometry("1500x900")
        self.configure(bg=BG)
        self._style()

        self.store = MapStore()
        self.cfg = AIConfig()
        self.chain: List[Effect] = []
        self.chain_lock = threading.Lock()
        self.selected: Optional[Effect] = None
        self.camera: Optional[Camera] = None
        self.ai_worker: Optional[AIWorker] = None
        self._logq: "queue.Queue[str]" = queue.Queue()

        self.process_width = tk.IntVar(value=args.width)
        self.bypass = tk.BooleanVar(value=False)
        self.split_view = tk.BooleanVar(value=False)
        self.show_hud = tk.BooleanVar(value=True)
        self.flip_var = tk.BooleanVar(value=True)
        self.source_var = tk.StringVar(value=str(args.source))
        self.rt = Runtime(args.width)
        for var, attr, cast in ((self.process_width, "process_width", int),
                                (self.bypass, "bypass", bool),
                                (self.split_view, "split_view", bool),
                                (self.show_hud, "show_hud", bool)):
            def mirror(*_, v=var, a=attr, c=cast):
                setattr(self.rt, a, c(v.get()))
            var.trace_add("write", mirror)
            mirror()

        self._build()
        self.processor = Processor(self)
        self.processor.start()
        self.start_ai()

        if args.preset in PRESETS:
            self.apply_preset(args.preset)
        else:
            self.apply_preset("Portrait")

        self.log(f"{APP} ready — {deps_report()}")
        if not fx_ai.TRANSFORMERS_OK:
            self.log("transformers/torch not found: AI effects will pass through. "
                     "pip install torch transformers")
        self.start_source()
        self.protocol("WM_DELETE_WINDOW", self.on_close)

        # Global shortcuts must not fire while the user is typing a prompt or
        # editing another text field.  The old direct bindings meant that a
        # space inside "minecraft man" toggled Bypass, while the letters s
        # and r could take a snapshot or start/stop recording.
        self.bind("<space>", self._shortcut_bypass)
        self.bind("<s>", self._shortcut_snapshot)
        self.bind("<r>", self._shortcut_record)

        self._closing = False
        self.after(16, self._tick_view)
        self.after(250, self._tick_status)

    @staticmethod
    def _editing_text(widget) -> bool:
        """True when keyboard input belongs to a text-like GUI control.

        Toplevel key bindings also receive events originating in child Text,
        Entry and Combobox widgets.  Ignore global hotkeys in those controls so
        prompt editing cannot accidentally bypass the entire effect chain.
        """
        try:
            cls = str(widget.winfo_class()).lower()
        except Exception:
            return False
        return cls in {
            "text", "entry", "tentry", "spinbox", "tspinbox",
            "combobox", "tcombobox",
        }

    def _shortcut_bypass(self, event):
        if self._editing_text(event.widget):
            return None
        self.bypass.set(not self.bypass.get())
        return "break"

    def _shortcut_snapshot(self, event):
        if self._editing_text(event.widget):
            return None
        self.snapshot()
        return "break"

    def _shortcut_record(self, event):
        if self._editing_text(event.widget):
            return None
        self.toggle_record()
        return "break"

    # ---------------------------------------------------------------- style
    def _style(self) -> None:
        s = ttk.Style(self)
        s.theme_use("clam")
        s.configure(".", background=BG, foreground=FG, fieldbackground=BG3,
                    bordercolor=BG3, lightcolor=BG3, darkcolor=BG, focuscolor=ACC)
        s.configure("TFrame", background=BG)
        s.configure("Card.TFrame", background=BG2)
        s.configure("TLabel", background=BG, foreground=FG)
        s.configure("Card.TLabel", background=BG2, foreground=FG)
        s.configure("Mut.TLabel", background=BG2, foreground=MUT)
        s.configure("Head.TLabel", background=BG2, foreground=ACC, font=("Segoe UI", 10, "bold"))
        s.configure("TButton", background=BG3, foreground=FG, borderwidth=0, padding=4)
        s.map("TButton", background=[("active", ACC)], foreground=[("active", BG)])
        s.configure("Acc.TButton", background=ACC, foreground=BG)
        s.configure("TCheckbutton", background=BG2, foreground=FG)
        s.map("TCheckbutton", background=[("active", BG2)])
        s.configure("TCombobox", fieldbackground=BG3, background=BG3, foreground=FG, arrowcolor=FG)
        s.configure("TEntry", fieldbackground=BG3, foreground=FG, insertcolor=FG)
        s.configure("TScale", background=BG2, troughcolor=BG3)
        s.configure("TNotebook", background=BG, borderwidth=0)
        s.configure("TNotebook.Tab", background=BG2, foreground=MUT, padding=(12, 6))
        s.map("TNotebook.Tab", background=[("selected", BG3)], foreground=[("selected", ACC)])

    # ---------------------------------------------------------------- build
    def _build(self) -> None:
        root = ttk.Frame(self, padding=8)
        root.pack(fill="both", expand=True)
        root.columnconfigure(1, weight=1)
        root.rowconfigure(0, weight=1)

        left = ttk.Frame(root, width=330)
        left.grid(row=0, column=0, sticky="ns", padx=(0, 8))
        left.grid_propagate(False)
        centre = ttk.Frame(root)
        centre.grid(row=0, column=1, sticky="nsew")
        right = ttk.Frame(root, width=360)
        right.grid(row=0, column=2, sticky="ns", padx=(8, 0))
        right.grid_propagate(False)

        self._build_left(left)
        self._build_centre(centre)
        self._build_right(right)

    # -- left column -------------------------------------------------------
    def _card(self, parent, title: str) -> ttk.Frame:
        outer = ttk.Frame(parent, style="Card.TFrame", padding=8)
        outer.pack(fill="x", pady=(0, 8))
        ttk.Label(outer, text=title, style="Head.TLabel").pack(anchor="w", pady=(0, 6))
        return outer

    def _build_left(self, p) -> None:
        c = self._card(p, "SOURCE")
        row = ttk.Frame(c, style="Card.TFrame")
        row.pack(fill="x")
        ttk.Entry(row, textvariable=self.source_var, width=14).pack(side="left", fill="x", expand=True)
        ttk.Button(row, text="File…", width=6, command=self.pick_file).pack(side="left", padx=4)
        ttk.Button(row, text="Open", width=6, style="Acc.TButton",
                   command=self.start_source).pack(side="left")
        ttk.Label(c, text="webcam index (0,1,…) or video path", style="Mut.TLabel").pack(anchor="w")

        row = ttk.Frame(c, style="Card.TFrame")
        row.pack(fill="x", pady=(6, 0))
        ttk.Label(row, text="Process width", style="Card.TLabel").pack(side="left")
        cb = ttk.Combobox(row, width=6, state="readonly",
                          values=("320", "400", "480", "640", "800", "960", "1280"))
        cb.set(str(self.process_width.get()))
        cb.bind("<<ComboboxSelected>>", lambda e: self.process_width.set(int(cb.get())))
        cb.pack(side="right")

        opts = ttk.Frame(c, style="Card.TFrame")
        opts.pack(fill="x", pady=(4, 0))
        ttk.Checkbutton(opts, text="Mirror", variable=self.flip_var,
                        command=self._apply_flip).pack(side="left")
        ttk.Checkbutton(opts, text="HUD", variable=self.show_hud).pack(side="left", padx=6)
        ttk.Checkbutton(opts, text="Split A/B", variable=self.split_view).pack(side="left")

        c = self._card(p, "OUTPUT")
        row = ttk.Frame(c, style="Card.TFrame")
        row.pack(fill="x")
        self.rec_btn = ttk.Button(row, text="● Record  (r)", command=self.toggle_record)
        self.rec_btn.pack(side="left", fill="x", expand=True)
        ttk.Button(row, text="Snapshot (s)", command=self.snapshot).pack(side="left", padx=(6, 0))
        ttk.Label(c, text=f"files land in ./{OUT_DIR}/", style="Mut.TLabel").pack(anchor="w", pady=(4, 0))

        c = self._card(p, "AI ENGINE")
        grid = ttk.Frame(c, style="Card.TFrame")
        grid.pack(fill="x")
        grid.columnconfigure(1, weight=1)

        ttk.Label(grid, text="Device", style="Card.TLabel").grid(row=0, column=0, sticky="w")
        dev = ttk.Combobox(grid, width=10, state="readonly", values=("cpu", "cuda", "mps"))
        dev.set(self.cfg.device)
        dev.bind("<<ComboboxSelected>>", lambda e: setattr(self.cfg, "device", dev.get()))
        dev.grid(row=0, column=1, sticky="ew", pady=2)

        ttk.Label(grid, text="AI fps", style="Card.TLabel").grid(row=1, column=0, sticky="w")
        self.aifps_lbl = ttk.Label(grid, text="10.0", style="Mut.TLabel")
        self.aifps_lbl.grid(row=1, column=2, padx=(4, 0))
        sc = ttk.Scale(grid, from_=0.5, to=30, value=self.cfg.fps,
                       command=lambda v: (setattr(self.cfg, "fps", float(v)),
                                          self.aifps_lbl.config(text=f"{float(v):.1f}")))
        sc.grid(row=1, column=1, sticky="ew", pady=2)

        nb = ttk.Notebook(c)
        nb.pack(fill="x", pady=(8, 0))
        self._tab_depth(nb)
        self._tab_seg(nb)
        self._tab_style(nb)
        self._tab_layers(nb)

        self.ai_status = ttk.Label(c, text="idle", style="Mut.TLabel", wraplength=300)
        self.ai_status.pack(anchor="w", pady=(6, 0))

        c = self._card(p, "LOG")
        self.logbox = tk.Text(c, height=7, bg=BG3, fg=MUT, bd=0, wrap="word",
                              font=("Consolas", 8), insertbackground=FG)
        self.logbox.pack(fill="both", expand=True)

    def _mk_entry(self, parent, row, label, values, attr, width=28):
        ttk.Label(parent, text=label, style="Card.TLabel").grid(row=row, column=0, sticky="w", pady=2)
        cb = ttk.Combobox(parent, values=values, width=width)
        cb.set(getattr(self.cfg, attr))
        cb.grid(row=row, column=1, sticky="ew", pady=2)
        def commit(*_):
            setattr(self.cfg, attr, cb.get().strip())
            self.log(f"{attr} -> {cb.get().strip()}")
        cb.bind("<<ComboboxSelected>>", commit)
        cb.bind("<Return>", commit)
        cb.bind("<FocusOut>", commit)
        return cb

    def _mk_scale(self, parent, row, label, lo, hi, attr, fmt="{:.2f}"):
        ttk.Label(parent, text=label, style="Card.TLabel").grid(row=row, column=0, sticky="w")
        val = ttk.Label(parent, text=fmt.format(getattr(self.cfg, attr)), style="Mut.TLabel", width=5)
        val.grid(row=row, column=2, sticky="e")
        sc = ttk.Scale(parent, from_=lo, to=hi, value=getattr(self.cfg, attr),
                       command=lambda v: (setattr(self.cfg, attr,
                                                  type(getattr(self.cfg, attr))(float(v))),
                                          val.config(text=fmt.format(float(v)))))
        sc.grid(row=row, column=1, sticky="ew", pady=2)
        return sc

    def _tab_depth(self, nb):
        f = ttk.Frame(nb, style="Card.TFrame", padding=6)
        f.columnconfigure(1, weight=1)
        nb.add(f, text="Depth")
        self._mk_entry(f, 0, "Model", DEPTH_MODELS, "depth_model")
        self._mk_scale(f, 1, "Smoothing", 0.0, 0.95, "depth_smooth")
        self._mk_scale(f, 2, "Resolution", 128, 768, "depth_res", "{:.0f}")
        inv = tk.BooleanVar(value=self.cfg.depth_invert)
        ttk.Checkbutton(f, text="Invert (if near/far look swapped)", variable=inv,
                        command=lambda: setattr(self.cfg, "depth_invert", inv.get())
                        ).grid(row=3, column=0, columnspan=3, sticky="w", pady=(4, 0))

    def _tab_seg(self, nb):
        f = ttk.Frame(nb, style="Card.TFrame", padding=6)
        f.columnconfigure(1, weight=1)
        nb.add(f, text="Segment")
        self._mk_entry(f, 0, "Model", SEG_MODELS, "seg_model")
        ttk.Label(f, text="Classes", style="Card.TLabel").grid(row=1, column=0, sticky="w")
        ent = ttk.Entry(f)
        ent.insert(0, self.cfg.seg_targets)
        ent.grid(row=1, column=1, columnspan=2, sticky="ew", pady=2)
        ent.bind("<KeyRelease>", lambda e: setattr(self.cfg, "seg_targets", ent.get()))
        self._mk_scale(f, 2, "Smoothing", 0.0, 0.95, "seg_smooth")
        ttk.Button(f, text="List model classes", command=self.show_labels
                   ).grid(row=3, column=0, columnspan=3, sticky="w", pady=(4, 0))

    def _tab_style(self, nb):
        f = ttk.Frame(nb, style="Card.TFrame", padding=6)
        f.columnconfigure(1, weight=1)
        nb.add(f, text="Diffusion")
        self._mk_entry(f, 0, "Model", STYLE_MODELS, "style_model")
        ttk.Label(f, text="Prompt", style="Card.TLabel").grid(row=1, column=0, sticky="nw")
        txt = tk.Text(f, height=3, width=24, bg=BG3, fg=FG, bd=0, wrap="word",
                      font=("Segoe UI", 8), insertbackground=FG)
        txt.insert("1.0", self.cfg.style_prompt)
        txt.grid(row=1, column=1, columnspan=2, sticky="ew", pady=2)
        txt.bind("<KeyRelease>",
                 lambda e: setattr(self.cfg, "style_prompt", txt.get("1.0", "end").strip()))
        self._mk_scale(f, 2, "Strength", 0.05, 0.95, "style_strength")
        self._mk_scale(f, 3, "Steps", 1, 8, "style_steps", "{:.0f}")
        self._mk_scale(f, 4, "Guidance", 0.0, 8.0, "style_guidance", "{:.1f}")
        self._mk_scale(f, 5, "Resolution", 256, 768, "style_res", "{:.0f}")
        ttk.Label(f, text="needs diffusers + a GPU to be fun", style="Mut.TLabel"
                  ).grid(row=6, column=0, columnspan=3, sticky="w", pady=(4, 0))


    def _tab_layers(self, nb):
        f = ttk.Frame(nb, style="Card.TFrame", padding=6)
        f.columnconfigure(1, weight=1)
        nb.add(f, text="Layers")

        ttk.Label(f, text="PERSON PROMPT", style="Head.TLabel").grid(
            row=0, column=0, columnspan=3, sticky="w", pady=(0, 2)
        )
        person = tk.Text(f, height=3, width=24, bg=BG3, fg=FG, bd=0, wrap="word",
                         font=("Segoe UI", 8), insertbackground=FG)
        person.insert("1.0", self.cfg.person_prompt)
        person.grid(row=1, column=0, columnspan=3, sticky="ew", pady=2)
        person.bind("<KeyRelease>",
                    lambda e: setattr(self.cfg, "person_prompt",
                                      person.get("1.0", "end").strip()))
        person_live = tk.BooleanVar(value=self.cfg.person_live)
        ttk.Button(
            f, text="Generate person", style="Acc.TButton",
            command=lambda: self._bump_generation("person")
        ).grid(row=2, column=0, sticky="ew", pady=2)
        ttk.Checkbutton(
            f, text="Live prompt", variable=person_live,
            command=lambda: setattr(self.cfg, "person_live", person_live.get())
        ).grid(row=2, column=1, sticky="w", padx=5)
        self._mk_scale(f, 3, "Person strength", 0.05, 0.95, "person_strength")

        ttk.Label(f, text="BACKGROUND PROMPT", style="Head.TLabel").grid(
            row=4, column=0, columnspan=3, sticky="w", pady=(8, 2)
        )
        back = tk.Text(f, height=3, width=24, bg=BG3, fg=FG, bd=0, wrap="word",
                       font=("Segoe UI", 8), insertbackground=FG)
        back.insert("1.0", self.cfg.background_prompt)
        back.grid(row=5, column=0, columnspan=3, sticky="ew", pady=2)
        back.bind("<KeyRelease>",
                  lambda e: setattr(self.cfg, "background_prompt",
                                    back.get("1.0", "end").strip()))
        back_live = tk.BooleanVar(value=self.cfg.background_live)
        ttk.Button(
            f, text="Generate background", style="Acc.TButton",
            command=lambda: self._bump_generation("background")
        ).grid(row=6, column=0, sticky="ew", pady=2)
        ttk.Checkbutton(
            f, text="Live prompt", variable=back_live,
            command=lambda: setattr(self.cfg, "background_live", back_live.get())
        ).grid(row=6, column=1, sticky="w", padx=5)
        self._mk_scale(f, 7, "Back strength", 0.05, 0.95, "background_strength")

        ttk.Separator(f).grid(row=8, column=0, columnspan=3, sticky="ew", pady=7)
        self._mk_scale(f, 9, "Layer steps", 1, 8, "layer_steps", "{:.0f}")
        self._mk_scale(f, 10, "Layer guidance", 0.0, 8.0, "layer_guidance", "{:.1f}")
        self._mk_scale(f, 11, "Layer resolution", 256, 768, "layer_res", "{:.0f}")
        ttk.Button(
            f, text="Generate both worlds", command=lambda: self._bump_generation("both")
        ).grid(row=12, column=0, columnspan=3, sticky="ew", pady=(5, 0))
        ttk.Label(
            f,
            text="Frozen by default. A prompt edit takes effect when you press Generate; "
                 "Live prompt continuously makes new keyframes.",
            style="Mut.TLabel", wraplength=280,
        ).grid(row=13, column=0, columnspan=3, sticky="w", pady=(4, 0))
        self._mk_entry(f, 14, "LP keyframe file", (), "person_keyframe_path", width=24)
        self._mk_entry(f, 15, "LP output file", (), "live_portrait_path", width=24)

    def _bump_generation(self, which: str) -> None:
        if which in ("person", "both"):
            self.cfg.person_revision += 1
            self.store.drop("person_style")
        if which in ("background", "both"):
            self.cfg.background_revision += 1
            self.store.drop("background_style")
        self.log(f"generate {which} requested")

    # -- centre ------------------------------------------------------------
    def _build_centre(self, p) -> None:
        p.rowconfigure(0, weight=1)
        p.columnconfigure(0, weight=1)
        self.canvas = tk.Canvas(p, bg="#05070a", highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self._imgid = self.canvas.create_image(0, 0, anchor="nw")
        bar = ttk.Frame(p, padding=(0, 6, 0, 0))
        bar.grid(row=1, column=0, sticky="ew")
        ttk.Checkbutton(bar, text="Bypass all (space)", variable=self.bypass).pack(side="left")
        self.stat = ttk.Label(bar, text="", foreground=MUT)
        self.stat.pack(side="right")

    # -- right column ------------------------------------------------------
    def _build_right(self, p) -> None:
        c = self._card(p, "PRESETS")
        row = ttk.Frame(c, style="Card.TFrame")
        row.pack(fill="x")
        self.preset_cb = ttk.Combobox(row, values=list(PRESETS), state="readonly", width=18)
        self.preset_cb.set("Portrait")
        self.preset_cb.pack(side="left", fill="x", expand=True)
        ttk.Button(row, text="Load", width=6, style="Acc.TButton",
                   command=lambda: self.apply_preset(self.preset_cb.get())).pack(side="left", padx=4)
        row2 = ttk.Frame(c, style="Card.TFrame")
        row2.pack(fill="x", pady=(4, 0))
        ttk.Button(row2, text="Save chain…", command=self.save_chain).pack(side="left", fill="x", expand=True)
        ttk.Button(row2, text="Load chain…", command=self.load_chain).pack(side="left", fill="x", expand=True, padx=(6, 0))

        c = self._card(p, "EFFECT CHAIN")
        row = ttk.Frame(c, style="Card.TFrame")
        row.pack(fill="x")
        names = []
        for g in ("depth", "segment", "diffusion", "layer", "classic"):
            for cls in EFFECT_CLASSES:
                if cls.group == g:
                    names.append(f"{GROUP_LABEL[g]} · {cls.name}")
        self.add_cb = ttk.Combobox(row, values=names, state="readonly", width=24)
        self.add_cb.pack(side="left", fill="x", expand=True)
        ttk.Button(row, text="+ Add", width=7, style="Acc.TButton",
                   command=self.add_selected).pack(side="left", padx=(4, 0))
        ttk.Button(row, text="Clear", width=6, command=self.clear_chain).pack(side="left", padx=(4, 0))

        self.chain_frame = ScrollFrame(c, height=210)
        self.chain_frame.pack(fill="x", pady=(6, 0))

        c = self._card(p, "PARAMETERS")
        self.param_head = ttk.Label(c, text="— nothing selected —", style="Card.TLabel")
        self.param_head.pack(anchor="w")
        self.param_note = ttk.Label(c, text="", style="Mut.TLabel", wraplength=320)
        self.param_note.pack(anchor="w", pady=(0, 4))
        self.param_frame = ScrollFrame(c, height=300)
        self.param_frame.pack(fill="both", expand=True)

    # ------------------------------------------------------------ chain ops
    def get_chain(self) -> List[Effect]:
        with self.chain_lock:
            return list(self.chain)

    def add_selected(self) -> None:
        txt = self.add_cb.get()
        if not txt:
            return
        nice = txt.split("·", 1)[-1].strip()
        cls = next((c for c in EFFECT_CLASSES if c.name == nice), None)
        if cls:
            self.add_effect(cls())

    def add_effect(self, fx: Effect) -> None:
        with self.chain_lock:
            self.chain.append(fx)
        self.select(fx)
        self.refresh_chain()

    def remove_effect(self, fx: Effect) -> None:
        with self.chain_lock:
            if fx in self.chain:
                self.chain.remove(fx)
        self.store.state.pop(fx.uid, None)
        if self.selected is fx:
            self.select(None)
        self.refresh_chain()

    def move_effect(self, fx: Effect, delta: int) -> None:
        with self.chain_lock:
            i = self.chain.index(fx)
            j = max(0, min(len(self.chain) - 1, i + delta))
            self.chain.insert(j, self.chain.pop(i))
        self.refresh_chain()

    def clear_chain(self) -> None:
        with self.chain_lock:
            self.chain = []
        self.store.clear_state()
        self.select(None)
        self.refresh_chain()

    def apply_preset(self, name: str) -> None:
        spec = PRESETS.get(name)
        if not spec:
            return
        with self.chain_lock:
            self.chain = build_chain(spec)
        self.store.clear_state()
        self.select(self.chain[0] if self.chain else None)
        self.refresh_chain()
        self.log(f"preset: {name}")

    def save_chain(self) -> None:
        path = filedialog.asksaveasfilename(defaultextension=".json",
                                            filetypes=[("FX chain", "*.json")])
        if not path:
            return
        data = {"chain": [fx.to_dict() for fx in self.get_chain()],
                "ai": {k: getattr(self.cfg, k) for k in vars(self.cfg)}}
        Path(path).write_text(json.dumps(data, indent=2))
        self.log(f"saved chain -> {path}")

    def load_chain(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("FX chain", "*.json")])
        if not path:
            return
        try:
            data = json.loads(Path(path).read_text())
            with self.chain_lock:
                self.chain = build_chain(data.get("chain", []))
            for k, v in (data.get("ai") or {}).items():
                if hasattr(self.cfg, k):
                    setattr(self.cfg, k, v)
            self.store.clear_state()
            self.select(self.chain[0] if self.chain else None)
            self.refresh_chain()
            self.log(f"loaded chain <- {path}")
        except Exception as exc:
            messagebox.showerror(APP, f"could not load: {exc}")

    # ------------------------------------------------------------- chain UI
    def refresh_chain(self) -> None:
        self.chain_frame.clear()
        inner = self.chain_frame.inner
        for i, fx in enumerate(self.get_chain()):
            sel = fx is self.selected
            row = tk.Frame(inner, bg=BG3 if sel else BG2, padx=4, pady=3)
            row.pack(fill="x", pady=1)
            var = tk.BooleanVar(value=fx.enabled)
            cbx = tk.Checkbutton(row, variable=var, bg=row["bg"], activebackground=row["bg"],
                                 selectcolor=BG, bd=0, highlightthickness=0,
                                 command=lambda f=fx, v=var: setattr(f, "enabled", v.get()))
            cbx.pack(side="left")
            missing = self._missing_dep(fx)
            colour = WARN if missing else (ACC if sel else FG)
            lab = tk.Label(row, text=f"{i+1}. {fx.name}", bg=row["bg"], fg=colour,
                           anchor="w", font=("Segoe UI", 9, "bold" if sel else "normal"))
            lab.pack(side="left", fill="x", expand=True)
            lab.bind("<Button-1>", lambda e, f=fx: self.after(0, self.select, f))
            for txt, cmd in (("✕", lambda f=fx: self.after(0, self.remove_effect, f)),
                             ("▼", lambda f=fx: self.after(0, self.move_effect, f, 1)),
                             ("▲", lambda f=fx: self.after(0, self.move_effect, f, -1))):
                tk.Button(row, text=txt, command=cmd, bg=BG, fg=MUT, bd=0, padx=5,
                          activebackground=ACC, highlightthickness=0).pack(side="right", padx=1)

    def _missing_dep(self, fx: Effect) -> bool:
        need = fx.requires()
        if not need:
            return False
        if ({"depth", "mask"} & need) and not fx_ai.TRANSFORMERS_OK:
            return True
        diffusion_maps = {"style", "person_style", "background_style"}
        if (diffusion_maps & need) and not fx_ai.DIFFUSERS_OK:
            return True
        # live_portrait is an intentionally external bridge map. It is not a
        # missing Python package in this app; the effect simply waits for it.
        return False

    def select(self, fx: Optional[Effect]) -> None:
        self.selected = fx
        self.refresh_chain()
        self.build_params()

    # -------------------------------------------------------- parameter UI
    def build_params(self) -> None:
        self.param_frame.clear()
        fx = self.selected
        if fx is None:
            self.param_head.config(text="— nothing selected —")
            self.param_note.config(text="Add an effect, then click its name to edit it here.")
            return
        need = ", ".join(sorted(fx.requires())) or "no model"
        self.param_head.config(text=f"{fx.name}   [{need}]")
        note = fx.blurb
        if self._missing_dep(fx):
            note += "  ⚠ required libraries missing — this effect passes through."
        self.param_note.config(text=note)
        inner = self.param_frame.inner
        for prm in fx.params:
            self._param_widget(inner, fx, prm)

    def _param_widget(self, parent, fx: Effect, prm: Param) -> None:
        row = ttk.Frame(parent, padding=(2, 3))
        row.pack(fill="x")
        row.columnconfigure(1, weight=1)
        ttk.Label(row, text=prm.label, width=15).grid(row=0, column=0, sticky="w")

        if prm.kind in ("float", "int"):
            fmt = "{:.0f}" if prm.kind == "int" else "{:.3g}"
            val = ttk.Label(row, text=fmt.format(fx.p(prm.key)), foreground=MUT, width=6)
            val.grid(row=0, column=2, sticky="e")

            def on_move(v, f=fx, k=prm.key, kind=prm.kind, lbl=val, fmt=fmt):
                x = int(round(float(v))) if kind == "int" else float(v)
                f.values[k] = x
                lbl.config(text=fmt.format(x))

            sc = ttk.Scale(row, from_=prm.lo, to=prm.hi, value=fx.p(prm.key), command=on_move)
            sc.grid(row=0, column=1, sticky="ew", padx=4)
            sc.bind("<Double-Button-1>", lambda e, s=sc, d=prm.default, f=on_move: (s.set(d), f(d)))

        elif prm.kind == "bool":
            var = tk.BooleanVar(value=fx.p(prm.key))
            cb = tk.Checkbutton(row, variable=var, bg=BG, activebackground=BG, selectcolor=BG3,
                                bd=0, highlightthickness=0,
                                command=lambda f=fx, k=prm.key, v=var: (f.values.__setitem__(k, v.get()),
                                                                        self.after(0, self.build_params),
                                                                        self.after(0, self.refresh_chain)))
            cb.grid(row=0, column=1, sticky="w", padx=4)

        elif prm.kind == "choice":
            cb = ttk.Combobox(row, values=list(prm.choices), state="readonly", width=12)
            cb.set(str(fx.p(prm.key)))
            cb.grid(row=0, column=1, sticky="ew", padx=4)
            cb.bind("<<ComboboxSelected>>",
                    lambda e, f=fx, k=prm.key, c=cb: (f.values.__setitem__(k, c.get()),
                                                      self.after(0, self.refresh_chain)))

        elif prm.kind == "color":
            btn = tk.Button(row, text=fx.p(prm.key), bg=fx.p(prm.key), fg="#000", bd=0, width=10)

            def pick(f=fx, k=prm.key, b=btn):
                rgb, hx = colorchooser.askcolor(color=f.p(k), title=f"{f.name} — {k}")
                if hx:
                    f.values[k] = hx
                    b.config(text=hx, bg=hx)

            btn.config(command=pick)
            btn.grid(row=0, column=1, sticky="w", padx=4)

    # ------------------------------------------------------------- runtime
    def start_source(self) -> None:
        if self.camera:
            self.camera.stop()
            time.sleep(0.15)
        raw = self.source_var.get().strip()
        src: object = int(raw) if raw.isdigit() else raw
        self.camera = Camera(src)
        self.camera.flip = self.flip_var.get()
        self.camera.start()
        self.log(f"source: {src!r}")

    def _apply_flip(self) -> None:
        if self.camera:
            self.camera.flip = self.flip_var.get()

    def start_ai(self) -> None:
        self.ai_worker = AIWorker(self.store, self.cfg, self.log)
        self.ai_worker.start()

    def pick_file(self) -> None:
        p = filedialog.askopenfilename(filetypes=[("Video", "*.mp4 *.mov *.avi *.mkv *.webm"),
                                                  ("All", "*.*")])
        if p:
            self.source_var.set(p)
            self.start_source()

    def toggle_record(self) -> None:
        if self.processor.writer is None:
            p = self.processor.start_record(max(10.0, self.processor.fps))
            if p:
                self.rec_btn.config(text="■ Stop")
                self.log(f"recording -> {p}")
        else:
            self.processor.stop_record()
            self.rec_btn.config(text="● Record  (r)")
            self.log("recording stopped")

    def snapshot(self) -> None:
        self.processor.snapshot_req = True

    def show_labels(self) -> None:
        labs = self.ai_worker.hub.seg_labels if self.ai_worker else []
        if not labs:
            messagebox.showinfo(APP, "Load a segmentation effect first — classes appear "
                                     "once the model has been fetched.")
            return
        messagebox.showinfo(f"{APP} — classes", ", ".join(labs))

    def log(self, msg: str) -> None:
        self._logq.put(f"{time.strftime('%H:%M:%S')}  {msg}")

    # ------------------------------------------------------------ tk loops
    def _tick_view(self) -> None:
        if self._closing:
            return
        frame = self.processor.display()
        if frame is not None:
            cw = max(1, self.canvas.winfo_width())
            ch = max(1, self.canvas.winfo_height())
            h, w = frame.shape[:2]
            s = min(cw / w, ch / h)
            if s > 0:
                nw, nh = max(1, int(w * s)), max(1, int(h * s))
                interp = cv2.INTER_LINEAR if s > 1 else cv2.INTER_AREA
                shown = cv2.resize(frame, (nw, nh), interpolation=interp)
                img = Image.fromarray(cv2.cvtColor(shown, cv2.COLOR_BGR2RGB))
                self._photo = ImageTk.PhotoImage(img)
                self.canvas.itemconfig(self._imgid, image=self._photo)
                self.canvas.coords(self._imgid, (cw - nw) // 2, (ch - nh) // 2)
        self.after(16, self._tick_view)

    def _tick_status(self) -> None:
        if self._closing:
            return
        while not self._logq.empty():
            self.logbox.insert("end", self._logq.get() + "\n")
            self.logbox.see("end")
            if float(self.logbox.index("end")) > 220:
                self.logbox.delete("1.0", "60.0")
        p = self.processor
        cam = self.camera
        bits = [f"{p.fps:4.1f} fps", f"{p.ms:5.1f} ms/frame"]
        if cam and cam.error:
            bits.append(f"⚠ {cam.error}")
        f = p.display()
        if f is not None:
            bits.append(f"{f.shape[1]}×{f.shape[0]}")
        self.stat.config(text="   ".join(bits))
        if self.ai_worker:
            r = self.ai_worker.rates
            keys = ("depth", "mask", "style", "person_style", "background_style", "live_portrait")
            live = "  ".join(f"{k}: {r.get(k, 0.0):.1f}/s" for k in keys
                              if r.get(k, 0.0) > 0.05)
            self.ai_status.config(text=(live or "no AI map requested") + f"   [{self.ai_worker.status}]")
        self.after(250, self._tick_status)

    def on_close(self) -> None:
        self._closing = True
        self.processor.stop_record()
        self.processor.stop()
        if self.camera:
            self.camera.stop()
        if self.ai_worker:
            self.ai_worker.stop()
        self.after(120, self.destroy)


def main() -> None:
    ap = argparse.ArgumentParser(description=APP)
    ap.add_argument("--source", default="0", help="webcam index or video path")
    ap.add_argument("--width", type=int, default=640, help="processing width (speed knob)")
    ap.add_argument("--preset", default="Portrait", help=f"one of: {', '.join(PRESETS)}")
    args = ap.parse_args()
    App(args).mainloop()


if __name__ == "__main__":
    main()
