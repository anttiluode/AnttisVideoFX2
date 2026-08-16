#!/usr/bin/env python3
"""Duplicate-aware finite-keyframe benchmark for AnttisVideoFX2.

The goal is deliberately mundane: turn "it becomes Picasso after a while" into
one repeatable number.  Run several renderers on the same prerecorded driving
clip and compare them against either:

* the real clip itself for an *unstyled self-reenactment/control* run; or
* a per-frame stylised oracle video for the actual FX task.

Whole-frame static-room pixels and exact/near duplicate source frames are
excluded from the keyframe budget.  A manual person crop is strongly preferred.

Example
-------
python transport_benchmark.py ^
  --target oracle_marble.mp4 --crop 180,40,310,420 ^
  --candidate phase=phase.mp4 ^
  --candidate gather=gather.mp4 ^
  --candidate hybrid=hybrid.mp4 ^
  --candidate liveportrait=liveportrait.mp4
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Optional

import cv2
import numpy as np

EPS = 1e-9


def parse_crop(text: str | None):
    if not text:
        return None
    vals = [int(x.strip()) for x in text.split(",")]
    if len(vals) != 4:
        raise ValueError("--crop must be x,y,w,h")
    x, y, w, h = vals
    if w <= 0 or h <= 0:
        raise ValueError("crop width/height must be positive")
    return x, y, w, h


def crop_frame(frame: np.ndarray, crop):
    if crop is None:
        return frame
    x, y, w, h = crop
    H, W = frame.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(W, x + w), min(H, y + h)
    if x1 <= x0 or y1 <= y0:
        raise ValueError("crop lies outside video")
    return frame[y0:y1, x0:x1]


def duplicate_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Mean 8-bit luma difference on a tiny thumbnail."""
    def small(x):
        g = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
        return cv2.resize(g, (64, 64), interpolation=cv2.INTER_AREA).astype(np.float32)
    aa, bb = small(a), small(b)
    return float(np.mean(np.abs(aa - bb)))


def global_ssim(a: np.ndarray, b: np.ndarray) -> float:
    """Cheap global SSIM-like control metric; no skimage dependency required."""
    x = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY).astype(np.float64) / 255.0
    y = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY).astype(np.float64) / 255.0
    mux, muy = float(x.mean()), float(y.mean())
    vx, vy = float(x.var()), float(y.var())
    cov = float(np.mean((x - mux) * (y - muy)))
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    return float(((2 * mux * muy + c1) * (2 * cov + c2)) /
                 ((mux * mux + muy * muy + c1) * (vx + vy + c2) + EPS))


def frame_metrics(target: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    if candidate.shape[:2] != target.shape[:2]:
        candidate = cv2.resize(candidate, (target.shape[1], target.shape[0]), interpolation=cv2.INTER_LINEAR)
    d = target.astype(np.float32) - candidate.astype(np.float32)
    mse = float(np.mean(d * d))
    l1 = float(np.mean(np.abs(d)) / 255.0)
    psnr = 99.0 if mse < 1e-12 else float(10.0 * math.log10((255.0 * 255.0) / mse))
    return {"l1": l1, "psnr": psnr, "ssim": global_ssim(target, candidate)}


def keyframe_budget(values: Iterable[float], threshold: float, *, higher_is_better: bool,
                    patience: int = 3) -> int:
    """Number of active/nonduplicate frames before persistent threshold failure."""
    bad = 0
    n = 0
    for v in values:
        n += 1
        failed = v < threshold if higher_is_better else v > threshold
        bad = bad + 1 if failed else 0
        if bad >= max(1, int(patience)):
            return max(0, n - bad)
    return n


@dataclass
class BenchmarkSummary:
    name: str
    frames: int
    active_frames: int
    duplicates_removed: int
    median_l1: float
    median_psnr: float
    median_ssim: float
    budget_metric: str
    budget_threshold: float
    keyframe_budget: int
    first_failure_raw_frame: Optional[int]


def compare_videos(target_path: str, candidate_path: str, *, name: str,
                   crop=None, duplicate_threshold: float = 0.25,
                   budget_metric: str = "psnr", budget_threshold: float = 25.0,
                   patience: int = 3, csv_path: Path | None = None) -> BenchmarkSummary:
    target_cap = cv2.VideoCapture(target_path)
    cand_cap = cv2.VideoCapture(candidate_path)
    if not target_cap.isOpened():
        raise FileNotFoundError(f"cannot open target {target_path}")
    if not cand_cap.isOpened():
        raise FileNotFoundError(f"cannot open candidate {candidate_path}")

    rows = []
    prev_target = None
    raw = active = duplicates = 0
    try:
        while True:
            ok_t, target = target_cap.read()
            ok_c, cand = cand_cap.read()
            if not ok_t or not ok_c:
                break
            raw += 1
            target = crop_frame(target, crop)
            cand = crop_frame(cand, crop)
            if cand.shape[:2] != target.shape[:2]:
                cand = cv2.resize(cand, (target.shape[1], target.shape[0]), interpolation=cv2.INTER_LINEAR)

            dup_d = None if prev_target is None else duplicate_distance(target, prev_target)
            is_dup = dup_d is not None and dup_d <= float(duplicate_threshold)
            prev_target = target.copy()
            if is_dup:
                duplicates += 1
                rows.append({"raw_frame": raw - 1, "active_frame": "", "duplicate": 1,
                             "duplicate_distance": dup_d, "l1": "", "psnr": "", "ssim": ""})
                continue
            active += 1
            m = frame_metrics(target, cand)
            rows.append({"raw_frame": raw - 1, "active_frame": active - 1, "duplicate": 0,
                         "duplicate_distance": "" if dup_d is None else dup_d, **m})
    finally:
        target_cap.release()
        cand_cap.release()

    active_rows = [r for r in rows if not r["duplicate"]]
    if not active_rows:
        raise RuntimeError("no active frames remained after duplicate filtering")
    values = [float(r[budget_metric]) for r in active_rows]
    higher = budget_metric in ("psnr", "ssim")
    budget = keyframe_budget(values, budget_threshold, higher_is_better=higher, patience=patience)
    first_failure_raw = None
    if budget < len(active_rows):
        first_failure_raw = int(active_rows[budget]["raw_frame"])

    if csv_path is not None:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    return BenchmarkSummary(
        name=name,
        frames=raw,
        active_frames=active,
        duplicates_removed=duplicates,
        median_l1=float(np.median([float(r["l1"]) for r in active_rows])),
        median_psnr=float(np.median([float(r["psnr"]) for r in active_rows])),
        median_ssim=float(np.median([float(r["ssim"]) for r in active_rows])),
        budget_metric=budget_metric,
        budget_threshold=float(budget_threshold),
        keyframe_budget=int(budget),
        first_failure_raw_frame=first_failure_raw,
    )


def parse_candidate(spec: str) -> tuple[str, str]:
    if "=" not in spec:
        path = Path(spec)
        return path.stem, spec
    name, path = spec.split("=", 1)
    return name.strip(), path.strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True,
                    help="ground-truth clip for an unstyled control, or per-frame stylised oracle")
    ap.add_argument("--candidate", action="append", required=True,
                    help="NAME=video.mp4; repeat for each architecture")
    ap.add_argument("--crop", default=None, help="person ROI as x,y,w,h; strongly recommended")
    ap.add_argument("--duplicate-threshold", type=float, default=0.25,
                    help="mean 8-bit thumbnail luma change below which a target frame is ignored")
    ap.add_argument("--budget-metric", choices=("psnr", "ssim", "l1"), default="psnr")
    ap.add_argument("--budget-threshold", type=float, default=25.0)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--csv-dir", default="benchmark_csv")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    crop = parse_crop(args.crop)
    summaries = []
    for spec in args.candidate:
        name, path = parse_candidate(spec)
        csv_path = Path(args.csv_dir) / f"{name}.csv" if args.csv_dir else None
        s = compare_videos(
            args.target, path, name=name, crop=crop,
            duplicate_threshold=args.duplicate_threshold,
            budget_metric=args.budget_metric,
            budget_threshold=args.budget_threshold,
            patience=args.patience,
            csv_path=csv_path,
        )
        summaries.append(s)

    summaries.sort(key=lambda s: s.keyframe_budget, reverse=True)
    print("\nTRANSPORT / KEYFRAME BUDGET")
    print("target:", args.target)
    if crop is None:
        print("WARNING: no person crop supplied; a static room can dominate pixel metrics")
    for s in summaries:
        print(
            f"{s.name:18s} B={s.keyframe_budget:4d}/{s.active_frames:<4d} "
            f"PSNR={s.median_psnr:5.2f} SSIM={s.median_ssim:6.3f} "
            f"L1={s.median_l1:6.4f} dup={s.duplicates_removed}"
        )

    payload = [asdict(s) for s in summaries]
    if args.json:
        Path(args.json).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    else:
        print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
