#!/usr/bin/env python3
"""Headless check: runs every effect and every preset on synthetic frames.

    python selftest.py            # correctness + timings, no camera, no GUI
    python selftest.py --render   # also writes selftest_<effect>.png thumbnails

Useful when installing on a new machine, or after editing fx_core.py.
"""
import argparse
import time

import cv2
import numpy as np

from fx_core import (EFFECT_CLASSES, PRESETS, FXContext, MapStore, build_chain,
                     chain_requirements, to_f32, to_u8)


def fake_scene(h=360, w=640, t=0.0):
    """A frame with structure: gradient sky, a bright disc, noise texture."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    img = np.zeros((h, w, 3), np.float32)
    img[..., 0] = 0.25 + 0.5 * yy / h
    img[..., 1] = 0.35 + 0.3 * np.sin(xx / 40 + t)
    img[..., 2] = 0.4 + 0.4 * xx / w
    r = np.sqrt((xx - w * 0.5 - 40 * np.sin(t)) ** 2 + (yy - h * 0.5) ** 2)
    disc = np.clip(1.2 - r / 70, 0, 1)
    img += disc[..., None] * np.array([0.9, 0.95, 1.0], np.float32)
    img += np.random.random_sample((h, w, 1)).astype(np.float32) * 0.05
    return to_u8(np.clip(img, 0, 1))


def fake_maps(h=360, w=640):
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    r = np.sqrt((xx - w * 0.5) ** 2 + (yy - h * 0.5) ** 2)
    depth = np.clip(1.15 - r / (0.55 * w), 0, 1).astype(np.float32)
    mask = (depth > 0.55).astype(np.float32)
    style = cv2.applyColorMap(to_u8(depth), cv2.COLORMAP_MAGMA).astype(np.float32) / 255.0
    return depth, mask, style


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--frames", type=int, default=6)
    args = ap.parse_args()

    store = MapStore()
    d, m, s = fake_maps()
    store.put("depth", d)
    store.put("mask", m)
    store.put("style", s)
    # Layered AI maps. The person and background are deliberately different so
    # the composite and PhaseRail code exercise both paths.
    person = np.clip(0.25 * to_f32(fake_scene()) + 0.75 * s, 0, 1).astype(np.float32)
    background = np.roll(s, 60, axis=1).astype(np.float32)
    store.put("person_style", person)
    store.put("background_style", background)
    store.put("live_portrait", person)

    print(f"{'effect':26s} {'ms/frame':>9s}   status")
    print("-" * 56)
    failures = []
    for cls in EFFECT_CLASSES:
        fx = cls()
        t_total, ok, err = 0.0, True, ""
        out = None
        try:
            for i in range(args.frames):
                frame = fake_scene(t=i * 0.1)
                ctx = FXContext(store, i * 0.1, i, frame.shape[:2])
                img = to_f32(frame)
                t0 = time.perf_counter()
                img = fx.apply(img, ctx)
                t_total += time.perf_counter() - t0
                assert img.shape == frame.shape, f"shape {img.shape}"
                assert img.dtype == np.float32, f"dtype {img.dtype}"
                assert np.isfinite(img).all(), "non-finite values"
                out = img
        except Exception as exc:
            ok, err = False, repr(exc)
            failures.append((cls.__name__, err))
        ms = 1000 * t_total / args.frames
        print(f"{cls.name:26s} {ms:8.2f}   {'ok' if ok else 'FAIL ' + err}")
        if args.render and ok and out is not None:
            cv2.imwrite(f"selftest_{cls.__name__}.png", to_u8(out))

    print("\npresets")
    print("-" * 56)
    for name, spec in PRESETS.items():
        chain = build_chain(spec)
        try:
            t0 = time.perf_counter()
            for i in range(args.frames):
                frame = fake_scene(t=i * 0.1)
                ctx = FXContext(store, i * 0.1, i, frame.shape[:2])
                img = to_f32(frame)
                for fx in chain:
                    img = fx.apply(img, ctx)
                assert np.isfinite(img).all()
            ms = 1000 * (time.perf_counter() - t0) / args.frames
            need = ", ".join(sorted(chain_requirements(chain))) or "none"
            print(f"{name:26s} {ms:8.2f}   ok   (needs: {need})")
            if args.render:
                cv2.imwrite(f"selftest_preset_{name.replace(' ', '_')}.png", to_u8(img))
        except Exception as exc:
            failures.append((name, repr(exc)))
            print(f"{name:26s} {'':8s}   FAIL {exc!r}")

    print()
    if failures:
        print(f"{len(failures)} failure(s):")
        for n, e in failures:
            print("  ", n, e)
        raise SystemExit(1)
    print("all effects and presets pass  ✔")


if __name__ == "__main__":
    main()
