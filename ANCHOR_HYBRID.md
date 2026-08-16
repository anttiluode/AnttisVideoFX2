# Anchor Hybrid — make the Picasso failure graceful

`Anchor Gather` fixed one real problem: recursive raster transport was losing high-frequency detail. It kept a pristine generated person and transported only a current-to-anchor address field, so every displayed frame sampled the sharp source once.

The failure moved exactly where expected: **correspondence**. After enough motion, the old face/hands remained sharp but attached to the wrong places. Fingers stayed behind, facial regions sheared, and the output became a layered Picasso rather than a blur.

This experiment treats that as two separate objects:

- **appearance memory** — the sharp generated anchor;
- **address memory** — where current pixels believe they came from in that anchor.

## What the new effect does

`Antti Anchor Hybrid` keeps Anchor Gather, then estimates an **address trust map** from:

1. local Jacobian strain/folding of the accumulated address field;
2. whether the transported old person mask still agrees with the current person mask;
3. whether the address points outside the original anchor.

A rigid translation is not punished merely for being large. Stretch, fold, stale ownership and newly exposed regions are.

Where trust is high, the sharp generated detail is used normally. Where trust is low, stale high-frequency detail is suppressed and only low-frequency structure is borrowed from the current camera. The intended failure is therefore **soft/current rather than sharp/wrong**.

When too much of the subject becomes untrustworthy, the effect requests one img2img donor through the existing `detail_metabolism_patch` worker. The donor is:

- aligned to the current fused person;
- rejected if low-frequency geometry changes too much;
- rejected if mid-band structure is poorly correlated;
- rejected if its fine-band energy is implausibly low or high;
- used only as a mid/fine-frequency transplant.

If accepted, that repaired current pose becomes the new sharp anchor and the address field resets to identity. This is a **soft re-anchor**, not a wholesale new frame.

The bounded band correlation is important: unrelated sharp noise cannot win merely by having lots of high-frequency energy.

## Run

```bat
run_anchor_hybrid.bat
```

or

```bat
python3.13 ai_video_fx_hybrid.py --preset "Antti Anchor Hybrid"
```

The HUD shows:

- `health` — mean address trust inside the current person;
- `bad` — fraction of subject pixels below the trust threshold;
- `strain` — mean accumulated local address deformation;
- `conf` — the existing PhaseRail instantaneous motion confidence;
- `DONOR` — an img2img soft re-anchor is in flight;
- `mid/fine/gain/geom` — acceptance diagnostics for the last donor.

Turn on **Show address trust map** to see the receiver directly.

## First kill gates

This should earn its complexity only if at least one of these happens on the same webcam motion that produced the Picasso failure:

1. stale fingers/face fragments disappear into a soft fallback before they become sharp displaced layers;
2. accepted donor re-anchors materially extend useful motion time without obvious whole-frame popping;
3. donor rejection catches sharp-but-unrelated generations that an energy-only metric would have accepted.

If the trust map stays bright while the correspondence is visibly wrong, the receiver is insufficient. If it correctly goes dark but the output is no better, the gating/fallback is insufficient. If donor refreshes happen constantly, the thresholds or the address model are insufficient. Those are separate failures and should stay separate.
