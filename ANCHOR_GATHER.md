# Antti Anchor Gather — address moves, image does not

This experiment came out of a failure of the multiscale-repair and detail-metabolism branches.

A carried generated person can remain alive geometrically while its visible texture turns into blur or sharp mush.  The important correction is that the current PhaseRail implementation is **not literally remapping yesterday's raster into today's raster every frame**.  It recursively evolves complex Gabor state (`target_z`, `output_z`) plus low/residual state.  Nevertheless the appearance state itself is recursive.

Anchor Gather removes recursive appearance from the experiment completely.

## Rule

Keep one immutable generated person keyframe `K0`.

Use the existing Gabor motion solver only to estimate the incremental current-to-previous displacement `d_t`.

Maintain a backward address field:

```text
Phi_t(x) = Phi_{t-1}(x + d_t(x))
```

Then render:

```text
I_t(x) = K0(Phi_t(x))
```

The address field is composed through time; the image is not.  Every displayed frame samples the pristine full-resolution generated person once.

This deliberately changes the expected failure mode:

```text
recursive appearance transport -> blur / phase mush / texture manufacture
source-anchored gather         -> wrong address / stretch / drift / occlusion failure
```

A sharp feature may end up slightly in the wrong place, but it should not become a fifty-generation photocopy merely because the keyframe is old.

## Run

```bat
python3.13 ai_video_fx_anchor.py --preset "Antti Anchor Gather"
```

or:

```text
run_anchor_gather.bat
```

Generate person and background normally. Leave Live prompt off.

## Monitor

The HUD reports:

```text
ANCHOR GATHER  motion 0.42  conf 0.36  addr 7.2px  stretch 0.031
```

- `motion`: current incremental displacement magnitude reported by the Gabor solver.
- `conf`: motion confidence.
- `addr`: mean distance the accumulated address field has moved away from its identity map.
- `stretch`: a cheap local distortion diagnostic for the address map.

`Show displacement map` displays the accumulated address displacement itself.

## Important first gate

Compare the same modest webcam motion under:

1. `Antti Layered PhaseRail`
2. `Antti Anchor Gather`

Do **not** ask which one tracks more degrees of freedom first. Ask the narrower question:

> after 10–30 seconds, does Anchor Gather keep the original generated texture materially sharper while PhaseRail softens/mushes?

If yes, the blur problem was at least partly recursive appearance-state aging and source-anchored gathering earns another step.

If Anchor Gather becomes equally blurry, inspect the full-resolution gather path and mask/compositor before inventing more theory.

If it remains sharp but drifts, tears, stretches, or cannot open the mouth/turn the head, that is the expected informative failure: **detail survived; the address field did not know enough geometry.**  Then the next work is correspondence/re-anchoring, not detail refresh.

## Synthetic gate

`test_anchor_gather.py` contains a deliberately simple stress test. A textured image is translated by the same subpixel motion for 50 frames.

- recursive raster warping repeatedly interpolates the raster and loses high-frequency energy;
- Anchor Gather composes the coordinate map and samples the original raster once.

The test does not claim this is the exact PhaseRail mechanism. It only freezes the engineering principle that motivates the experiment.

## Relation to the neuron work — narrow version

The useful transfer is **address vs content**, not a claim that the video code is a neuron.

Earlier receiver/observability work kept forcing a separation between:

```text
what state exists in the system
and
what a particular receiver can recover about where it came from
```

Anchor Gather makes that separation explicit in software:

```text
K0      = content memory
Phi_t   = address / correspondence field
I_t     = receiver query: sample K0 at Phi_t(x)
```

The content does not have to carry its own location history through repeated state updates.  Location is a separate field that tells the current pixel where to query the stable source.

That is a concrete design lesson from the older thinking.  It is not evidence for dendritic field-query mechanisms, Clockfield, or any new neural physics.

## Why this may explain part of the old SplatWorld surprise

A compact splat/Gabor face model re-synthesizes an image from a persistent parameterization each frame instead of repeatedly copying a raster forward.  Anchor Gather approaches the same stability principle from the other direction: preserve a canonical appearance source and let a smaller dynamical object carry time.

If this branch works, the natural hybrid is:

```text
immutable / occasionally refreshed appearance source
                    +
          living low-DOF address field
                    +
       rare geometry re-anchor on drift
```

Only then should img2img detail metabolism be reintroduced as a source-refresh mechanism for genuinely new views or disocclusions, rather than as a band-aid for recursive blur.
