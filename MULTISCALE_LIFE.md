# Antti Multiscale Life — v0.1

The goal is no longer "generate a new video frame every frame."  The experiment asks whether one generated person can remain a useful dynamical object for a long time if different parts of its representation are allowed to have different ages.

Initial **hypothesis**, not result:

```text
silhouette / coarse geometry     ~100 frames
low-frequency appearance          ~60 frames
medium structure                   ~25 frames
fine texture                        ~8 frames
```

The numbers are editable in the effect UI.  They are supposed to lose if the live measurements say something else.

## Run

```bat
python3.13 ai_video_fx_life.py --preset "Antti Multiscale Life"
```

or:

```text
run_multiscale_life.bat
```

Generate a person and background exactly as in the ordinary layered effect.  Leave **Live prompt** off.

## What v0.1 actually does

The existing PhaseRail already carries two complex Gabor states:

- `target_z`: a separately transported reference derived from the generated keyframe;
- `output_z`: the visible transported state.

The five Gabor scales are ordered fine to coarse (`0.32, 0.18, 0.10, 0.055, 0.03` cycles/pixel).  v0.1 groups them as:

```text
scale 0       -> fine      -> default life 8 frames
scales 1,2    -> medium    -> default life 25 frames
scale 3       -> low       -> default life 60 frames
scale 4       -> coarse    -> default life 100 frames
```

When a scale reaches its life, `output_z` is re-locked toward the current `target_z` at that scale.  Crucially, `target_z` has been phase-advanced with the same motion estimate, so this is **not** a snap back to the original pose.  It is an attempt to repair a drifting band from a transported memory.

No new diffusion call is made by the repair operation.

The **Repair strength** control is a phasor-safe blend: amplitude and unit phase are blended separately.  A direct complex linear interpolation was deliberately avoided because opposite phases can cancel and manufacture blur.

## The live monitor

The lower-left HUD has four rows:

```text
coarse  vis 0.98  z 0.99  age  42/100
low     vis 0.94  z 0.97  age  42/60
medium  vis 0.81  z 0.93  age  17/25
fine    vis 0.53  z 0.88  age   7/8
```

Two different sensors are shown on purpose:

- `vis` is image-space band-energy retention relative to the first carried frame after the generated keyframe was accepted.  It uses an independent Gaussian/Laplacian pyramid, so it can catch visible blur even if PhaseRail's own state says everything is fine.
- `z` is internal PhaseRail health: `1 / (1 + relative squared error)` between the visible complex state and the transported reference for that scale group.
- `age` is frames since that group was last repaired, followed by its configured lifetime.

A white outline around a row for one frame means that scale group was actually re-locked on that frame.

`vis > 1` is allowed in the raw measurement because motion/noise can add band energy.  The bar itself clips at 1; the printed number does not.

## The first kill gate

Do **not** tune the four lifetimes from one flattering moment.

Run the same motion twice:

1. **Multiscale repair OFF** (`Multiscale repair = false`).
2. **Multiscale repair ON** with the default 8/25/60/100 clocks.

Record at least 20–30 seconds each.

The hypothesis earns another step only if both are visible:

- with repair OFF, fine/medium `vis` health tends to deteriorate materially faster than coarse/low during ordinary motion;
- with repair ON, the short-lived bands recover/retain more visible energy without obvious periodic pose snapping or worse global artifacts.

If all four bands decay together, the multiscale-lifetime story is wrong for this representation.  If `z` stays healthy while `vis` dies, the damage is outside the Gabor state (likely residual/reconstruction/output path) and the next repair target should be there instead of adding more Gabor logic.

## Why the sensor comes before more machinery

TinyAvatar/SplatWorld and the PhaseRail experiments repeatedly showed a qualitative failure: a generated object can continue to move while becoming soft.  One scalar "keyframe bad" score cannot tell us what has aged.

This experiment therefore separates:

```text
coarse geometry
low appearance
medium structure
fine texture
```

before deciding how to repair them.  The visual lifetime order is the scientific object.  The 8/25/60/100 scheduler is only the first actuator built around it.

## Important limitation of v0.1

The periodic repair currently acts on the **five complex Gabor scales only**.  PhaseRail also has explicit low-pass and residual paths.  In particular, visible fine-detail loss may live in the residual path rather than in `output_z`.

That is why the independent `vis` monitor exists.  If the fine `z` health repeatedly resets to ~1 while visible fine health keeps falling, v0.1 has localized the failure: repairing Gabor phase is not enough, and the residual must become an independently aged/repaired state in v0.2.

That failure would be useful, not a null result.

## Files

| file | role |
|---|---|
| `multiscale_life.py` | independent visible-band sensor + 5-scale lifetime scheduler |
| `fx_phase_rail_life.py` | PhaseRail wrapper with per-scale phasor re-lock |
| `fx_multiscale_life.py` | UI effect, HUD and preset |
| `ai_video_fx_life.py` | launcher |
| `run_multiscale_life.bat` | Windows launcher |
| `test_multiscale_life.py` | blur/scheduler/effect integration gates |

## Next step if v0.1 is promising

Do not immediately add diffusion refresh.

First make the low-pass and residual states independently addressable.  Then the architecture can become:

```text
coarse state     long memory / rare repair
low appearance   medium-long memory
mid Gabor state  shorter re-lock
fine Gabor + residual   rapid repair
```

Only after we know which state is actually dying should an expensive diffusion keyframe be used as the final repair source.
