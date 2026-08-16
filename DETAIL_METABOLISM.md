# Antti Detail Metabolism — v0.2

This branch replaces the failed assumption that PhaseRail can recover lost
visible detail by periodically re-locking its own internal Gabor state.

The new question is:

> Can a generated image remain a persistent living video state if expensive
> img2img is used only as a source of *new detail*, rather than as a sequence of
> replacement frames?

## Run

```bat
python3.13 ai_video_fx_metabolism.py --preset "Antti Detail Metabolism"
```

or:

```text
run_detail_metabolism.bat
```

Generate the initial person and background normally. Leave **Live prompt** off.
The original person keyframe remains the identity/style origin; detail donors do
not replace the `person_style` slot.

## Loop

```text
initial generated person K0
          |
          v
   PhaseRail living carrier Lt
          |
          v
  fine-detail debt sensor
          |
       debt high?
       /       \
     no         yes
     |           |
 continue    img2img donor Ft
 living           |
                  v
         flow-align Ft -> Lt
                  |
          donor quality gates
          /             \
       reject           accept
                          |
                 transplant mid/fine only
                          |
                repaired current image Rt
                          |
                 install Rt as the next
                 persistent PhaseRail target
                          |
                         live
```

A donor is therefore a **proposal**, not an authority.

## Detail debt

The first sensor is intentionally simple and local to the failure seen by eye.
A Laplacian split gives low / medium / fine image bands.  The scalar

```text
fine energy / medium energy
```

is recorded when a detail state is accepted.  Later frames are compared against
that accepted ratio:

```text
health = current_ratio / accepted_ratio
debt   = max(0, 1 - health)
```

Using a ratio makes global exposure changes much less important than simple raw
high-frequency energy. It is still only a blur/detail sensor, not an identity
metric. Noise or breakup can create extra high-frequency power, so v0.2 also
quality-gates the donor and does not claim that `health > 1` means the image is
better.

## The img2img donor

`detail_metabolism_patch.py` adds a fourth diffusion mailbox without modifying
the old person/background channels:

```text
detail_request_image -> img2img -> detail_donor
```

The effect publishes its *currently carried generated person* as the img2img
init image only when debt crosses threshold.  The original person prompt is
reused.  Default donor settings are deliberately conservative:

```text
strength 0.32
steps    4
guidance 0
```

These settings are effect parameters and can be changed live.

While a donor is being generated, the old representation keeps moving.  When
the donor arrives it may therefore be a few frames behind the current pose.
The first implementation uses dense Farneback flow to warp the donor into the
current carrier coordinates.

## Acceptance gates

The donor must satisfy two independent conditions before it can modify the
living state:

1. **fine detail gain** — donor fine-band energy must exceed the current
   carrier by at least `Minimum detail gain`;
2. **low-frequency geometry compatibility** — after flow alignment, separately
   contrast-normalized blurred images must remain within `Maximum geometry
   error`.

A sharp but pose-changing donor should therefore be rejected. A geometrically
compatible donor that does not actually add detail should also be rejected.

These are engineering gates, not perceptual or identity guarantees.

## Frequency transplant

Accepted donors do not RGB-crossfade with the living image and do not replace
it wholesale.  Both images are decomposed as:

```text
image = low + medium + fine
```

and the repaired image is:

```text
low       = 100% living carrier
medium    = mix(living, donor, Fresh medium detail)
fine      = mix(living, donor, Fresh fine detail)
```

Defaults:

```text
medium donor share 0.30
fine donor share   0.90
```

The replacement delta is restricted to the person ownership mask.

The important final step is **not** a one-frame overlay: the repaired image is
fed back into `LayerPhaseRail.set_target()`.  It becomes the next persistent
appearance memory and is transported from then onward.

## What to look for

The HUD shows:

```text
DETAIL METABOLISM
health 0.71 debt 0.29 WAITING DONOR ok=2 reject=1
                 gain 1.24 geom 0.18 flow 2.7
```

A successful cycle should look like:

```text
sharp generated person
      -> lives / softens
      -> debt rises
      -> WAITING DONOR while old state keeps moving
      -> accepted donor
      -> visible detail improves without a pose jump
      -> debt resets near zero
      -> repaired state continues living
```

The interesting failure modes are equally useful:

- **donors rejected for geometry**: img2img strength/model is changing pose too
  much; lower strength or improve conditioning/alignment;
- **donors accepted but no visible sharpening**: Laplacian transplant is not
  targeting the real loss;
- **detail sharpens but identity changes**: v0.3 needs a real K0 reference-image
  conditioning or identity/style gate; prompt-only anchoring is insufficient;
- **debt never rises while image visibly blurs**: the fine/medium energy ratio is
  the wrong sensor and needs phase/coherence/perceptual information;
- **debt rises immediately from ordinary motion**: the sensor is measuring pose,
  not detail.

## Why keep the original keyframe?

Three states now have different jobs:

```text
K0  original generated person       -> identity/style origin
Lt  current PhaseRail carrier       -> current living pose/continuity
Ft  occasional img2img donor        -> candidate fresh detail
```

v0.2 does not yet use K0 as a second model conditioning image because the
current diffusers pipeline is single-image img2img.  K0 is preserved and never
overwritten, which leaves the correct architectural slot for a later
reference-image adapter or identity metric if prompt-only donors drift.

## First gate

Use one prompt and one motion sequence. Compare:

1. ordinary `Antti Layered PhaseRail`;
2. `Antti Detail Metabolism` with automatic detail maintenance on.

Record at least ~30 seconds.  The idea earns another step only if donor cycles
produce visibly sharper persistence **without obvious keyframe jumps**, and
accepted-donor count is much lower than generating every frame.

Do not tune thresholds to one flattering Santa take and call that validation.
The purpose of v0.2 is to establish whether *new information injected only into
expired bands* is a useful maintenance operation at all.
