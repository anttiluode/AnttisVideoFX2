# LivePortrait control + measurable transport benchmark

## Important archaeological correction

LivePortrait was **already in the system before this pass**. The original
`AIvideoFX` / `AnttisVideoFX2` architecture already contained:

- `ExternalPortraitLayer` in `fx_core.py`;
- `PortraitBridgeWorker` in `fx_ai.py`;
- automatic export of `layer_person_keyframe.png` whenever a prompted person was generated;
- a watcher for `live_portrait_latest.png`;
- editable keyframe/output filenames in the Layers UI.

So the idea was not newly imported from the literature. It had been designed as
an optional module but never completed end-to-end. The missing wire was mundane:
the app never exported the **driving webcam frame**, and there was no worker that
actually ran LivePortrait and returned a frame.

This pass completes that wire without vendoring LivePortrait into the Python
3.13 application.

## Why keep LivePortrait in a separate process?

The official LivePortrait project currently expects its own environment (the
official instructions use Python 3.10) and model stack. AnttisVideoFX2 already
has a large Torch / Diffusers / Transformers environment. Joining them would make
a control experiment unnecessarily fragile.

The bridge is therefore:

```text
AnttisVideoFX2 (Python 3.13)
    generated appearance -> layer_person_keyframe.png
    current camera       -> live_portrait_drive.png
                                   |
                                   v
LivePortrait env (Python 3.10) -> live_portrait_latest.png
                               -> live_portrait_alpha.png
                                   |
                                   v
Antti LivePortrait Assist
    PhaseRail body + LivePortrait face + generated background
```

This is an intentionally **unfair baseline**. It asks one narrow question:

> If the learned/canonical portrait correspondence owns the face while our
> PhaseRail still owns the body, does the face stop becoming Picasso?

If yes, stop tuning face confidence thresholds. The representation was missing
a portrait prior. If no, the failure is elsewhere (source stylisation, crop,
compositing, driver/source mismatch, etc.).

## Run

Install/prepare the official LivePortrait checkout and weights according to its
own instructions. Then set the two environment variables once in a Windows shell:

```bat
set LIVEPORTRAIT_HOME=E:\path\to\LivePortrait
set LIVEPORTRAIT_ENV=LivePortrait
```

Start the external worker:

```bat
run_liveportrait_worker.bat
```

In a second terminal:

```bat
run_liveportrait_assist.bat
```

Select/generate a person and background as usual. The first generated person is
the LivePortrait **source appearance**; the webcam is only the **driver**.

The worker follows the image-source relative-motion branch of the official
LivePortrait pipeline: source appearance features are extracted once; each live
driving crop supplies pose/expression keypoints; stitching is applied; the face
is decoded and placed back into the current driving face coordinates. The alpha
is exported separately so real-camera pixels do not silently become persistent
identity state.

If the generated/stylised source itself is too strange for the face detector,
the worker can borrow only the *crop transform* from the current real driving
face while still extracting appearance from the generated source.

## The new independent cycle receiver

Anchor Hybrid's old trust map was Jacobian strain + current/old person-mask
agreement. The latest screenshots showed that this receiver often already knew
it was in trouble (`health` low, `bad` high), but a smooth wrong correspondence
can still have a healthy Jacobian.

`cycle_consistency.py` therefore maintains an independent long-range pair of
Farneback histories:

```text
current --current->anchor--> anchor --anchor->current--> current_hat
```

The loop error is measured in rail pixels. It is composed over time, not merely
a one-frame t<->t-1 check. Anchor Hybrid now multiplies its old trust by this
cycle trust. The HUD adds `cyc`.

This is still only a **receiver**. It can say that a correspondence is lost; it
cannot generate an unseen ear, cheek, finger, profile or shoulder. That is the
reason for the LivePortrait control.

## Benchmark: stop scoring by "looks Picasso"

`transport_benchmark.py` gives each architecture a finite keyframe budget `B`.
Use the exact same prerecorded motion for every candidate.

For an unstyled control, the target can be the original video itself. For the
actual stylised task, use a per-frame stylised oracle as target (for example,
independently img2img every original frame with frozen prompt/settings). This is
important: comparing a successful marble person directly to the real webcam
would punish the style we are trying to preserve.

Prefer a person crop:

```bat
python3.13 transport_benchmark.py ^
  --target oracle_marble.mp4 ^
  --crop 180,40,310,420 ^
  --candidate phase=phase.mp4 ^
  --candidate gather=gather.mp4 ^
  --candidate hybrid=hybrid.mp4 ^
  --candidate liveportrait=liveportrait.mp4
```

The scorer:

- removes exact/near duplicate **target** frames before counting budget;
- ignores the static room if a person crop is supplied;
- reports L1, PSNR and a dependency-free structural similarity control;
- defines `B` as active/nonduplicate frames before a metric stays beyond the
  threshold for `--patience` consecutive active frames;
- writes per-frame CSVs so the failure can be inspected rather than hidden in a
  single average.

The default budget is PSNR >= 25 dB for 3-frame patience. That default is a
starting calibration, not a scientific constant. Pick the threshold on a small
human-labelled set and then freeze it before comparing architectures.

## What would count as a result?

1. **LivePortrait face >> PhaseRail face, same body:** learned portrait
   correspondence was the missing face representation.
2. **Both fail at the same moment:** look at source generation/cropping or
   ownership/compositing rather than motion representation.
3. **Cycle error rises before visual Picasso:** useful predictive receiver.
4. **Cycle stays low while visible correspondence is wrong:** even long-range
   low-level flow cannot see the semantic failure; use the portrait prior, do
   not invent Cycle Receiver V2 forever.
5. **LivePortrait face works but hands/torso fail:** expected. It is a portrait
   control, not a full-body solution.

The strategic point remains the same: the interesting product is a stylised
video-FX instrument. LivePortrait is used here as a strong component/control,
not as a claim that we invented portrait animation.
