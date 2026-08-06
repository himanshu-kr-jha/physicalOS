# PhysicalOS

**The physical world tells the truth.**

Feed it ordinary video of a real place — a road, a construction site, a building
facade — and it returns a geolocated 3D digital twin where **every claim is one
click from the pixels that justify it**.

Not a dashboard of numbers. A map where clicking a pothole opens the actual
frame, with the bounding box drawn on it and the model's own sentence explaining
why it called it.

Everything runs on CPU. A GPU buys a better point cloud; nothing requires one.

---

## Contents

- [Quickstart](#quickstart-no-api-key-no-gpu)
- [The pipeline](#the-pipeline)
- [Running on your own video](#running-on-your-own-video) ← start here for real footage
  - [The studio](#the-short-way--the-studio) — upload in a browser
  - [Importing a findings CSV](#already-have-the-findings-attach-a-csv) — no detector needed
- [Camera calibration](#camera-calibration-the-thing-that-decides-accuracy)
- [Perception backends](#perception-backends)
- [Domains](#domains-the-taxonomy-is-data-not-code)
- [Absence: defects that are things missing](#absence-defects-that-are-things-missing)
- [Point clouds](#point-clouds-two-routes)
- [The viewer](#the-viewer)
- [Google Earth export](#google-earth-export)
- [Satellite basemap](#satellite-basemap)
- [PDF report](#pdf-report)
- [Run directory and data contracts](#run-directory-and-data-contracts)
- [How positions are computed](#how-positions-are-computed)
- [Measured results](#measured-results-on-real-footage)
- [Things that bit us](#things-that-bit-us)
- [Project layout](#project-layout)
- [Licences](#licences)

---

## Quickstart (no API key, no GPU)

```bash
bash setup.sh      # deps, viewer build, .env, and a report of what this box can run
bash start.sh      # → http://127.0.0.1:8090/studio  (upload)  and  /  (viewer)
```

`setup.sh` checks `uv`, `node`, `npm`, `ffmpeg` and `ffprobe` **before** installing
anything — a missing `ffprobe` otherwise surfaces much later as what looks like a
corrupt video. It never overwrites an existing `.env`. It finishes by printing which
perception backends this machine can actually use, resolved through the same function
the upload form calls, so the two cannot disagree.

`start.sh` serves the newest run under `runs/` and rebuilds the viewer only when
`viewer/src` is newer than the bundle. `bash start.sh --dev` runs Vite with hot
reload instead and pins the API to `:8000`, because that proxy target is hard-coded
in `viewer/vite.config.ts`.

Nothing processed yet? The studio still works — upload there. Or build the committed
synthetic sample, which has known ground truth:

```bash
uv run python scripts/make_sample.py
uv run pos run \
  --video samples/road/road.mp4 \
  --gpx   samples/road/track.gpx \
  --truth samples/road/truth.json \
  --out   run
uv run pos serve --run run --port 8090
```

Expected:

```
60 keyframes | 30.0s | origin 12.97160, 77.59460 | heading 90deg
68 detections -> 29 findings
Street Quality Index: 74.7/100  (grade C)
47 buildings, 90 roads
```

The first three lines are deterministic and should match exactly. The building
and road counts come from a **live Overpass query** against OpenStreetMap, so they
drift as people edit the map — this figure was 48 buildings a day earlier. A
different count there is not a broken install.

Port 8000 is a common conflict; pick another with `--port`. For viewer
development instead of a build: `cd viewer && npm run dev` (port 5173, proxies
`/api` to 8000).

### Prove it is correct, not just pretty

```bash
uv run python scripts/verify_sample.py
```

24 checks, non-zero exit on failure. On the committed sample:
**29/29 objects recovered, median position error 0.40 m, max 1.02 m.**

That number means something because the fixtures are built *independently*.
`scripts/make_sample.py` places objects in world coordinates and projects them to
pixels with its own copy of the pinhole model; the pipeline then inverts that
with `pos/geo.py`. Agreement is real evidence the geometry is right. Had
`pos/geo.py` built the fixtures too, the check would be circular and worthless.

---

## The pipeline

Twelve commands. `pos run` chains the core ones; each is also runnable alone, so
you never re-pay for a step you did not change.

```
video.mp4 + track.gpx
    │
    ├─ doctor ──────▶ preflight: VFR, clock offset, calibration sanity
    ├─ ingest ──────▶ keyframes, each with a real lat/lon/heading
    ├─ perceive ────▶ per-frame findings + boxes + severity + reasoning
    ├─ localize ────▶ 2D box ⇄ ground plane ⇄ GPS → absolute lat/lon
    ├─ cluster ─────▶ repeat sightings merged; absences inferred
    ├─ score ───────▶ 20 m segments, Quality Index, heatmap colour
    ├─ twin ────────▶ OpenStreetMap buildings + roads
    ├─ depthcloud ──▶ point cloud from the video, on CPU
    ├─ pointcloud ──▶ point cloud from a lingbot-map GPU pass
    ├─ basemap ─────▶ satellite imagery under the route
    ├─ verify ──────▶ re-examine the shakier findings (opt-in)
    ├─ kml ─────────▶ self-contained KMZ for Google Earth
    ├─ report ──────▶ paginated PDF inspection report
    └─ serve ───────▶ FastAPI + viewer + SSE live stream
```

| Command | Does |
|---|---|
| `pos domains` | list inspection taxonomies |
| `pos doctor` | **run this first** — catches silent failures before you spend anything |
| `pos ingest` | ffmpeg keyframes + GPX interpolation + heading |
| `pos perceive` | run a detector over every keyframe |
| `pos localize` | project boxes onto the ground plane |
| `pos cluster` | dedupe into findings; infer absences |
| `pos score` | segment the route and score it |
| `pos twin` | fetch OSM context geometry |
| `pos depthcloud` | CPU point cloud via monocular depth |
| `pos pointcloud` | convert lingbot-map NPZ predictions |
| `pos basemap` | fetch + stitch satellite imagery for the ground plane |
| `pos segment` | segment the drivable carriageway in every keyframe |
| `pos video` | render the drive as an annotated MP4: road mask, defects, HUD |
| `pos verify` | second-pass confirmation of weak findings. **Opt-in — see the measurement below** |
| `pos kml` | export a KMZ for Google Earth |
| `pos report` | write a PDF inspection report |
| `pos import-csv` | import findings from a CSV someone else produced — no detector needed |
| `pos export-csv` | export findings as a CSV with chainage, severity and confidence |
| `pos run` | ingest → perceive → localize → cluster → score → twin |
| `pos serve` | serve a run directory; `--studio` adds the upload page |

**Perception is cached** by image + prompt + model hash. Re-clustering after a
radius tweak, or re-scoring after changing weights, costs nothing.

---

## Running on your own video

### The short way — the studio

```bash
bash start.sh          # then open http://127.0.0.1:8090/studio
```

Drop in a video and its GPX, press Process, watch the pipeline log stream past, and
open the result. Clock offset is detected server-side; calibration can be solved from
the clip itself (`camera = auto`) and is validated before it is used. Every run under
`runs/` stays browsable at `?run=<name>` without a restart.

The page prints the actual calibration numbers for whichever camera you pick and says
plainly what picking the wrong one costs — it is the one judgement the form cannot
make for you.

### Already have the findings? Attach a CSV

If the defects were surveyed by someone else — a field team's spreadsheet, or a run
exported with `pos export-csv` and edited — attach it as the third file. It
**replaces detection entirely**: the rows already carry a position, so nothing is
detected and nothing is projected. No detector model, no API key, and it finishes in
a couple of minutes instead of an hour.

The video and GPX are still required, and still do their work: they produce the
route, the keyframes, the satellite basemap, the OSM buildings and the playable clip.
So a CSV import gets the full viewer — chase camera, basemap, 3D buildings, evidence
panel — not a bare scatter of markers.

```csv
object_id,frame,time,defect_name,lat,lon,chainage_km,frame_url,severity,confidence
1,00002,0:01.839,Waterlogging,28.65715505,77.20363533,0.002,frames/00002.jpg,3,0.410
2,00004,0:01.839,Garbage Accumulation,28.65717255,77.20361470,0.006,,3,0.594
```

- `time` is **video** time — `M:SS`, `H:MM:SS` or plain seconds. Sub-second matters:
  scoring assigns a finding to the segment whose time window contains it.
- `lat` / `lon` are decimal degrees. Blank borrows the nearest keyframe's position.
- `defect_name` is matched against the domain taxonomy's **labels** first, then its
  keys — so "Danger Element" resolves to `hazard` rather than inventing a class.
- `severity` (1–5) and `confidence` (0–1) are optional but drive the Quality Index,
  which is `weight × severity × confidence`. Omit them and every finding imports at
  severity 3 / confidence 1.0, and the index is indicative rather than reproduced.
  The importer says so when that happens.

Anything it cannot read — a bad cell, a duplicate `object_id`, a class the taxonomy
does not define — is counted and reported rather than dropped or thrown. Same on the
CLI:

```bash
uv run pos import-csv defects.csv --run runs/myrun
uv run pos export-csv --run runs/myrun --out defects.csv
```

Round-trip is lossless: exporting a scored run and re-importing it reproduces the
same findings, classes, severities, timestamps and Quality Index.

### The scripted way

```bash
# 1. preflight — read the offset it prints
uv run pos doctor --video my.mp4 --gpx my.gpx --camera dashcam

# 2. everything else, in one go
bash scripts/run_pipeline.sh my.mp4 my.gpx dashcam myrun <offset> 8090
```

`scripts/run_pipeline.sh` chains doctor → `pos run` → `pos depthcloud` →
`pos basemap` → `pos kml` + `pos report` → viewer build, then prints the score and
per-class counts and leaves you a `.kmz` and a `.pdf`. It hard-codes `--fps 1` and
`--heading-baseline 15`, and it refuses to silently pretend it has the full
ensemble: with no ONNX model it falls back to `cosmos` and says so; with no API
key, to `onnx`, and says what that loses. Override the model path with
`POS_ONNX=/path/to/model.onnx`.

### The long way

Do it step by step when you want to re-run a single stage. Steps 1 and 2 are what
separate a usable result from a plausible-looking wrong one.

```bash
V=road_videos/test_4/myclip.mp4
G=road_videos/test_4/myclip.gpx

# 1. PREFLIGHT — reports the time offset and flags silent problems
uv run pos doctor --video "$V" --gpx "$G" --camera dashcam

# 2. CALIBRATE — from vehicle motion + GPS, no tape measure needed
uv run python scripts/calibrate_from_motion.py \
  --video "$V" --gpx "$G" \
  --height 1.10 --min-speed 1.2 --dt 0.90 \
  --roi-top 0.36 --roi-bottom 0.88 --horizon 190 \
  --time-offset <from doctor> \
  --write configs/camera/mycam.yaml

# 3. VERIFY the calibration took
uv run pos doctor --video "$V" --gpx "$G" --camera mycam

# 4. RUN
uv run pos run --video "$V" --gpx "$G" \
  --camera mycam --domain road_pci --backend ensemble \
  --fps 1 --time-offset <from doctor> --heading-baseline 15 \
  --out runs/myclip

# 5. POINT CLOUD (optional, CPU, ~1.5 s/frame)
uv run pos depthcloud --run runs/myclip --camera mycam --stride 5

# 6. VIEW
uv run pos serve --run runs/myclip --port 8090
```

### What `pos doctor` catches

Every one of these fails *silently* — the pipeline runs, produces findings, and
every position is quietly wrong.

| Check | Why it matters |
|---|---|
| **Clock offset** | Computed from the video's `creation_time` vs the GPX start. On real footage this was **−2.71 s ≈ 6 m** of error on every finding. |
| `creation_time` start-vs-end | Many cameras write it when the file is *finalised*. Doctor tests both readings and keeps whichever overlaps the GPS. |
| Variable frame rate | Most phones record VFR. Handled, but reported. |
| Uncalibrated camera | Warns if the config is still the shipped defaults. |
| `pitch_offset_frac == 0` | Assumes the horizon sits exactly at mid-frame. Rarely true. |
| Track shorter than video | Fails outright — the GPX does not cover the clip. |
| Implausible speed | Stationary stretches give unusable headings. |
| API key present | Otherwise only `--backend mock` works. |

### Flags worth knowing

| Flag | When |
|---|---|
| `--fps 1` | halves API cost. At walking or bike pace, 1 fps is still ~2 m spacing |
| `--time-offset` | from doctor. **Always pass it** |
| `--heading-baseline 15` | **essential when walking or cycling** — see below |
| `--classes-per-call 4` | more recall, much worse precision. Off by default |
| `--model-path` | point `onnx`/`ensemble` at a different ONNX file |

Before spending a full run on a new clip, smoke-test perception on a few frames.
`--limit` is on `pos perceive`, not `pos run` — `run` writes a complete run
directory, and a truncated one would have findings that disagree with its own
score:

```bash
uv run pos ingest --video "$V" --gpx "$G" --camera mycam --out runs/probe
uv run pos perceive --run runs/probe --backend ensemble --limit 5
```

**`--heading-baseline`** deserves explanation. Heading comes from the direction
between two GPS fixes, so its accuracy is (distance travelled between them)
against (GPS noise). Measured on synthetic tracks with 3 m of noise:

| | ±2 samples (old) | 15 m baseline |
|---|---|---|
| driving 8 m/s | 6.4° | 6.4° |
| **walking 1.4 m/s** | **36.4°** | **11.6°** |
| **walking, 5 m GPS noise** | **55.3°** | **16.9°** |

A 30° heading error throws a finding 8 m away about **4 m sideways** — the
difference between carriageway and footpath. Use 8 (default) for driving, 15–20
for walking or cycling. The cost is that heading lags through corners.

---

## Camera calibration: the thing that decides accuracy

Forward range is `Z = f·h / (v − cy)`. It depends on focal length `f`, camera
height `h`, and the horizon row `cy`. Get these wrong and every distance is wrong
proportionally — silently, with no error anywhere.

Concretely: **KITTI's real vertical FOV is 29.13°, the shipped default is 58°.**
Same pixel, 2.6 m vs 6.2 m.

Three ways to get them, best first.

### 1. Two ground markers — `scripts/calibrate.py`

Most accurate, but needs the camera still mounted. Park, put a marker at 5 m and
another at 15 m dead ahead, grab a frame, read off the pixel row where each meets
the ground:

```bash
uv run python scripts/calibrate.py \
  --height 1.35 --frame f.jpg --near 5:612 --far 15:451 \
  --write configs/camera/mycam.yaml
```

Closed-form from two equations. It round-trips your own measurements back through
`pos/geo.py` and prints the re-projected distances, so you can see it reproduce
the numbers you fed it.

Validated by synthesising two marker rows from a known camera and inverting them:
exact sub-pixel rows recover **vfov 58.00° / pitch 0.0000 from a true 58.0 / 0.0**.
Rounding those rows to whole pixels — which is what you will actually do reading
them off a frame — gives 57.82°, a 0.3% error. So eyeballing the row to the
nearest pixel is good enough; the dominant error is your height measurement, not
your row-reading.

### 2. Vehicle motion + GPS — `scripts/calibrate_from_motion.py`

For footage already shot. Drive forward at speed `s` for `dt` and a fixed road
feature moves from `Z₁` to `Z₂ = Z₁ − s·dt`. Track it between two frames:

```
s·dt = f·h · ( 1/(v₁ − cy) − 1/(v₂ − cy) )
```

Every tracked feature is one equation; GPS supplies `s`. **Forward range depends
on f and h only through their product**, which this measures directly — so
distances hold regardless of how you split it between the two. The split affects
only lateral offsets.

```bash
uv run python scripts/calibrate_from_motion.py \
  --video v.mp4 --gpx t.gpx --height 1.10 \
  --min-speed 1.2 --dt 0.90 --horizon 190 \
  --roi-top 0.36 --roi-bottom 0.88 \
  --write configs/camera/mycam.yaml
```

Tuning notes, learned the hard way:

- **`--horizon` is often required.** Fitting both the product *and* `cy` needs
  tracked features spread across a wide band of rows. At low speed only near
  features move enough to track, they land in a narrow band, and `cy` becomes
  unidentifiable — the search then pins to whichever bound it started from
  (observed on a 2.3 m/s bike clip: `cy` pinned at both −648 and +108, giving
  vfov 2.7° and 20.1°). Read the horizon off a frame and pass it.
- **`--min-speed`** defaults to 3.0 m/s. Walking is 1.4, bike ~2.3 — lower it or
  every frame pair is skipped and you get "no usable pairs".
- **`--dt`** must give roughly 2 m of travel: 0.3 s at driving speed, 0.9–1.0 s
  slower.
- **`--roi-bottom`** must stay above the bonnet when shooting from inside a car.
- **`--exclude-cols 0.42,0.72`** skips a vehicle ahead — it moves with you, so it
  has zero parallax and corrupts the fit.
- **Re-fit per clip.** Two clips from the same bike an hour apart differed by
  **28%** in fitted product (2116 vs 2708 px·m) — mount shift or road gradient.

### 3. A dataset with published intrinsics — `scripts/import_kitti.py`

```bash
B=https://s3.eu-central-1.amazonaws.com/avg-kitti          # public, no account
curl -O $B/raw_data/2011_09_26_calib.zip
curl -C - -O $B/raw_data/2011_09_26_drive_0002/2011_09_26_drive_0002_sync.zip
unzip -q '*.zip'

uv run python scripts/import_kitti.py \
  --drive 2011_09_26/2011_09_26_drive_0002_sync --calib 2011_09_26 \
  --out samples/kitti
```

Derives vfov and pitch from `P_rect_02` — exact, not estimated — and writes
`configs/camera/kitti.yaml`. Useful for validating the geometry against a
reference dataset. KITTI is **CC BY-NC-SA 3.0: non-commercial**, so validate with
it, don't ship its frames.

---

## Perception backends

| Backend | Needs | Use |
|---|---|---|
| `mock` *(default)* | nothing | Offline fixtures. A fresh clone works with no key, no network, no GPU. |
| `cosmos` | `NVIDIA_API_KEY` | Hosted VLM. Broad class coverage, human-readable reasoning. Cached. |
| `onnx` | the post_cons model | **Local, free, offline.** Transverse + longitudinal cracking, tight boxes. ~2.3 s/frame CPU. |
| **`ensemble`** | both | **Best results.** YOLO geometry + VLM coverage. |
| `locate-anything` | CUDA GPU, ~12 GB | Optional box refiner. **NVIDIA non-commercial — research only**, gated behind `POS_ACCEPT_NONCOMMERCIAL=1`. |

`mock` stamps every fabricated finding `[SYNTHETIC FIXTURE]` so it can never be
mistaken for real perception.

### Why the ensemble wins

The two models fail in opposite directions. Measured on the same real footage:

| class | VLM | ONNX | **Ensemble** |
|---|---|---|---|
| pothole | 2 | **8** | 8 |
| waterlogging | **5** | 0 | 5 |
| ravelling | 0 | **1** | 1 |
| garbage / hazard / footpath | 0 | 0 | **8** |

| | detections | findings | Index | grade | classes |
|---|---|---|---|---|---|
| VLM only | 16 | 7 | 74.4 | C | 2 |
| ONNX only | 27 | 9 | 59.7 | D | 2 |
| **Ensemble** | **49** | **24** | **15.9** | **F** | **7** |

The ONNX model finds 4× the potholes; only the VLM sees water, refuse and
hazards. **Merge rule:** same class and IoU ≥ 0.45 → fused, YOLO's box kept (it
is the better localiser), confidences combined, and *both* explanations retained
so the evidence panel shows two independent models concurred. Different classes
are never merged — a pothole inside a waterlogged stretch is two separate facts.
A VLM box covering more than 35% of the frame with no YOLO support is demoted to
≤0.45 confidence and labelled *"area indication, not a point"*.

### Sliced inference — `--tile 640`

```bash
uv run pos perceive --run runs/x --backend ensemble --model-path "$ONNX" --tile 640
```

The net wants 640×640; a road frame is 1920×1080. Letterboxing therefore shrinks
everything **3×**: the median detected object is 108 px in-frame but only 36 px at
the input, and 21% land under 20 px — about the limit the model can resolve.
Anything further down the road is smaller still, which is exactly the recall being
lost.

Running the model on 640 px tiles involves no downscaling at all. Measured on 25
real Kohima frames, same model, same frames:

| | detections | median min-dimension | smallest | median confidence | wall time |
|---|---|---|---|---|---|
| single pass | 6 | 137 px | 73 px | 0.38 | 16 s |
| **`--tile 640`** | **17** (2.83×) | **50 px** | **15 px** | 0.39 | 116 s (7.2×) |

The shift toward *smaller* objects is the point — it is recovering distant defects,
not inventing large ones. Confidence is unchanged, so the extra detections are not
low-quality padding, and two classes appear (`depression`, `alligator_crack`) that
the single pass missed entirely.

**The full frame is kept alongside the tiles.** Tiles alone *lose* large objects: a
defect spanning a tile boundary fragments and each piece falls below threshold —
measured going 1 → 0 on two frames. The union lost nothing on any frame tested.

Off by default: 7× the CPU time is real, even though it costs no API calls.

### On roboflow/supervision

Worth addressing directly, since it is the obvious library for this. It was
evaluated and **the idea was adopted while the dependency was declined.**

| supervision feature | verdict |
|---|---|
| `InferenceSlicer` (SAHI) | **The right idea** — it is what `--tile` above implements. |
| `ByteTrack` | Assumes video-rate frames with small inter-frame motion. Keyframes here are ~2 m apart and an approaching object's box moves a long way, so its constant-velocity Kalman prior does not hold. The tracker in `pos/cluster.py` uses a domain prior instead — objects move *down* the frame as you approach — and already lifted corroborated objects 5 → 22 on real footage. |
| metrics (mAP, PR) | `scripts/score_perception.py` already does this and reproduces the published figures. |
| annotators, zones | Not needed; boxes are burned in with Pillow, and zone counting does not apply to a moving survey. |
| dataset IO (COCO/YOLO) | Genuinely useful *later*, for the labelled eval set. Not needed yet. |

It would add **15 packages including opencv-python, matplotlib and scipy**. The one
piece that addresses the measured bottleneck is ~40 lines reusing the `nms()` and
`letterbox_params()` already in `pos/perception/onnx_yolo.py`, and writing it here
keeps the class-wise NMS this project deliberately uses (overlapping boxes of
*different* distress types are legitimate — a pothole inside an alligator-cracked
patch). For a project whose selling point is a lean CPU-only footprint, 15 packages
for 40 lines is the wrong trade.

If dataset interchange or a proper tracker becomes the bottleneck, revisit it —
supervision is MIT, so there is no licence obstacle.

### Which hosted model — measured, not assumed

`models.list()` advertises models an account may not be entitled to invoke. On
the account tested, `cosmos-reason2-8b`, `vila` and `neva-22b` all appear in the
listing but **404 on call** — which is why `CosmosDetector` has a fallback chain
and a `probe()` step.

| Model | Verdict |
|---|---|
| **`nvidia/llama-3.1-nemotron-nano-vl-8b-v1`** | **Default.** Boxes within a few units of truth. Conservative. |
| `nvidia/nemotron-nano-12b-v2-vl` | Finds more, hallucinates boxes including whole-frame `[0,0,1000,1000]`. |
| `meta/llama-3.2-90b-vision-instruct` | Obeys the JSON shape, emits degenerate `[0,0,0,0]`. |
| `meta/llama-3.2-11b-vision-instruct` | Ignores the JSON contract, returns prose. |

Text-only models (`llama-3.3-70b-instruct` and similar) cannot be used at all —
no vision encoder.

---

## Domains: the taxonomy is data, not code

Five ship. Adding one means writing a YAML file — no Python.

```bash
uv run pos domains
# building_facade       Building / Facade Inspection    (11 classes)
# construction_safety   Construction Site Safety        (12 classes)
# road                  Road & Street Assessment        (15 classes)
# road_pci              Road Distress (PCI + Ensemble)  (18 classes)
# utility_pole          Overhead Line & Pole Audit      (12 classes)
```

| Domain | Looks for |
|---|---|
| `road` | potholes, cracking, pavement distress, waterlogging, footpath presence/damage/obstruction, streetlights, faded markings, signage, encroachment, garbage, hazards |
| **`road_pci`** | the full 11-class PCI taxonomy (alligator / edge / longitudinal / transverse cracking, bleeding, depression, patching, pothole, ravelling, rutting, shoving) **plus** VLM-only classes and absence rules. The ONNX model currently covers two of them (transverse + longitudinal cracking); the VLM covers the rest, so **use this with `--backend ensemble`.** |
| `construction_safety` | missing hard hat / hi-vis / harness, unprotected edges, missing barricades, unsafe scaffold, blocked egress, exposed rebar, worker-near-plant |
| `building_facade` | structural vs surface cracking, spalling, exposed reinforcement, seepage, drainage defects, glazing damage, render loss, unsafe projections |
| `utility_pole` | leaning poles, conductor sag, conductors down, damaged insulators, transformer defects, vegetation encroachment, unauthorised taps, low service drops |

Each class declares:

```yaml
- key: pothole
  label: Pothole
  geometry: point          # point | segment | area
  cluster_radius_m: 3.0    # how far apart two sightings are different objects
  weight: 4.5              # score penalty. 0.0 = asset: inventoried, not penalised
  alert: true              # red alert card in live mode
  color: "#991b1b"
  hint: >                  # this text goes into the VLM prompt
    A bowl-shaped cavity with broken edges and visible depth. Not a dark
    stain, shadow, patch or manhole cover.
```

Write hints as **"X, not Y"** — the dominant failure is a model confidently
reporting a lookalike, and naming the lookalike suppresses it. `prompt_context`
sets the persona and evidence standard and is the single biggest quality lever in
the file.

---

## Absence: defects that are things missing

"No street lighting" is not something you can point at. Asking a per-frame
detector *"is there a lighting gap?"* asks it to reason about what lies outside
the frame, and models answer by guessing — prompting for absence classes directly
produced **12 hazards where 1 existed**.

So absence is **inferred from coverage over distance**. `pos/absence.py` walks the
route, places each detected asset at its route distance, and emits a finding for
every gap exceeding a threshold:

```yaml
absence:
  - key: streetlight_missing
    label: No Street Lighting
    asset: streetlight      # the presence class whose absence we infer
    min_gap_m: 40.0         # must exceed normal 25-35 m pole spacing
    weight: 2.0
  - key: footpath_missing
    asset: footpath
    min_gap_m: 30.0
    weight: 2.5
    alert: true
```

Two safeguards:

- **Never confuses "we didn't look" with "it isn't there."** The ONNX model has no
  streetlight class, so under `--backend onnx` every route would look unlit. Rules
  whose asset the active backend cannot detect are skipped, and it says so.
- **Confidence is capped at 0.90.** A missed detection is indistinguishable from a
  real gap, so confidence scales with how far past threshold the gap runs. On a
  67 m route it came out 0.55 — deliberately modest, because that is a short
  stretch on which to claim a road is unlit.

In the UI these appear as a wide hovering ring (not a pin — absence is a property
of a stretch), a `MISSING` tag in the legend, an **Asset coverage** panel with
✓/✕ per asset, and an evidence panel with **no bounding box** and an explicit
banner. Drawing a box would assert a location that does not exist.

---

## Point clouds: two routes

| | `pos depthcloud` | `pos pointcloud` |
|---|---|---|
| Needs | **CPU only** | CUDA GPU (RunPod / Colab) |
| Time | ~1.5 s/frame — 110 s for 45 frames | ~10–25 min rented |
| Model | Depth Anything V2 Small, **Apache-2.0** | lingbot-map, Apache-2.0 |
| Quality | per-frame back-projection; thickens on overlap | multi-frame fusion with drift correction |

### CPU route

```bash
uv run pos depthcloud --run runs/myclip --camera mycam --stride 5 --max-range 25
```

Monocular depth is only relative — "nearer than that", not "7 m away". Your
calibration supplies the missing scale: any pixel below the horizon has a
computable true distance, so it samples ground pixels, pairs each one's predicted
inverse depth `d` with its known distance `Z`, and least-squares fits

```
1/Z = a·d + b
```

Two unknowns, hundreds of samples per frame. That converts the whole map to
metres; each pixel then back-projects, rotates by heading, and offsets by its GPS
fix, so all frames land in one local ENU frame. On real footage: **3.5 M raw →
352 k points**, 38% within ±0.4 m of ground level, with the GPS track running
down the middle of the reconstructed carriageway. The model downloads itself on
first use (37 MB, cached under `~/.cache/physicalos/depth`).

### GPU route

```bash
# on a RunPod pod (any 24 GB+ GPU), after runpodctl send
bash lingbot_gpu_pass.sh myclip.mp4 out

# back home
runpodctl receive <CODE>
mkdir -p lingbot_out && tar xzf predictions.tar.gz -C lingbot_out
uv run pos pointcloud --run runs/myclip --preds lingbot_out
```

The script runs a `--first_k 20` smoke test before committing the full clip,
passes lingbot's own `outdoor_drive.yaml` preset (max_depth 250 m, sky masking,
follow-then-birdeye camera), downloads both `lingbot-map.pt` **and**
`skyseg_batch.onnx` (the config requires it), and tars only the NPZ for transfer.
`pos pointcloud` then fits lingbot's solved trajectory to your GPX with a
similarity transform — scale, yaw, translation, yaw only because letting roll and
pitch float on a noisy monocular trajectory tilts the whole street — so the cloud
lands in the same local ENU metres as everything else.

Skip both and the viewer just shows the OSM twin. Nothing breaks.

---

## The viewer

React 19 + React-Three-Fiber, served by FastAPI out of `viewer/dist`.

| Feature | What it does |
|---|---|
| **Evidence panel** | Click a marker → the real frame, box drawn on it, the model's reasoning, lat/lon, range, and every other frame that saw the same object. Opens on the *closest* sighting, since the first is the furthest and least legible. |
| **Quality heatmap** | Road segments coloured green→red. Click one to jump to its worst finding. |
| **Markers** | Cone = defect (apex at the spot), sphere = asset, wide ring = absence. Height scales with severity. |
| **Video panel** | The source clip beside the map, synced to the timeline, **always 1×**. |
| **Side-by-side** | `◫ side by side` splits map and video so neither hides the other. Pure CSS, so the 3D canvas resizes rather than remounting — camera and scene survive the toggle. |
| **Timeline** | Scrub or play; findings appear as they were first seen. |
| **Live drive** | Streams findings over SSE with alert cards. The same endpoint replays a saved run — one code path, not two. |
| **Asset coverage** | ✓/✕ per asset; click to jump to the missing stretch. |
| **Layers** | point cloud / buildings / heatmap / markers / route. |

All three clocks — video element, SSE stream, timeline scrub — run at **1×** so
they stay in step. Anything faster leaves the footage lagging its own markers.

Hard-refresh (Ctrl+Shift+R) after a rebuild: the bundle filename changes and all
runs share one origin cache.

---

## Google Earth export

```bash
uv run pos kml --run runs/kohima4 --out kohima4.kmz
```

One self-contained file — no API key, no network, evidence photos and marker icons
packed inside. Opens in **Google Earth Pro** (free), Earth Web, QGIS and ArcGIS.

| Folder | What it is |
|---|---|
| Findings | one placemark each, severity-coloured, popup carries the evidence photo with the box burned in plus the model's own sentence |
| Missing assets | a **line along the gap**, not a pin — absence is a property of a stretch, so a point would claim a location that does not exist |
| Quality heatmap | 20 m segments, green→red, clamped to ground |
| Route | the driven line plus a `gx:Track`, which makes Google Earth show its **time slider** and replay the drive |
| Tour | a `gx:Tour` flythrough of the worst findings in route order |

On the 279 m Kohima run: **3.0 MB, 57 findings, 13 segments, 2 missing-asset
stretches, 57 photos, 41 tour stops.**

**Why this and not Cesium + Google's photorealistic 3D tiles.** Those tiles cover
about **2,500 cities, high-density urban only**. Paris has coverage; rural Kohima
and Urida have none, so that route would look spectacular on one clip and be a
flat grey plane on the others — and it needs a GCP project with billing. What *is*
available everywhere is Google Earth's satellite imagery and worldwide 3D terrain,
which is what this export relies on.

**For a demo video:** Google Earth Pro's *Tools → Movie Maker* records the tour
straight to MP4. That beats screen-capturing a browser.

Google Earth cannot show the point cloud (KML has no representation for 400,000
loose points), the SSE live alerts, the synced video panel, or the coverage panel.
Those stay in the dashboard — the two are complements, not substitutes.

Two conventions this code gets right, because both fail silently:

- **KML colour is `aabbggrr`, not `#rrggbb`.** Byte-swapping wrong inverts the
  heatmap — greens exactly where the worst potholes are. Unit-tested:
  `_kml_color("#ea580c") == "ff0c58ea"`.
- **KML coordinates are `lon,lat,alt`** — longitude first, the opposite of every
  other file here. One helper writes that order.

---

## Satellite basemap

```bash
uv run pos basemap --run runs/kohima4 --zoom 18
```

Replaces the dark grey ground plane with the actual place, which is what lets a
viewer judge whether a finding sits on the road or in a field beside it.

Tiles are fetched and **stitched server-side once** into a single texture, cached
in the run directory. A tile pyramid in three.js would mean a loader, LOD juggling
and hundreds of requests per page view; a route is a few hundred metres, so one
JPEG does it. On the Kohima run: 8 tiles → 1024×512 px at **0.54 m/px** covering
551 × 276 m.

**Licence — read this.** Imagery is not public domain.

| `--provider` | Terms |
|---|---|
| `esri` *(default)* | Esri World Imagery. Free for **non-commercial** use, attribution required. Worldwide sub-metre coverage, rural India included. |
| `osm` | OpenStreetMap, ODbL. A **street** map, not satellite — useful for checking alignment. Do not bulk-fetch. |
| `mapbox` | Needs `--token`. Clear commercial path. |

**Google's tiles are deliberately unavailable**: they may not be used outside
Google's own APIs. Use `pos kml` and Google Earth for Google imagery.

The attribution string travels in `basemap.json` and the viewer renders it from
there. Removing it is a licence breach, not a styling choice.

Verified independently: drawing OSM road geometry on the fetched imagery lands it
on the visible track, and the GPS route follows it — three independent sources
(Esri imagery, OSM vectors, your GPX) agreeing. If the route sits beside the road
by a constant offset, suspect the imagery's own georeferencing before the
pipeline's.

---

## PDF report

```bash
uv run pos report --run runs/kohima4 --out kohima4.pdf
```

Cover with the score, per-class counts and asset coverage; the segment table
worst-first as a dispatch order; then **one page per finding** with its photograph,
box, coordinates, range and the model's reason verbatim. The Kohima run gives 59
pages / 10.4 MB.

Every page footer names the **perception backend**, and a `mock` run is stamped
`*** SYNTHETIC FIXTURES - NOT REAL PERCEPTION ***`. A report that looked
authoritative while hiding which model produced it would launder a guess into a
record.

---

## Run directory and data contracts

Self-describing. The viewer needs nothing else.

```
runs/myclip/
├── manifest.json      run metadata + score summary + domain taxonomy
├── frames.json        keyframe → lat/lon/heading/timestamp
├── detections.ndjson  raw per-frame observations (one JSON per line)
├── findings.json      deduplicated assets, each with its evidence frames
├── segments.json      20 m segments with Quality Index + heatmap colour
├── coverage.json      per-asset coverage, driving the absence panel
│                      findings.json also carries pos_method / pos_residual_m /
│                      n_rays / parallax_deg per finding, and anchor per detection
├── twin.json          OSM buildings + roads
├── cloud.ply          point cloud, if generated
├── basemap.jpg        stitched satellite texture      (pos basemap)
├── basemap.json       its extent in local metres + attribution
├── verification.json  second-pass verdicts, incl. a dry_run flag  (pos verify)
├── export.kmz         cached Google Earth archive     (served by /api/kml)
├── review.mp4         annotated render                (pos video)
├── defects.csv        findings as a spreadsheet       (pos export-csv)
├── frames/00042.jpg   the evidence images
└── .cache/            VLM responses and Overpass results, keyed by content hash
```

`manifest.backend` records what produced the findings, and is load-bearing: `mock`
means **synthetic fixtures**, `csv_import` means they came from a supplied CSV, and
`pos verify` and the PDF footer both read it. A run imported from CSV has no
`detections.ndjson` — there were no per-frame observations to record.

Field names, as defined in `pos/schema.py`:

| File | Fields |
|---|---|
| `frames.json` | `frame_id`, `t_sec`, `ts`, `lat`, `lon`, `heading_deg`, `width`, `height`, `path` |
| `detections.ndjson` | `frame_id`, `cls`, `box`, `severity`, `confidence`, `evidence`, `range_m`, `lat`, `lon` |
| `findings.json` | `finding_id`, `cls`, `lat`, `lon`, `severity`, `confidence`, `t_sec`, `evidence[]` |
| `segments.json` | `seg_id`, `start`, `end`, `length_m`, `quality_index`, `color` |
| `coverage.json` | `asset`, `label`, `seen`, `n_sightings`, `max_gap_m`, `detectable` |

```json
// manifest.json — values from the synthetic sample run
{
  "run_id": "run", "domain": "road", "created": "2026-07-25T16:56:25Z",
  "origin": [12.9716, 77.5946], "n_frames": 60, "n_detections": 68,
  "n_findings": 29, "duration_sec": 30.0, "backend": "mock",
  "summary": { "quality_index": 74.7, "grade": "C", "route_length_m": 235.8 },
  "has_video": true, "has_pointcloud": false
}
```

**Box convention**, defined once in `pos/schema.py` and nowhere else:
`[x1, y1, x2, y2]`, normalised **0–1000**, origin **top-left**. The viewer
converts to percentages in exactly one function, `EvidencePanel.tsx:pct` — this
is the easiest thing in the project to get wrong.

**Two clocks, kept separate.** `t_sec` is *video* time — what the timeline scrubs
and what `video.currentTime` needs. `ts` is a *GPS* wall-clock stamp (ISO-8601
UTC). `--time-offset` shifts only the GPS lookup. Folding it into `t_sec` put the
footage out of step with its own markers by exactly the offset.

### HTTP API

| Endpoint | Returns |
|---|---|
| `/api/manifest` | run metadata + the domain taxonomy (labels, colours, alert and absence flags) |
| `/api/findings` | all findings with evidence |
| `/api/segments` | scored segments |
| `/api/frames` | keyframe index |
| `/api/coverage` | per-asset coverage rows |
| `/api/twin` | OSM buildings + roads |
| `/api/video` | the source clip, **with HTTP Range support** |
| `/api/cloud.ply` | binary point cloud |
| `/api/basemap` | satellite extent + attribution, or `{"available": false}` |
| `/api/basemap.jpg` | the stitched texture |
| `/api/kml` | KMZ, built on demand and rebuilt when `findings.json` is newer |
| `/api/report.pdf` | PDF report, same staleness rule; `?download=1` forces a save |
| `/api/review.mp4` | the annotated render, when `pos video` has run |
| `/stream?speed=1` | SSE findings in time order, real-time by default |

Every run-scoped endpoint takes `?run=<name>` to serve any run under `--runs-dir`,
so a new upload is viewable without a restart.

With `pos serve --studio`, four more:

| Endpoint | Does |
|---|---|
| `/studio` | the upload page — video + GPX, optionally a findings CSV |
| `POST /api/upload` | accepts `video`, `gpx`, optional `csv`, starts a job, returns 202 |
| `/api/runs` | every processed run, with findings count, index and grade |
| `/api/jobs/{id}/events` | pipeline progress as SSE, so the page needs no polling loop |

---

## How positions are computed

Monocular ground-plane projection (`pos/geo.py`). For a box's bottom-centre
pixel `(u, v)`:

```
f  = (H/2) / tan(θv/2)         focal length in pixels
α  = atan((v − cy) / f)        angle below the optical axis
Z  = h / tan(α)                forward ground distance  ( = f·h/(v − cy) )
X  = (u − cx) · Z / f          lateral offset
```

then rotated by the frame's heading ψ and offset from its GPS fix:

```
East  =  X·cos ψ + Z·sin ψ
North = −X·sin ψ + Z·cos ψ
```

### Sample by distance, not by time

```bash
uv run pos run ... --spacing-m 2.0      # instead of --fps
```

`--fps` couples sampling to time, but everything downstream cares about distance.
Measured on real footage:

| | keyframe spacing | objects seen once |
|---|---|---|
| bike, 2.0 m/s, 1 fps | 2.6 m | 27 / 57 |
| **vehicle, 6.0 m/s, 1 fps** | **6.4 m (4.7–18.7)** | **153 / 158** |
| same vehicle, `--spacing-m 2.0` | **2.12 m (2.0–2.6)** | — |

Two separate problems with a fixed fps. Objects get seen **once**, so there is
nothing to corroborate or track. And coverage is **non-uniform** — that 18.7 m gap
is a stretch where the vehicle sped up and the survey effectively skipped 18 m of
road. Distance-based sampling fixes both, at the cost of more frames (573 vs 193
on that clip, a re-ingest not a re-shoot since the source is 30 fps).

Stationary stretches contribute nothing, which is correct: twenty frames of a
stopped vehicle are twenty copies of one observation.

### How well is each position actually known?

Findings now carry their own accuracy rather than sharing one disclaimer:

| `pos_method` | meaning |
|---|---|
| `triangulated` | bearing rays from 2+ camera positions; `pos_residual_m` is a real metre figure |
| `ground_plane` | one projection, flat-ground assumption, ±2–4 m |
| `camera` | above the horizon, pinned to the camera, never ranged |

Honest limits:

- **±2–4 m** with a correctly measured camera height on a rigid mount. Not survey
  grade. Handheld or bike-mounted, expect ±3–5 m — pitch varies continuously and
  there is no per-frame correction.
- Assumes a **flat ground plane**. Hills and slopes introduce error; on Kohima's
  hills the calibration residual was ~1 m.
- Detections **above the horizon** cannot be ranged (a streetlight head, a
  third-storey crack). They are pinned to the camera with `range_m: null` rather
  than given a fabricated distance.
- Beyond `max_range_m` the projection is **refused, not guessed** — a few pixels
  above the horizon means hundreds of metres.
- **Area-type defects used to bunch toward the camera.** A box covering half the
  frame has its bottom edge at the frame bottom, so it localised to minimum range
  — 6 of 16 detections sat at the 2.4 m floor. Fixed: `area` and `segment` classes
  now project from the box **centre**, recorded as `Detection.anchor`.

### Tracking beats spatial clustering — measured

Detections are now associated **across adjacent keyframes** by box overlap and
motion, before any spatial clustering. This matters because the position error
(median 2.46 m between two projections of one object) is comparable to the pothole
cluster radius (3.0 m), so purely spatial clustering splits one pothole into two
findings. Tracking is immune to that: it matches on how the box moves, not on
where the projection landed.

| | presence findings | seen 2+ times | duplicates within 2× radius |
|---|---|---|---|
| Kohima, spatial only | 55 | 30 | 18 |
| Kohima, **track-then-cluster** | **50** | 28 | **11** |
| Urida, spatial only | 158 | 5 | 25 |
| Urida, **track-then-cluster** | **140** | **22** | **15** |

Same detections in every row — only the clustering changed. On the Urida clip,
objects with corroborating evidence went from 5 to 22.

One subtlety worth recording: tracks must also be merged **with each other**, not
just have singletons attached. A track broken in two by a missed detection is one
object, and an early version of this that only absorbed singletons made the
duplicate count *worse* than pure spatial clustering (2 → 6 on Kohima).

### Triangulation: works, but barely applies to forward-facing video

Two bearing rays from two camera positions fix a point without any ground-plane
assumption, so pitch error and ground flatness drop out. It is implemented
(`pos/geo.py:triangulate`) and gated on parallax and residual.

**It applies to far fewer findings than expected, and the reason is geometric.**
Planning this, parallax was estimated as `2·atan(baseline/2/range)` — which is only
valid when the baseline is *perpendicular* to the view. Driving toward a pothole the
baseline lies *along* the view, which is the degenerate case. Measured on real
footage: **median parallax 2.6°, not the 33.8° predicted**, and only 3 of 23 tracks
clear 8°.

So triangulation helps laterally-offset objects (roadside hazards, footpath,
poles) and does almost nothing for defects in the wheel path. It is kept because
it genuinely improves the cases it covers and refuses the rest rather than
silently substituting a worse number — `pos_method` says which you got.

Also worth knowing: with exactly 2 rays the residual is **0 by construction** (two
equations, two unknowns), so it only becomes a useful consistency filter at 3+
sightings.

A related idea was tried and **rejected**: fitting the monotonic drift in
per-sighting estimates to recover a range-scale error. On real tracks the implied
scale ranged 0.18–2.07 and implied position shifts up to +36 m — with 3 points and
2.7 m of noise the slope is unstable and dividing by it amplifies the noise.

---

## Measured results on real footage

| Run | Footage | Backend | Findings | Index | Note |
|---|---|---|---|---|---|
| `run` | synthetic sample | mock | 29 | 74.7 C | ground truth known; 0.40 m median error |
| `runs/e2e` | synthetic sample | cosmos | 22 | 87.5 B | live VLM: 64% precision, 48% recall |
| `runs/kitti_real` | Karlsruhe, real | cosmos | **0** | 100 A | correct — German asphalt is flawless |
| `runs/paris` | Paris, real | cosmos | **0** | 100 A | correct — "road in good condition" |
| `runs/kohima` | Kohima rural, 67 m | cosmos | 7 | 74.4 C | VLM alone: water yes, potholes mostly missed |
| `runs/kohima_onnx` | same | onnx | 9 | 59.7 D | ONNX alone: potholes yes, water invisible |
| **`runs/kohima_ens`** | same | **ensemble** | **24** | **15.9 F** | + absence inference |
| **`runs/kohima4`** | Kohima, 279 m | **ensemble** | **57** | **43.5 E** | 28 potholes, 14 waterlogging, 344 k-point cloud |

Two things this table says plainly:

- **Zero findings can be the correct answer.** KITTI and Paris returned nothing
  because those roads are sound. Asked openly, the model replied *"the road
  surface appears to be in good condition, with clear lane markings and no visible
  potholes or debris."* That is the pipeline working, not failing.
- **The index is length-normalised.** `kohima` scores worse than `kohima4` (15.9
  vs 43.5) despite being the shorter clip, because it concentrates 24 findings
  into 67 m while `kohima4` spreads 57 over 279 m.

**Known weakness: recall.** On the synthetic sample, 48%. Box precision degrades
on real photographs — of 5 real pothole photos the VLM got 4 classes right but
only 2 usable boxes, the rest returning whole-frame. The ONNX model largely fixes
this for pavement distress.

### The verification pass — implemented, measured, and NOT enabled

`pos verify` re-examines the shakier findings: it crops each candidate's clearest
sighting and re-asks a sceptical prompt that names the lookalikes to reject. It
checks only **singletons below 0.85 confidence** — findings seen twice or more
already run at ~83% precision — which is 6 of 29 on the sample, 25 of 57 on Kohima.

Measure it yourself:

```bash
uv run python scripts/score_perception.py --run runs/e2e --compare runs/e2e_verified
```

**It made things worse, so it is off by default and not wired into `pos run`.**

| | findings | precision | recall | F1 |
|---|---|---|---|---|
| baseline | 22 | 63.6% | 48.3% | 54.9 |
| verified | 16 | 62.5% | **34.5%** | **44.4** |

It dropped 6 findings, of which only 2 were false positives — **4 were real**.
Precision barely moved; recall fell 14 points.

Two things went wrong, both worth recording:

- **The substrate is unfair to it.** Its rejections read as *correct* reasoning
  about synthetic imagery: *"a uniform dark area with a yellow border, no visible
  hole"* is exactly what a flat-shaded synthetic pothole is. The verifier is
  trained on photographs; the sample has no depth cues.
- **On real footage it abstains instead.** On Kohima: 25 checked, 2 rejected,
  **23 unsure**. That is the prompt's fault — pairing *"do not confirm out of
  politeness"* with *"unsure is a valid answer, use it rather than guessing"*
  over-corrected the yes-bias into abstention-bias. Median crop was 307×183 px and
  the one it *did* reject confidently was 840×512, so too few pixels is a secondary
  factor at most.

Next thing to try: show the full frame with the box drawn — what a human inspector
actually looks at — instead of a bare crop, and drop the explicit invitation to
answer "unsure". Until that is measured, the honest state is that the mechanism
works and the quality gain does not exist yet.

---

## Things that bit us

Each of these produced plausible-looking wrong output rather than an error.
Recorded so they are not re-learned.

| Problem | Symptom | Fix |
|---|---|---|
| **ffmpeg `fps` filter** | picks the frame nearest each interval's *centre* — 0.20 s off, ~1.6 m at 8 m/s | select source frames by index |
| **Variable frame rate** | `r_frame_rate` is nominal; real spacing drifts — 0.43 s, ~3.5 m | read each frame's true PTS |
| **`creation_time`** | is often the recording *END*, not start. Read as start: no GPS overlap at all | test both, keep whichever overlaps |
| **Two clocks conflated** | `--time-offset` folded into `t_sec`, so video sat 2.71 s out of step with its own markers | `t_sec` = video time, `ts` = GPS time |
| **Heading at low speed** | 1 Hz fixes 1.4 m apart against 3 m GPS noise → 36° median error | distance-based `--heading-baseline` |
| **Whole-frame boxes** | model right about the class, box useless; projected a fake position 2 m ahead | reject boxes >92% on both axes |
| **Voxel downsampler** | assumed a filled volume, but a road cloud is a 2D *surface* → 9,974 points against a 500,000 budget | bisect on voxel size |
| **Unidentifiable horizon** | at low speed the calibration fit pinned to whichever bound it started from, giving vfov 2.7° | supply `--horizon`; warn on boundary hits |
| **KITTI is 1242×375** | odd height; libx264 refuses | pad one row, recompute pitch against the *encoded* height |
| **zustand selector** | a selector building a new array → infinite re-render (React #185) | memoised hook over stable fields |
| **Overpass failure written as an empty twin** | a rate-limited query became `{"buildings":[],"roads":[]}` on disk — indistinguishable from "nowhere here has buildings", and re-running looked pointless. 13 of 30 runs had no 3D buildings; one bbox that reported 0 returned 78 on retry | never persist a fetch failure. Write nothing, say why, stay retryable |
| **CSV label slugified into a class key** | `export-csv` writes the label "Danger Element"; slugifying gave `danger_element` where the taxonomy says `hazard`. Fallback colour, absent from the legend filter, and scored at the default weight — 35 of 83 findings, index off by 4 | resolve `defect_name` against the taxonomy's labels, then its keys |
| **CSV time truncated to whole seconds** | `int(t_sec)` slid findings up to ~5 m along the route at survey speed, into the neighbouring 20 m scoring segment. Every marker still looked correctly placed | write `M:SS.mmm`; `parse_time` already read it |
| **CSV carried no severity** | every import became a flat severity 3, so the Quality Index — `weight × severity × confidence` — read 6.0 against a true 16.6 | severity + confidence as optional columns; report when defaulted |
| **Imported run claimed `backend: mock`** | `pos ingest` leaves that default, and `mock` means SYNTHETIC FIXTURES here — so the run table and PDF footer called real survey data fake | set `backend: csv_import` on import |

---

## Project layout

```
configs/camera/*.yaml        calibration: dashcam, car_dash, bike_kohima, kitti
configs/domains/*.yaml       inspection taxonomies — the main extension point

pos/schema.py                data contracts, single source of truth
pos/config.py                camera + domain + absence-rule loading
pos/geo.py                   geodesy and ground-plane projection
pos/ingest.py                keyframes, true PTS, GPX interpolation, heading
pos/doctor.py                preflight checks and clock-offset detection
pos/perception/base.py       Detector protocol, prompt building, strict-JSON parse
pos/perception/cosmos.py     hosted VLM, cached, with model fallback
pos/perception/onnx_yolo.py  post_cons YOLOv8 pavement distress + class-wise NMS
pos/perception/ensemble.py   merges the two
pos/perception/mock.py       offline fixtures
pos/cluster.py               localise, then dedupe into findings
pos/absence.py               infer defects from missing coverage; gap extents
pos/perception/verify.py     second-pass confirmation (opt-in — see measurement)
pos/score.py                 segments + Quality Index
pos/twin.py                  Overpass → buildings/roads
pos/depthcloud.py            CPU monocular-depth point cloud
pos/pointcloud.py            lingbot NPZ → georeferenced PLY
pos/basemap.py               satellite tiles → one stitched georeferenced texture
pos/kmlexport.py             run → self-contained KMZ for Google Earth
pos/report.py                run → paginated PDF inspection report
pos/import_csv.py            external CSV → findings, resolved against the taxonomy
pos/jobs.py                  studio job runner: subprocess pipeline + SSE progress
pos/server.py                FastAPI: run dir, video with Range, SSE, upload
pos/studio.html              the upload page — one dependency-free file, no build
pos/cli.py                   the twenty commands

viewer/src/Scene.tsx         3D twin: buildings, heatmap, markers, cloud
viewer/src/EvidencePanel.tsx the click-through to pixels
viewer/src/VideoPanel.tsx    source video, synced, 1×
viewer/src/panels.tsx        score, legend, coverage, layers, timeline
viewer/src/store.ts          one zustand store, local ENU projection

setup.sh                           one-shot install + capability report
start.sh                           serve studio + viewer; --dev for hot reload

scripts/run_pipeline.sh            whole pipeline on one clip, start to end
scripts/make_sample.py             synthetic sample generator
scripts/verify_sample.py           24-check correctness gate (needs the sample)
scripts/test_geometry.py           19 closed-form geometry checks (no fixtures)
scripts/test_import_csv.py         51 CSV-importer checks (no fixtures, no network)
scripts/test_onnx_contract.py      post_cons weights vs decoder (class count, index order)
scripts/score_perception.py        precision / recall / F1 vs ground truth
scripts/calibrate.py               two-marker calibration
scripts/calibrate_from_motion.py   calibration from motion + GPS
scripts/import_kitti.py            KITTI → video + GPX + exact camera config
scripts/lingbot_gpu_pass.sh        RunPod reconstruction recipe
```

Roughly 11,400 lines of Python in `pos/` across 33 files, 3,500 more in `scripts/`
across 9, 2,300 of viewer across 8, and 10 config files.

---

## Licences

| Component | Licence | Note |
|---|---|---|
| PhysicalOS | yours | |
| map3d patterns | MIT | building-extrusion approach |
| lingbot-map | Apache 2.0 | commercially safe |
| Depth Anything V2 | Apache 2.0 | commercially safe |
| reportlab | BSD | commercially safe |
| **Esri World Imagery** (`pos basemap` default) | **non-commercial + attribution** | commercial use needs an ArcGIS licence. Attribution is rendered from `basemap.json` — do not strip it |
| **post_cons YOLOv8 weights** | **AGPL-3.0** (Ultralytics) | copyleft, network-triggered — take advice before shipping in a closed product |
| Cosmos / NVIDIA hosted models | NVIDIA API terms | |
| **LocateAnything-3B** | **NVIDIA non-commercial** | research and academic use only |
| **KITTI** | **CC BY-NC-SA 3.0** | non-commercial; validate with it, don't ship its frames |
| OpenStreetMap data | ODbL | attribution required |

The ones to watch are the **AGPL YOLO weights** and the **non-commercial** model,
imagery and dataset licences. Everything else on the default path — `mock`,
`cosmos`, `depthcloud`, `kml`, `report`, OSM — is commercially clean. Google's map
tiles are absent by design: they may not be used outside Google's own APIs, which
is why imagery-in-the-dashboard and Google-Earth-export are two separate features
rather than one.

**What is reused from map3d.** `THREE.Shape` + `extrudeGeometry` for buildings,
drei `Html` hover cards, `Line`, `Sky`/`Environment`, and the React 19 / R3F 9 /
three 0.173 stack. Reworked into local ENU metres instead of map3d's single
`scale = 51000` constant, so one scene unit is one metre. map3d's
`src/api/axios.ts` calls `api.fleet.cartesiancs.com`, a proprietary backend, so
OSM is fetched server-side via Overpass in `pos/twin.py` instead.
