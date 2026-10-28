# P01 — Perception risk spike (Muse Spark 1.3, SAM 3.1, OpenCV)

Date: 2026-10-28. Scope: spike only, no production code. All spike code
lives in `scripts/spikes/`; raw API responses are cached under
`runs/spikes/cache/` (re-runs cost USD 0). This tool supports decisions;
it is not a safety certification.

## Assets and ground truth

Manuel's hand-drawn A and photo C were not in the repo when the spike
ran, so the spike used project-generated stand-ins with exact ground
truth, registered in `data/ASSETS.md`:

| ID | File | Role | Ground truth |
| --- | --- | --- | --- |
| A (substitute) | `data/samples/spike_plan_hand.png` | Hand-drawn style plan (jittered strokes, door arcs), same layout as B | `spike_plan_hand_gt.json` + `_gtmask.png` |
| B | `data/samples/spike_plan_synth.png` | Synthetic rectangle plan (800x600, 0.02 m/px, "13.6 m" scale line) | `spike_plan_synth_gt.json` + `_gtmask.png` |
| C (substitute) | `data/samples/spike_aerial_synth.png` | Synthetic top-down open space (building, stage, 3 tents, fence, 6 trees, 4 cars) | `spike_aerial_synth_gt.json` + `_gtmask.png` |

GT per plan: 4 doors (2 of them exits), 10 wall rectangles, 1 stairs
block. Matching rule: greedy box IoU >= 0.30. Because A and C are
synthetic stand-ins, every conclusion about real hand-drawn plans and
real aerial photos must be re-checked when Manuel supplies the real
assets (re-run commands at the end; cost is cents).

## Method

1. **Spark vision** (`muse-spark-1.3`, Responses API, `text.format`
   json_schema, strict): image type, exits/doors with boxes (native
   0-1000 grounding grid, converted to pixels), visible dimension
   annotations, overall confidence. One call per image.
2. **SAM concept** (`sam-3.1`, Responses streaming, one request per
   concept, `metadata.include_confidence="true"`): plans `door`,
   `doorway`, `wall`, `stairs`; aerial `building`, `tent`, `stage`,
   `fence`, `tree`, `car`. Zero matches is a valid result, not an error.
3. **SAM box prompts**: one request per plan image carrying all Spark
   door boxes as object prompts (docs grammar, `w`/`h` = image size).
4. **OpenCV** on A and B: Otsu threshold + morphological close for
   walls; walkable = outer-contour interior minus dilated walls; exit
   candidates = gaps along the outer wall band.
5. **Metrics** vs GT: walkable-zone IoU, exits/doors detected vs real
   (recall, precision, mean box IoU), SAM counts kept at confidence
   0.3 / 0.5 / 0.7. Wire output parsed by regex; masks decoded with the
   official `meta-sam-parser` 0.1.0.

Control experiment (transport check): the same SAM concept pipeline on
a natural photo (Wikimedia avocado image from the Meta docs) returned
2 records at confidence 0.97. The pipeline works; the zero results
below are a domain effect, not a transport bug. The production client
in `src/egresstwin/meta/` cannot reproduce these calls yet (see
"Production gaps").

## Results

### Task 1 — Muse Spark 1.3 vision (structured output)

| Image | Type read | Scale read | Doors found | Recall vs 4 doors | Precision vs doors | Mean box IoU | Conf. |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A hand substitute | floor_plan | "13.6 m" (exact) | 4 | 1.00 | 1.00 | 0.92 | 0.92 |
| B synth plan | floor_plan | "13.6 m" (exact) | 4 | 1.00 | 1.00 | 0.90 | 0.90 |
| C aerial substitute | other (not aerial_photo) | none (none present) | 0 | n/a | n/a | n/a | 0.95 |

Notes: against the 2 true exits, Spark recall is also 1.00 with
precision 0.50, because Spark returns interior doors too and does not
distinguish exits from doors. The dimension annotation box was located
correctly in both plans. Spark vision is the strongest component in
the spike. Cost driver: ~720 input + ~2,100-2,360 output tokens per
plan image (reasoning included), about USD 0.010 per image.

### Task 2 — SAM 3.1 concept prompts (one request per concept)

| Image | Concept | Matches | Mean conf. | Recall vs GT | Kept @0.3/0.5/0.7 |
| --- | --- | --- | --- | --- | --- |
| A | door / doorway / wall / stairs | 0 / 0 / 0 / 0 | n/a | 0.00 (all) | 0/0/0 |
| B | door / doorway / wall / stairs | 0 / 0 / 0 / 0 | n/a | 0.00 (all) | 0/0/0 |
| C | building / tent / stage / fence / tree / car | 0 / 0 / 0 / 0 / 0 / 0 | n/a | 0.00 (all) | 0/0/0 |
| Control: natural photo | avocado | 2 | 0.97 | n/a | 2/2/2 |

No confidence threshold can rescue an empty result: SAM concept mode
returned an empty `output_text` (a valid zero-match outcome) for every
concept on both line plans and on the flat synthetic aerial. On the
natural-photo control it worked immediately. This confirms the spike
hypothesis: SAM 3.1 concept prompting, trained mostly on natural
images, does not transfer to line drawings or flat vector-style
graphics.

### Task 3 — SAM 3.1 box prompts seeded by Spark boxes

| Image | Prompted objects | Returned | Recall vs 4 doors | Precision | Mean box IoU | Conf. range |
| --- | --- | --- | --- | --- | --- | --- |
| A hand substitute | 4 | 4 | 1.00 | 1.00 | 0.98 | 0.957-0.980 |
| B synth plan | 4 | 4 | 1.00 | 1.00 | 0.95 | 0.957-0.980 |
| C aerial substitute | 0 (Spark found none) | 0 | n/a | n/a | n/a | n/a |

Box-prompt mode is reliable on plans when Spark supplies the boxes:
SAM refines each seeded box into a tight box plus mask. It inherits
Spark's recall ceiling, so it cannot find doors Spark missed.

### Task 4+5 — OpenCV walls and walkable-zone IoU

| Image | Interior-only baseline | OpenCV | SAM-wall only | OpenCV + SAM |
| --- | --- | --- | --- | --- |
| A hand substitute | 0.8505 | 0.8872 | 0.8505 (= baseline) | 0.8872 (= OpenCV) |
| B synth plan | 0.8548 | 0.9097 | 0.8548 (= baseline) | 0.9097 (= OpenCV) |

GT walkable share of frame: 0.5833 (plans). SAM `wall` contributed
zero pixels (no matches), so "SAM walls" degenerates to the baseline
and the combination equals OpenCV alone. OpenCV exit-gap detection on
B found exactly the 2 exits (gap boxes `[182,530,77,10]` and
`[730,422,10,57]`; centres inside the GT exit boxes, 2/2, no false
positives) — their thin-strip boxes score IoU 0.00 against the
slab-inclusive GT door boxes, so gap detection is judged by centre
hit, not box IoU. On the jittered hand-style A, the same detector
fragmented into 10 candidates (wall loop breaks), so gap detection
needs the manual-correction step on real hand-drawn input.

Aerial walkable check: with zero SAM obstacle boxes, the SAM-derived
walkable prediction is "everything walkable", IoU 0.7978, exactly the
GT walkable share — a vacuous score, reported only to show the
baseline. Overlays: `docs/figures/spike/A_hand_substitute_overlay.png`,
`B_synth_plan_overlay.png`, `C_aerial_substitute_overlay.png`,
`A_hand_substitute_opencv.png`, `B_synth_plan_opencv.png`.

## Decision (explicit)

1. **Main path: floor plans.** The plan pipeline works end to end on
   both the clean synthetic plan and the hand-drawn style variant.
   The aerial-photo path is **deferred** (consistent with cut priority
   4 in `INSTRUCCIONES.md`): on the flat synthetic aerial, Spark
   classified the image as "other" and found no objects, and SAM
   concept mode found nothing. That does not prove real aerial photos
   fail — the natural-photo control suggests they may work — but with
   no real photo tested, the photo path carries unretired risk and
   must not be the demo's main path. Re-run this spike with Manuel's
   real C before investing in it.
2. **SAM concepts used: none on plans, at any threshold.** `door`,
   `doorway`, `wall` and `stairs` all returned zero matches; threshold
   choice is moot. SAM is used **only in box-prompt mode**, seeded by
   Spark boxes, with confidence threshold **0.5** (all observed box
   prompt records scored 0.957-0.980, so 0.5 keeps everything real
   while filtering junk). If the photo path is revived, re-test
   concept mode there first; the control photo scored 0.97.
3. **Walls come from OpenCV, not SAM.** OpenCV threshold + morphology
   reached walkable IoU 0.91 (clean plan) / 0.89 (hand style) vs the
   0.85 interior-only baseline; SAM `wall` added zero pixels. SAM
   masks are still useful for door refinement (Task 3), not walls.
4. **Spark role: both.** Spark detects the doors (recall 1.00,
   precision 1.00 vs doors on both plans) **and** reads the scale
   annotation exactly ("13.6 m" in both plans, correctly localised).
   Two limits: Spark does not separate exits from interior doors
   (precision 0.50 vs exits), so exit classification needs the
   EXIT-label/outer-wall heuristic plus the planned manual correction;
   and Spark missed everything on the synthetic aerial.

## Production gaps found (for the next prompts, not fixed here)

- `src/egresstwin/meta/client.py` `_real_segment` uses a chat
  completion with an inline base64 string; the documented SAM 3.1
  transport is Responses streaming with `input_text`/`input_image`
  parts and a special-token output grammar. It must be rewritten.
- `src/egresstwin/meta/spark.py` has no vision or structured-output
  helper; Spark 1.3 image reads need Responses `input_image` plus
  `text.format` json_schema, with the 0-1000 grounding grid converted
  to pixels.
- `meta-sam-parser` (PyPI, 0.1.0 verified here; docs require >= 0.0.6
  for confidence) is not in `pyproject.toml`; add it when the
  production SAM transport lands.
- SAM box prompts always return `one_bit` masks; concept prompts
  default to lossless. Parse fields by key, never by position.

## Cost

Spike ledger `runs/spikes/costs_spike.jsonl`: 19 charged calls,
USD 0.0658 (3 Spark vision + 14 SAM concept + 2 SAM box-prompt; the
third box-prompt call was skipped because Spark returned no boxes on
C). Throwaway probes (transport checks + natural-photo control):
~USD 0.0133. **Total ≈ USD 0.08, under the USD 2 cap.** Cached re-runs
cost USD 0.

## Reproduce

```bash
.venv/Scripts/python.exe scripts/spikes/generate_assets.py
.venv/Scripts/python.exe scripts/spikes/run_spike.py
```

Requires `MODEL_API_KEY` in the environment and the spike-only parser:
`uv pip install --python .venv/Scripts/python.exe meta-sam-parser`.
Full numbers: `runs/spikes/results.json` (regenerated each run).
