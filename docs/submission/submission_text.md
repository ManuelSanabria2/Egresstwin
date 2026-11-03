# Submission texts

## Short version (form field / one paragraph)

EgressTwin turns a photo of a floor plan into an evacuation digital
twin - and only recommends changes it can prove. Muse Spark and
SAM 3.1 read the plan into a correctable layout, a deterministic
social-force simulator (verified 5/6 against the standard RiMEA
test cases) measures it across replicas, a Spark agent proposes
improvements and stress-tests them N-1, and a deterministic verifier
stamps the report VERIFIED only when every number checks out. Built
in the hackathon for about USD 0.59 of API spend in total, including
two real end-to-end Muse Spark agent runs. 83 tests pass. A recorded
real run replays offline in the web app's demo mode.

## Long version (project page)

### The problem

Evacuation planning is static while crowds are not. Building managers,
schools and venues rarely get to ask "what if this exit is blocked, or
two metres wider?" and receive a measured answer - and AI agents
usually answer such questions with confident prose and invented
numbers.

### What EgressTwin does

Upload an image of a space and EgressTwin builds its evacuation
digital twin:

1. **Perception.** Muse Spark reads the plan (doors, exits, scale),
   SAM 3.1 segments with Spark-seeded box prompts, OpenCV extracts
   walls. The result is a correctable layout with per-element
   confidences and warnings - the human can fix any misread by hand.
2. **Simulation.** A deterministic social-force engine (Helbing,
   Farkas and Vicsek, *Nature* 2000) on a geodesic floor field
   simulates the evacuation across replicas and reports metrics with
   confidence intervals: T95, total time, flow per exit, peak
   density, hotspots. Timed events (a blocked exit at second 30, a
   hazard zone) recompute only the affected routes.
3. **Optimization.** A Muse Spark tool-calling agent inspects the
   layout, runs a baseline, proposes modifications (widen an exit,
   add an exit, remove an obstacle), simulates them, and must pass
   an N-1 stress test - each exit blocked in turn - before
   recommending anything.
4. **Verification.** A deterministic verifier (code, not an LLM)
   re-checks every claimed number against the stored results,
   requires statistical significance, and stamps the report
   VERIFIED, PARTIALLY_VERIFIED or FAILED, claim by claim. An
   honest "change nothing" is verified as such. In real runs with
   muse-spark-1.3 and 200 people: on a hand-drawn plan the agent
   added an exit and widened another, T95 95.46 s -> 42.18 s, N-1
   tested, VERIFIED 14/14 (USD 0.1056, 11 turns). On an office plan
   it found a faster layout (T95 38.28 s -> 31.75 s) that failed N-1
   (59.38 s with one exit blocked), so it recommended nothing; that
   report was VERIFIED 3/3 after the verifier rejected two earlier
   drafts (USD 0.2288, 20 turns).

### Evidence, not adjectives

- Simulator verified against the standard RiMEA test cases:
  5 of 6 pass. The failure (fundamental diagram at crush density)
  is published with its technical explanation; the criterion fixed
  before the run was not relaxed.
- Real-run baseline on the hand-drawn plan (200 people, 20 replicas):
  T95 95.46 s, 95% CI 88.81-102.10. Traces with real response ids and
  costs are in `data/real_runs/`.
- Parallel sweep on a 4-core / 8-thread Ryzen 3 3300X, no GPU:
  8 scenarios x 5 replicas with 200 people took 433.7 s on 1 worker
  process, 187.0 s on 4 (speedup 2.32) and 142.5 s on 8 (speedup
  3.04). Results were identical at every process count. Scaling is
  sub-linear: 8 processes share 4 physical cores, and a few runs keep
  one person inside until the time cap.
- 83 tests pass, `ruff` clean. Total API spend for the whole
  project: about USD 0.59 (USD 0.5391 in the ledgers), behind a hard
  budget guard.
- Runs with no network and no API key: the web app's demo mode
  replays the recorded real run; `python scripts/demo.py --mock` runs
  the flow with a driver labelled "scripted".

### Built during the hackathon

Everything - perception pipeline, simulator, sweep, agent, verifier,
FastAPI app and single-page UI - was written between 27 and 31
October 2026, one descriptive commit per step (see NEW_WORK.md).
Only published scientific parameters (Helbing 2000; RiMEA criteria;
Weidmann 1993) were reused, re-implemented and cited.

### What is next

Aerial-photo input (spike-tested, deferred), MPI sweep backends,
and calibration against real evacuation drills.

EgressTwin is decision support, not a safety certification.
