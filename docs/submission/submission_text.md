# Submission texts

## Short version (form field / one paragraph)

EgressTwin turns a photo of a floor plan into an evacuation digital
twin - and only recommends changes it can prove. Muse Spark and
SAM 3.1 read the plan into a correctable layout, a deterministic
social-force simulator (verified 5/6 against the standard RiMEA
test cases) measures it across replicas, a Spark agent proposes
improvements and stress-tests them N-1, and a deterministic verifier
stamps the report VERIFIED only when every number checks out. Built
in the hackathon for about USD 0.08 of API spend; 70 tests; runs
fully offline in mock/demo mode.

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
   honest "change nothing" is verified as such - on the bundled
   demo plan the agent explored widening both exits, found no
   significant improvement, recommended nothing, and was stamped
   VERIFIED (2/2 checks). In the acceptance test on a clogged room,
   it widened an exit from 0.5 m to 2.4 m (T95 from a censored 60 s
   to 4.6 s), passed N-1, and earned VERIFIED with 10/10 claims.

### Evidence, not adjectives

- Simulator verified against the standard RiMEA test cases:
  5 of 6 pass. The failure (fundamental diagram at crush density)
  is published with its technical explanation; the criterion fixed
  before the run was not relaxed.
- Demo baseline (12 people, 3 replicas): T95 10.9 s, 95% CI
  8.4-13.4.
- Performance on a 4-core Ryzen, no GPU: 30 people in 1.75 s,
  300 in 36.26 s; a 4-scenario x 3-replica sweep in 34.82 s
  (20.7 scenarios/min).
- 70 tests pass, `ruff` clean. Total API spend for the whole
  project: about USD 0.08, behind a hard budget guard.
- Runs with no network and no API key: `python scripts/demo.py --mock`,
  or the web app's cached demo mode.

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
