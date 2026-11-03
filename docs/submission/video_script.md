# Demo video script (3 minutes)

Recording rules: 1080p, web UI in light theme with large type, cursor
movements slow. Voice-over text below is in English (international
judges); bracketed notes are screen directions in Spanish and are not
read aloud. Total spoken text is about 520 words, roughly 3.5 minutes
at a calm 150 words per minute: speak a little faster or trim the
0:45-1:05 segment to fit 3:00. If a live step stalls, cut to the cached demo
("Load demo" button): it replays the same flow with no network.

Prepare before recording: server running with
`.venv/Scripts/python.exe -m uvicorn egresstwin.app.api:app --app-dir src --port 8077`,
browser at http://127.0.0.1:8077, sample plan
`data/samples/spike_plan_hand.png` on the desktop. Every number below
comes from the recorded real run in `data/real_runs/spike_plan_hand/`
(Muse Spark 1.3 through the Meta Model API, 2026-11-03). The "Load
demo" button replays exactly that run. A live run takes about 86
minutes with 200 people on this machine, so the video uses the replay
for the agent part and says so on screen.

## 0:00-0:20 - Problem and why it matters

[Screen: black or title card "EgressTwin", then a crowded corridor
photo substitute: the bundled plan image fading in.]

Voice: "When a building has to empty fast, the plan on the wall is
static - but crowds are not. Exit widths, a blocked door, a bad layout:
today, testing any of that means hiring specialists or guessing.
EgressTwin lets anyone turn a picture of a space into a simulation -
and it only recommends changes it can prove."

## 0:20-2:20 - Live flow

### 0:20-0:45 - Image to layout

[Screen: upload `spike_plan_hand.png` in the Layout tab; the overlay
appears with exits E1 and E2 and two warnings.]

Voice: "I upload a hand-drawn floor plan. Muse Spark reads the
drawing - doors, exits, the scale - SAM 3.1 segments the doors, and
OpenCV extracts the walls. What I get is not a picture: it is a
measurable layout, and it is honest about its doubts. Here it warns
that some passages are too narrow for a body and were closed."

### 0:45-1:05 - Correct and simulate the baseline

[Screen: move or paint one wall correction if desired; switch to the
Simulation tab; run; agents evacuate; metrics appear.]

Voice: "If the machine misread something, I correct it by hand -
add an exit, paint a wall. Then the simulator runs: a social-force
engine, the Helbing model, on a geodesic floor field. Deterministic:
same seed, same result. Two hundred people, twenty replicas: ninety-
five percent are out after ninety-five seconds, with a confidence
interval from eighty-nine to one hundred and two. Not a single number
- a measurement."

### 1:05-1:45 - The agent optimizes

[Screen: "Load demo"; Agent tab; scroll the trace cards: each turn
card shows muse-spark-1.3, its response id and its cost. Caption:
"recorded real run, replayed".]

Voice: "Now the agent works - this is a recorded real run with Muse
Spark, replayed. It inspects the layout, runs the baseline, finds that
one exit carries eighty percent of the people, and proposes changes.
Every proposal runs in the simulator, twenty replicas at a time,
spread over eight worker processes, and is compared with statistics.
The agent cannot invent numbers: every figure it cites must come from
a tool result."

### 1:45-2:05 - VERIFIED stamp

[Screen: report panel; the VERIFIED stamp; claim-by-claim checks.]

Voice: "And here is the point of the whole project. A deterministic
verifier - plain code, no language model - rechecks every claim
against the stored simulations. The recommendation: add one exit on
the east wall and widen the second exit. T95 drops from ninety-five
seconds to forty-two. Fourteen of fourteen checks: VERIFIED."

### 2:05-2:20 - N-1 stress test

[Screen: stress test result; optionally block exit E1 from t = 0 in
the Simulation tab and replay.]

Voice: "Before recommending, the agent must survive an N-1 test: each
exit blocked in turn. On a second plan, an office, Muse Spark found a
layout that was faster - thirty-two seconds instead of thirty-eight -
but with one door blocked it got much worse. So it recommended nothing,
and that honest answer was verified too. A recommendation that
collapses when one door closes never reaches the user."

## 2:20-2:40 - Parallel performance

[Screen: Performance tab; `bench_parallel.png`; the table from
`data/demo/bench.json` (`sweep_parallel`).]

Voice: "Optimization means many simulations, so they run as sweeps
spread over worker processes, with identical results to a serial run.
On a four-core Ryzen with no GPU, forty simulations of two hundred
people take four hundred and thirty-four seconds on one process and
one hundred and forty-three on eight - a three-times speedup. Not
linear: eight processes share four cores, and a few runs keep one
person inside until the time cap."

## 2:40-3:00 - Built during the hackathon, next steps

[Screen: repository page or file tree; then back to the VERIFIED
report. End card: repo URL and "Decision support, not a safety
certification."]

Voice: "All of this was built during the hackathon: perception,
simulator, agent, verifier, web app - and the simulator itself
verified against the standard RiMEA cases: five of six pass, and the
one failure is published, not hidden. Both real agent runs together
cost thirty-three cents of API time; the whole project, under sixty
cents. Next: aerial photos, multi-machine sweeps, and calibration with
real evacuation drills. EgressTwin: recommendations an agent can
prove."

## Fallbacks while recording

- The agent segment uses the replay of the recorded real run; say
  so on screen. Do not present a `--mock` run as the model: its trace
  is labelled "scripted".
- If a number on screen differs from this script, read the screen,
  never the script: the script's numbers are the committed demo
  cache values and should match.
- Keep the mouse off the VERIFIED stamp until 1:45; it is the
  visual payoff of the video.
