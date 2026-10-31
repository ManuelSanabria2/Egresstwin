# Demo video script (3 minutes)

Recording rules: 1080p, web UI in light theme with large type, cursor
movements slow. Voice-over text below is in English (international
judges); bracketed notes are screen directions in Spanish and are not
read aloud. Total spoken text is about 430 words, roughly 3 minutes
at a calm pace. If a live step stalls, cut to the cached demo
("Load demo" button): it replays the same flow with no network.

Prepare before recording: server running with
`.venv/Scripts/python.exe -m uvicorn egresstwin.app.api:app --app-dir src --port 8077`,
browser at http://127.0.0.1:8077, sample plan
`data/samples/spike_plan_synth.png` on the desktop.

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

[Screen: upload `spike_plan_synth.png` in the Layout tab; show the
scale step; the overlay appears with exits E1 and E2.]

Voice: "I upload a floor plan. Muse Spark reads the drawing - doors,
exits, the thirteen-point-six-metre scale annotation - SAM 3.1
segments the elements, and OpenCV extracts the walls. What I get is
not a picture: it is a measurable layout, a walkable grid with exits
and distances, and every element carries a confidence and warnings."

### 0:45-1:05 - Correct and simulate the baseline

[Screen: move or paint one wall correction if desired; switch to the
Simulation tab; run; agents evacuate; metrics appear.]

Voice: "If the machine misread something, I correct it by hand -
add an exit, paint a wall. Then the simulator runs: a social-force
engine, the Helbing model, on a geodesic floor field. Deterministic:
same seed, same result. Twelve people, three replicas: T95 of
ten-point-nine seconds, with a confidence interval. Not a single
number - a measurement."

### 1:05-1:45 - The agent optimizes

[Screen: Agent tab; start optimization; trace cards appear live over
SSE: get_layout_summary, get_baseline, run_scenarios, compare.]

Voice: "Now the agent works. Muse Spark inspects the layout, runs
the baseline, and proposes changes - here, widening both exits. Every
proposal is executed by the simulator, across replicas, and compared
with statistics. The agent cannot invent numbers: every figure it
cites must come from a tool result."

### 1:45-2:05 - VERIFIED stamp

[Screen: report panel; the VERIFIED stamp; claim-by-claim checks.]

Voice: "And here is the point of the whole project. A deterministic
verifier - plain code, no language model - rechecks every claim
against the stored simulations. On this plan, widening did not beat
the baseline significantly, so the honest report recommends changing
nothing... and that honesty is what gets verified. Two of two checks,
VERIFIED."

### 2:05-2:20 - N-1 stress test

[Screen: stress test result; optionally block exit E1 from t = 0 in
the Simulation tab and replay.]

Voice: "Before any real recommendation, the agent must also survive
an N-1 test: each exit blocked in turn. In our acceptance test, the
agent widened a clogged exit, passed N-1, and earned VERIFIED with
ten of ten claims. A recommendation that collapses when one door
closes never reaches the user."

## 2:20-2:40 - Parallel performance

[Screen: Performance tab; `bench_scaling.png` and
`bench_throughput.png`; the numbers from `data/demo/bench.json`.]

Voice: "Optimization means many simulations, so they run as sweeps:
scenarios times replicas. On a four-core Ryzen with no GPU: thirty
people in under two seconds, three hundred in thirty-six, and a
four-scenario sweep with three replicas each in thirty-five
seconds - about twenty-one scenarios per minute."

## 2:40-3:00 - Built during the hackathon, next steps

[Screen: repository page or file tree; then back to the VERIFIED
report. End card: repo URL and "Decision support, not a safety
certification."]

Voice: "All of this was built during the hackathon: perception,
simulator, agent, verifier, web app - seventy tests, and the
simulator itself verified against the standard RiMEA cases: five of
six pass, and the one failure is published, not hidden. Total API
spend: eight cents. Next: aerial photos, MPI sweeps, and calibration
with real evacuation drills. EgressTwin: recommendations an agent
can prove."

## Fallbacks while recording

- If the real API is slow, narrate over the cached demo replay; the
  trace, stamp and figures are identical in structure.
- If a number on screen differs from this script, read the screen,
  never the script: the script's numbers are the committed demo
  cache values and should match.
- Keep the mouse off the VERIFIED stamp until 1:45; it is the
  visual payoff of the video.
