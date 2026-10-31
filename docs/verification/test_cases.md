# Verification test cases for the EgressTwin simulator

This document selects the standard test cases that verify the
simulator itself (as opposed to the agent, whose recommendations the
deterministic verifier checks against the simulator). Every case is
summarised in our own words. No part of the source documents is
copied into this repository. Where a datum does not appear in the
freely accessible source, it is marked PENDING instead of being
invented.

## Sources

1. **RiMEA** - *Richtlinie fuer Mikroskopische
   Entfluchtungsanalysen* (Guideline for Microscopic Evacuation
   Analysis), RiMEA e.V. Current version **4.1.1** (2025,
   German/English PDF linked from the official site). Annex 2
   (*Vorlaeufige Anleitung zur Validierung / Verifizierung von
   Simulationsprogrammen*) defines 14 component tests. Case
   summaries below follow the official per-test pages at
   https://rimea.de (Annex 2, Tests 1, 5, 6, 11 and 12), accessed
   2026-10-30.
2. **IMO** - *Revised Guidelines on Evacuation Analysis for New and
   Existing Passenger Ships*, MSC.1/Circ.1533, 6 June 2016. Its
   Annex 3 recommends component tests for evacuation simulation
   tools that overlap the RiMEA set (corridor walking speed,
   pre-movement time assignment, exit flow, route choice). Used
   here only as a cross-reference: the full IMO text is not freely
   downloadable, so no IMO numeric criterion is quoted. PENDING:
   Manuel to supply the circular if exact IMO wording is wanted.
3. **Weidmann (1993)** - *Transporttechnik der Fussgaenger*,
   Schriftenreihe des IVT Nr. 90, ETH Zuerich. Empirical
   speed-density relation (fundamental diagram). We use the
   Kladek-form parameterisation as quoted in the secondary
   literature (Nikolic, Bierlaire and co-authors, 2014, quoting
   Weidmann via Zhang 2012): free speed 1.34 m/s, jam density
   5.4 people/m^2, shape parameter 1.913 people/m^2, giving
   v(rho) = 1.34 * (1 - exp(-1.913 * (1/rho - 1/5.4))). The same
   diagram is replotted in Seyfried and co-authors (2005),
   arXiv:physics/0506170. The original 1993 report was not
   consulted directly. PENDING: confirm against the original if a
   printed copy becomes available.

## Chosen cases (6)

### V1 - Free walking speed in a straight corridor (RiMEA Test 1)

One person walks a corridor 40 m long and 2 m wide at a set speed
of 1.33 m/s. Allowing 0.40 m body size, 1 s reaction and 5 percent
speed tolerance, the source requires a travel time between 26 s
and 34 s. This is the purest check of the driving force and the
integrator. Implemented with the desired speed fixed at 1.33 m/s.

### V2 - Movement around a corner (RiMEA Test 6)

Twenty people approach a left-turning corner and must round it
successfully without crossing walls. Qualitative criterion: all
twenty leave, and no recorded position is ever inside a wall.
PENDING: the exact corridor dimensions are only in Figure 4 of
the source (an image, not text). We use our own L-shaped corridor
(2 m wide arms) and say so; Manuel to confirm if exact-figure
replication is required.

### V3 - Pre-movement (reaction) time distribution (RiMEA Test 5)

Ten people in an 8 m x 5 m room with a 1 m exit centred on the
5 m wall. Reaction times are uniform between 10 s and 100 s.
Criterion: each person starts moving at their assigned time. This
case motivated a small engine feature: per-agent pre-movement
times sampled from a uniform range (Scenario fields
`pre_movement_min_s` / `pre_movement_max_s`). People who have not
started yet do not move and cannot be captured by an exit; in this
implementation they also do not yet exert or receive social
forces, which is acceptable in this sparse room and is stated
here as a known simplification.

### V4 - Bottleneck between two rooms (RiMEA Test 12)

Two rooms joined by a corridor, filled with 150 adults, reaction
time zero. Because the corridor limits the flow, congestion may
form only in the first room, not in the second. The source itself
says this case documents model behaviour and must not be used as
an exclusion criterion until the empirical question is settled;
we report it in that spirit. Operationalisation (ours, labelled
as such): peak smoothed density in room 1 exceeds that in room 2,
and the second room never jams (its occupancy stays well below
the first room's). PENDING: exact geometry is only in Figure 9;
room and corridor sizes are ours.

### V5 - Exit assignment and route choice (RiMEA Test 11)

A public space with two exits, filled from the left at high
density (1,000 adults in the source). Expected behaviour: people
prefer the nearer exit 1 and congestion forms there, yet some
individuals also use the alternative exit 2. We run 200 people
(a documented reduction for runtime; the criterion is the split
pattern, not the headcount) and require: both exits used, the
nearer exit carrying the majority. PENDING: room dimensions and
exit positions are only in Figure 8; ours are a 20 m x 10 m room
with the near exit on the left wall and the far exit on the
right wall, population filling the left part.

### V6 - Fundamental diagram vs Weidmann (empirical)

Speed falls as density rises. The source curve is the Weidmann
relation quoted above; it gives no numeric tolerance, so the
pass tolerance is a project choice and is labelled as such:
mean speed in the lowest density bin within 20 percent of the
curve, speed monotonically decreasing across density bins, and
overall RMSE against the curve reported (pass if at most
0.30 m/s). True periodic boundary conditions do not exist in
this engine; the case measures agents walking a long corridor
toward an exit and bins their instantaneous speed by local
density (people within a 2 m window). This adaptation is stated
in the report.

## Cases not chosen, and why

- RiMEA Tests 2 and 3 (stairs up/down): the model has no stairs
  or height. Out of scope for a floor-plan twin.
- RiMEA Test 4 (specific flow through a cross-section): the
  accessible page defines the quantity but its numeric relation
  is in a figure. PENDING the figure values; the door-flow
  behaviour is meanwhile covered indirectly by V4 and V6.
- RiMEA Test 7 (demographic parameters) and Test 8 (parameter
  analysis): population demographics beyond radius and desired
  speed are not modelled.
- RiMEA Test 9 (large crowd leaving a large public space):
  closest to a full product scenario; V5 covers its exit-choice
  core at smaller scale.
- RiMEA Test 10 (assignment of escape routes): route assignment
  to specific groups is not a product feature; nearest-exit
  assignment is covered by V5.
- RiMEA Test 13 (congestion before a staircase): no stairs.
- RiMEA Test 14 (route choice between floors): the source itself
  only asks to document the behaviour (short, long, mixed or
  configurable); multi-floor routes are out of scope.

## Open questions for Manuel

1. Exact figure geometries for RiMEA Tests 6, 11 and 12
   (Figures 4, 8, 9) if literal replication is wanted; a PDF of
   RiMEA 4.1.1 in the repo inbox would resolve all three.
2. RiMEA Table 1 demographic walking speeds (referenced by
   Tests 11 and 12, not reproduced on the web pages).
3. IMO MSC.1/Circ.1533 full text if IMO-numbered criteria should
   be quoted verbatim.
4. Weidmann (1993) original, to confirm the Kladek parameters
   quoted from secondary literature.
