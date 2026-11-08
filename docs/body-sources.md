# Rupert's body: numbers and where they come from

Every number here could end up said on camera ("Rupert is limited to human strength").
Status: **CONFIRMED** = checked against a source. **TO VERIFY** = reasonable but not yet checked. Nothing TO VERIFY may be claimed on screen.

The code is in `rupert/body.py` (size, mass, strength) and `rupert/env.py` (muscle speed limits).

## Size
| Item | Value | Status | Source / reason |
|---|---|---|---|
| Height | 1.84 m | Design choice | Rupert is "tall and thin". Measured from the built model. |
| Mass | 70 kg | Design choice | Lean adult male. Bolt: ~1.95 m, ~94 kg (TO VERIFY) |
| Limb thickness | stock (not scaled up) | Design choice | Keeps him thin; the masses below are set directly. |

## Mass of each body part (% of total), de Leva (1996), adult male
| Part | % | Status |
|---|---|---|
| Head | 6.94 | CONFIRMED |
| Trunk (total) | 43.46 | CONFIRMED |
| …upper / middle / lower trunk | 15.96 / 16.33 / 11.17 | TO VERIFY (paper Table 4) |
| Upper arm | 2.71 | CONFIRMED |
| Forearm | 1.62 | CONFIRMED |
| Hand | 0.61 | TO VERIFY |
| Thigh | 14.16 | TO VERIFY |
| Shank (shin) | 4.33 | TO VERIFY |
| Foot | 1.37 | TO VERIFY |

Source: de Leva P. (1996) *Adjustments to Zatsiorsky-Seluyanov's segment inertia parameters.* J Biomech 29(9):1223–1230. doi:10.1016/0021-9290(95)00178-6. Checked values via [has-motion wiki table](https://wiki.has-motion.com/Adjusted_Zatsiorsky-Seluyanov%27s_segment_inertia_parameters); full paper PDF: [ebm.ufabc.edu.br](https://ebm.ufabc.edu.br/wp-content/uploads/2013/12/Leva-1996.pdf).

## Peak joint strength (N·m), strongest each joint can ever push (body v2, from 2026-11-07)
Rupert has the tested leg strength of an **ordinary (non-athletic) young man**.

Source: Harbo T., Brincks J., Andersen H. (2012) *Maximal isokinetic and isometric muscle strength of major muscle groups related to age, body mass, height, and sex in 178 healthy subjects.* Eur J Appl Physiol 112:267–275. doi:10.1007/s00421-011-1975-3. The subjects were healthy non-athletic Danish volunteers. We use **Table 5, men under 30** (mean 24 years, 1.79 m, 74 kg; Rupert is 1.84 m, 70 kg).

Values marked "x1.15" were only measured at 60°/s, which reads a little lower than a full-strength static push. We multiply by 1.15 to estimate the static value; the paper's own static/60°/s ratios are 1.23 (knee) and 1.08 (hip).

| Joint action | Value | Paper value (men <30) | Status |
|---|---|---|---|
| Hip extension (thigh back) | 227 | 197 ± 58 at 60°/s, x1.15 | CONFIRMED source, estimated static |
| Hip flexion (thigh forward) | 167 | 167 ± 37 static | CONFIRMED |
| Knee extension (straighten) | 265 | 265 ± 73 static | CONFIRMED |
| Knee flexion (bend) | 122 | 106 ± 30 at 60°/s, x1.15 | CONFIRMED source, estimated static |
| Ankle plantarflexion (toes down, push off) | 147 | 128 ± 24 at 60°/s, x1.15 | CONFIRMED source, estimated static |
| Ankle dorsiflexion (toes up) | 44 | 44 ± 11 static | CONFIRMED |
| Shoulders | 60 | shoulder abduction 60 ± 14 static | CONFIRMED (one action used for all shoulder directions) |
| Elbow | 50 | elbow flexion 50 ± 18 static | CONFIRMED (used both ways) |
| Hip sideways / twist | 150 / 80 | not in paper | TO VERIFY |
| Ankle sideways | 60 | not in paper | TO VERIFY |
| Waist bend / side / twist | 200 / 150 / 100 | not in paper | TO VERIFY |

On camera, it's fair to say: "Rupert has the leg strength of an average 24-year-old man, measured in a lab."

**Body v1** (EXP-001 to EXP-007) was a "strong trained adult": hip 300, knee 250, ankle 250, shoulders 80 and elbows 60, the same both ways. With instant muscles it reached Bolt-like speed (100 m in 10.0 s, EXP-007). Old runs are still replayed with v1 (`env.load_for_run`).

## Muscle switch-on delay (body v2)
| Item | Value | Status |
|---|---|---|
| Time to switch on | 10 ms | CONFIRMED: Zajac (1989), Winters (1990), as given in [OpenSim's first-order activation docs](https://opensimconfluence.atlassian.net/wiki/spaces/OpenSim/pages/53090590/First-Order+Activation+Dynamics) |
| Time to switch off | 40 ms | CONFIRMED: same source. Thelen (2003) uses 15 / 50 ms. |
Model: first-order lag. Each 20 ms command moves a muscle about 92% of the way when it's switching on and about 46% when it's switching off. Body v1 muscles were instant.

Searched 2026-11-06: Schache et al. (2011) *Effect of running speed on lower limb joint kinetics*, MSSE, doi:10.1249/MSS.0b013e3182084929, is the right paper for sprint joint torques, but the values need the full text ([abstract](https://researchprofiles.canberra.edu.au/en/publications/effect-of-running-speed-on-lower-limb-joint-kinetics/)).

## Muscle speed limit (joint speed at which torque falls to zero)
| Joint | Value | Status |
|---|---|---|
| Hip | 25 rad/s | TO VERIFY. Elite sprint hip speed in stance ≈ 640–670 °/s ≈ 11–12 rad/s ([Konstanz ISBS paper](https://ojs.ub.uni-konstanz.de/cpa/article/view/6908/6203)), so 25 leaves headroom. |
| Knee, ankle | 35 rad/s | TO VERIFY |
| Everything else | 20 rad/s | TO VERIFY |
Model: straight-line force–velocity (simpler than the real Hill curve). The real curve is steeper, so ours is slightly generous at mid speeds.

## Joint range (body v3, from 2026-11-08)
| Joint | Value | Status |
|---|---|---|
| Hip sideways: outwards / inwards | 33° / 14° (stock model: 25° / 5°) | CONFIRMED: passive normative values 33.0 ± 4.8 / 13.8 ± 6.3, AAOS-based table in Sports Health (2020), [PMC8167346 table 1](https://pmc.ncbi.nlm.nih.gov/articles/PMC8167346/table/table1-1941738120953418) |
| Hip backwards (extension) | 20° (unchanged) | Within norms: same table gives 17.0 ± 3.9 |
| Everything else | stock MuJoCo humanoid | TO VERIFY |
Why it changed: fitting real runners onto Rupert, his hip hit the 5° inwards limit 30–50% of the time (runners put their feet under their middle).

## Known simplifications (be honest about these on camera)
- Muscles are motors at each joint, not individual muscles. No fatigue.
- Hip, knee and ankle (forward/back) have a different strength each way (body v2). The other joints are still the same both ways.
- Our strength is a non-athletic man's tested strength, applied at every joint angle. Real strength also changes with joint angle; ours doesn't.
- Feet are two rigid capsules; no toes or arch spring.
