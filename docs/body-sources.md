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

## Peak joint strength (N·m), strongest each joint can ever push
| Joint | Value | Status |
|---|---|---|
| Hip forward/back | 300 | TO VERIFY |
| Knee | 250 | TO VERIFY |
| Ankle up/down | 250 | TO VERIFY (real ankles push down far harder than they lift; our motor is the same both ways) |
| Hip sideways / twist | 150 / 80 | TO VERIFY |
| Ankle sideways | 60 | TO VERIFY |
| Waist bend / side / twist | 200 / 150 / 100 | TO VERIFY |
| Shoulders / elbows | 80 / 60 | TO VERIFY |

Searched 2026-11-06: Schache et al. (2011) *Effect of running speed on lower limb joint kinetics*, MSSE, doi:10.1249/MSS.0b013e3182084929, is the right paper for sprint joint torques, but the values need the full text ([abstract](https://researchprofiles.canberra.edu.au/en/publications/effect-of-running-speed-on-lower-limb-joint-kinetics/)).

## Muscle speed limit (joint speed at which torque falls to zero)
| Joint | Value | Status |
|---|---|---|
| Hip | 25 rad/s | TO VERIFY. Elite sprint hip speed in stance ≈ 640–670 °/s ≈ 11–12 rad/s ([Konstanz ISBS paper](https://ojs.ub.uni-konstanz.de/cpa/article/view/6908/6203)), so 25 leaves headroom. |
| Knee, ankle | 35 rad/s | TO VERIFY |
| Everything else | 20 rad/s | TO VERIFY |
Model: straight-line force–velocity (simpler than the real Hill curve). The real curve is steeper, so ours is slightly generous at mid speeds.

## Known simplifications (be honest about these on camera)
- Muscles are motors at each joint, not individual muscles. No fatigue.
- Joint motors are equally strong in both directions.
- Feet are two rigid capsules; no toes or arch spring.
