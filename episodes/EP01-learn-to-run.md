# EP01 — Rupert learns to run

**Status:** Planning

## Goal
Rupert goes from collapsing to running, ending on his first 100m time trial. The gap to Bolt's 9.58 sets up EP02.

## Milestone 1 is done when
- [ ] Rupert genuinely learns to run, with no pre-made animation anywhere
- [ ] At least five distinct stages are captured as clips
- [ ] The final run looks like running, not a hop or a crawl
- [ ] Clips render at 1080p with the Rupert model on the track
- [ ] Total cloud spend is under £15

## Build steps
1. ~~Set up the GitHub repo and the app skeleton~~ (done 2026-11-05; Tom confirmed install check and humanoid viewer work)
2. Get the humanoid training in free Google Colab to prove the code works end to end
3. Run the full training on a rented GPU, saving checkpoints
4. Run the moment-finder and pick the stages
5. Build the Blender scene (track, Rupert, cameras) and render the clips
6. Tom watches and decides whether it's a video

## Experiments
- EXP-001: local smoke test, pipeline works (CPU, 17 min)
- EXP-006: stable training (no permanent collapse) but he learned to **run on his knees** (best 20.8 s). Also a 'stand still and survive' phase at 20-40M. Great comedy clip: step_0000728432640.
- EXP-005: first full 1B run (scoring v3). **Real running stride at 345M: 100 m in 15.2 s, 8/8 attempts.** Then training collapsed (falls at ~2.7 s); fixed with a slower learning rate. Best clip: step_0000345047040.
- EXP-004: first RunPod run (4090, 8.5 min for 100M steps). Upright now, but stiff-legged bouncing with a 'zombie arm'. Best 100 m 21.8 s.
- EXP-003: rebuilt Rupert on Colab. Learned to finish 100 m (best 19.2 s) but as a hunched shuffle. Great early-stage clips (collapse, lunge-and-fall, first shuffle).
- REVIEW 2026-11-06: stock setup unfit for EP01 (wrong body size, scoring rewards standing, no early checkpoints, no stats/replay). Rebuilt. See decisions.md.
- EXP-002: first Colab T4 run, reward 1.5 -> ~138, jump near step 60M. x-axis ran to ~88M not the 20M requested (unexplained). Checkpoints not saved (lost). Clip not rendered.

## Story beats (fill in as clips come in)
- 
