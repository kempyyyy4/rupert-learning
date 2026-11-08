# Real-runner recordings (motion capture)

`run_reference.npz` holds 26 short clips of real people running, fitted onto Rupert's body (body v3). The style judge learns from them. The file has `qpos` per frame at 50 frames/s, `starts` (where each clip begins), `fps` and `trials`.

- **Source:** CMU Graphics Lab Motion Capture Database, http://mocap.cs.cmu.edu
- **Trials:** subjects 9, 16 and 35, straight running and jogging; the list is in `rupert/mocap.py`.
- **Credit (required wording, for the video description):** "The data used in this project was obtained from mocap.cs.cmu.edu. The database was created with funding from NSF EIA-0196217."
- **Licence:** the data may be used in commercial products, but may not be resold, even converted. Raw files (`cmu/`) are kept out of git.

## How to rebuild
1. Download the raw `.asf` and `.amc` files into `mocap/cmu/`. The URLs follow `http://mocap.cs.cmu.edu/subjects/09/09_01.amc`.
2. Run `python -m rupert.retarget --video`. This rewrites `run_reference.npz` and `run_reference.mp4` (a check video of Rupert doing the real runs).
