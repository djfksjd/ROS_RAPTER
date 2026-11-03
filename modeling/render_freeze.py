"""Side views of R-03 (0.92 m leg) and the frozen 0.80 m leg (rev3_l80.json) at the stand nominal pose (the reset
pose, no physics step).
Usage: .venv-sim/bin/python modeling/render_freeze.py -> docs/design/media/freeze_side.png"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'sim/rl'))
from run_env import RunEnv  # noqa: E402

KW = dict(arch8=True, real_mass=True, randomize=False, seed=1, model_rev=3, springs='c', nominal=(-.5273, 1., -.9997),
          render_mode='rgb_array')
CASES = [('R-03 leg 0.92 m (0.22 : 0.38 : 0.32)', {}),
         ('frozen leg 0.80 m (0.17 : 0.321 : 0.309)', dict(rev3_config=str(ROOT/'sim/rl/rev3_l80.json')))]
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
for k, (title, kw) in enumerate(CASES):
    e = RunEnv(**KW, **kw)
    e.reset(seed=1)
    m, d = e.model, e.data
    d.qvel[:] = 0.
    mujoco.mj_forward(m, d)
    r = mujoco.Renderer(m, 480, 560)
    cam = mujoco.MjvCamera(); cam.lookat[:] = [-.3, 0, .45]; cam.distance, cam.azimuth, cam.elevation = 2.4, 90., -22.
    m.vis.headlight.ambient[:] = .45
    r.update_scene(d, cam)
    ax[k].imshow(r.render()[40:]); ax[k].axis("off")  # crop the dark sky band
    ax[k].set_title(f'{title}\nhip height {d.xpos[e.base][2]-.03:.2f} m, mass {e.mass:.2f} kg, knee band '
                    f'{e.knee_band[0]:.2f}-{e.knee_band[1]:.2f} rad', fontsize=9)
fig.suptitle('Design freeze F4 (2026-11-03): stand nominal pose, MuJoCo model (not hardware)', fontsize=10)
fig.tight_layout(); fig.savefig(ROOT/'docs/design/media/freeze_side.png', dpi=110)
