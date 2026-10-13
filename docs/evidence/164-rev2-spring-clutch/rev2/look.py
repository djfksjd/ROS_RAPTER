"""Zero-action hold on model_rev 2: time series and rendered frames."""
import json, sys
from pathlib import Path
import numpy as np, mujoco
ROOT = Path.cwd(); sys.path.insert(0, str(ROOT/'sim/rl')); sys.path.insert(0, str(ROOT/'modeling'))
from run_env import RunEnv
from train_run import env_kwargs
out = Path(sys.argv[1]); rev = int(sys.argv[2])
kw = env_kwargs(json.loads((ROOT/'sim/rl/runs/scr_a8_h2/args.json').read_text()))
if rev == 2:
    kw['springs_override'] = {'knee_pitch': [170, .956, 'uni+', 'latch']}
env = RunEnv(**{**kw, 'level': 0., 'episode_s': 30., 'model_rev': rev}, randomize=False, seed=7, render_mode='rgb_array')
env.reset(); m, d = env.model, env.data; env.command[:] = 0; env.resample_steps = 0
r = mujoco.Renderer(m, 480, 640); cam = mujoco.MjvCamera(); cam.distance, cam.azimuth, cam.elevation = 2.6, 90, -8
from PIL import Image
class iio:
    imwrite = staticmethod(lambda p, a: Image.fromarray(a).save(p))
com = lambda: d.subtree_com[env.root].copy()
print('t   base_z pitch  comx-toe_x  knee  hip  kneeTau hipTau  grfL grfR  term')
for i in range(100):
    if i % 10 == 0:
        cam.lookat[:] = [d.xpos[env.base][0], 0, .45]; r.update_scene(d, cam); iio.imwrite(out/f'stand_rev{rev}_{i:03d}.png', r.render())
    _, _, term, _, info = env.step(np.zeros(len(env.policy_idx)))
    R = d.xmat[env.base].reshape(3, 3); pitch = np.arctan2(-R[2, 0], R[0, 0])
    pads = [d.geom_xpos[g][0] for g in env.pad_geoms]
    k, h = env.ix['left_knee_pitch_joint'], env.ix['left_hip_pitch_joint']
    if i % 5 == 0 or term:
        print(f"{d.time:4.2f} {d.xpos[env.base][2]:.3f} {pitch:+.3f} com_x {com()[0]:+.3f} pads[{min(pads):+.3f},{max(pads):+.3f}] {d.qpos[env.q_adr[k]]:.3f} {d.qpos[env.q_adr[h]]:+.3f} {d.actuator_force[env.act[k]]:+6.1f} {d.actuator_force[env.act[h]]:+6.1f} {info['grf']['left']:.2f} {info['grf']['right']:.2f} hit={info['terms']['collision']} tilt={info['tilt']:.2f}")
    if term:
        cam.lookat[:] = [d.xpos[env.base][0], 0, .45]; r.update_scene(d, cam); iio.imwrite(out/f'stand_rev{rev}_end.png', r.render()); break
