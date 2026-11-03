"""Evidence 197: knee load split of l80_r1 while running (randomize off, realism on, constant command): motor torque
(actuator force) vs passive joint torque (ratchet spring + joint damping/friction, d.qfrc_passive), share of steps the
ratchet pawl is engaged, stance knee angle range. Usage: knee_probe.py RUN_DIR CMD SECONDS"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'sim/rl'))
import stage_gates as sg  # noqa: E402

d, cmd, secs = Path(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3])
a = json.loads((d/'args.json').read_text())
kw = {**sg.env_kwargs(a), 'level': 0., 'episode_s': secs+5., 'disturb_items': [], 'roll_impulse': None, 'yaw_impulse': None,
      'cmd_max': [0., 0., 0.], 'init_speed': False}
env = sg.RunEnv(**kw, randomize=False, seed=6800); env.init_seed = 6800
model, venv = sg.load(str(d/'model.zip'), env); venv.reset()
env.resample_steps = 10**9; env.command = np.array([cmd, 0., 0.])
obs = venv.normalize_obs(env._obs()[None]); dd = env.data
i = env.active.index('left_knee_pitch_joint'); dof = env.v_adr[i]
mot, pas, armed, q_st, loaded = [], [], [], [], []
for k in range(int(secs/.02)):
    raw, _, term, _, info = env.step(model.predict(obs, deterministic=True)[0][0]); obs = venv.normalize_obs(raw[None])
    if k < 150:
        continue
    mot.append(float(dd.actuator_force[env.act][i])); pas.append(float(dd.qfrc_passive[dof]))
    armed.append(bool(info['latched'].get('left', False)))
    ld = bool(info['loaded']['left']); loaded.append(ld)
    if ld:
        q_st.append(float(dd.qpos[env.q_adr[i]]))
    if term:
        break
mot, pas, loaded = np.array(mot), np.array(pas), np.array(loaded)
# same motion, stiffer ratchet spring (k x f): the spring torque scales with k at the same deflection, the joint torque
# need is unchanged, so motor = motor + passive - f*passive (kinematic replay, an estimate: the gait would change)
alt = {f'k{int(500*f)}': round(float(np.sqrt(np.mean((mot-(f-1.)*pas)**2))), 1) for f in (1.5, 2., 3.)}
rms = lambda x: round(float(np.sqrt(np.mean(x**2))), 1)
print(json.dumps(dict(run=d.name, cmd=cmd, motor_rms_nm=rms(mot), motor_rms_stance=rms(mot[loaded]), passive_rms_nm=rms(pas),
                      passive_rms_stance=rms(pas[loaded]), pawl_engaged_share=round(float(np.mean(armed)), 2),
                      stance_share=round(float(loaded.mean()), 2),
                      stance_mean_motor_nm=round(float(mot[loaded].mean()), 1), stance_mean_passive_nm=round(float(pas[loaded].mean()), 1),
                      stance_share_opposing=round(float(np.mean(mot[loaded]*pas[loaded] < 0)), 2),
                      motor_rms_if_stiffer_spring=alt,
                      stance_knee_rad=[round(float(np.min(q_st)), 2), round(float(np.max(q_st)), 2)] if q_st else None)))
