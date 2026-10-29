"""Randomized-fall ablation (evaluate_run episode pattern as evidence 165 temp_probe.py): randomize on, one group
returned to nominal. argv: run_dir group cmd episodes [seed0=5000]. Groups: all (nothing off), impulse (yaw/roll impulses + push),
only_yaw / only_roll / only_push (just that impulse kept, everything else on), plant (body mass, foot friction, tau_max), init (start pose noise, start speed, start heat), delay (action/obs delay
at nominal 2 steps), sensor (IMU bias, encoder offset), motor (motor/friction/thermal/ambient draws at nominal)."""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, 'sim/rl')
import realism
from evaluate import load
from run_env import RunEnv
from train_run import env_kwargs
d, grp, vx, n = Path(sys.argv[1]), sys.argv[2], float(sys.argv[3]), int(sys.argv[4]); secs = 60.
seed0 = int(sys.argv[5]) if len(sys.argv) > 5 else 5000
kw = {**env_kwargs(json.loads((d/'args.json').read_text())), 'level': 0., 'episode_s': secs+5.}
if grp == 'impulse':
    kw.update(yaw_impulse=None, roll_impulse=None, disturb_items=[i for i in ('pitch', 'yaw', 'trip', 'touchdown')])
if grp in ('only_yaw', 'only_roll', 'only_push'):  # every other randomization stays on, only one of the three impulses
    kw.update(yaw_impulse=kw['yaw_impulse'] if grp == 'only_yaw' else None, roll_impulse=kw['roll_impulse'] if grp == 'only_roll' else None,
              disturb_items=['pitch', 'yaw', 'trip', 'touchdown']+(['push'] if grp == 'only_push' else []))
if grp == 'init':
    kw.update(init_noise=0., init_speed=False)
if grp == 'plant':
    _orig = RunEnv._randomize
    def _nom(self):
        r = self.randomize; self.randomize = False
        try: _orig(self)
        finally: self.randomize = r
    RunEnv._randomize = _nom
if grp == 'motor':
    realism.Realism._draw = lambda self, r, nominal=1., size=None: nominal if size is None else np.full(size, nominal, float)
_rz_reset = realism.Realism.reset
def _rz(self, env):
    _rz_reset(self, env)
    dd = env.data
    if grp == 'delay' and self.on['delay']:
        c = self.cfg['delay']; self.k_act, self.k_obs = int(c['action_steps']['nominal']), int(c['obs_steps']['nominal'])
        self.act_line = realism.DelayLine(self.k_act, dd.ctrl[env.act].copy())
        if self.k_obs: self.obs_line = realism.DelayLine(self.k_obs, self.sensed)
    if grp == 'sensor':
        if self.on['imu']: self.gyro_bias, self.accel_bias = np.zeros(3), np.zeros(3)
        if self.on['encoder']: self.enc_offset = np.zeros(self.n)
realism.Realism.reset = _rz
_imp = RunEnv.apply_impulse
def _log(self, axis, impulse, duration=.04, body=None):
    self.imp_log.append((round(float(self.data.time), 2), axis, round(float(impulse), 3)))
    return _imp(self, axis, impulse, duration, body)
RunEnv.apply_impulse = _log
rows = []
for ep in range(n):
    env = RunEnv(**kw, randomize=True, seed=seed0+ep); env.init_seed = None
    model, venv = load(str(d/'model.zip'), env)
    venv.reset(); env.imp_log = []
    if grp == 'init': env.heat[:] = 0.
    env.command = np.array([vx, 0., 0.]); env.resample_steps = 0
    obs = venv.normalize_obs(env._obs()[None])
    tw, vxs, t_fall = 0., [], None
    for t in range(int(secs/.02)):
        raw, _, term, trunc, info = env.step(model.predict(obs, deterministic=True)[0][0])
        obs = venv.normalize_obs(raw[None])
        if term: t_fall = round((t+1)*.02, 2); break
        w = np.asarray(info.get('realism', {}).get('t_winding', []), float); tw = max(tw, float(np.nanmax(w))) if w.size else tw
        vxs.append(info['v_body'][0])
    rows.append(dict(seed=seed0+ep, t_fall=t_fall, vx=round(float(np.mean(vxs[100:])), 2) if len(vxs) > 100 else None, tw=round(tw, 1), impulses=env.imp_log))
print(json.dumps(dict(run=d.name, group=grp, cmd=vx, n=n, falls=sum(r['t_fall'] is not None for r in rows), episodes=rows)), flush=True)
