"""Geometry check (no training): can the swing knee reach the latch rest angle q0 before the foot meets the ground?
Usage: python geometry_probe.py <out.json> [run=rev2_n10]. Run from the project root.
Part A  flat-foot leg geometry as a function of the knee angle (pantograph: metatarsus pitch = base pitch + hip + c).
Part B  touchdown / lift-off states of the deterministic policy (seed 96001, 20 s) and two counterfactuals in which only the
        knee is set to q0: (1) hip unchanged, (2) hip re-solved so the foot keeps its fore-aft position. Each gives the pad
        clearance the foot would have had; negative = the foot would be below the ground.
Part C  the same counterfactual over the whole late swing (from peak flexion to touchdown)."""
import json, sys
from pathlib import Path
import numpy as np, mujoco
ROOT = Path.cwd(); sys.path.insert(0, str(ROOT/'sim/rl'))
from evaluate import load
from run_env import RunEnv
from train_run import env_kwargs
RUN = ROOT/'sim/rl/runs'/(sys.argv[2] if len(sys.argv) > 2 else 'rev2_n10'); Q0 = 0.956
kw = env_kwargs(json.loads((RUN/'args.json').read_text()))
env = RunEnv(**{**kw, 'episode_s': 30., 'level': 0.}, randomize=False, seed=7); env.reset(seed=1); m = env.model; sd = mujoco.MjData(m)
C = float(env.q0[env.ix['left_knee_pitch_joint']]+env.q0[env.ix['left_ankle_pitch_joint']])
J = {s: {j: env.q_adr[env.ix[f'{s}_{j}_joint']] for j in ('hip_pitch', 'knee_pitch', 'ankle_pitch')} for s in ('left', 'right')}
PADS = {s: [g for g in env.pad_geoms if m.body(m.geom_bodyid[g]).name.startswith(s)] for s in ('left', 'right')}
THIGH = {s: m.body(f'{s}_thigh_link').id for s in ('left', 'right')}; FOOT = env.foot_body


def fk(qpos, side, knee=None, hip=None):
    """Kinematics of a copy of qpos with the knee (and coupled ankle) and optionally the hip of one side replaced."""
    sd.qpos[:] = qpos
    if knee is not None:
        sd.qpos[J[side]['knee_pitch']] = knee; sd.qpos[J[side]['ankle_pitch']] = C-knee
    if hip is not None:
        sd.qpos[J[side]['hip_pitch']] = hip
    mujoco.mj_kinematics(m, sd); mujoco.mj_comPos(m, sd)
    ext = lambda g: m.geom_size[g][0] if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE else abs(sd.geom_xmat[g].reshape(3, 3)[2]) @ m.geom_size[g]
    R = sd.xmat[FOOT[side]].reshape(3, 3); hd = sd.xmat[env.root+1].reshape(3, 3)[:, 0]; hd = np.array([hd[0], hd[1], 0.])/np.hypot(hd[0], hd[1])
    rel = sd.xpos[FOOT[side]]-sd.xpos[THIGH[side]]; px = [(sd.geom_xpos[g]-sd.subtree_com[env.root]) @ hd for g in PADS[side]]
    return dict(clear=float(min(sd.geom_xpos[g][2]-ext(g) for g in PADS[side])), foot_x_hip=float(rel @ hd), foot_z_hip=float(rel[2]),
                foot_x_com=float((sd.xpos[FOOT[side]]-sd.subtree_com[env.root]) @ hd), pad_x_com=[float(min(px)), float(max(px))],
                foot_pitch=float(np.arcsin(-R[2, 0])), hip_z=float(sd.xpos[THIGH[side]][2]))


def solve_hip(qpos, side, knee, target_x):
    """Hip pitch that keeps the foot's fore-aft position relative to the hip (bisection around the current hip angle)."""
    h0 = qpos[J[side]['hip_pitch']]; g = lambda h: fk(qpos, side, knee, h)['foot_x_hip']-target_x
    hs = np.linspace(h0-.6, h0+.6, 121); v = [g(h) for h in hs]
    i = [k for k in range(120) if v[k]*v[k+1] <= 0]
    if not i:
        return None
    k = min(i, key=lambda k: abs(hs[k]-h0)); lo, hi = hs[k], hs[k+1]
    for _ in range(40):
        mid = (lo+hi)/2; lo, hi = (mid, hi) if g(lo)*g(mid) > 0 else (lo, mid)
    return float((lo+hi)/2)


# ---- Part A: flat foot, level trunk; only the knee varies
env.reset(seed=1); q = env.data.qpos.copy(); q[3:7] = [1, 0, 0, 0]; q[:3] = [0, 0, 1.]
A = []
for k in (0.74, 0.80, 0.90, 0.956, 1.00, 1.05, 1.10, 1.20, 1.288, 1.40, 1.50):
    qq = q.copy()
    for s in ('left', 'right'):
        qq[J[s]['knee_pitch']] = k; qq[J[s]['ankle_pitch']] = C-k
    f = fk(qq, 'left'); A.append(dict(knee=k, hip_height=f['hip_z']-f['clear'], foot_x_hip=f['foot_x_hip'], foot_x_com=f['foot_x_com'], pad_x_com=f['pad_x_com'], foot_pitch=f['foot_pitch']))
    print('A', json.dumps({a: (np.round(b, 4).tolist()) for a, b in A[-1].items()}))

# ---- Parts B and C
out = dict(q0=Q0, coupling_c=C, links=dict(thigh=.22, shin=.38, metatarsus=.32), flat_foot=A, speeds={})
ms = lambda x: [float(np.mean(x)), float(np.min(x)), float(np.max(x))] if len(x) else None
for cmd in (2., 4., 7.):
    env.init_seed = 96001; model, venv = load(str(RUN/'model.zip'), env); venv.reset(); venv.reset()
    env.command = np.array([cmd, 0., 0.]); env.resample_steps = 0; obs = venv.normalize_obs(env._obs()[None]); d = env.data
    hist = []
    for k in range(1000):
        a = model.predict(obs, deterministic=True)[0][0]; raw, _, term, _, info = env.step(a); obs = venv.normalize_obs(raw[None])
        hist.append((d.qpos.copy(), dict(info['loaded']), float(info['v_body'][0])))
        if term: break
    res = {}
    for s in ('left', 'right'):
        ev = dict(td=[], lo=[]); late = []
        for i in range(251, len(hist)):
            qp, ld, vx = hist[i]; was = hist[i-1][1][s]
            if ld[s] and not was:  # touchdown: state of the first loaded control step
                f = fk(qp, s); kq = float(qp[J[s]['knee_pitch']]); hq = float(qp[J[s]['hip_pitch']])
                c1 = fk(qp, s, Q0); h2 = solve_hip(qp, s, Q0, f['foot_x_hip']); c2 = fk(qp, s, Q0, h2) if h2 is not None else None
                # late swing: from the peak knee flexion of this swing to the last unloaded step
                j = i-1; sw = []
                while j > 0 and not hist[j][1][s]:
                    sw.append(j); j -= 1
                sw = sw[::-1]
                if len(sw) > 2:
                    pk = int(np.argmax([hist[t][0][J[s]['knee_pitch']] for t in sw])); seg = sw[pk:]
                    kn = [float(hist[t][0][J[s]['knee_pitch']]) for t in seg]; cl = [fk(hist[t][0], s)['clear'] for t in seg]
                    cf = [fk(hist[t][0], s, Q0)['clear'] for t in seg]
                    late.append(dict(knee_min=min(kn), knee_last=kn[-1], clear_last=cl[-1], cf_clear_max=max(cf), cf_clear_last=cf[-1],
                                     cf_positive_frac=float(np.mean(np.array(cf) > 0)), steps=len(seg)))
                ev['td'].append(dict(knee=kq, hip=hq, foot_pitch=f['foot_pitch'], foot_x_com=f['foot_x_com'], foot_x_hip=f['foot_x_hip'], hip_z=f['hip_z'], vx=vx,
                                     cf_same_hip_clear=c1['clear'], cf_same_hip_dx=c1['foot_x_hip']-f['foot_x_hip'],
                                     cf_same_x_clear=c2 and c2['clear'], cf_same_x_dhip=c2 and h2-hq, cf_same_x_foot_pitch=c2 and c2['foot_pitch']))
            if was and not ld[s]:
                qp0 = hist[i-1][0]; f = fk(qp0, s); ev['lo'].append(dict(knee=float(qp0[J[s]['knee_pitch']]), hip=float(qp0[J[s]['hip_pitch']]), foot_x_com=f['foot_x_com'], foot_pitch=f['foot_pitch'], hip_z=f['hip_z']))
        col = lambda L, k: [e[k] for e in L if e[k] is not None]
        res[s] = dict(n_td=len(ev['td']), td={k: ms(col(ev['td'], k)) for k in ev['td'][0]}, lo={k: ms(col(ev['lo'], k)) for k in ev['lo'][0]},
                      late_swing={k: ms(col(late, k)) for k in late[0]}, swings_where_q0_fits=int(sum(e['cf_clear_max'] > 0 for e in late)), swings=len(late))
        t = res[s]['td']; l = res[s]['late_swing']
        print(cmd, s, 'td knee %.3f hip %.3f foot_pitch %.3f x_com %.3f hip_z %.3f | lo knee %.3f x_com %.3f' % (t['knee'][0], t['hip'][0], t['foot_pitch'][0], t['foot_x_com'][0], t['hip_z'][0], res[s]['lo']['knee'][0], res[s]['lo']['foot_x_com'][0]))
        print('    knee=q0 same hip: clear %.4f (min %.4f max %.4f) dx %.4f | same x: clear %.4f dhip %.4f foot_pitch %.3f' % (*t['cf_same_hip_clear'], t['cf_same_hip_dx'][0], t['cf_same_x_clear'][0], t['cf_same_x_dhip'][0], t['cf_same_x_foot_pitch'][0]))
        print('    late swing: knee_min %.3f clear_last %.4f cf_clear_max mean %.4f max %.4f, swings where q0 fits %d/%d' % (l['knee_min'][0], l['clear_last'][0], l['cf_clear_max'][0], l['cf_clear_max'][2], res[s]['swings_where_q0_fits'], res[s]['swings']))
    out['speeds'][f'{cmd:g}'] = res
json.dump(out, open(sys.argv[1], 'w'), indent=1)
