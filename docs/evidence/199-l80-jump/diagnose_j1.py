"""Bounded, read-only j1 diagnosis. No learn(), run writes or normalizer updates."""
import csv
import hashlib
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
E = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'sim/rl'))
from evaluate import load
from run_env import RunEnv
from train_run import env_kwargs
from jump_task import JumpTask
from jump_probe import Recorder, intervals, longest, make, keyframes, action_at, reset_stationary, measurements
from check_training_j1 import metrics, stop_reason, judge_text, BLOCK, MAX_STEPS
from launch_j1 import verify_inputs, plan
from stable_baselines3 import PPO

OUT = E/'eval/j1_diagnosis'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit():
    verify_inputs()
    run = ROOT/'sim/rl/runs/l80_j1'
    args = json.loads((run/'args.json').read_text())
    assert all(args.get(k) == v for k, v in plan().items())
    hist = json.loads((run/'jump_checks.json').read_text())
    with (run/'progress.csv').open() as f:
        log = list(csv.DictReader(f))
    lengths = [float(r['rollout/ep_len_mean']) for r in log if r.get('rollout/ep_len_mean')]
    stats=[float(v) for r in log for k,v in r.items() if v and (k.startswith('train/') or k=='rollout/ep_rew_mean')]
    assert np.isfinite(stats).all()
    checks = []
    for i, h in enumerate(hist):
        rows = [json.loads(l) for l in (E/f'eval/l80_j1/quick_{i+1}M.jsonl').read_text().splitlines()]
        m = metrics(rows)
        assert all(m[k] == h[k] for k in m)
        assert h['added_steps'] == (i+1)*BLOCK
        # All observed updates are finite and episode lengths never approach collapse.
        reason = stop_reason(i+1, lengths, hist[:i], m)
        assert reason == h['decision']
        judge_text(OUT/f'quick_{i+1}M_judge.txt', rows)
        checks.append({**h, 'seeds': [e['seed'] for r in rows for e in r['episodes']]})
    hashes = {n: digest(run/n) for n in ('model.zip', 'vecnorm.pkl', 'args.json')}
    assert all(v == digest(run.with_name('l80_j1_3M')/n) for n, v in hashes.items())
    return dict(checks=checks, added_steps=hist[-1]['added_steps'], budget=MAX_STEPS,
        declared_ceiling=8_000_000, remaining_unused=MAX_STEPS-hist[-1]['added_steps'],
        initial_steps=json.loads((run/'initial-policy.json').read_text())['steps'],
        final_hashes=hashes, matches_3M_snapshot=True, input_contract_matches=True,
        training_statistics_finite=True,
        episode_length_range=[min(lengths), max(lengths)],
        first10_length_mean=float(np.mean(lengths[:10])), last10_length_mean=float(np.mean(lengths[-10:])))


def model_input_audit():
    run=ROOT/'sim/rl/runs/l80_j1';model=PPO.load(run/'model.zip',device='cpu')
    with (run/'vecnorm.pkl').open('rb') as f:vn=pickle.load(f)
    columns={key:dict(request_l2=float(torch.linalg.vector_norm(w[:,-1])),
                     command_abs_max=float(torch.abs(w[:,6:9]).max()))
             for key,w in model.policy.state_dict().items()
             if key.endswith('.0.weight') and w.ndim==2 and w.shape[1]==56}
    return model.num_timesteps,dict(norm_reward=vn.norm_reward,obs_rms_type=type(vn.obs_rms).__name__,
        request_mean=float(vn.obs_rms.mean[-1]),request_std=float(np.sqrt(vn.obs_rms.var[-1]+vn.epsilon)),
        return_std=float(np.sqrt(vn.ret_rms.var+vn.epsilon)),first_layer_columns=columns,
        alpha_6hz=float(1-np.exp(-2*np.pi*6*.02)),step_response_at_100ms=float(1-np.exp(-2*np.pi*6*.1)))


def rewards(rows, selected):
    values = [r for r in rows if selected(r)]
    seconds = .02*len(values)
    keys = rows[0]['weighted'].keys()
    sums = {k: sum(r['weighted'][k] for r in values) for k in keys}
    return dict(seconds=seconds, sums=sums, per_second={k:v/max(seconds,.02) for k,v in sums.items()},
        bonus=sum(r['bonus'] for r in values), total=sum(r['reward'] for r in values))


def policy_probe(name, seed, stochastic):
    run = ROOT/'sim/rl/runs'/name
    args = json.loads((run/'args.json').read_text())
    env = RunEnv(**{**env_kwargs(args), 'episode_s':12.}, randomize=True, seed=seed)
    model, vn = load(str(run/'model.zip'), env)
    # Source policy keeps its 55 inputs; attach only passive measurements + task bonuses.
    source = env.jump_task is None
    if source:
        env.jump_task = JumpTask(dict(auto_request_s=2., flight_bonus=10., success_bonus=30.))
    torch.manual_seed(seed)
    raw, _ = env.reset(seed=seed)
    rows, deltas, angles, normalized = [], [], [], []
    def obs(raw):
        return vn.normalize_obs((raw[:-1] if source else raw)[None])
    norm_meta = dict(return_std=float(np.sqrt(vn.ret_rms.var+vn.epsilon)), gamma=model.gamma,
        std=model.policy.log_std.exp().detach().cpu().numpy().tolist(),
        request_mean=None if source else float(vn.obs_rms.mean[-1]),
        request_std=None if source else float(np.sqrt(vn.obs_rms.var[-1]+vn.epsilon)),
        command_means=vn.obs_rms.mean[6:9].tolist(),
        command_stds=np.sqrt(vn.obs_rms.var[6:9]+vn.epsilon).tolist())
    fell = None
    with Recorder(env) as rec:
        for k in range(600):
            if not source:
                pair = np.repeat(raw[None], 2, axis=0); pair[:, -1] = [0., 1.]
                pair = vn.normalize_obs(pair)
                with torch.no_grad():
                    means = model.policy.get_distribution(torch.as_tensor(pair, dtype=torch.float32)).distribution.mean.cpu().numpy()
                delta = np.clip(means[1], -1, 1)-np.clip(means[0], -1, 1)
                deltas.append(delta); angles.append(delta*env.scale[env.policy_idx]); normalized.append(pair[:, -1])
            action = model.predict(obs(raw), deterministic=not stochastic)[0][0]
            raw, reward, term, trunc, info = env.step(action)
            weighted = {k:.02*env.weights[k]*float(v) for k,v in info['terms'].items()}
            bonus = reward-sum(weighted.values())+(10. if term else 0.)
            rows.append(dict(t=float(env.data.time), request=env.jump_task.command, reward=reward,
                weighted=weighted, bonus=bonus, flight=bool(info['flight']), vz=float(info['v_body'][2])))
            if term or trunc:
                fell = float(env.data.time) if term else None
                break
    a = rec.arrays(); task = env.jump_task.result()
    request = task['request_time_s']; mask = a['t'] >= (request if request is not None else 12.)
    off = (a['feet_n'] == 0).all(axis=1) & mask
    reference = task['standing_height_m']
    flights = [dict(start_s=float(a['t'][s]), duration_s=(e-s)*.001,
                    apex_rise_m=float(a['xyz'][s:e,2].max()-reference)) for s,e in intervals(off)]
    out = dict(policy=name, seed=seed, stochastic=stochastic, task=task, t_fall=fell,
        elapsed_s=float(env.data.time), normalization=norm_meta, max_flight_s=longest(off), flights=flights,
        max_apex_rise_m=float(a['xyz'][mask,2].max()-reference) if mask.any() else None,
        min_rise_m=float(a['xyz'][mask,2].min()-reference) if mask.any() else None,
        max_upward_vz=float(a['velocity'][mask,2].max()) if mask.any() else None,
        reward_before=rewards(rows, lambda r:r['request']==0.),
        reward_requested=rewards(rows, lambda r:r['request']==1.),
        reward_flight=rewards(rows, lambda r:r['flight']),
        counterfactual=None if source else dict(states=len(deltas),
            action_rms=float(np.sqrt(np.mean(np.square(deltas)))), action_abs_max=float(np.abs(deltas).max()),
            angle_abs_max_rad=float(np.abs(angles).max()), per_joint_rms=np.sqrt(np.mean(np.square(deltas),axis=0)).tolist(),
            normalized_request_0_1=normalized[0].tolist()),
        active=[env.active[i] for i in env.policy_idx])
    stem=f'{name}_{seed}_{"sampled" if stochastic else "mean"}'
    keys=list(rows[0]['weighted'])
    np.savez_compressed(OUT/f'{stem}.npz', **a,
        control_t=[r['t'] for r in rows], reward_keys=keys,
        weighted_rewards=[[r['weighted'][key] for key in keys] for r in rows],
        control_rewards=[r['reward'] for r in rows], control_bonus=[r['bonus'] for r in rows])
    vn.close()
    return out


def cem_probe(lpf):
    env = make()
    # Score the historical trajectory using j1's standing terms; only reward bookkeeping changes.
    jargs = json.loads((ROOT/'sim/rl/runs/l80_j1/args.json').read_text())
    env.weights.update(json.loads(jargs['weights']))
    # Preserve the historical frozen-model 6 Hz reproduction; disable only the filter for the pair.
    if lpf is None:
        env.action_alpha = None
    summary = json.loads((E/'eval/search_summary.json').read_text())
    selected = summary['selected']['forward']['score']
    assert selected['candidate'] == 13 and selected['result']['event']['apex_rise_m'] > .14
    x = selected['x']
    reset_stationary(env)
    env.command = np.zeros(3); env.resample_steps = 0
    times, values = keyframes(env, x); rows=[]; fell=None
    with Recorder(env) as rec:
        for k in range(180):
            _, reward, term, trunc, info = env.step(action_at(k*.02,times,values))
            rows.append(dict(t=float(env.data.time), reward=reward, bonus=0.,
                weighted={key:.02*env.weights[key]*float(v) for key,v in info['terms'].items()},
                flight=bool(info['flight'])))
            if term or rec.rows['body_hit'][-20:].count(True):
                fell=float(env.data.time);break
    a=rec.arrays(); result=measurements(env,a,fell,launch_after=times[2])
    result.update(action_lpf_hz=lpf, alpha=env.action_alpha, keyframe_times_s=times.tolist(),
                  candidate=13, reward=rewards(rows,lambda r:True),
                  reward_flight=rewards(rows,lambda r:r['flight']),
                  reward_push=rewards(rows,lambda r:times[2] <= r['t'] <= times[3]))
    np.savez_compressed(OUT/f'cem_lpf_{lpf}.npz',**a)
    env.close()
    return result


def main():
    torch.set_num_threads(1)
    OUT.mkdir(parents=True, exist_ok=False)
    start=time.process_time(); result=dict(audit=audit(), episodes=[])
    result['audit']['final_steps'],result['saved_model_input_audit']=model_input_audit()
    for name in ('l80_s1p','l80_j1'):
        for stochastic in (False,True):
            for seed in range(9600,9604):
                row=policy_probe(name,seed,stochastic);result['episodes'].append(row)
                print(name,seed,stochastic,row['task'],flush=True)
                assert time.process_time()-start < 1800, 'diagnostic CPU budget'
    result['cem']=[cem_probe(6.),cem_probe(None)]
    result['cpu_s']=time.process_time()-start
    (OUT/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print('wrote',OUT/'summary.json',flush=True)


if __name__=='__main__': main()
