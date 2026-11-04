"""Replay selected keyframes through RunEnv.step, fresh env, nominal and pose jitter only."""
import hashlib
import json
import time
import numpy as np
from jump_probe import E, ROOT, RUN, make, rollout, loads, save_trace, Recorder, reset_stationary, measurements


def detail(env, a, result):
    event = result['event']
    if not event: return {}
    start, stop = event['start'], event['stop']
    return dict(apex_rise_from_first_ms_m=float(a['xyz'][start:stop, 2].max()-a['xyz'][0, 2]),
                peak_base_z_m=float(a['xyz'][start:stop, 2].max()), takeoff_velocity_mps=a['velocity'][start].tolist(),
                takeoff_tilt_rad=float(a['tilt'][start]), landing_tilt_rad=float(a['tilt'][stop]) if stop < len(a['t']) else None,
                body_contacts_during_flight=int(a['body_hit'][start:stop].sum()))


def main():
    start = time.process_time(); summary = json.loads((E/'eval/search_summary.json').read_text())
    out = E/'eval/replay.jsonl'
    if out.exists(): raise SystemExit('Refusing overwrite')
    with out.open('x') as log:
        for mode, selected in summary['selected'].items():
            for label, source in selected.items():
                for seed in [None, 9100, 9101, 9102]:
                    env = make()
                    result, a = rollout(env, source['x'], seed=seed, capture=True)
                    tag = f"replay_{mode}_{label}_{'nominal' if seed is None else seed}"
                    save_trace(E/f'eval/{tag}.npz', a, env)
                    row = dict(mode=mode, selection=label, candidate=source['candidate'], seed=seed, result=result,
                               detail=detail(env, a, result), loads=loads(env, a, result['event']), trace=tag+'.npz')
                    log.write(json.dumps(row)+'\n'); log.flush()
                    print(json.dumps({k:v for k,v in row.items() if k != 'loads'}), flush=True)
                    env.close()
    # Diagnostic: nominal pose hold, without a jumping keyframe; not a new optimization candidate.
    env = make(); reset_stationary(env); env.command = np.zeros(3); env.resample_steps = 0
    fallen = None
    with Recorder(env) as rec:
        for k in range(180):
            _, _, term, _, _ = env.step(np.zeros(len(env.policy_idx)))
            if term or rec.rows['body_hit'][-20:].count(True):
                fallen = float(env.data.time); break
    a = rec.arrays(); save_trace(E/'eval/nominal_hold.npz', a, env)
    baseline = dict(result=measurements(env, a, fallen), loads=loads(env, a))
    (E/'eval/nominal_hold.json').write_text(json.dumps(baseline, indent=2)+'\n')
    env.close()
    sources = [E/'jump_probe.py', E/'replay_jump.py', ROOT/'sim/rl/rev3_l80.json', ROOT/'sim/rl/realism_robstride_l80.json', RUN/'args.json']
    (E/'eval/replay_audit.json').write_text(json.dumps(dict(cpu_s=time.process_time()-start, inputs_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}), indent=2)+'\n')

if __name__ == '__main__': main()
