"""Finite CEM, two modes; run only after reading the pre-declaration in README.md."""
import hashlib
import json
import time
from pathlib import Path
import numpy as np
from jump_probe import E, ROOT, RUN, GUESS, encode, make, rollout, score, loads, save_trace

INPUTS = [ROOT/'sim/rl/rev3_l80.json', ROOT/'sim/rl/realism_robstride_l80.json', RUN/'args.json', RUN/'model.zip',
          Path(__file__), E/'jump_probe.py']

def hashes():
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in INPUTS}

def main():
    path = E/'eval/search.jsonl'
    if path.exists():
        raise SystemExit('Refusing to overwrite a declared search. Preserve the existing evidence.')
    before = hashes(); start = time.process_time(); wall = time.monotonic()
    env = make(); selected = {}; counts = {}; history = []
    partial = E/'eval/search_partial.jsonl'
    cached = [json.loads(line) for line in partial.read_text().splitlines()] if partial.exists() else []
    with path.open('x') as log:
        for mode, seed in [('vertical', 19901), ('forward', 19902)]:
            rng = np.random.default_rng(seed); best_score = -np.inf; highest = -np.inf; best_land = -np.inf
            count = 0; mode_best = {}
            for restart in range(2):
                mean = encode(GUESS) if restart == 0 else rng.uniform(-.6, .6, len(GUESS))
                std = np.full(len(mean), .65)
                for generation in range(10):
                    xs = np.clip(rng.normal(mean, std, (32, len(mean))), -1, 1)
                    xs[0] = mean
                    scores = []
                    for x in xs:
                        if time.process_time()-start > 5900:
                            raise RuntimeError('CPU budget guard: stop and inspect saved candidates; no automatic extension')
                        cached_row = next((r for r in cached if r['mode'] == mode and r['candidate'] == count+1), None)
                        if cached_row:
                            assert np.array_equal(x, cached_row['x'])
                            result, a = cached_row['result'], None
                        else:
                            result, a = rollout(env, x, capture=True)
                        if a is not None and not all(np.isfinite(v).all() for v in a.values()):
                            raise RuntimeError('Non-finite instrumented state')
                        s = score(result, mode); scores.append(s); count += 1
                        row = dict(mode=mode, restart=restart, generation=generation, candidate=count, score=s,
                                   x=x.tolist(), cpu_s=time.process_time()-start, result=result)
                        log.write(json.dumps(row)+'\n'); log.flush()
                        ev = result['event']; metric = (ev['apex_rise_m'] if mode == 'vertical' else ev['distance_m']) if ev else -np.inf
                        landed = ev and ev['landed'] and result['t_fall'] is None and result['stand_after_landing_s'] >= 1.
                        eligible = landed and (abs(ev['landing_offset_m']) <= .25 if mode == 'vertical' else ev['distance_m'] > 0)
                        for label, improves in [('score', s > best_score), ('extreme', metric > highest), ('landed', eligible and metric > best_land)]:
                            if improves:
                                mode_best[label] = row
                                if a is not None:
                                    mode_best[label]['loads'] = loads(env, a, ev)
                                    save_trace(E/f'eval/{mode}_{label}.npz', a, env)
                        best_score = max(best_score, s); highest = max(highest, metric)
                        if eligible: best_land = max(best_land, metric)
                    elite = xs[np.argsort(scores)[-8:]]
                    mean = .25*mean+.75*elite.mean(axis=0)
                    std = np.maximum(.10, .25*std+.75*elite.std(axis=0))
                    progress = dict(mode=mode, restart=restart, generation=generation, candidates=count,
                                    best_score=best_score, best_metric=highest, landed_found='landed' in mode_best,
                                    cpu_s=time.process_time()-start)
                    history.append(progress); print(json.dumps(progress), flush=True)
                    (E/'eval/search_progress.json').write_text(json.dumps(dict(history=history, selected={**selected, mode:mode_best}), indent=2)+'\n')
            selected[mode] = mode_best; counts[mode] = count
    after = hashes()
    assert before == after
    summary = dict(input_sha256=before, hashes_unchanged=True, cached_candidates=len(cached), candidates=counts, cpu_s=time.process_time()-start,
                   wall_s=time.monotonic()-wall, selected=selected, history=history)
    (E/'eval/search_summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    env.close()

if __name__ == '__main__': main()
