import sys, json
from multiprocessing import Pool
OUT = sys.argv[0]; run = sys.argv[1]; sys.argv = ['x', run, '/dev/null', '32']
sys.path.insert(0, 'output/pdf/RAPTOR_최종문서_20261012/evidence/rev2/train')
import disturb_tol as dt
if __name__ == '__main__':
    tasks = [('roll', s, c, 96001+i) for s in (1.25, 1.5, 1.75, 2.0) for c in (4., 5.5, 7.) for i in range(32)]
    with Pool(8) as p: R = p.map(dt.trial, tasks, chunksize=2)
    json.dump(R, open(OUT.replace('.py', f'_{run}.json'), 'w'))
    for s in (1.25, 1.5, 1.75, 2.0): print(run, s, [f"{sum(r['fell'] for r in R if r['size']==s and r['cmd']==c)}/32" for c in (4., 5.5, 7.)])
    print([ (r['cmd'], r['seed'], round(r['t_after'],2)) for r in R if r['fell'] and r['size']==2.0])
