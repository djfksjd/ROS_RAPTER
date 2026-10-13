# argv: run out.json 32 overrides-json   (same positions disturb_tol.py reads, so spawned workers agree)
import sys, json
from multiprocessing import Pool
sys.path.insert(0, 'output/pdf/RAPTOR_최종문서_20261012/evidence/rev2/train')
import disturb_tol as dt
if __name__ == '__main__':
    tasks = [('roll', s, c, 96001+i) for s in (1.5, 2.0, 3.0) for c in (4., 7.) for i in range(32)]
    with Pool(8) as p: R = p.map(dt.trial, tasks, chunksize=2)
    json.dump(R, open(sys.argv[2], 'w'))
    for s in (1.5, 2.0, 3.0): print(sys.argv[2].split('/')[-1], s, [f"{sum(r['fell'] for r in R if r['size']==s and r['cmd']==c)}/32" for c in (4., 7.)])
