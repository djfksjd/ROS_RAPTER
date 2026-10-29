#!/bin/bash
# zero-shot terrain sweep, run from repo root: 2 policies x 3 commands x 28 grounds x 8 seeds
source .venv-sim/bin/activate
E=docs/evidence/173-terrain-zero-shot
{ for run in rev2_hr05_ev_r01 rev2_hd40_ev_r01; do for v in 1 2 4; do
    echo "$run flat 0 $v"
    for a in 0.01 0.02 0.03 0.04 0.05 0.06 0.08; do echo "$run rough $a $v"; done
    for g in 5 10 15 20 25; do echo "$run slope_up $g $v"; echo "$run slope_down $g $v"; done
    for g in 5 10 15 20; do echo "$run side_slope $g $v"; done
    for r in 0.02 0.04 0.06 0.08 0.10 0.12; do echo "$run stairs $r $v"; done
  done; done; } | xargs -P 8 -L 1 sh -c 'python '$E'/eval/terrain_probe.py sim/rl/runs/$0 $1 $2 $3 8 2>/dev/null' > $E/eval/terrain.jsonl
wc -l $E/eval/terrain.jsonl
