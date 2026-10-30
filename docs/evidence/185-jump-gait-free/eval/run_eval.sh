#!/bin/bash
# post-training evaluation of rev2_jp3 (height scan + boxes terrain, from rev2_tr4), run from repo root
set -u
source .venv-sim/bin/activate
E=docs/evidence/185-jump-gait-free; mkdir -p $E/eval $E/media $E/policy
R=sim/rl/runs; X=docs/evidence/182-drop-ledge-baseline/eval
mk() {  # mk name rate realism(1/0): flat-ground eval copy (terrain removed, scan kept)
  d=$R/rev2_jp3_$1; mkdir -p $d; cp $R/rev2_jp3/{model.zip,vecnorm.pkl} $d/
  python3 -c "
import json;a=json.load(open('$R/rev2_jp3/args.json'));a.pop('terrain',None);a['roll_impulse'][0]=$2;a['name']='rev2_jp3_$1'
a.pop('realism',None) if not $3 else a.__setitem__('realism','sim/rl/realism_robstride_v2.json')
json.dump(a,open('$d/args.json','w'),indent=1)"
}
mk st_r01 0.1 1; mk st_r02 0.2 1; mk st_norz 0.1 0
cp $R/rev2_jp3/{model.zip,vecnorm.pkl,args.json,progress.csv} $E/policy/
# 1) drop / ledge (the target), jp3
{ for v in 1 2 4; do
    for h in 0.1 0.2 0.3 0.4 0.5 0.6 0.8 1.0; do echo "rev2_jp3_st_r01 drop $h $v"; done
    for h in 0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45; do echo "rev2_jp3_st_r01 ledge $h $v"; done
  done; } | xargs -P 8 -L 1 sh -c 'python '$X'/edge_probe.py sim/rl/runs/$0 $1 $2 $3 8 2>/dev/null' > $E/eval/edge.jsonl
echo edge done
# 2) terrain suite (regression check against tr4)
T=docs/evidence/173-terrain-zero-shot/eval/terrain_probe.py
{ for v in 1 2 4; do
    echo "rev2_jp3_st_r01 flat 0 $v"
    for a in 0.01 0.02 0.03 0.04 0.05 0.06 0.08; do echo "rev2_jp3_st_r01 rough $a $v"; done
    for g in 5 10 15 20 25; do echo "rev2_jp3_st_r01 slope_up $g $v"; echo "rev2_jp3_st_r01 slope_down $g $v"; done
    for g in 5 10 15 20; do echo "rev2_jp3_st_r01 side_slope $g $v"; done
    for r in 0.02 0.04 0.06 0.08 0.10 0.12; do echo "rev2_jp3_st_r01 stairs $r $v"; done
  done; } | xargs -P 8 -L 1 sh -c 'python '$T' sim/rl/runs/$0 $1 $2 $3 8 2>/dev/null' > $E/eval/terrain.jsonl
echo terrain done
# 3) flat: impulses, nominal speed, 10-minute heat
python docs/evidence/165-realism-robstride-training/eval/temp_probe.py $R/rev2_jp3_st_r01 nodr 8 2>/dev/null > $E/eval/probe_nominal.jsonl &
{ for c in r01 r02 norz; do for v in 2 4 5.5 7; do echo "$R/rev2_jp3_st_$c all $v"; done; done; } \
  | xargs -P 7 -L 1 sh -c 'python docs/evidence/166-realism-dr-fall-ablation/eval/ablate.py $0 $1 $2 24 2>/dev/null' > $E/eval/ablate.jsonl
wait
{ for m in plate worst; do for v in 4 7; do echo "$v $m"; done; done; } | xargs -P 4 -L 1 sh -c 'python docs/evidence/167-robstride-public-data-fit/eval/long_run.py '$R'/rev2_jp3_st_r01 $0 600 $1 2>/dev/null' > $E/eval/long_run.jsonl
echo flat done
# 4) videos (2 m/s, seed 5000)
M=$E/media; P=$X/edge_probe.py
python $P $R/rev2_jp3_st_r01 drop 0.5 2 1 5000 $M/jp3_drop50cm_2mps.mp4 2>/dev/null > $E/eval/video_runs.jsonl &
python $P $R/rev2_jp3_st_r01 drop 0.8 2 1 5000 $M/jp3_drop80cm_2mps.mp4 2>/dev/null >> $E/eval/video_runs.jsonl &
python $P $R/rev2_jp3_st_r01 ledge 0.3 2 1 5000 $M/jp3_ledge30cm_2mps.mp4 2>/dev/null >> $E/eval/video_runs.jsonl &
python $P $R/rev2_jp3_st_r01 ledge 0.4 2 1 5000 $M/jp3_ledge40cm_2mps.mp4 2>/dev/null >> $E/eval/video_runs.jsonl &
wait
for f in jp3_drop50cm_2mps jp3_ledge30cm_2mps jp3_ledge40cm_2mps; do ffmpeg -loglevel error -y -i $M/$f.mp4 -vf "fps=12,scale=480:-1" $M/$f.gif; done
echo all done
