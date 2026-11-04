#!/bin/bash
# Evidence 198: criteria, budget, checks and stopping rules are declared in README.md.
# From the repository root; do not overwrite an existing run.
set -euo pipefail
NAME=${NAME:-l80_t1}
if [[ -e "sim/rl/runs/$NAME" ]]; then
  echo "Run exists: sim/rl/runs/$NAME; choose a new NAME after reviewing its evidence" >&2
  exit 1
fi
PY=.venv-sim/bin/python
MIX=$("$PY" -c 'import json; print(json.dumps(json.load(open("docs/evidence/198-l80-turn/commands.json"))))')
W='{"track_lin": 5, "progress": 3, "track_yaw": 4, "yaw_err": -1, "stand": -5, "gait": -8, "fold": 0.35, "extend": 0.3, "ankle_static": -0.2, "retract": 0.7, "leg_sym": -6, "heat": -2, "spring_store": 0.03, "spring_release": -0.06, "joint_limit": -30, "self_collision": -5, "hold": -3, "stand_support": -5, "slip_lin": -5, "action_rate": -2}'
exec "$PY" -u sim/rl/train_run.py --name "$NAME" --steps 8000000 --envs 8 --model-rev 3 --arch8 --real-mass --tail active --springs c \
  --springs-override '{"knee_pitch": [500, 0.956, "uni+", "ratchet"]}' --ankle-clutch --target-ramp \
  --rev3-config sim/rl/rev3_l80.json --realism sim/rl/realism_robstride_l80.json --contact-tc 0.005 --nominal -0.5273 1.0 -0.9997 \
  --cmd 6 0 1.5 --cmd-final 6 0 1.5 --fixed-curriculum --cmd-mix "$MIX" --hold-settle 0.1 \
  --episode-s 20 --level 0.5 \
  --disturb-items push pitch yaw --roll-impulse 0.1 0.1 0.5 --init-noise 0.04 --heat-limit 0.8 --track-sigma-frac 0.3 \
  --scale-override '{"hip_roll": 0.5}' --symmetric --action-lpf 6 --kp-err 0.2 --obs-hold '{"noise":0.02,"walk":0.005}' \
  --init sim/rl/runs/l80_r1_6M/model.zip --resume-action-std 0.15 --weights "$W" \
  --checkpoint-check docs/evidence/198-l80-turn/check_training.py
