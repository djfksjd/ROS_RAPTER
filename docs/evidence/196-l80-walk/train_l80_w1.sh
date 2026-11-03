#!/bin/bash
# Evidence 196: stage 2 walking on the frozen 0.80 m leg, continued from the stage 1a pass of evidence 195.
# 1a settings plus a command range: forward 0-1.5 m/s, lateral +-0.2, yaw +-0.5 rad/s, 25 % zero commands (stops),
# commands resampled inside the episode (stop <-> start). Budget 8M steps (+ one 4M extension); kill rules in README.md.
source .venv-sim/bin/activate
W='{"track_lin": 5, "progress": 3, "track_yaw": 4, "yaw_err": -1, "stand": -5, "gait": -8, "fold": 0.35, "extend": 0.3, "ankle_static": -0.2, "retract": 0.7, "leg_sym": -6, "heat": -2, "spring_store": 0.03, "spring_release": -0.06, "joint_limit": -30, "self_collision": -5, "hold": -3, "stand_support": -5, "slip_lin": -5, "action_rate": -2}'
python sim/rl/train_run.py --name ${NAME:-l80_w1} --steps ${STEPS:-8000000} --envs 8 --model-rev 3 --arch8 --real-mass --tail active --springs c \
  --springs-override '{"knee_pitch": [500, 0.956, "uni+", "ratchet"]}' --ankle-clutch --target-ramp \
  --rev3-config sim/rl/rev3_l80.json --realism sim/rl/realism_robstride_l80.json --contact-tc 0.005 --nominal -0.5273 1.0 -0.9997 \
  --cmd 1.5 0.2 0.5 --cmd-final 1.5 0.2 0.5 --zero-cmd 0.25 --top-cmd 0.3 --no-init-speed --episode-s 20 --fixed-curriculum --level 0.5 \
  --disturb-items push pitch yaw --roll-impulse 0.1 0.1 0.5 --init-noise 0.04 --heat-limit 0.8 --track-sigma-frac 0.3 \
  --scale-override '{"hip_roll": 0.5}' --symmetric --action-lpf 6 --kp-err 0.2 --obs-hold '{"noise":0.02,"walk":0.005}' \
  --init ${INIT:?stage 1a pass model.zip} --resume-action-std ${STD:-0.2} --weights "$W"
