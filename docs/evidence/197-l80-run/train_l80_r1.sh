#!/bin/bash
# Evidence 197: stage 3 running on the frozen 0.80 m leg, continued from the stage 2 pass l80_w2 (evidence 196).
# Walk settings plus: forward command range grown by the deterministic evaluation curriculum (EvalCurriculum: x1.15 when
# 4 eval episodes at the current top speed fall at most once and reach 80 % of it) from 3 to 11.1 m/s; 10 % zero commands,
# 10 % in-place turns (cmd_turn), hold anchor after the stop (hold_settle 0.1), init speed on, command inputs widened
# (train_run.widen_command_inputs). Budget 8M steps; kill / B-A rules in README.md.
source .venv-sim/bin/activate
W='{"track_lin": 5, "progress": 3, "track_yaw": 4, "yaw_err": -1, "stand": -5, "gait": -8, "fold": 0.35, "extend": 0.3, "ankle_static": -0.2, "retract": 0.7, "leg_sym": -6, "heat": -2, "spring_store": 0.03, "spring_release": -0.06, "joint_limit": -30, "self_collision": -5, "hold": -3, "stand_support": -5, "slip_lin": -5, "action_rate": -2}'
python sim/rl/train_run.py --name ${NAME:-l80_r1} --steps ${STEPS:-8000000} --envs 8 --model-rev 3 --arch8 --real-mass --tail active --springs c \
  --springs-override '{"knee_pitch": [500, 0.956, "uni+", "ratchet"]}' --ankle-clutch --target-ramp \
  --rev3-config sim/rl/rev3_l80.json --realism sim/rl/realism_robstride_l80.json --contact-tc 0.005 --nominal -0.5273 1.0 -0.9997 \
  --cmd 3 0.3 1.0 --cmd-final 11.1 0.3 1.0 --eval-curriculum --zero-cmd 0.1 --top-cmd 0.3 --cmd-turn 0.1 --hold-settle 0.1 \
  --episode-s 20 --level 0.5 \
  --disturb-items push pitch yaw --roll-impulse 0.1 0.1 0.5 --init-noise 0.04 --heat-limit 0.8 --track-sigma-frac 0.3 \
  --scale-override '{"hip_roll": 0.5}' --symmetric --action-lpf 6 --kp-err 0.2 --obs-hold '{"noise":0.02,"walk":0.005}' \
  --init ${INIT:-sim/rl/runs/l80_w2/model.zip} --resume-action-std ${STD:-0.2} --weights "$W"
