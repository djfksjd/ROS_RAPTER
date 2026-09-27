# Actual evidence — 2026-09-27

Screenshots are captured from the running local noVNC desktop. Logs and JSON files
are actual program outputs; they are not reconstructed terminal examples.
Trailing whitespace in the simulation log was normalized; message content is unchanged.

| File | What it establishes |
|---|---|
| `03-installation-controllers.png`, `installation.txt` | Ubuntu/Jazzy/Gazebo versions, successful build, active controllers |
| `09-gazebo-detailed-robot.png` | Actual Gazebo GUI with detailed Raptor model |
| `12-rviz.png` | Actual RViz RobotModel and Global Status OK; dark material issue remains |
| `raptor-concept-render.png` | Blender appearance render only, not simulation evidence |
| `01-motion-initial-failure.*` | Free-body individual-joint trial fell; retained failure |
| `02-fixture-joint-test.json` | 10 small joint commands with body fixed for bench testing |
| `04-static-standing.json`, `09-detailed-static-standing.json` | Limited-duration free-body neutral stance with active control |
| `06-predefined-step.json`, `07-alternating-step.json` | Predefined trajectories tracked, without proof of walking |
| `06-gazebo-step-experiment.png` | Actual primitive-model gait experiment GUI |
| `08-*` | IMU/RGB-D/contact message evidence |
| `10-step-contact-test.json` | Both feet remained in contact during intended lift |
| `11-larger-shift-step.json` | Right foot unloaded, then robot tilted beyond limit: failed gait |
| `05-*` | Qwen → ROS gate STAND, unsupported mission rejection, STOP latch |
| `13-*` | Actual NanoJev decision model → ROS STAND target reached |
| `*evaluation.json` | Development-set model experiments, not independent final evaluation |
| `*-holdout.json` | Fixed 38-command comparison; no robotics execution success implied |
| `safety-tests.txt` | Pure command contract and gate-policy unit tests |
| `model-contract.txt` | Same physics and 10 actuated joints with detailed visuals |
| `model-backup-assets.json` | HF model asset size/hash verification receipt |

A successful joint target or static pose does not establish stable walking,
field autonomy, mechanical manufacturability or flexible-tail dynamics.

## Passive toe follow-up

See [passive toe results](../PASSIVE_TOES.ko.md) for files 15–28 and their initial
conditions. Files 14–15, 18, 21, 23 and 28 record failures; they are not success
evidence. File 24 used the original startup-relative IMU reference; file 27
repeats the aligned slope test with explicit world ENU IMU orientation.
Screenshots 20 and 25 are actual Gazebo views of the small step and 5° ramp.
None establishes rough-terrain walking or steep-slope climbing.

Files 30–33 retain the failed faster balance response, Qwen/NanoJev/STOP
revalidation with passive joint feedback, final hardware interface query and build.

File 34 verifies that refused development probes emit no trajectory commands.

- `35-reference-appearance-build.txt`: 외형 개선 후 colcon/URDF 검사.
- `36-reference-appearance-control.txt`: 재실행 후 active controller와 10개 claimed interface.
- `37-reference-appearance-gazebo.png`: 개선 메시가 표시된 실제 Gazebo 화면.

- `38-passive-toe-visual-build.txt`: 수동 발 상세 메시 적용 후 빌드/URDF 검사.
- `39-passive-toe-visual-contract.txt`: 비시각 모델 불변, 10 active + 12 passive 확인.
- `40-passive-toe-visual-gazebo.png`: 실제 Gazebo 수동 발 메시 표시 확인.

- `41-gait-contact-details.*`: roll .28의 실제 접촉 상세 재현.
- `42-gait-larger-shift.*`, `43-gait-support-analysis.json`: roll .39 전도 및 FK/contact hull 분석.
- `44-gait-middle-shift.*`, `45-gait-middle-support-analysis.json`: roll .36 전도 및 분석.
