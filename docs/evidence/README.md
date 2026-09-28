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

- `46-gait-imu-cancel.*`: 실제 IMU 한계 초과와 action 취소 승인/STATUS_CANCELED.
- `47-gait-forward-crouch.*`, `48-forward-crouch-analysis.json`: hip -.05 비교 실패.
- `49-gait-middle-crouch.*`: hip -.10 비교, shift 이후 감시 중 실패.

- `50-gait-transition-*`: 시간 연속 IMU/관절/접촉 재현.
- `51-toe-stiffness-contract.txt`: 강성 외 모델 불변/SDF 변환 확인.
- `52-stiff-toe-*`: 스프링 강성 3배 비교, 발목 추종 오차 실패.

- `53-joint-load-interfaces.txt`: 10 active 제어/effort 상태 확인.
- `54-joint-load-*`: 실제 속도/전달 effort 계측, 발목 속도 포화와 추종 실패.

- `55-control-path-sources.txt`: 설치 버전과 gz_ros2_control/gz-sim/DART 소스 경로 대조.
- `55-joint-load-reanalysis.json`: 54번 재분석, 명령으로 설명 안 되는 속도 한도 표본과 모서리 접촉.
- `56-lateral-support-feasibility.json`: 평평한 발 정적 한 발 지지의 측방 COM 여유(오프라인 FK).
- `57-mujoco-crouch-standing.json`, `57-mujoco-zero-start-failure.json`: MuJoCo 변환 모델 정적 기립 비교.
- `58-mujoco-lateral-rocking.json`: MuJoCo open-loop 흔들기/보폭 견고성(60s, 질량·마찰 변화). 보행 성공 아님.
- `59-mujoco-rocking-robustness.json`: elliptic cone 모델 흔들기 open/closed-loop 9조건 60s.
- `60-mujoco-stride-clean-steps.json`: 보폭+발목 피드백 깨끗한 걸음 지표, servo kv 30/50/100 비교. 조건부 결과.
- `61-gazebo-servo-step-*.json`: 고정 베이스 servo 계단 응답(발목/무릎/hip, 발가락 유무, 0.5ms, Bullet-FS). DART 발목 포화 결함 재현.
- `62-gazebo-bullet-featherstone-standing.txt`: Bullet-FS 전신 기립 실패 기록.
- `63-digitigrade-leg-design.txt`: 디지티그레이드 설계 FK·MuJoCo 비교. Gazebo 항목은 legacy였다는 정정 포함.
- `63-digitigrade-gazebo-standing.png`: 정정 대상 — legacy 모델 화면(leg_design 미전달).
- `64-*`: morphloom 외형 digitigrade Gazebo 정지 기립, Blender 렌더, morphloom run report.
- `65-wip-*`: 꼬리·머리 변경 WIP Blender 미리보기(미검증 시점).
- `66-*`: 0.95m 꼬리 MuJoCo 재검증, Gazebo 기립·줄무늬 해소, RViz GLB 표시, Blender 렌더.
- `74-rocking/`: 흔들목마 모델 실험 F(자유 흔들림) — 첫 뜸→착지는 대체로 예측 안, kv20 이탈·고유 주기 추론 불일치로 정지.
- `73-lateral/`: 측방 안정 0단계(정지 하중비·거울 보행·낮은 자세 FK)와 실험 A(MuJoCo) — 예측 불일치로 정지.
- `72-diagnosis/`: Gazebo 빠른 보행 roll 전도 진단(1단계) — 시계열·주파수·설정 대조·가설 3개.
- `71-*`: 보행 동기 꼬리(설계 노트 규칙) MuJoCo 탐색·60초 확인, Gazebo 비교, /gait/phase 노드.
- `70-*`: 꼬리 균형(yaw·pitch) MuJoCo 탐색, 속도 sweep, Gazebo 재시험, 감시기 개선(지속 30ms, 명령 속도 검사).
- `69-*`: digitigrade Gazebo 보폭 시도 — 감시기 STOP 2회(발목 속도 포화, 기울기)로 불합격, MuJoCo kv 교차 점검.
- `68-*`: digitigrade Gazebo 흔들기 재현(포화 감시 STOP 포함)과 MuJoCo 비교.
- `67-*`: 참고 모습 스타일(`leg_design:=digitigrade_low`) 낮은 자세·MuJoCo 비교·Gazebo 기립·Blender 렌더·morphloom run report.
