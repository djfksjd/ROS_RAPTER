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
- `75-feasibility/`: 현재 구동기 사양 물리 타당성(MuJoCo, kv 30) — 점프 속도 배율 스윕, 낙하 착지, 경사 정적, 달리기 해석, 권고 사양. 정책 학습 없음.
- `74-rocking/`: 흔들목마 모델 F·E·D·C, 2단계(PLL·anti-pump), 분리 실험, 긴 램프 보행(Gazebo 5/5·거울 3/3·꼬리 4/4), 기준 대비표.
- `73-lateral/`: 측방 안정 0단계(정지 하중비·거울 보행·낮은 자세 FK)와 실험 A(MuJoCo) — 예측 불일치로 정지.
- `72-diagnosis/`: Gazebo 빠른 보행 roll 전도 진단(1단계) — 시계열·주파수·설정 대조·가설 3개.
- `71-*`: 보행 동기 꼬리(설계 노트 규칙) MuJoCo 탐색·60초 확인, Gazebo 비교, /gait/phase 노드.
- `70-*`: 꼬리 균형(yaw·pitch) MuJoCo 탐색, 속도 sweep, Gazebo 재시험, 감시기 개선(지속 30ms, 명령 속도 검사).
- `69-*`: digitigrade Gazebo 보폭 시도 — 감시기 STOP 2회(발목 속도 포화, 기울기)로 불합격, MuJoCo kv 교차 점검.
- `68-*`: digitigrade Gazebo 흔들기 재현(포화 감시 STOP 포함)과 MuJoCo 비교.
- `67-*`: 참고 모습 스타일(`leg_design:=digitigrade_low`) 낮은 자세·MuJoCo 비교·Gazebo 기립·Blender 렌더·morphloom run report.
- [77 — 발목 roll 추가(12 능동축)](77-ankle-roll/README.md): Gazebo 12축 제어기·정지 확인, 옆 경사 정지 10축 10°까지 → 12축 20°까지.
- [78 — RL 기준선 결함](78-rl-baseline/README.md) · [79 — 12축 평지 RL 보행](79-rl-walk12/README.md) · [80 — 발바닥 형상 비교](80-sole/README.md) · [81 — Gazebo 이식 판정](81-sim2sim/README.md) · [82 — 무작위화 영향 분리·펄스 조정](82-dr-ablation/README.md) · [83 — 낮은 자세·R-02 첫 학습](83-r02/README.md)

- [87 — 동일 조건 구조 비교 재개](87-structure-comparison/README.md): 평가 경계값·실패 누락 수정, 8축/12축 × 골반 위치의 학습 seed 0 평가. 8축 후방 골반 지속 6.17 m/s(가정한 모터/발열), 회전 추종·40 km/h 미달.

- [88 회전 원인 진단과 대칭 벌점 분리](88-turn-ablation/README.md): 두 후보 회전 실패, sym0 발열 경계 실패 보존, yaw 보상 단일 변수 후속 시험.

- [89 회전 접촉 모멘트](89-turn-wrench/README.md): 부호·비간섭 검증, 1 ms 운동량 수지, 명령 분포 단일 변수 학습.

- [90 회전 평가 원값·명령 시점 수정](90-turn-evaluation/README.md): 회귀 검사, 출발별 원값, 기준선 회전0/20 통과.

- [91 회전 명령 분포 비교](91-turn-curriculum/README.md): 두 후보 회전·고속발열 실패, 입력 확장 전도, 탐색 분산 후속 비교.

- [92 기립·가감속·정지 기준선](92-motion-baseline/README.md): 전도0과 동작 성공을 분리, 8회 모두 실패, 전환 평가 하니스.

- [93: 탐색 분산 비교](93-turn-exploration/README.md) — 전 축std0.25 회전 실패·전도11/30·발열 퇴행, 미채택.

- [94: 좌우 반사와yaw 보정](94-turn-reflection/README.md) — 6.4 독립회전117/120·전도0/180·직진발열통과이나3회실패로미채택.

- [95: yaw 상태 피드백](95-yaw-feedback/README.md) — 보정량 제한으로전도0/30이나회전19/20·미채택.

- [96: 정차 진단·반사 평균 학습 초기화](96-stand-and-symmetric-initialization/README.md) — 고정목표25회실패·초기800상태일치·별도학습중.

- [97: 반사 PPO 학습 평가](97-symmetric-learning-evaluation/README.md) — 503,808추가학습·회전/기립미달·명령순서비교중.

- [98: 명령순서대조](98-command-sequence-comparison/README.md) — 기립/전환무전도개선·회전/정지미달·roll목표비교중.

- [99 — 고관절 roll 목표 범위 비교](99-roll-target-range/README.md): 기존 전도0/18·회전0/12, 확대18/18회전전전도.

- [100 — 선택적 선형 속도 오차 보상 비교](100-dense-speed-reward/README.md): 대조0/비교-2 학습 시작, 평가 미완료.

- [101 — 회전축 한정 탐색](101-targeted-turn-exploration/README.md): 고관절roll·꼬리yaw에만std0.25, 평가대기.

- [102 — 회전 구동기 토크 계측](102-turn-actuator-telemetry/README.md):12롤아웃·6조건계측on/off정확일치,모터부족확정못함.

- [103 — 배우 clipping 진단](103-actor-clipping-diagnosis/README.md):상쇄clip가설실표본에서미관측·배우미수정.

- [104 — 숨은 스프링 상태](104-hidden-spring-state/README.md):관측/명령동일합성반사실3조건에서다음물리상태차이,실패원인확정아님.

- [105 — 접지상태관측초기화](105-contact-observation-initialization/README.md):46/48특징·8행동,source와9초기rollout정확일치·학습효과미검증.

- [106 — loaded 관측 비교 평가](106-contact-observation-evaluation/README.md):중간양쪽전도0/30·회전0/20,최종학습진행중.

- [107: loaded 관측의 학습 활용 진단](107-contact-observation-use/README.md) — 추가 입력 비영·가상 반전 출력 민감도, 실제 궤적 계측동등.

- [108: 명령별 보상·접촉 분포](108-mode-reward-distribution/README.md) — 영점 명령 단발지지와 stand항 적용범위, 계측 비간섭 확인.

- [109: 영점 명령 양발 지지 항 비교](109-stand-support-ablation/README.md) — 기본/이동 보상·물리 동등 검증, 0/−5 대조 학습 중.

- [110: 양발 지지 항 비교 평가](110-stand-support-evaluation/README.md) — 중간250k 보존 및 평가 중, 최종 독립 평가 대기.

- [111: 기립 위상 입력 고정 진단](111-stand-phase-input/README.md) — 기존 정책의 입력만 멈추면4/4전도, 미채택.

- [112: 탐색 분포·수집 명령 비율](112-action-distribution/README.md) — 과도한전축잡음가설미지지·실제yaw명령약5%.

- [113: 회전 명령 노출2초/6초](113-turn-command-exposure/README.md) — episode길이/다른명령보존·대조학습중.

- [114: 회전명령 노출2/6초 평가](114-turn-duration-evaluation/README.md) — 중간250k 보존 및원래2초회전기준평가중.

- [115: 초기 반사 배우 첫 PPO 업데이트](115-first-ppo-update/README.md) — 4k 전후각도8/8, 즉시퇴행가설미지지·통합미완료.

- [116 — PPO 추가학습 회전 퇴행](116-progressive-ppo-update/README.md): 16k6/8·64k0/8, 진단출발재사용·미채택.

- [117 — PPO KL 중단 비교](117-ppo-kl-guard/README.md): 새출발18/20 vs1/20, 최소속도미달·미채택.

- [118 — KL·속도벌점 비교](118-kl-speed-reward/README.md): 회전12/20·속도10/20절충·미채택.

- [119 — 회전속도 반응계측](119-turn-speed-sensitivity/README.md): 8회·예측계측궤적불변.

- [120 — 회전 속도입력 보정 진단](120-turn-speed-feedforward/README.md): 초기검증실패보존·v2평가진행.

- [121 — 유계 속도입력 피드백](121-bounded-speed-feedback/README.md): 동시통과4/8·미채택.

- [122 — 회전 위상주파수](122-turn-phase-rate/README.md): 동시통과3/8·미채택.

- [123 — 위상·속도 상호작용](123-phase-speed-interaction/README.md): 동시통과4/8·미달.

- [124 — 회전 전진/횡방향 수지](124-longitudinal-turn-balance/README.md): 1ms6000샘플·궤적비간섭확인.

- [125 — 제한 횡입력 피드백](125-lateral-feedback/README.md): 동시통과0/8·미달.

- [126 — 횡속도 보상 학습](126-lateral-reward/README.md): 회전0/20·최소속도0/20·미채택.

- [127 — 첫 배치 value/advantage](127-first-rollout-advantages/README.md): 각4096샘플·정책불변.

- [128 — KL·카탈로그 평가](128-kl-catalogue-evaluation/README.md): 속도/전환미달·미채택.

- [129 — 첫 PPO gradient 분리](129-first-gradient/README.md): actor/value/entropy 및 이전Adam이력 진단.

- [130 — optimizer 이력 비교](130-optimizer-history/README.md): 초기화계약통과·평가진행.

- [131 — 학습률 비교](131-learning-rate/README.md): 초기화계약통과·학습진행.

- [132: 회전 입력·속도 피드백 2×2 비교](132-input-factor-comparison/README.md) — 개발 회전 동시 6/8, 미채택.

- [133: 좌우 회전 접촉·CoM 수지](133-turn-contact-balance/README.md) — 좌회전 전진 일·하중 차이, 계측 궤적 일치.

- [134: 회전 진입 시점 비교](134-turn-entry-phase/README.md) — 각도40/40·속도24/40, 위상 선택만으로 미해결.

- [135: 정차 무릎 스프링 하중 대조](135-stand-knee-spring-load/README.md) — 초기부하영향은 있으나해제만으로기립미해결.

- [136: 기립 토크·접촉 지지](136-stand-support-torque/README.md) — 원스프링초기무릎제한접근과평가초기속도확인.

- [137: 초기속도0 기립 대조](137-zero-initial-speed-stand/README.md) — 초기속도제거만으로기립미해결.

- [138: 무릎 스프링 정지각 대조](138-stand-spring-rest-angle/README.md) — 예압제거만으로기립미해결.

- [139: 고관절 국소 응답](139-stand-action-response/README.md) — 피드백부호검증.
- [140: 고관절 자세 PD](140-stand-orientation-feedback/README.md) — 기준/정지각후보모두기립실패.

- [141: 수동 발가락 강성 대조](141-stand-toe-stiffness/README.md) — 강성3배만으로기립미해결.

- [142: 정차 전용 학습 경로](142-stand-skill-training/README.md) — 원기구/관측 보존·PPO pilot 진행, 성능미확정.

- [143: 회전입력 시작 필터](143-turn-yaw-input-filter/README.md) — 속도일부개선이나동시통과퇴행·미채택.

- [144: 정차학습 최종실패/접촉](144-stand-skill-evaluation/README.md) — 503808step완료·기립0/4·꼬리접촉.
- [145: terminal벌점 대조](145-stand-terminal-penalty/README.md) — 초기정책/기구유지·벌점100학습진행.

- [146: 속도입력과정책시계](146-turn-internal-clock/README.md) — 원기준동시통과개선없음·미채택.

- [147: 벌점100 기립최종평가](147-stand-penalty-evaluation/README.md) — 503808step완료·기립0/4·미채택.

- [148: 원기구정적평형/순방향확인](148-static-equilibrium/README.md) — 정적후보2·고정목표기립6회실패.

- [149: 평형 주변 고관절 자세PD](149-equilibrium-orientation-feedback/README.md) — 6회기립전도·상태선형화로원인분리예정.

- [150: 평형선형화/구동영향](150-local-linearization/README.md) — 불안정모드·국소수치모델검증.
- [151: 전체상태피드백60초](151-local-state-feedback/README.md) — 근처20무전도·열포함17/20·전체goal미완료.

- [152: 전류상한/원초기자세](152-stand-current-limit/README.md) — 평형근처60초20/20·원시작0/4·부분관측/전환미완료.

- [153: 기존 관측 기반 기립 추정기](153-stand-observer/README.md) — 평형 근처20/20·원시작0/4·실물 관측 경로 미검증.

- [154: 원시작 연동·접촉·기립 접근](154-reset-contact-and-approach/README.md) — 단일 초기화 수정은 모두0/4; 관절 기반 전환의 공중 진입 확인.

- [155: 원시작 접근 궤적 최적화](155-stand-trajectory-shooting/README.md) — 640평가/32재생·원기립0/32·짧은 목적함수의 한계 확인.

- [156: peak envelope/초기 제동](156-stand-peak-envelope/README.md) — 원시작개발1회60초통과/정확재현, 고정후독립0/20·미채택.

- [157: 관측 기반 제동/접근 피드백](157-observation-conditioned-braking/README.md) — 원성공재현·4변형각0/21·부호/입력검사29검증.

- [158: 접촉·관측기 전환](158-contact-and-handoff/README.md) — 1ms비간섭계측·전환후포화확인·혼합/추정이력변경미채택.

- [159: servo 목표경계/예측검증](159-servo-target-bounds/README.md) — 포화일부감소·개발1/21·국소전이와비선형전이오프라인비교.

- [160: 비선형추정·실제상태제어분리](160-nonlinear-observer/README.md) — 각1/21유지·truth진단도0/2·지지자세/capture영역문제분리.

- [161: 평형 후보/초기구간](161-equilibrium-family-and-prefix/README.md) — 정적3·원capture미해결·21중16은초기gate이미위반.

- [162 초기hip/tailcapture](162-early-hip-tail-capture/README.md): 원개발105대조, 개선실패·접촉이탈진단,기준선보존.

- [163 무릎수직지지](163-knee-vertical-support/README.md): pulse48·개발42,접촉상태별부호변화/미채택.
