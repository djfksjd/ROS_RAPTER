# 체크포인트

## 2026-09-29 저녁 R-02 (최신)

R-02 모델링 정비 완료(evidence 84): 단일 원천 Xacro, 외형 메시, 최종 모델 정책 runs/r02C_flat. 재개: 40km/h 요구 사양서 → T1 → T2.
모델 재생성: .venv-sim/bin/python modeling/build_r02.py. 실행 예: train.py --design ../raptor_r02.xml --crouch -0.5271 1.8762 -1.8762 --weights '{"track_lin":4,"track_yaw":2,"tall":-300}'.

## 2026-09-29 12축 RL 단계 B

평평한 패드 12축 MuJoCo 정책. 단계 A → B1d(지연) → B2(잡음·발가락·무게중심) → B3(펄스·초기 속도) 학습 완료.
evidence/82: 학습 넘어짐 급증은 확률적 행동 때문, 결정적 평가 주원인은 힘 펄스 40N. `sim/rl/dr_ablation.py`로 재현.
B4·B5 모두 기준 불합격(evidence 82 §7-8), B4가 기준선. 재개 지점: 사용자 결정(§8 선택지 1~3) 후 진행.
학습 산출물은 `sim/rl/runs/`(Git 제외). Gazebo 실험 컨테이너는 정지 상태.

## 2026-09-28 외형 단계 (최신)

디지티그레이드 다리 설계와 morphloom 외형 완료(review 초안). DIGITIGRADE_APPEARANCE.ko.md.
꼬리 0.95m·쐐기형 머리·디테일 반영 후 MuJoCo·Gazebo 기립·RViz 재검증 완료, 줄무늬 음영 해소(evidence/66).
참고 모습 스타일 digitigrade_low 추가(evidence/67): 기립·흔들기 통과, 보폭 걷기 깨끗한 걸음 기준 미달, Gazebo 정지 기립.
Gazebo digitigrade 흔들기 재현·STOP 감시 완료(evidence/68). MuJoCo와 위상 ~46ms 차이.
MuJoCo JTCLikeServo로 위상 차이 46→~21ms 설명, 보폭 결과 영향 작음.
Gazebo 보폭 시도 불합격(evidence/69): STOP 2회, MuJoCo actuator 근사 불일치 확인.
꼬리 균형·속도 탐색 완료(evidence/70): 꼬리 효과 작음, 개루프 보행 한계 ~0.16m/s(MuJoCo), Gazebo 보폭 불합격 유지.
보행 동기 꼬리 완료(evidence/71): MuJoCo 빠른 보행 kv 전 범위 전도 없음, Gazebo는 roll 전도·위상 추정 부정확.
꼬리 동결. roll 전도 진단 완료(evidence/72-diagnosis), 사용자 확인 대기(1단계 끝에서 정지 지시).
0단계·실험 A(MuJoCo) 완료 후 예측 20% 이상 불일치로 정지(evidence/73-lateral). 사용자 판단 대기.
흔들목마 모델 1·2단계 진행 완료(evidence/74-rocking): 기본 보행 = 2.5Hz 개루프 + 긴 raised-cosine 램프(Gazebo 5/5).
재개 지점: Fy/Fz(0.41) 대 착지 발바닥(외전 시 0.05) 맞바꿈에 대한 사용자 결정, 또는 actuator 사양 범위로 kv20 재평가.
Gazebo 실행은 --experiment leg_design:=digitigrade. 실험 컨테이너는 정지 상태.

## 2026-09-28 재개 결과

재개 첫 작업 2번을 기록·소스·오프라인 계산으로 수행했다. 상세 LATERAL_SUPPORT_FEASIBILITY.ko.md.
- physics 1ms/DART 6.13.2, 제어 100Hz, position 경로 = 10ms 지연 위치의 P 속도 명령(유효 gain 30/s).
- 54번 발목 속도 한도 표본 다수는 명령 포화로 설명되지 않음. shift 중 양발 모서리 접촉 확인.
- 평평한 발 정적 한 발 지지: 전체 관절 한도에서도 측방 COM 여유 음수(-0.008m).
- 떨림 기전(접촉+속도 constraint vs 지연 servo 진동)은 미확정. 1ms 단위 계측이 필요하나
  방향 결정 전에는 우선순위가 낮다.

사용자가 A(동적 흔들기 보행, 10축 유지)를 선택했다. 라이선스/revision은 기존 조사 문서를 재사용했다.
MuJoCo 모델 대응과 정적 기립 비교까지 완료: MUJOCO_MJX_ROADMAP.ko.md 진행 기록.
open-loop 흔들기 시험 완료(58): 교대 지지는 μ1에서만 안정, 보폭 전진은 깨끗한 걸음 미확립.
closed-loop 단계 완료(59, 60): 교대 지지 견고, 깨끗한 걸음은 servo kv 가정(30)에서만.
Gazebo servo 식별 완료(61, 62): DART 발목 포화 결함, kv는 고정 베이스로 결정 불가. SIMULATION_METHODOLOGY.ko.md.
다음 재개 지점: actuator 후보 사양 범위 확정(사용자) → MuJoCo actuator 모델 교체·범위 평가. 실험 컨테이너는 정지 상태.

## 이전 일시정지 체크포인트 — 2026-09-27

사용자 요청: 다음 체크포인트까지만 진행하고 잠시 중단, 모두 백업.

## 마지막 완료 작업

브랜치 `test/joint-load-observation`.
10개 position command interface를 유지하고 10개 읽기 전용 effort state를 추가했다.
관절 속도와 전달 effort를 시간 연속 계측한다. 제어 gain/토크 한도 변경 없음.
53번은 인터페이스, 54번은 toe stiffness scale=3 비교 계측 결과다.

설치: gz_ros2_control 1.2.20-1noble.20260904.032933, 실제 position gain .3.
[동일 버전 소스](https://github.com/ros-controls/gz_ros2_control/blob/1.2.20/gz_ros2_control/src/gz_system.cpp)
에서 position 오차를 update_rate와 gain으로 속도 목표로 변환하는 경로를 확인했다.
effort는 JointTransmittedWrench의 관절축 투영이며 모터 명령 토크와 동일하다고 단정하지 않는다.

54번: 80개 표본, 버퍼 초과 0. shift 중 왼발목 속도가 ±2.5rad/s에 반복 도달.
실험은 tracking error .14852rad로 실패했다. 전달 effort 일부 표본은 37.33Nm였지만,
관측만으로 모터 토크 포화나 유일한 원인을 확정하지 않는다.
강성 기본값은 1, gain .3, 속도 제한 2.5, effort 제한 80을 유지했다.

## 재개 첫 작업

1. git status, WORK_STATE 확인. 이 체크포인트에서 새 실험을 자동 재시작하지 않는다.
2. 발목 속도 포화와 접촉/수치 진동을 구분: 실제 SDF/physics 시간 간격과 설치된 physics 구현 확인.
3. 새 근거에 따라 한 변수 비교. 무작정 gain/토크/속도 한도를 높이지 않는다.
4. 이후 안정 지지 → 보행 → 지형 → MuJoCo/MJX → AI mission/evaluation 순서를 유지.

## 남은 상태

외형은 참고 이미지에 아직 미달. 안정 보행/급경사 대응/강화학습/탐색 mission은 미완료.
Qwen/NanoJev 기본 STAND/STOP 및 ROS 환경 복구는 기존 검증 기록에 있다.
프로젝트 Docker 컨테이너와 이번 세션의 Ollama 서버를 종료한 상태로 일시정지한다.
다른 사용자 Docker 자원과 앱은 종료하지 않는다.

## 백업 범위

GitHub: 소스, Blender 원본, 메시, 문서, 실제 검증 기록과 개발 브랜치.
비공개 Hugging Face: 최신 전체 Git bundle 및 기존 Qwen/NanoJev 모델 자산.
.env/토큰/가상환경/build/install/log는 백업 제외. 재현 지침은 저장소에 있다.

## 2026-09-28 사용 한도로 중단 (WIP → 같은 날 검증 완료, evidence/66)

사용자 결정: 꼬리·머리·디테일 반영(다리 길이·자세 유지). 레퍼런스 Tripo GLB는 참고용만(저장소 미포함).
완료: digitigrade 꼬리 링크 0.5→0.95m(0.8kg, 폭 0.065, legacy URDF 수치 동일 확인), 쐐기형 머리·꼬리 16마디·
기어열·볼트·배선 등 디테일(349부품, morphloom review-pass), 링크 GLB 탄젠트 포함 재출력(14MB).
미검증: 꼬리 변경 후 MuJoCo 기립·흔들기·보폭, Gazebo 기립·줄무늬 음영 해소, RViz GLB 표시.
재개: MuJoCo 재검증 → Gazebo(spawn_z 0.1075) 기립·스크린샷 → RViz → 결과 문서화.
