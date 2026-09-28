# MuJoCo·MJX 강화학습 후속 작업

사용자 요청(2026-09-27): ROS 2·Gazebo 작업을 먼저 진행한 뒤 MuJoCo+MJX에서
걷기·균형 강화학습을 하고 정책을 ROS 제어와 연결한다. 현재 문서는 계획이며,
MuJoCo 설치·모델 변환·학습을 완료했다는 보고가 아니다.

## 기준과 순서

1. Gazebo에서 10 active DOF, 수동 발가락·꼬리 구분, 접촉·IMU, 기립 및 보행 기준 측정.
2. 해당 Xacro의 관절축·질량·관성·제한·collision·수동 spring을 MuJoCo MJCF로 변환.
   임의로 다른 로봇으로 교체하지 않는다. floating base를 추가하더라도 actuator는 10개.
3. 무중력 관절축, 중력 낙하, 지면 접촉, 정적 자세를 두 엔진에서 비교한다.
   엔진 간 차이를 동일한 물리라고 주장하지 않는다.
4. CPU 단일 환경 → MJX-JAX 소규모 vmap batch의 처리량·메모리·안정성을 측정한다.
5. 기립 → 작은 속도 추종 → 평지 보행 → 작은 단차·경사 curriculum 순서로 학습한다.
   보상은 속도 추종·넘어짐·미끄럼·발 접촉·동작 크기·에너지/관절 제한을 함께 평가한다.
6. 여러 seed와 미사용 지형에서 predefined controller와 비교한다. 높은 reward만으로
   보행 성공이라고 판단하지 않는다. 낙상률, 추종 오차, 미끄럼, 지형별 성공률을 기록한다.
7. ROS mission gate와 STOP 우선순위를 유지한 정책 어댑터로 Gazebo에서 재검증한다.
   Qwen/NanoJev가 직접 motor action을 생성하는 구조로 바꾸지 않는다.

## 실제 Mac과 가속 지원

확인된 Mac은 Apple M5 / 24GB RAM이며 M5 Pro로 확인된 것은 아니다.
[MuJoCo MJX 공식 문서](https://mujoco.readthedocs.io/en/latest/mjx.html)는
MJX-JAX 실행 대상으로 Apple Silicon을 포함한다. 다만
[JAX 설치 문서](https://docs.jax.dev/en/latest/installation.html)는 Apple GPU 지원을
experimental로 구분하고 macOS의 일반 설치 경로로 CPU를 안내한다.
따라서 Apple Silicon에서 실행 가능하다는 사실을 M5 GPU 가속 보장으로 해석하지 않는다.
설치 시 실제 `jax.devices()`와 workload benchmark를 기록한다. MJX-Warp의 NVIDIA
최적화 경로와 MJX-JAX 경로도 구분한다.

기존 AI 가상환경과 분리해 버전을 고정한다. 대규모 병렬 환경이나 유료 클라우드
자원을 임의로 생성하지 않는다. 학습 코드·환경 설정·seed·평가 결과는 Git,
큰 checkpoint는 비공개 HF 백업에 저장하고 복원도 검증한다.

## 진행 기록 — 2026-09-28 (사용자 선택 A: 동적 흔들기 보행)

측방 정적 한 발 지지가 운동학적으로 불가능하다는 결과([LATERAL_SUPPORT_FEASIBILITY](LATERAL_SUPPORT_FEASIBILITY.ko.md))
뒤, 사용자가 10축을 유지한 동적 보행 방향을 선택했다. 위 순서의 2~3단계 일부를 수행했다.

- 환경: 별도 `.venv-sim`(Python 3.14, mujoco 3.14.0 Apache-2.0, placo 0.10.0 MIT, 설치 약 582MB).
  MJX/JAX는 아직 설치·측정하지 않았다.
- `sim/generate_urdf.sh`: 기존 ROS 이미지에서 xacro만 실행(시뮬레이션 없음).
  `sim/build_model.py`: URDF → `sim/raptor.xml`. floating base 1개, actuator 정확히 10개.
  수동 발가락 12개는 Gazebo spring 값, 발가락 μ0.8, 지면 μ1, self-collision 없음, 1ms step.
- `sim/raptor_servo.py`: gz_ros2_control position 경로 근사(10ms 갱신 위치, P 30/s,
  속도 한도 clip, effort 한도). DART의 외력 속도 한도 constraint는 재현하지 않는다.
- `tests/test_sim_contract.py`: URDF와 질량(17.3kg)·관절축·범위·actuator 한도·
  zero pose COM 일치 3개 통과. `.venv-ai`에서는 skip.

### 정적 기립 비교 (evidence/57-*)

| 조건 | Gazebo(DART) 기록 | MuJoCo |
|---|---|---|
| zero pose에서 시작 | 16번 통과 | **실패**, 0.68s 뒤로 전도 |
| crouch(hip -0.15, knee 0.4, ankle -0.25)에서 시작 | 기존 crouched_start 사용 자세 | **10s 통과**, 최대 기울기 0.012rad |

확인한 원인: zero pose의 COM은 뒤꿈치 모서리보다 1.8cm 앞(뒤로 0.028rad 기울면 전도).
MuJoCo 연성 접촉은 초기 압력중심(+0.002m)을 COM(-0.042m)으로 옮기는 동안 발이 회전하고
그 과도 응답이 작은 여유를 넘는다. 관절 잠금 강체는 서 있고, 서보·내장 position actuator·
접촉 강성·발가락 접촉 유무를 한 가지씩 바꿔도 zero pose는 넘어졌다.
Gazebo 통과는 강체 접촉에서의 결과이며 zero pose의 여유가 작다는 점을 가리지 않는다.
앞쪽으로 기울인 자세는 COM이 강체 뒤꿈치 상자(x -0.06~+0.03m)를 벗어나 스프링 발가락에
기대면서 앞으로 넘어졌다. 이후 MuJoCo 작업의 기본 초기 자세는 crouch로 둔다.

이 결과는 정적 지지다. 교대 지지·보행·강화학습 성공이 아니다.
다음: crouch 기준으로 동적 측방 흔들기(몸통 roll = 지지 다리 hip roll 관계) 궤적을
MuJoCo에서 open-loop 시험하고, 접촉·기울기·발 들림을 기록한다.

### 측방 흔들기 open-loop 시험 (evidence/58-mujoco-lateral-rocking.json, `sim/rock_probe.py`)

crouch 시작, 양쪽 hip roll 평행 사인 r=A·sin(2πft). 발 하중 <5% 체중 = unloaded.
tilt 0.25rad 초과 시 STOP(hold) latch — 회복 기능이 아니다.

| 시험 | 결과 |
|---|---|
| A 0.05~0.2, f 0.5~2.25Hz 탐색 | 1.5~2.25Hz, A 0.05~0.10에서 20주기 안정 limit cycle(roll 최대 0.06~0.13rad 수렴). 양발 교대 unloaded 각 ~46%, 동시 unloaded 없음, 들린 발 최저점 28~51mm |
| 1.0~1.25Hz 큰 진폭 | 옆으로 전도, 민감(1.25Hz에서 A .08 전도·.10 생존) |
| 흔들기 + 무릎 들기 | 거의 모두 pitch 전도 |
| 흔들기 + hip pitch 보폭 (A .08, stride .05) | 60s +2.1m, 질량 ±10%·μ0.5/0.8 생존. 그러나 접촉점 미끄럼 평균 7.6~8cm/s(몸 속도 3.5cm/s보다 큼), 발당 주기당 착지 ~3회, 동시 unloaded 21~28% |
| 흔들기만 (A .08, 2Hz) | μ1 60s 안정, 발당 주기당 착지 1회, 미끄럼 2.7cm/s, 60s 동안 뒤로 0.4m. 질량 ±10% 통과, **μ0.8·0.5에서 전도** |

판정:
- **확인됨(MuJoCo 한정):** 10축 그대로 동적 교대 지지(흔들기)가 존재한다. 다만 open-loop이며 마찰에 민감하다.
- **확립되지 않음:** 보폭 추가 시 전진은 미끄럼·반복 착지가 섞인 운동이며 깨끗한 걸음이 확인되지 않았다.
  안정 보행으로 보고하지 않는다.
- open-loop 수치 탐색은 여기서 중단한다.

UNI_AI(`gpt-6-sol`, 2,018 tokens) 검토 반영: "보행 아님" 대신 "깨끗한 걸음 미확립"으로 표현,
다음 단계 지표에 착지 debounce·걸음당 누적 stance 미끄럼 추가, RL 전에 작은 closed-loop 리듬을 먼저 시험.

다음: 측정한 몸통 roll 위상으로 동기화하는 위상 진동자(위상 보정 게인 제한, 주기당 진폭 조정 0.05~0.10rad).
통과 조건: μ0.5/0.8/1.0 × 질량 ±10%에서 60s 직립, 교대 unloaded, 지속 동시 unloaded 없음.
그 뒤에 보폭을 추가하고 깨끗한 걸음 조건(발당 주기당 debounce 착지 ≈1, 전진, 누적 미끄럼 ≪ 보폭)을 적용한다.
