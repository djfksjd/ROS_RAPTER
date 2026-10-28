# 시뮬레이션 실물화 계층 (2026-10-29)

사용자 지시(2026-10-29): 실물 로봇에 적용할 물리·열·모터 발열·지연·잡음을 시뮬레이션에 넣는다.
코드·단위시험·짧은 회귀 rollout만 했다. **PPO 학습은 실행하지 않았다.** 모든 수치는 가정이며
실제 모터 값은 별도 조사 세션의 `docs/design/motor_candidates.json`(아직 없음)으로 바꿔야 한다.

## 사용

```python
RunEnv(..., realism=None)    # 기본: 꺼짐. 이전 코드 경로 그대로(아래 회귀 확인)
RunEnv(..., realism=True)    # sim/rl/realism_default.json 전체
RunEnv(..., realism='my.json' | {...})            # 기본 설정 위에 덮어쓰기
RunEnv(..., realism={'enable': ['motor', 'thermal']})  # 나열한 블록만 켬
```

블록: `motor`, `friction`, `backlash`, `thermal`, `delay`, `imu`, `encoder`, `battery`.
기본 설정 파일에서 `backlash`, `battery`는 꺼져 있고 나머지는 켜져 있다. `thermal.observe`(관측에
권선 온도 추가)는 기본 off다. 무작위 범위 `[lo, hi]`는 `randomize=True`일 때만 에피소드마다 뽑고,
아니면 `nominal`(배율은 1)을 쓴다. 센서 잡음과 바이어스 랜덤워크는 항상 적용하며 시드로 재현된다.
계층은 env의 `self.rng`와 분리된 난수열을 쓴다.

## 모델 (`sim/rl/realism.py`)

| 항목 | 내용 |
|---|---|
| 모터 스펙 | `gear_ratio, gear_efficiency, kt, r_phase, j_rotor, v_bus, i_peak, i_cont, t_winding_max, r_th_wh, r_th_ha, c_th_w, c_th_h` (SI, 온도는 °C, `temp_unit: "K"` 가능), 선택 `backdrive_efficiency`. 한 개 스펙, 목록, `{"motors"/"candidates": 목록 또는 {이름: 스펙}}`, 값 자리의 `{"value": x, "source": ...}`을 읽는다. 빠진 필드는 이름을 대고 오류 |
| 관절 배정 | 설정 `motor.assign`: 관절 종류 → 모터 이름, `external_ratio`(수 또는 arch8의 `"fourbar"`), `link_efficiency`. 토크 0인 결합 관절(arch8 발목)은 제외 |
| 토크 한계 | 전류 `i ≤ i_peak·감액`, 전압 `+τ: i ≤ (V − Kt·ω_m)/R(T)`, `−τ: i ≤ (V + Kt·ω_m)/R(T)`, ω_m = N·q̇. 구동 사분면 `τ = i·Kt·N·η`(사용자 식 τ ≤ (V − Kt·ω_m)/R·Kt·N·η), 역구동(제동) `τ = i·Kt·N/η_b` |
| 반사 관성 | `j_rotor·N²` → `dof_armature` (모델 개정 2의 AK10-9 값을 그 관절에서 대체) |
| 마찰 | 관절 쿨롱 → `dof_frictionloss`, 점성 → `dof_damping` (모델 기존 값에 더함) |
| 백래시 | 서보 위치 오차를 백래시 절반만큼 줄이는 데드밴드(틈 안에서는 위치 복원 토크 없음, 속도항은 남음) |
| 열 | 모터별 권선-하우징-주변 2노드 RC, 손실 `loss_factor·I²·R0·(1 + 0.00393·(T_w − 25))`, 1 ms 명시 오일러. `T_max − derate_margin_k`부터 `T_max`까지 선형 감액, `T_max`에서 차단, 감액 시작 온도 아래로 식으면 해제 |
| 지연 | 행동·관측 각각 물리 스텝(1 ms) 단위 링 버퍼, 에피소드마다 범위에서 정수 추출 |
| IMU | 자이로 백색잡음 + 바이어스 랜덤워크, 가속도(중력 방향 관측에 /9.81로 더함) 백색잡음 + 바이어스, 속도 추정 잡음 |
| 엔코더 | 오프셋 + 잡음 후 `2π/2^bits` 양자화, 속도 잡음 |
| 배터리 | `V = V_oc − R_int·I_bus`, `I_bus = (Σ 모터 전기 입력 + 보조부하)/V`, 직전 물리 스텝 값 사용, 회생 off면 음수 전력 버림 |

PD 이득은 바꾸지 않는다. 모터 모델은 매 물리 스텝의 `actuator_forcerange`만 정한다.
기존 `heat` 지표·보상은 그대로이고, 새 값은 `info['realism']`(`t_winding`, `t_housing`, `current`,
`derate`, `cutoff`, `v_bus`, `i_bus`, 지연 스텝)에만 나온다(모터 없는 관절은 NaN).

DC 등가 모델이다: 전류 i = τ_m/Kt 하나, 저항 R 하나, 역기전력 상수 = Kt. 3상 상저항·상전류 피크
정의로 주어진 값은 등가로 바꾸거나 `thermal.loss_factor`로 맞춰야 한다.

## 파일

- `sim/rl/realism.py` — 계층 본체, 스펙 로더
- `sim/rl/realism_default.json` — 기본 설정과 무작위 범위. 블록마다 `source: ""`, `source_todo: "TODO: docs/design/sim2real_params.json"`
- `sim/rl/motor_specs/PLACEHOLDER_not_real.json` — **실제 모터 아님**(이름·내용에 명시, 사용 시 경고)
- `sim/rl/realism_regression.py` — main과의 비트 단위 비교
- `sim/rl/realism_bench.py` — 초당 스텝 수
- `tests/test_sim_realism.py` — 단위시험 17개

## 확인 결과 (2026-10-29, 이 컨테이너, mujoco 3.14.0, Python 3.13)

1. **기본값 회귀**: `git worktree`로 `origin/main`(3670406)을 따로 꺼내 같은 시드·같은 고정 행동열(200 제어 스텝,
   전도 시 재설정 포함)로 6조건(기본 무작위화, 결정적, arch8+target_ramp+외란, 개정2+arch12+real_mass,
   꼬리 고정+스프링 b+접촉 관측+외란, arch8+래칫+보상 가중) rollout. 관측·보상·종료·qpos·qvel·ctrl·heat
   모두 바이트 단위 같음(`ALL IDENTICAL`).
2. **단위시험**: 열 정상상태 = T_a + P·(R_wh+R_ha) (상대오차 1e-6), 온도계수 포함 고정점 해석해 일치,
   토크-속도 한계 식 일치, 롤아웃 중 힘이 한계 안, 행동 지연 k = 7·23 스텝 정확 이동, 관측 지연 20·40 스텝
   정확 이동, 감액 ¼ 구간에서 토크 0.75배, 차단·히스테리시스 해제, 백래시, 엔코더 양자화, 배터리 강하,
   시드 재현. 전체 `tests/` 171개 통과(main 154개 + 신규 17개).
3. **속도**(단일 프로세스, 4코어 컨테이너, 1000 스텝 최대값, 측정마다 ±10% 흔들림): T1 기본에서 off 약 430–450,
   모터만 약 0.7배, 모터+열 약 0.55배, 기본 전체 약 0.5배(~215 스텝/초), 백래시·배터리까지 약 0.43배.
   arch8+개정2에서는 off 138, 전체 0.75배. 추가 비용은 대부분 물리 스텝마다 도는 Python/numpy 연산이다.

## 한계·미검증

- 자리표시 스펙이라 모든 결과는 보고용이 아니다. 실제 값과 `sim2real_params.json` 출처가 필요하다.
- `RecoverEnv`는 자체 `step`을 써서 모터 한계·관측만 적용되고 열·지연은 갱신되지 않는다.
- 회전자 관성 토크는 전류(발열)에 넣지 않았다. 감속기 손실은 효율 상수뿐(부하·속도 의존 없음).
- 실물화 on 상태의 정책 학습·보행 성능은 미검증이다(학습 금지 범위).
