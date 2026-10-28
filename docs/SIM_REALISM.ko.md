# 시뮬레이션 실물화 계층 (2026-10-29)

사용자 지시(2026-10-29): 실물 로봇에 적용할 물리·열·모터 발열·지연·잡음을 시뮬레이션에 넣는다.
코드·단위시험·짧은 회귀 rollout만 했다. **PPO 학습은 실행하지 않았다.** 기본 설정의 수치는 가정이다.
실제 모터 값은 `docs/design/motor_candidates.json`(모터 조사)을 읽는 `sim/rl/realism_robstride.json`으로 쓸 수 있다
(아래 「RobStride 제안 조합」, 열 값은 추정).

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
| 토크 한계 | 부호 있는 전류 i의 허용 구간 = 전류 구간 `[−i_lim, i_lim]`(i_lim = i_peak·감액) ∩ 전압 구간 `[(−V − E)/R(T), (V − E)/R(T)]`, E = Kt·ω_m, ω_m = N·q̇ (`current_window`). 구동 사분면(i·q̇ ≥ 0) `τ = i·Kt·N·η`(사용자 식 τ ≤ (V − Kt·ω_m)/R·Kt·N·η), 역구동(제동) `τ = i·Kt·N/η_b`. 무부하 속도 위(|E| > V)에서는 회생 전류만 가능: 예 E 60 V, V 48 V, R 1 Ω, 20 A → −20 ~ −12 A. **교집합이 비면**(|E| > V + i_lim·R, 역기전력이 브리지로 밀어 넣는 전류가 한계를 넘음) 구간을 **회생 전류 강제** `i = −sign(E)·i_lim` 한 점으로 둔다(차단 중 i_lim = 0이면 0). 실제로는 환류 다이오드 전류가 한계를 넘을 수 있으나 모델은 한계에서 자른다 |
| 반사 관성 | `j_rotor·N²` → `dof_armature` (모델 개정 2의 AK10-9 값을 그 관절에서 대체). **4절 링크 무릎은 무릎 작동 구간의 평균 N² 상수 근사**(개정 2 규칙): 실제 `j·N(q)²`의 각도 변화와 속도항 `j·N·N′·q̇²`은 없다 |
| 마찰 | 관절 쿨롱 → `dof_frictionloss`, 점성 → `dof_damping` (모델 기존 값에 더함) |
| 백래시 | **유격 상태 모델**(관절마다 숨은 모터측 위치 θ·속도 ω, 상태 −1/0/+1). 맞물림: 모터가 접촉면(q ± b/2)에 붙어 함께 움직이고 서보 토크가 전달되며 회전자 관성 J가 관절 armature로 들어간다. 전달 토크 `τ_s − J·q̈`의 부호가 당기는 쪽이 되면 떨어진다. 유격: 관절에는 구동 토크 0, 모터는 자기 서보 토크로 틈을 가로지른다(PD 암시적 적분). 반대 면에 닫히는 속도로 닿으면 충격량 `P = (1+e)(ω − q̇)·J·J_l/(J + J_l)`, `J_l = 1/(M⁻¹)_ii`를 한 스텝 토크 P/dt로 관절에 준다(e = `restitution`, 기본 0 소성). 서보는 모터측 위치를 본다. 시작은 틈 가운데. J는 그 관절의 armature(없으면 `motor_inertia_min` 1e-3 kg·m²) |
| 열 | 모터별 권선-하우징-주변 2노드 RC, 손실 `loss_factor·I²·R0·(1 + 0.00393·(T_w − 25))`, 1 ms 명시 오일러. `T_max − derate_margin_k`부터 `T_max`까지 선형 감액, `T_max`에서 차단, 감액 시작 온도 아래로 식으면 해제 |
| 지연 | 행동·관측 각각 물리 스텝(1 ms) 단위 링 버퍼, 에피소드마다 범위에서 정수 추출 |
| IMU | 자이로 백색잡음 + 바이어스 랜덤워크, 가속도(중력 방향 관측에 /9.81로 더함) 백색잡음 + 바이어스, 속도 추정 잡음. 바이어스 랜덤워크는 **경과 시뮬레이션 시간** `std = rw·√Δt`로 갱신(같은 시각에 여러 번 읽어도 분산 불변; 백색잡음은 읽을 때마다 새로 뽑음) |
| 엔코더 | 오프셋 + 잡음 후 `2π/2^bits` 양자화, 속도 잡음 |
| 배터리 | `V = V_oc − R_int·I_bus`, `I_bus = (Σ 모터 전기 입력 + 보조부하)/V`, 직전 물리 스텝 값 사용, 회생 off면 음수 전력 버림 |

PD 이득은 바꾸지 않는다. 모터 모델은 매 물리 스텝의 `actuator_forcerange`만 정한다.
기존 `heat` 지표·보상은 그대로이고, 새 값은 `info['realism']`(`t_winding`, `t_housing`, `current`,
`derate`, `cutoff`, `v_bus`, `i_bus`, 지연 스텝)에만 나온다(모터 없는 관절은 NaN).

모터 관절의 전류·발열은 백래시가 켜져 있으면 관절 토크가 아니라 서보(모터) 토크로 계산하고, 역기전력은 모터측 속도로 계산한다.
`info['realism']`에 `backlash_side`(관절별 −1/0/+1), `backlash_impacts`, `backlash_max_impulse`가 더해진다.

DC 등가 모델이다: 전류 i = τ_m/Kt 하나, 저항 R 하나, 역기전력 상수 = Kt. 3상 상저항·상전류 피크
정의로 주어진 값은 등가로 바꾸거나 `thermal.loss_factor`로 맞춰야 한다.

## RobStride 제안 조합 (`sim/rl/realism_robstride.json`)

`RunEnv(arch8=True, model_rev=2, realism='sim/rl/realism_robstride.json')`. 기본 설정 위에 덮어쓴다.

| 관절 | 모듈 | 외부 감속 · 링크 효율 | 정지 토크(관절) |
|---|---|---|---|
| hip_pitch | RS03 | 1 · 1.0 | 60 N·m |
| hip_roll | RS06 | 1.5 벨트 · 0.9 | 48.6 N·m |
| knee_pitch | RS03 | 4절 링크 N(q) · 0.9 | 60·N(q)·0.9 |
| tail_pitch | RS03 | 1 · 1.0 | 60 N·m |
| tail_yaw | RS06 | 1 · 1.0 | 36 N·m |
| ankle_pitch·ankle_roll | `null` (arch8 연동·잠금, env 자체 한계) | | |

- 스펙 로더가 `docs/design/motor_candidates.json`의 `modules`를 직접 읽는다(`survey_module_spec`).
  출력측 값이므로 `kt = kt_out/내장비`, 내장 감속 효율 1(kt_out에 포함으로 간주), `j_rotor = j_out/내장비²`,
  `i_peak = τ_peak/kt_out`(공개 피크 토크를 재현하는 DC 등가 전류; 제조사 상전류 A peak는 쓰지 않음),
  `r_phase = r_line`(선간). 역기전력도 같은 kt: 무부하 속도 V/kt_out은 RS03 20.3 rad/s(공개 20.4), RS06 44.0 rad/s(공개 50.3, 12 % 보수적).
- `thermal.loss_factor` 1.5(**추정**): kt가 상전류 A rms당, 저항이 선간이면 Y결선 동손 3·I²·R_상 = 1.5·I²·R_선간.
- **열저항·열용량은 어느 제조사도 공개하지 않아 전부 추정**이다. `spec_override`에 값마다 `estimate: true`와 출처를 적었다.
  RS03 ← maxon EC 90 flat 260 W(980 g): R_wh 1.82, R_ha 1.74 K/W, 시간상수 57 s·258 s → C_w 31.3, C_h 148 J/K.
  RS06 ← maxon EC 90 flat 90 W: R_wh 2.99, R_ha 1.89 K/W, 52.6 s·281 s → C_w 17.6, C_h 148.7 J/K.
  출처는 웹 검색 요약이다(maxongroup.com·farnell.com 원문은 이 세션 네트워크 정책으로 차단되어 열지 못함).
- **일관성 점검:** 위 값(자유 공기, 방열판 없음)이면 RobStride 방열판 정격(RS03 20 N·m, RS06 11 N·m)을 25 °C에서 연속으로 내면
  정상상태 권선 온도가 145 °C 보호값을 넘는다(RS03 ΔT 약 363 K, RS06 약 526 K, 온도계수 포함). 즉 이 설정은 방열판 장착보다 **비관적**이다.
  RS03의 R_wh·R_ha를 0.545배 하면 정격 20 N·m에서 ΔT 120 K(145 °C)로 정격과 맞는다. 실제 장착 방열은 미측정이다.
- 모터별 출처(`source`, RobStride 공식 문서 URL)는 스펙에 남는다.

## 파일

- `sim/rl/realism.py` — 계층 본체, 스펙 로더
- `sim/rl/realism_default.json` — 기본 설정과 무작위 범위. 블록마다 `source`에 비교한 `docs/design/sim2real_params.json` 항목 이름(같은 값인지, 범위 안인지, 더 좁은 가정인지)을 적었다. 그 파일에 검증 값이 없는 항목은 `source_todo`로 남긴다. 값 자체는 바꾸지 않았다
- `sim/rl/realism_robstride.json` — 제안 모터 조합(위)
- `sim/rl/realism_smoke.py` — 설정 하나로 짧은 rollout(학습 없음): NaN, 권선 온도·전류, 초당 스텝
- `sim/rl/motor_specs/PLACEHOLDER_not_real.json` — **실제 모터 아님**(이름·내용에 명시, 사용 시 경고)
- `sim/rl/realism_regression.py` — main과의 비트 단위 비교
- `sim/rl/realism_bench.py` — 초당 스텝 수
- `tests/test_sim_realism.py` — 단위시험 24개

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

## 통합·수정 확인 (2026-10-28, 브랜치 `claude/rapter-sim-realism-integration-lwhobm`, mujoco 3.14.0, Python 3.13)

외부 검토 지적 4건 수정: 전류 구간(교집합·회생 강제), IMU 바이어스 시간 기준, 유격 상태 백래시, 4절 반사 관성 한계 명시.
1. **기본값 회귀**: `origin/main`(3670406) worktree 대비 6조건 `ALL IDENTICAL`(sha256 같음).
2. **단위시험**: 전체 `tests/` 178개 통과(29개 환경 의존 생략, main 154 + 실물화 24). 신규: 과속 전류 구간(E 60 V 예 −20 ~ −12 A, 빈 교집합 강제), 토크-속도 표(무부하 속도 위 포함),
   IMU(같은 시각 반복 읽기 불변, 증분 std = rw·√Δt, 긴 구간 1회 = rw), 백래시(유격 중 관절 토크 0·armature 0, 재접촉 충격량 = 해석식, 토크 반전 시 분리, 맞물림 유지, rollout 유한·충격 발생),
   motor_candidates 로더(RS03/RS06 피크·연속 토크·관성 재현, 열 값 없으면 오류), RobStride 설정의 관절별 정지 토크·armature.
3. **RobStride 짧은 rollout**(`realism_smoke.py`, arch8 + model_rev=2, 평활 무작위 행동, 1000 제어 스텝 = 20 s, 넘어지면 재설정, 학습 없음):

| 조건 | NaN 스텝 | 재설정 | 권선 최고 °C | 전류 최고 A | 스텝/초 off → on |
|---|---|---|---|---|---|
| 기본 블록, 무작위화 off | 0 | 42 | 25.6–29.7 | 18.2–33.0 | 178 → 152 (0.86×) |
| 기본 블록, 무작위화 on(시드 1) | 0 | 42 | 38.9–42.0 (주변 온도 10–40 °C에서 추출, 시작값=주변) | 19.9–32.7 | 269 → 140 (0.52×) |
| + 백래시·배터리, 무작위화 off | 0 | 41 | 25.7–30.0 | 21.2–32.5 | 221 → 100 (0.45×) |
| + 백래시·배터리, 무작위화 on | 0 | 34 | 38.6–41.5 | 22.3–34.5 | 202 → 90 (0.44×) |

   권선 온도·전류(`t_winding`, `t_housing`, `current`, `derate`, `cutoff`)가 `info['realism']`에 나온다. 전류 최고값은 DC 등가 i_peak(RS03 25.4 A, RS06 33.0 A, 무작위화 ±5 %)에 닿는다.
   off 기준 속도가 실행마다 178–269로 흔들려(같은 컨테이너, 단일 프로세스) 배율은 대략값이다. 무작위 행동이라 자주 넘어지며 보행이 아니다. 20 s 온도는 연속 정격과 무관하다(시간상수 수 분).

## 한계·미검증

- 기본 설정은 자리표시 스펙이라 보고용이 아니다. RobStride 설정도 열 값은 유사 모터 추정, 전기 값은 DC 등가 근사다.
- 백래시: 접촉은 강체(충격량 + 맞물림)이며 기어 비틀림 강성·접촉 시간은 없다. 충격량은 한 물리 스텝(1 ms) 토크로 주며, 관절측 관성은 직전 스텝의 질량 행렬로 계산하고 지면 접촉 구속은 넣지 않는다(실제 전달 충격과 다를 수 있음). 분리 판정은 직전 스텝 가속도를 쓴다. env 자체의 `heat`·출력 지표는 관절 토크(충격 포함)를 본다. 모터 모델 없이 켜면 J가 armature(rev1은 0 → 최솟값 1e-3)라 관성이 실제와 다르다.
- 4절 링크 무릎의 반사 관성은 평균 N² 상수다(위 표).
- `RecoverEnv`는 자체 `step`을 써서 모터 한계·관측만 적용되고 열·지연은 갱신되지 않는다. 백래시도 그 경로에서는 갱신되지 않으므로(회전자 관성이 관절에서 빠진 채 남음) `RecoverEnv`에서는 켜지 않는다.
- 회전자 관성 토크는 전류(발열)에 넣지 않았다. 감속기 손실은 효율 상수뿐(부하·속도 의존 없음).
- 실물화 on 상태의 정책 학습·보행 성능은 미검증이다(학습 금지 범위).
