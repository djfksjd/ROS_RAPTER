# 센서 구성과 sim-to-real 변수 — 2026-10-28

조사·문서 작업이다. 학습·평가·로봇 실험은 하지 않았다. 기계 판독용 목록: [`sim2real_params.json`](sim2real_params.json).

## 결론

1. **현재 학습 환경(`RunEnv`)에는 관측 잡음과 지연이 전혀 없다.** `RunEnv._observe()`가 `_obs()`를 그대로
   돌려주기 때문에(`sim/rl/run_env.py:553-554`) 부모 `RaptorEnv`의 잡음·0/20 ms 지연 경로(`raptor_env.py:211-221`)를 쓰지 않는다.
   몸통 선속도 3개는 시뮬레이터 참값이다(`run_env.py:562-563`). 접촉 판정(GRF > 0.15 W, `run_env.py:880-881`)도 참값이며,
   무릎 걸쇠·래칫·클러치의 연결 논리가 이 값에 의존한다(`run_env.py:750-809`). 실물에서 가장 먼저 깨질 수 있는 가정이다.
2. 이미 있는 무작위화는 **몸통 질량 ×0.9–1.15, 발·바닥 마찰 ×0.625–1.5625, 토크 한계 ×0.9–1.1, 초기 열 지표 0–0.8, 외란 5종**이다.
   **빠진 것**: 링크 질량·관성·질량중심, PD 이득, 관절 감쇠·마찰, 스프링 강성, 백래시, 지연, 센서 잡음·오프셋, 배터리 전압,
   온도에 따른 권선 저항·토크 상수 변화, 경사·요철 지형.
3. 센서는 [표 1](#1-센서-구성-제안)처럼 구성한다. 몸통 IMU 1개(+예비 1개), 관절 출력측 절대 엔코더, 구동기 내장 모터측 엔코더·전류·온도,
   발 접촉(모멘텀 관측기 + 하중 센서), 권선 NTC, 배터리 V/I, 래칫 홀 스위치. 카메라는 선택이다.
4. **수치 근거의 한계.** 이 세션의 네트워크 정책이 arxiv.org, science.org, IEEE, 부품사 사이트를 막았다. 그래서 다음처럼 나뉜다.
   - 논문 표(Hwangbo 2019, Lee 2020, Miki 2022, Hound 2312.17507, Ji 2022, Siekmann 2021, Radosavovic 2024): **값을 하나도 확인하지 못했다.** 값을 비워 두었다(`verified: no`).
   - 데이터시트 수치: 검색 요약에서만 봤다(`partial`).
   - 공개 코드 6개 저장소의 고정 커밋 값: 직접 읽었다(`yes_code`). legged_gym, Isaac Lab, cassie_rl_walking(Li 2024 공식 코드), osudrl roadrunner/apex, humanoid-gym, unitree_rl_gym.
5. 몸통 속도 추정 오차의 문헌 값은 모두 **저속 보행** 결과다(< 0.02 m/s RMS, 0.05–0.07 m/s RMSE, 위치 드리프트 < 5 %).
   **5–7 m/s 이족 달리기의 추정 오차 근거는 찾지 못했다.** 따라서 46 관측의 속도 항은 실측 전까지 미검증 가정이다.

검증 수준: `yes_repo` 이 저장소 코드(커밋 3670406), `yes_code` 공개 코드 직접 확인, `partial` 검색 요약(원문 미열람), `no` 값 미확보.

## 1. 센서 구성 제안

질량·가격·지연은 원문을 확인한 것이 거의 없다. 표에 없는 값은 미확보다.

| 기능 | 1순위 후보 | 대안 | 주요 사양 (검증) | 이유 |
|---|---|---|---|---|
| 몸통 IMU | TDK ICM-42688-P | Bosch BMI088 | ICM: 자이로 2.8 mdps/√Hz, 가속 65–70 µg/√Hz, SPI (partial). BMI088: 자이로 0.014 °/s/√Hz, SPI (partial) | 저가·고 ODR. 바이어스 불안정성은 두 칩 모두 미확보 |
| 상태 추정용 산업 IMU (선택) | MicroStrain 3DM-CV7 | Xsens MTi-630, VectorNav VN-100 | CV7: 바이어스 1.5 °/h, ARW 0.14 °/√h (partial). MTi-630 8 °/h. VN-100 5 °/h typ (partial) | 장시간 자세·속도 추정 |
| 관절 출력측 절대 엔코더 | RLS AksIM-2 | MPS MA600A, ams AS5047P | AksIM-2: 17–20 bit, ±0.025°, 55 µs (partial). MA600: INL < 0.6° (보정 후 < 0.1°), 단종 예정 (partial) | 4절 링크 무릎·1:1 연동 발목은 링크 탄성 때문에 모터각과 관절각이 다르다(증거 154의 equality 오차 0.019–0.055 rad) |
| 모터측 엔코더·전류·온도 | 구동기 내장 (AK10-9 V3 이중 엔코더) | RobStride 03 (공식 매뉴얼 미확보) | AK10-9 V3: 내부 14 bit, 외부 15 bit. MIT 응답에 온도 필드가 있으나 권선인지 MOSFET인지 미확인 (partial) | 추가 부품 없음 |
| 발 접촉 | 관절 토크·모멘텀 관측기 (Bledt 2018 방식) | 하중 센서 TE FX29, FlexiForce A201 (445 N 등급) | A201: 반복성 < ±2.5 %, 히스테리시스 < 4.5 % FS (partial). FSR 402는 수십 N 이하라 착지 하중에 부적합 (partial) | 클러치·래칫 논리가 접촉 판정에 의존 |
| 권선 온도 | 권선 매립 NTC 10 kΩ | 구동기 내장 온도값 | Murata NCP18XH103F03RB: B25/50 3380 K ±1 %, 단종 예정 (partial). 열 시정수 미확보 | 열 모델 식별·보호 |
| 배터리 V/I | TI INA228 + 션트 | Allegro ACS772 (홀) | INA228: 20 bit, 이득 오차 ±0.05 % max, 변환 50 µs–4.12 ms (partial) | 전압 강하에 따른 토크-속도 한계 |
| 래칫·걸쇠 걸림 | 연속 동작 홀 스위치 (Allegro A1120 계열) | Omron D2F 마이크로스위치 | DRV5032는 5–80 Hz 샘플링이라 부적합 (partial) | 시뮬레이터의 `latched` 상태를 실물에서 관측 |
| 카메라·깊이 (선택) | Orbbec Gemini 336 | RealSense D435i, OAK-D Pro W | Gemini 336: 2 m에서 ≤ 1.5 %, 99 g (partial). D435i: 2 m에서 ≤ 2 % (partial). 지연 미확보 | 지형 인지는 현 단계 범위 밖 |

### 몸통 속도 추정 오차 (문헌)

| 출처 | 값 | 조건 | 검증 |
|---|---|---|---|
| Bloesch et al. RSS 2012 (StarlETH, 다리 운동학 + IMU EKF) | 속도 RMS < 0.02 m/s, 위치 드리프트는 이동거리의 최대 10 % | 저속 사족 | partial |
| Hartley et al. IJRR, arXiv 1904.09251 (Cassie contact-aided InEKF) | 최종 위치 오차 < 이동거리의 5 % | 60 s 보행 약 15 m | partial |
| arXiv 2104.04238 (IMU + 추적 카메라 + 다리 InEKF) | 속도 RMSE 0.070/0.066/0.051 m/s | 저속, 미끄러운 지면 | partial |
| Camurri et al. 2020 Pronto | 이동거리의 약 1.67 % 드리프트 | 실내 예비 시험 | partial |
| Ji et al. 2022 RA-L (KAIST 동시 학습 추정기) | 오차 수치 미확보 (3.75 m/s 주행만 확인) | Mini Cheetah급 | no |

참고로 공개 코드가 정책 학습에 넣는 속도 관측 잡음은 legged_gym·Isaac Lab ±0.1 m/s 균등, cassie_rl_walking ±0.04 m/s다(yes_code).

## 2. sim-to-real 변수 (출처별)

| 변수 | 범위 | 출처 (고정 커밋:줄) | 검증 |
|---|---|---|---|
| 마찰 | 0.5–1.25 | legged_gym@8fa29ac `legged_robot_config.py:123` | yes_code |
|  | 0.1–1.25 | unitree_rl_gym@276801e `g1_config.py:30` | yes_code |
|  | 0.1–2.0 | humanoid-gym@ae46e20 `humanoid_config.py:151` | yes_code |
|  | 0.3–3.0 | cassie_rl_walking@45c830e `env_randomlizer.py:32` | yes_code |
|  | ×0.8–1.2 | roadrunner@55a092e `dynamics_randomization.json` | yes_code |
|  | 고정(정지 0.8 / 동적 0.6, 반발 0) | IsaacLab@210ee79 `velocity_env_cfg.py:202-204` | yes_code |
| 접촉 solref (반발·강성) | ×0.9–1.1 | roadrunner `solref` | yes_code |
| 몸통 질량 | ×0.8–1.25 log-uniform | IsaacLab `:209-218` | yes_code |
|  | −1–+1 kg (기본 off), ANYmal-C ±5 kg | legged_gym `:124-125`, `anymal_c_rough_config.py:79-81` | yes_code |
| 모든 링크 질량 | ×0.5–1.5 / ×0.75–1.25 | cassie_rl_walking `:68` / roadrunner `mass` | yes_code |
| 관성 | ×0.7–1.3 | cassie_rl_walking `:73` | yes_code |
| 질량중심 | x,y ±5 cm, z ±1 cm (몸통) | IsaacLab `:221-227` | yes_code |
|  | ipos ×0.7–1.3 + 골반 ±0.1 m, 링크 ±0.01 m | cassie_rl_walking `:31, 78-83` | yes_code |
| 관절 감쇠 | ×0.7–4.0 / ×0.5–3.5 / ×0.3–5.0 | cassie_rl_walking `:54` / roadrunner / apex@82c44af `cassie.py:153-154` | yes_code |
| 스프링 강성 | ×0.8–1.2 (Digit ×0.9–1.1) | cassie_rl_walking `:39`, roadrunner `spring` | yes_code |
| PD 이득 | ×0.7–1.3 | cassie_rl_walking `:38` | yes_code |
| 토크 효율 | ×0.9–1.0 | roadrunner `genericenv.py:217` | yes_code |
| 행동 잡음 | 곱셈 0.02 σ | humanoid-gym `:160` | yes_code |
| 상태·행동 지연 | 0.45–28.6 ms (따로 샘플) | cassie_rl_walking `:33` | yes_code |
| 토크 지연 | 2.0–3.5 ms (4–7 × 0.5 ms) | roadrunner `genericenv.py:216`, `cassie_fast.xml:5` | yes_code |
| 행동 저역 지연 | a ← (1−d)a + d·a_prev, d ∈ [0, 0.5] | humanoid-gym `:159` | yes_code |
| 엔코더 오프셋 (에피소드 고정) | ±0.05 rad / ±0.01 rad | roadrunner / apex `cassie.py:151` | yes_code |
| 관절각·관절속도 잡음 | ±0.01 rad, ±1.5 rad/s | legged_gym `:170-171`, IsaacLab `:174-175` | yes_code |
| 자이로·중력 방향 잡음 | ±0.2 rad/s, ±0.05 | legged_gym `:173-174` | yes_code |
| 자세·자이로·가속 잡음 | ±0.002 rad, ±0.01 rad/s, ±0.1 m/s² | cassie_rl_walking `:34-36` | yes_code |
| 밀기 | ±1 m/s / 15 s, ±0.5 m/s / 10–15 s, ±1.5 m/s / 5 s | legged_gym `:126-128`, IsaacLab `:267-271`, G1 `:34-35` | yes_code |
| 바닥 경사 | ±0.03 rad / ±0.05 rad | cassie_rl_walking, apex / roadrunner | yes_code |
| 액추에이터 네트워크, 지연 | 미확보 | Hwangbo 2019 Sci. Robot. (arXiv 1901.08652) | no |
| DR 표 전체 | 미확보 | Lee 2020 (2010.11251), Miki 2022 (2201.08117), Choi 2023 Raibo, Ji 2022 (2202.05481), Siekmann 2021 (2011.01387, 2105.08328), Radosavovic 2024 (2303.03381) | no |
| MOR 제약·토크 클리핑·배터리 | 미확보 | Hound, Shin et al. arXiv 2312.17507 | no |

주의: 공개 코드의 값이 해당 논문 표와 같다는 보장은 없다. apex는 Siekmann 2021 이전 커밋이다.

### 모터 열 모델 (제안, 수치 대부분 미확보)

현재 `heat`는 (τ_motor/τ_cont)²의 3 s 저역 통과 지표다(`run_env.py:100-104, 836-839`; τ_cont = 등급 최대 토크의 0.4).
온도·저항·토크 상수 변화가 없다. 실물 대응 모델은 다음 순서로 만든다.

- 권선 1차(또는 권선-하우징 2노드) 모델: C_w·dT_w/dt = P_cu − (T_w − T_h)/R_th, P_cu ∝ R(T_w)·i².
- R(T) = R_25·(1 + α_Cu·(T − 25)). α_Cu = 0.00393 /K (IEC 60028 표준값, 원문 미열람 → `no`).
- K_t(T) = K_t25·(1 + β·(T − 25)). NdFeB 잔류자속 계수 −0.09 ~ −0.12 %/K (partial). 자석 등급은 미확인.
- AK10-9 V3: K_t 0.16 N·m/A, 상간 저항 248 mΩ, 정격/최대 18/53 N·m (모두 partial). 코드의 B급은 최대 48 N·m로 가정한다(`run_env.py:87`).
- R_th, C_w(시정수), 최대 권선 온도: **미확보.** 데이터시트에 없으면 계단 전류 가열·냉각 시험으로 식별해야 한다.

## 3. 격차 분석 (현재 `run_env.py` 대비)

| 항목 | 현재 모델 (`run_env.py`) | 참고 범위 (yes_code) | 상태 |
|---|---|---|---|
| 몸통 질량 | ×0.9–1.15 (`:457`) | ×0.8–1.25 (IsaacLab) | 있음 |
| 링크 질량·관성 | 고정 | ×0.5–1.5 / ×0.7–1.3 (Cassie) | **없음** |
| 질량중심 | 없음 (부모 dr2의 ±3 cm는 RunEnv에서 사용 안 함, `:246`) | 몸통 ±5 cm (IsaacLab) | **없음** |
| 발·바닥 마찰 | ×0.625–1.5625 (rev2, `:458-460`) | 0.1–3.0 (여러 출처) | 있음, 범위 좁은 편 |
| 반발·접촉 강성 | 고정 | solref ×0.9–1.1 | **없음** |
| 토크 한계 | ×0.9–1.1 (`:463`) | 토크 효율 ×0.9–1.0 | 있음 |
| 토크-속도-전력 곡선 | 등급별 곡선 (`:174-178, 584-604`) | — | 있음 (곡선 형태는 가정) |
| PD 이득 | 고정 kp = τ/0.1, kd = kp/45 (`:58-59`) | ×0.7–1.3 | **없음** |
| 관절 감쇠·마찰 | 없음 | 감쇠 ×0.3–5.0 | **없음** |
| 스프링 강성 (무릎·발목) | 고정 | ×0.8–1.2 | **없음.** 스프링 설계에 직접 영향 |
| 백래시 | 없음 | 수치 미확보 | **없음** |
| 기어·링크 효율 | 0.9 고정 (`:90`) | — | 고정 가정 |
| 로터 반사 관성 | rev2 armature (`:412-419`) | — | 있음, 무작위화 없음 |
| 목표 보간 | `target_ramp` 선택 (`:220-224`) | — | 있음 |
| 제어·관측 지연 | **없음** | 0.45–28.6 ms, 토크 2–3.5 ms | **없음** |
| 관측 잡음 | **없음** (`:553-554`) | 관절 ±0.01 rad, ±1.5 rad/s, 자이로 ±0.2 rad/s | **없음** |
| 엔코더 오프셋 | 없음 | ±0.01–0.05 rad | **없음** |
| 몸통 속도 관측 | 시뮬레이터 참값 (`:562-563`) | ±0.04–0.1 m/s | **없음** (추정기 가정) |
| 접촉 판정 | 참값 GRF > 0.15 W (`:880-881`) | — | **센서 모델 없음** |
| 열 | (τ/τ_cont)² 저역 지표, 초기 0–0.8 (`:510`) | — | 지표만. 온도·R(T)·K_t(T) **없음** |
| 배터리 전압 강하 | 없음 (P_max 고정) | 수치 미확보 (Hound 미열람) | **없음** |
| 지형 | 평지만 (`:308`) | 경사 ±0.03–0.05 rad | **없음** |
| 외란 | pitch·yaw·발 걸림·착지·밀기, 선택 roll·yaw 충격 (`:697-713, 744-745, 889-890`) | 밀기 ±0.5–1.5 m/s | 있음 |
| 초기 상태 | 관절 σ 0.02 rad, 전진 속도 0–0.3·cmd (`:483, 520`) | 관절 ×0.5–1.5, 몸통 속도 ±0.5 m/s | 있음 |

## 4. 다음 한 단계

차단된 호스트(arxiv.org, science.org, ieeexplore.ieee.org, 부품사 데이터시트 도메인)를 환경 네트워크 설정에서 허용한다.
그다음 `verified: no/partial` 항목을 원문으로 채운다. 코드 변경(지연·잡음·스프링 강성 무작위화 추가)은 이 문서 범위 밖이며 사용자 지시가 필요하다.
