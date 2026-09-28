# 지속 작업 상태

이 파일을 먼저 읽고, git status와 해당 단계의 증거만 확인한다. 비밀값 기록 금지.

## 운영 방식

사용자 요청: 남은 작업을 순서대로 반복 구현·검증하는 랄프 방식.
전용 Ralph 플러그인은 로컬 검색에서 발견되지 않았다. 세션의 지속 목표 기능으로 관리한다.
명시적 토큰 한도는 없으므로 임의 한도를 설정하지 않는다.
긴 출력은 파일로 저장하고 요약만 읽는다. 중복 조사와 불필요한 에이전트 실행을 피한다.
모델 자동 라우팅은 설정하지 않았다. 실제 사용하지 않은 모델을 사용했다고 기록하지 않는다.

## 불변 조건

- 10 active DOF, 현재 선택형 passive toe 12개. 14축으로 회귀 금지.
- Ubuntu24.04/Jazzy/Harmonic, 기존 저장소 유지. 한 문제씩 최소 수정.
- 시각 모델·동역학·제작 가능성 검증을 구분한다.
- GitHub와 비공개 HF 백업, .env/토큰/빌드 산출물 제외. force push 금지.

## 현재 기준

- 브랜치 feature/reference-appearance, 시작 커밋 2533235.
- ROS 빌드, URDF, active controllers 검증: evidence/35~37.
- 참고 외형 1차 개선. 전시 자세는 물리 검증 전.
- 평지와 제한된 8mm 단차/정렬된 5도 경사 정적 시험 통과. 보행 미완료.
- Qwen/NanoJev STAND/STOP 연결 확인. 탐색/이동 mission 미완료.
- MuJoCo/MJX/RL 아직 미설치·미실행.

## 순서와 통과 조건

1. 외형: 전시/시뮬레이션 발 구조 일치, 메시 로딩, 물리 계약 불변 확인.
2. 제어: 실제 접촉과 발 이탈 측정 → 한 발 지지 → 반복 보행. 실패는 보존.
3. 지형: 평지, 단차, 경사를 수치화해 성공/실패 범위를 기록.
4. MuJoCo/MJX: Gazebo 제어 단계 이후 모델 대응, 장치/속도 검증, 학습/교차 검증.
5. AI: 허용 mission 연결, STOP 우선, 독립 명령 평가와 end-to-end 지연.
6. 부품 제작 설계: 치수/하중/구동기 선택의 미정 사항을 명시. 이미지로 단정 금지.

## 현재 상태 — 2026-09-28, 외형·디지티그레이드 다리 작업

브랜치 feature/digitigrade-appearance. 상세 DIGITIGRADE_APPEARANCE.ko.md.
leg_design:=digitigrade 추가(기본 legacy, 기존 결과 재현 가능). 10축 유지. morphloom 외형 23개 링크 GLB.
MuJoCo에서 지지 여유·보폭 견고성 개선, Gazebo 정지 기립 확인. evidence 63의 Gazebo 기록은 legacy였음을 정정.
꼬리 0.95m·쐐기형 머리·349부품 디테일 후 재검증(evidence/66): MuJoCo 기립·흔들기 9/9·보폭 결과 유지,
Gazebo 정지 기립, RViz GLB 표시 확인. 줄무늬 원인은 Gazebo가 적용하지 않는 텍스처 타일링 확장 → 이미지 없이 출력.
참고 모습 스타일 leg_design:=digitigrade_low 추가(evidence/67): 낮은 자세(hip_drop 0.06, 발목 위치 한도 ±0.95,
crouch -0.80/1.65), 은색·크롬 외형. MuJoCo 기립·흔들기 9/9 통과, 보폭 걷기는 전도 없으나 깨끗한 걸음 기준 미달.
Gazebo 정지 기립 확인. 보행 개발 기준 설계는 digitigrade 유지.
Gazebo 흔들기 재현(evidence/68): digitigrade A 0.08/2Hz 120주기 STOP 없이 교대 지지, 포화 감시 STOP 동작 확인.
MuJoCo 대비 몸통 roll 진폭 일치, 위상 ~46ms 지연(기준 미달). JTC horizon 50→20ms로 추종 개선.
MuJoCo JTCLikeServo로 위상 차이 대부분 설명(46→~21ms), JTC 지연에도 MuJoCo 보폭 결과 유지.
Gazebo 보폭 시도(evidence/69): 불합격 — 보폭 최대 후 ~2.5초 안에 STOP 2회(발목 속도 포화, 기울기).
MuJoCo 속도 actuator 근사로는 Gazebo servo 응답을 재현 못 함. MuJoCo 보폭 성공은 두 엔진 기준 미확인.
꼬리 균형(evidence/70): 꼬리 pitch 피드백은 MuJoCo에서 작은 개선, 꼬리 yaw는 몸통을 돌려 채택 안 함.
속도 MuJoCo 최고 ~0.16m/s(kv30, stride 0.08, 2.5Hz), kv 범위 전체 통과 못 함. Gazebo 보폭은 여전히 불합격.
MuJoCo 전도 감시를 중력 기울기로 수정(이전엔 yaw 포함). Gazebo 감시기: 속도 30ms 지속 + 명령 속도 사전 검사.
보행 동기 꼬리(evidence/71): 위상 φ(왼발 착지=0) 피드포워드 A 0.015·φ0 0.3 + yaw 각속도 피드백 -0.1(τ 0.02 필터, 변화율 1rad/s).
MuJoCo 빠른 보행이 kv 20~100 모두 60초 전도 없음(yaw -15~46%), 기본 보행 kv30은 악화. Gazebo는 yaw -25%지만 roll로 ~6.5초 전도,
접촉 메시지 기반 위상 추정이 부정확(주기 0.53~1.05s vs 0.4s). /gait/phase 읽기 전용 노드 추가.
꼬리 작업 동결(c2c1b03, 사용자 결정). roll 전도 진단(evidence/72-diagnosis, 1단계): Gazebo는 3회 모두 왼쪽 전도,
정상 보행 중 ~0.9Hz 저주파 측방 sway(CoM 0.10m)가 있고 마지막 2~3걸음에 급변, 미끄럼은 주원인 아님.
MuJoCo는 kv가 낮을수록 같은 sway가 커짐(kv20 전도는 pitch). 가설: 고관절 roll 유효 강성·지연, 접촉 모델, 보행 좌우 비대칭+착지 보정 부재.
측방 안정(evidence/73-lateral): 정지 하중비 0.500(두 엔진), 거울 보행에서 전도 방향이 오른쪽으로 바뀜 → 제어(궤적) 문제.
실험 A 예측(몸통 평행이동 6.3cm, Fy/Fz 1.82)은 발목 roll 부재로 몸통이 도는 구조라 -76~-85% 불일치, LIP 이중 지지 0.06s도 실제 0.01~0.02s.
규칙에 따라 정지. 다음: 사용자 판단(모델 수정 후 실험 B~D 또는 2단계 착지 피드백). 모터 사양 범위는 사용자 결정 대기.

## 이전 상태 — 방향 A(동적 흔들기 보행)

브랜치 analysis/lateral-support-feasibility. 상세 LATERAL_SUPPORT_FEASIBILITY.ko.md.
새 Gazebo 실험 없음. 54번 재분석 + 설치 버전 소스 대조 + 오프라인 FK.
55: 발목 ±2.5rad/s 표본 12개 중 7개는 위치 명령 포화로 설명 불가, shift 내내 양발 모서리 접촉.
56: 평평한 발 정적 한 발 지지의 측방 COM 여유 최대 -0.008m(한도·기울기 무시), 몸통 수직 -0.110m.
결론: ankle roll 없는 현재 기하에서 준정적 교대 지지는 운동학적으로 찾지 못함.
roll/crouch 수치 탐색·gain/토크/속도 한도 상향은 중단. 동적 보행 불가능 증명은 아님.
사용자 선택: A 동적 흔들기 보행(10축 유지). B 기하 변경/D ankle roll은 선택하지 않음.
MuJoCo 모델 변환·계약 시험 통과, crouch 시작 정적 기립 10s 통과, zero pose는 MuJoCo 실패(57).
58: MuJoCo 흔들기(2Hz, A .08)로 동적 교대 지지 확인(μ1). μ0.8/0.5 전도, open-loop 한정.
보폭 추가 전진은 미끄럼·반복 착지 혼합, 깨끗한 걸음 미확립. open-loop 수치 탐색 중단.
59: 마찰 cone을 elliptic으로 수정 후 흔들기 open-loop μ0.5~1.0×질량±10% 60s 9/9. roll 위상 피드백 필수 아님.
60: 보폭+발목 pitch 피드백은 servo kv30에서만 명목 깨끗한 걸음(+4.9m/60s). kv50/100 실패 → 조건부, 보행 미확정.
61: Gazebo servo 식별(사용자 승인) — 기본 DART에서 발목 속도 포화 시 목표 초과·한도 고착 결함 재현. Bullet-FS는 관절은 정상이나 전신 사용 불가(62).
고정 베이스 시험은 kv를 정하지 못함. 방법론: SIMULATION_METHODOLOGY.ko.md.
다음: actuator 후보 사양 범위(사용자 확인) → MuJoCo actuator를 토크 한도 PD+속도-토크+지연으로 교체해 60 재평가. Gazebo에는 포화 감시 추가. MJX/JAX 미설치.
기본 강성/gain/속도/effort 제한 변경 없음. 안정 보행/강화학습 등 전체 목표 미완료.

## 알려진 실패

BALANCE_NOTES.ko.md 및 PASSIVE_TOES.ko.md를 필요할 때만 읽는다.
응답 속도만 높인 균형 보정은 실패했다. 동일 튜닝을 새 근거 없이 반복하지 않는다.
