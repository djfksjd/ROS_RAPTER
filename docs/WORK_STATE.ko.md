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

## 지금 진행

브랜치 fix/gait-imu-cancel (4b94162에서 분기).
발 메시 통합은 완료. 한 발 지지 실험 계측 개선 및 세 비교 실행 완료.
41: roll .28는 오른발 실제 접촉이 남음.
42/43: roll .39는 오른발 이탈했으나 COM이 접촉 hull 밖, 전도.
44/45: roll .36도 hull margin -5.29mm, 이후 전도.
상세: GAIT_CONTACT_ANALYSIS.ko.md. 새 analyzer는 실제 passive joint 표본까지 사용.
46: IMU .25378rad에서 ROS action 취소 승인, terminal CANCELED 확인. 10개 테스트 통과.
47/48: crouch hip -.05 전환 중 pitch 전도, 취소 승인.
49: hip -.10도 shift 후 settle 중 기울기 초과, 다음 action 발행 안 함.
50: 시간 연속 79표본으로 접촉이 한 줄 발가락에 집중/관절 변형/pitch 증가 확인.
51: toe_stiffness_scale 3배 외 모든 URDF 동일, SDF 강성/10active+12passive 확인.
52: 강성 3배는 pitch가 줄었으나 왼발목 추종 오차 .06213rad로 실패.
현재 브랜치 experiment/toe-stiffness. 기본 강성 1과 기본 자세는 유지.
다음: 실제 plugin 구현/발목 속도 및 하중을 확인하고 추종 오차 원인을 구분한다.
상세 TOE_STIFFNESS_EXPERIMENT.ko.md. 무근거 gain 증가나 숫자 탐색 반복 금지.
시각 모델 전체 완성·안정 보행·강화학습은 아직 완료되지 않았다.

## 알려진 실패

BALANCE_NOTES.ko.md 및 PASSIVE_TOES.ko.md를 필요할 때만 읽는다.
응답 속도만 높인 균형 보정은 실패했다. 동일 튜닝을 새 근거 없이 반복하지 않는다.
