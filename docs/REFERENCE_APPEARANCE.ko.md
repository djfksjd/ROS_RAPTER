# 참고 이미지 외형 개선 — 2026-09-27

브랜치: `feature/reference-appearance`.

## 이번 변경

실제 Blender 생성 코드와 Gazebo용 link-local COLLADA 메시를 개선했다.
몸통 측면을 다각형 패널로, 센서 상부를 경사진 덮개로 변경하고 하부 장비 포드,
보호 배선, 다리 액추에이터 외형과 가늘어지는 다리 덮개를 추가했다.
꼬리는 네모 블록 대신 어두운 중심부와 좌우 분절 패널로 표현한다.

![실제 Blender 모델 렌더](evidence/raptor-concept-render.png)

렌더는 hip -0.65 / knee 1.30 / ankle -0.65 rad의 **전시 자세**다.
지면 높이는 발 메시 최저점으로 계산한다. 이 자세의 물리적 기립·보행은 검증하지 않았다.
Gazebo 시작 자세는 기존 설정을 사용한다. 렌더의 발은 Gazebo의 선택형 12개 수동 발가락과 동일한 메시·관절 배치다.
렌더는 수동 관절 각도를 0으로 표시하며 동역학 결과를 나타내지 않는다.
URDF 관절, collision, inertia, 제어 설정은 이번에 변경하지 않았다.
꼬리 마디도 시각 표현이며 유연 동역학 구현이 아니다.

## 목표 이미지까지 남은 차이

- 몸통과 골반의 비율, 관절 하우징과 외피 연결부를 더 정교하게 조정해야 한다.
- 참고 이미지의 깊게 굽힌 자세를 실제 중력 아래에서 유지하는 제어가 필요하다.
- 수동 발가락에 공통 부품 메시를 적용했다. 발가락 움직임과 외형의 실제 접촉 대조는 계속한다.
- 꼬리의 연속적인 곡선 및 수동 유연성은 별도 물리 검증이 필요하다.
- 나사·베어링·감속기·배선 경로는 외형 표현이며 제작 가능한 CAD 설계가 아니다.

## 이후 검증 순서

1. 외형과 부품 모델링 개선 및 Gazebo 화면 대조.
2. 무게중심 이동, 한 발 지지, 발 이탈 확인 후 반복 보행.
3. 평지 → 낮은 단차 → 경사 → 불규칙 지형. 실패 사례와 수치 기록.
4. MuJoCo 모델 동역학 비교, MJX 장치/속도 확인, 강화학습과 Gazebo 재검증.
5. 이동·복귀·탐색 mission과 Qwen/NanoJev 연결, 독립 평가 데이터 확장.

Qwen/NanoJev의 STAND/STOP 연결은 기존 검증에 포함되지만 탐색 실행은 미완료다.
급경사 보행·강화학습 성공을 주장하지 않는다.

## 이번 검증 결과

- Blender 5.2.1 실제 렌더 성공, 이미지 육안 확인.
- colcon 2개 패키지 빌드 및 `check_urdf` 성공: [로그](evidence/35-reference-appearance-build.txt).
- Gazebo 재실행 후 두 controller active, 10 position interface claimed: [로그](evidence/36-reference-appearance-control.txt).
- [실제 Gazebo 화면](evidence/37-reference-appearance-gazebo.png): 기존 시작 자세에서 새 메시 확인.
- 생성 스크립트 Python 구문과 COLLADA 52개 XML 파싱 성공.

## 수동 발가락 외형 통합

상세 메시를 선택형 passive toe/heel에 연결했다. Blender 전시에도 같은 메시를 배치한다.
[빌드](evidence/38-passive-toe-visual-build.txt), [계약 검사](evidence/39-passive-toe-visual-contract.txt):
시각 요소를 제거한 변경 전후 URDF 전체가 동일하며 10 active + 12 passive를 유지한다.
