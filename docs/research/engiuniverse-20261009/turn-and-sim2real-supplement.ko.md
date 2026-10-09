# 2026-10-09 추가 조사 — 회전 후 복귀·접촉 계획·구동기 식별

영상 전편 완시청 결과가 아니다. 영어·중국어 논문, 공식 GitHub, 개발자 Reddit 원 글에서 확인한 범위를 구분한다. 검색 스니펫만 확인한 자료는 위 목록에 포함하지 않는다.

## Dynamic Bipedal Maneuvers through Sim-to-Real Reinforcement Learning

[Dynamic Bipedal Maneuvers through Sim-to-Real Reinforcement Learning](https://arxiv.org/html/2207.07835v1) — III-A/D/E, IV-B, V 본문

Cassie의 회전 정책을 기존 달리기 상태에서 초기화하고, 회전 후 기존 정책의 rollout으로 epilogue 보상을 계산한다. 저속 실물 성공과 고속 실물 불일치를 함께 보고한다.

랩터 적용 판단: 회전 후 명령 복귀 구간을 평가하고, 참고 접촉 일정의 실현 가능성을 먼저 확인한다.

## UMich-BipedLab/cassie_alip_mpc

[UMich-BipedLab/cassie_alip_mpc](https://github.com/UMich-BipedLab/cassie_alip_mpc) — README 전체 구성·모델 가정·환경 요구

ALIP 기반 발 디딤 MPC와 하위 가상 구속 제어를 나눈 공식 구현이다. 구간별 평면 지형과 CoM 각운동량 항 제거 등의 가정 및 Cassie 전용 환경이 있다.

랩터 적용 판단: 발 디딤 계획의 참고 대상으로 유지하며 현재 랩터에 실행 가능한 패키지로 취급하지 않는다.

## Pollen Robotics Microduck RL

[Pollen Robotics Microduck RL](https://github.com/pollen-robotics/microduck_rl) — README tasks/actuator/backlash/model sections

공식 코드 설명은 MuJoCo Warp/PPO, BAM 구동기, 전압·지연·마찰 무작위화와 백래시 출력측 인코더를 다룬다. 모든 접촉 형상을 쓰는 모델은 현재 작업에 사용하지 않는다고 명시한다.

랩터 적용 판단: 실물 구동기 식별·관측 위치·충돌 형상 검증 절차를 참고한다. CUDA 훈련 경로를 Mac CPU에 그대로 적용하지 않는다.

## LIPM-Guided Reinforcement Learning for Stable and Perceptive Locomotion in Bipedal Robots

[LIPM-Guided Reinforcement Learning for Stable and Perceptive Locomotion in Bipedal Robots](https://arxiv.org/html/2509.09106v2) — IV-C/D, V-A 선택 본문

LIPM 안정성 보상과 속도 방향·크기 항, 이중 critic을 사용한다. 설명 식과 표의 부호·가중치 표기가 일부 다르므로 원 구현 확인 없이 그대로 복사할 수 없다.

랩터 적용 판단: 회전 중 heading과 CoM 속도 방향의 분리 진단을 참고한다. 비행·스프링·꼬리가 있는 랩터에 LIPM 가정을 강제하지 않는다.

## 融合落脚点调整策略的双足机器人运动控制研究

[融合落脚点调整策略的双足机器人运动控制研究](https://newetds.lib.tsinghua.edu.cn/qh/paper/summary?dbCode=ETDQH&sysId=302696) — 공식 중국어·영어 초록만

2025년 학위논문 공식 초록은 모델 기반 MPC/WBC와 RL 평면 발 디딤 오프셋의 조합 및 실물 비교를 기술한다. 전체 126쪽 원문은 확보하지 않았다.

랩터 적용 판단: 계획기 전체를 학습 정책으로 대체하는 대신 제한된 발 디딤 보정이라는 비교 후보를 둔다.

## DRL与CPG融合的双足机器人行走控制研究

[DRL与CPG融合的双足机器人行走控制研究](https://zgjl.cbpt.cnki.net/portal/journal/portal/client/paper/fbd5244d979cb8c0fa22e8ac75174574) — 공식 초록·서지 정보만

Walker2d 시뮬레이션에서 CPG 출력·학습 보정의 조합을 평가한 연구다. 원문 미확보이며 3차원 회전·실물·랩터에서의 성능 근거가 아니다. 페이지의 출판·최종 심사 날짜 순서가 맞지 않아 서지는 추가 확인이 필요하다.

랩터 적용 판단: 고정 위상 입력을 변경하면 즉시 실패했던 랩터 기록과 함께 비교하며 CPG 추가를 바로 시행하지 않는다.

## Bimo 개발자 sim-to-real 경험

[Bimo 개발자 sim-to-real 경험](https://www.reddit.com/r/robotics/comments/1q2vj4o/finally_got_simtoreal_working_on_my_opensource/) — 개발자 원 게시글·직접 답변

개발자는 사인파 실물 궤적과 시뮬레이션을 비교하며 구동기 매개변수를 맞추고 유격·지연을 반영했다고 설명한다. 자체 경험담이며 독립 재현 결과가 아니다.

랩터 적용 판단: 공식 구동기 식별 실험 설계의 질문을 얻는 데 사용한다. 성능 수치나 일반 법칙의 검증 근거로 삼지 않는다.

## 실제 작업과의 연결

132 진단에서 회전·속도 입력 조합은 개발 회전 6/8 동시 통과로 남은 좌회전 속도 실패를 해결하지 못했다. 단일 수치를 계속 조절하기 전에 1ms 접촉력·접촉 부위·CoM 운동량 방향을 좌우 비교하는 별도 진단을 시작했다. 보상이나 물리를 변경하지 않으며 계측 off/on 궤적 일치를 확인한다. 논문 성공을 랩터 성공으로 보고하지 않는다.

OK-Robot 채널 영상 DUeT5S07IbI의 신규 자막 요청은 HTTP 429로 실패했다. 접근 실패 로그를 보존했으며 43편 전체 자막 검토/영상 완시청 0편의 기존 범위를 유지한다.
