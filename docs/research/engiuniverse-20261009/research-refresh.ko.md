# 채널 전편 조사 재점검과 원 자료 보완 — 2026-10-09

이번 문서는 기존 종합 보고서의 추가 조사분이다. 공개 목록121개(일반79·Shorts42), 전체 자막 분석43개, 연속 영상·음성 완시청0개, 미확보78개라는 범위를 유지한다. YouTube3편 자막 재시도는 모두HTTP429, 공개 영상 페이지 접근도 실패했다. Google Scholar 직접조회는 접근 불가, 채널 블로그는403이었다. 검색 결과와 원 논문/코드로 보완했지만 이를 미확보 영상의 완시청으로 계산하지 않는다.

## 랩터 적용 연결

| 채널에서 다룬 주제 | 적용 후보 | 먼저 검증할 조건 |
|---|---|---|
| 로봇의 몸·다리·감각 | 스프링 하중 경로, 구동기 식별, 접촉 센서 | 실제 토크–속도·응답·유격·반복 하중 |
| Isaac Sim·Sim2Real | 관측/행동/단위/지연 일치, 접촉 모델 대조 | 현재 MuJoCo/ROS 경로를 기준으로 실측 후 비교 |
| Agentic Robotics·ER2 | 작업 ID, 상태 확인, 허용 행동 호출 | STOP 우선, 현재 상태 실행 가능성, 실제 완료 응답 |
| ASPIRE·실패 디버깅 | 실패 로그에서 수정 가설 만들기 | 고정 성공 기준, 별도 평가, 기준선 보존 |
| Force·Tactile·T-Rex | 접지·하중·미끄럼 및 원시각 이력 | 손 조작 결과를 발 보행 성공률로 이전하지 않음 |
| VLA·World Model·GR00T | 미래 관측/행동 구조 및 데이터 관리 | 신체·관절·실시간 요구에 맞는 자체 평가 |
| OK-Robot(영상 본문 미확보) | 사전학습 인식과 실행 API 결합 | 집기 하드웨어/교정 필요, 랩터 보행 정책 별도 |

표의 주제 연결은 기존43편 자막 검토와 원 자료에 대한 분석자의 적용 제안이다. 이미 구현·검증된 랩터 기능으로 보고하지 않는다.

## 이번에 확인한 원 자료


### Robust Recovery Controller for a Quadrupedal Robot using Deep Reinforcement Learning

뒤집기·기립·보행 정책과 행동 선택기를 나눠 학습한다. 넘어짐 중 접촉 기반 상태추정이 불안정해 self-righting 관측에서 선형 위치·속도를 제외한다. 평지 시험이며 큰 경사·거친 지형은 한계로 남긴다. [Robust Recovery Controller for a Quadrupedal Robot using Deep Reinforcement Learning](https://arxiv.org/abs/1901.07517)

**확인 범위:** PDF §II-B,D–H 및 §III–IV 선택 구간. **랩터 적용 판단:** 회복 모드 관측/전환 조건을 별도로 설계; 사족 구조의 성공률을 랩터에 이전하지 않음.

### Learning Agile Soccer Skills for a Bipedal Robot with Deep Reinforcement Learning

회복은 앞/뒤 기립 경로의 세 핵심 자세로 유도한 별도 정책을 학습하고 축구 정책과 통합한다. 40Hz 관절 목표/행동 필터와 최근5관측 이력을 사용한다. 실물 get-up 평가는 어깨 높이 임계값이며 지속 정차 시험이 아니다. 이상 구동기·배터리 민감도·지연 한계를 명시한다. [Learning Agile Soccer Skills for a Bipedal Robot with Deep Reinforcement Learning](https://arxiv.org/html/2304.13653v2)

**확인 범위:** Stage1 Get-Up; Environment; Limitations; Supplement Get-Up Ability. **랩터 적용 판단:** 유효 접촉 핵심 자세를 먼저 찾되 60초 정차·열·STOP 기준 유지.

### AnyBipe: An End-to-End Framework for Training and Deploying Bipedal Robots Guided by Language Models

언어 모델로 보상을 수정해도 공통 속도·각속도·생존 평가를 유지한다. TableIII는 pointfoot 실물 각속도 .06rad/s/목표 .10, 정차5.8초/목표30초를 보고한다. TableV에서 P1만 실물, H1/Turin은 시뮬레이션이다. [AnyBipe: An End-to-End Framework for Training and Deploying Bipedal Robots Guided by Language Models](https://arxiv.org/pdf/2409.08904)

**확인 범위:** §III-B,C; Tables I,III,V; Conclusion. **랩터 적용 판단:** 학습 보상 자동 수정과 동결 평가 분리; 안전 체크 통과를 모든 과제 성공으로 취급하지 않음.

### AnyBipe official repository

공개 README는 IsaacGym·ROS Noetic·Ubuntu20.04 경로이며 ROS2 지원 가능성은 언급하지만 프로젝트에 즉시 호환되는 Jazzy 구현 증거가 아니다. [AnyBipe official repository](https://github.com/sjtu-mvasl-robotics/AnyBipe)

**확인 범위:** README Prerequisites/Installation. **랩터 적용 판단:** 보상/평가 구조 참고; 기존 Jazzy/Harmonic/MuJoCo 재작성 또는 이관 없음.

### Bimo actuator and observation source at pinned tree a2b2cc5

소스에 명령 지연 링 버퍼·1차 응답·방향 의존 유격·PD·마찰·속도 의존 토크 제한이 있다. 정책 관측은 IMU Euler와 마지막 명령이다. 코드 모델의 존재는 제조사 보증이나 랩터 실측을 뜻하지 않는다. [Bimo actuator and observation source at pinned tree a2b2cc5](https://github.com/mekion/the-bimo-project/tree/a2b2cc5b0f7a817394d07898bac2aa15604b0c59)

**확인 범위:** STS3215.py compute/reset/config; bimo_task_env.py _get_observations. **랩터 적용 판단:** 제어 갱신 주기와 구동기 응답 대역폭을 분리해 측정; 소스의 수치/관측 축소를 그대로 복사하지 않음.

### Design and experimental verification of a clutched parallel elastic actuation mechanism for legged locomotion

병렬 인장 스프링이 있는 수직축 제한 홉핑 기구의 모델 식별과 관절 가동성 문제를 다루고 클러치 시제품을 통합했다는 초록이다. 무릎 걸쇠 제작 도면·반복 하중·치수는 본문 미확보로 확인하지 못했다. [Design and experimental verification of a clutched parallel elastic actuation mechanism for legged locomotion](https://open.metu.edu.tr/handle/11511/96080)

**확인 범위:** OpenMETU abstract lines62–64; PDF fetch unavailable. **랩터 적용 판단:** 클러치를 별도 단축 벤치에서 식별하는 연구 후보; 랩터 걸쇠 제작 규격으로 사용하지 않음.

### 有躯干双足机器人被动行走及其稳定器

Matlab 몸통 포함 이족 모델의 내리막 무구동 보행 두 형태는 불안정하며 전체상태 선형 피드백 안정기를 설계했다고 초록이 설명한다. [有躯干双足机器人被动行走及其稳定器](https://cje.ustb.edu.cn/cn/article/doi/10.13374/j.issn1001-053x.2012.01.011)

**확인 범위:** 중국어 摘要. **랩터 적용 판단:** 수동 보행 가능과 안정성 구분; 평지 수동 정차 불가능의 증명으로 해석하지 않음.

### OK-Robot official repository requirements

Stretch Dex Wrist·LiDAR iPhone·GPU workstation, AnyGrasp 라이선스/체크포인트와 교정 URDF를 요구한다. 추가 학습 없는 모듈 결합은 기존 사전학습 모델 및 로봇 기반 기능을 사용한다. [OK-Robot official repository requirements](https://github.com/ok-robot/ok-robot)

**확인 범위:** README Hardware and software requirements; Installation; Roadmap. **랩터 적용 판단:** 인식/실행 모듈 API 분리만 참고; 팔 없는 랩터에 pick 기능 또는 무학습 보행을 약속하지 않음.

### OK-Robot official project analysis

프로젝트는 10가정 171 pick-and-drop 시도와58.5% 성공을 설명하고, 의미 메모리 검색·어려운 조작 자세·하드웨어를 주요 실패 원인으로 구분한다. [OK-Robot official project analysis](https://ok-robot.github.io/)

**확인 범위:** Analysis failure modes. **랩터 적용 판단:** VLM 정답률과 전체 행동 성공률을 분리하고 단계별 실패 로그 확보.

## 해석에서 중요한 차이

AnyBipe의 안전 체크와 높은 학습 보상은 명령 추종·지속 정차 달성을 뜻하지 않는다. 논문TableIII의 미달 사례가 그 차이를 보여 준다. 회복 논문은 유용한 접촉 순서/정책 분리를 제시하지만 ANYmal사족·OP3팔 포함 휴머노이드와 현재 랩터의 가동 범위는 다르다. Bimo개발자의 Reddit경험은 현재 pinned코드와 연결해 읽었으며 제3자 실험 재현은 하지 않았다.

Bimo소스의 IMU+명령만 쓰는 관측은 그 플랫폼의 선택이다. 현재 랩터의 q/qd·접촉 정보 제거를 정당화하지 않는다. 유격/지연/구동기 응답 모델은 가져올 수 있는 개념이지만 수치는 실제 랩터 부품 식별 후 정해야 한다. source파일을 내려받아 읽었으며 실행/설치하지 않았다.

## 현재 개발과 다음 검증

증거161 기준 원8축 시작21조건 중16은 .6초 접근 종료 전에 이동/기울기/비행 기준을 위반했다. 평형근처 안정과 원시작 기립은 다른 검증이다. 따라서 초기 접근의 감속·자세 토크 분배를 먼저 비교하고, 이후 전체60초 정차·기존4m/s회전·직진/가상열·STOP카탈로그 조건을 유지한다. 이번 조사로 기준선이나 기본 정책을 교체하지 않았다.

## 방법 및 미완료 범위

영어/중국어 검색, arXiv/학술지/대학 원문 페이지, 공식GitHub README/고정트리 소스, 개발자 직접Reddit답변을 대조했다. PDF2개는 내려받고 텍스트로 추출했으며 필요한 절만 읽었다. 논문 전문 전체 정독이나 실험 재현이라고 보고하지 않는다. 출처·근거·주장은 기존JSONL에 append했고 중복ID·참조·확보자막수·파일SHA256을 검사했다. 영상78개의 내용 확인과 영상 화면/음성 완시청은 여전히 미완료다.
