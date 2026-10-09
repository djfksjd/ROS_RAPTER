# 체크포인트

## 현재 상태 — 사용자 요청으로 일시정지 (2026-10-09)

지속 goal 상태는 **paused**다. 아래 active 기록은 실험 당시의 이력이며 자동 재개 지시가 아니다. 새 실험을 시작하지 않는다. 마지막 완료는 증거163이며 실행 중인 랩터 학습/실험 프로세스는 없다.

재개 절차와 핵심 artifact는 [일시정지 체크포인트](checkpoints/2026-10-09-pause/README.md)에 보존했다. 다음 한 단계는 .1/.2/.24초 동일 원시작 상태에서 hip/knee/tail의 수평·수직·pitch 결합 응답을 측정하는 것이다. 기준선미교체·전체목표미완료.



## 2026-10-09 지속 목표와 회전 개선 진행

사용자의 `/goal` 요청에 따라 목표가 active다. 회전 추종 개선 → 직진·발열 퇴행 검사 → 기립·가속·감속·정지 전환 검증 순서로 계속한다.
회전 원인 계측 10회, 대조/대칭 벌점 제거 학습 각 253,952 스텝, 짧은 평가 60회 및 60초 발열 12회 완료.
회전 추종 두 후보 모두 실패. sym0은 7 m/s 발열 2/3, 최대 1.002336으로 실패. 기준선 `scr_a8_h2`는 보존·미교체.
yaw16 단일 보상 변경 학습·평가도 완료: 회전 실패, 60초 직진·발열·속도추종 6/6 통과.
7명령 후기 최소6.366271 m/s, 최대 발열0.862793이나 회전 미달로 미채택.
yawerr8도 학습·평가 완료: 회전4조건 실패, 60초 직진 전도·발열6/6 통과이나 속도추종0/6.
20초 연속 회전 명령 진단8회도 추종 실패·전도0으로, 2초 시험 시간만의 문제는 아니다.
이번 학습·평가 프로세스는 정상 종료했다. 같은 물리 상태의 yaw 명령 반사실 예측도6회 완료:
정규화 관측·8개 정책 출력·제한 목표각에 명령 차이가 반영된다. 완전한 명령 누락은 검사 상태에서 배제했다.
접촉력 부호·좌표 교정과1 ms 모멘트 계측12회 완료. 복사본 계측의 궤적 비간섭을 확인했고
최대 운동량 수지 잔차0.004060 N·m·s, 전도0. 지면 yaw모멘트가 항상0인 것은 아니다.
명령 최대 속도7→4 비교는 각503,808스텝 학습·평가 완료. 두 후보 회전0/20 통과,
control7은 전도0/30이나7명령 발열0/3 통과, focus4는7명령 전도1/30·발열0/3 통과로 둘 다 미채택.
yaw16 입력 확장12회에서는±8입력4/4 전도. 단순 입력 증폭도 미채택. [증거91](evidence/91-turn-curriculum/README.md).
탐색 분산 비교와 평가도 완료했다. 대조는 회전0/20·전도0/30, 모든축std0.25는 회전0/20·전도11/30.
비교 측4/7명령 발열·속도 추종 모두0/6 통과로 미채택. 대조도7명령 발열0/3로 미채택.
기준선과 실패 모델을 보존했다. [증거93](evidence/93-turn-exploration/README.md).
기립·가속·감속·정지 전환은 여전히 미완료이며 기준선은 보존한다.
같은구조 고정관절목표25회도기립실패. 반사평균을학습가능한배우로옮겨800상태 초기출력·가치차이0 확인.
`sym_train_pilot_20261009`503,808스텝학습완료·회전0/20·7명령추종0/3·기립/전환4회전도,미채택.
명령순서대조/카탈로그각503,808스텝학습·최종평가완료. 회전양쪽0/20·7명령추종0/3.
카탈로그기립/전환4회전도0이나이동·정지미달,전체0/4. roll목표±.3/.45 비교완료: 확대18/18회전전전도·미채택. [증거99](evidence/99-roll-target-range/README.md). [증거98](evidence/98-command-sequence-comparison/README.md). 영점샘플러결함가설은철회. [증거97](evidence/97-symmetric-learning-evaluation/README.md). [증거96](evidence/96-stand-and-symmetric-initialization/README.md).
후속 yaw피드백gain4는전도5/30으로미채택. 보정량±0.4 제한과heading rate사용은
전도0/30·회전19/20이지만−1/17026미달.1ms접촉비교2회완료·계측궤적정확일치.
평균정책기립/전환4회모두전도·실패. 회전·이후1초최소속도도다수미달해통합후보미확보. [증거95](evidence/95-yaw-feedback/README.md).
반사 추론평균·내부yaw6배 보정에서 회전14/20·전도0/30. 개발출발로6.4를고정하고
독립30출발180회 완료: 회전117/120·전도0/180,3회회전미달로미채택.
60초직진·발열은양모드각6/6통과. 다음은실패조건의속도저하와yaw상태피드백보정. 원정책은미교체. [증거94](evidence/94-turn-reflection/README.md).


회전 평가의 명령 관측1주기 지연·첫 각도 증분 누락을 수정하고 출발별 원값 저장을 추가했다.
새 규약 기준선30회 재평가: 전도0, 회전20회 통과0. [증거90](evidence/90-turn-evaluation/README.md).

기립·전환 기준선8회 완료: 전도0이나 통과0/8. 명령0에서60초 최대 이동24.735–36.305m,
4m/s 뒤3초 정지 실패. 걷기0.4 추종도 실패. 최종20회 검증이 아닌2출발 초기 진단이다.
새RunEnv전환 하니스와 관련29개 검사 통과. [증거92](evidence/92-motion-baseline/README.md).

채널 전편 조사 요청도 이어간다. `/Users/danny/Documents/Raptor_Engiuniverse_Research_20261009/`에
공개 목록 121개(일반79·Shorts42), 확보 자막43편, 전 구간 자막 읽기43편을 구분한
`full-review-coverage.jsonl`·`full-review-notes.ko.md`를 저장했다. 영상 전편 완시청으로 보고하지 않는다.
추가 자막 HTTP429·영상 다운로드 HTTP403, Google Scholar 결과 도구 접근 실패. 원 논문·공식 GitHub·Reddit 조사와
남은0편 전체 자막 검토는 계속할 수 있다. 메타데이터 확보·발췌 검토·자막 전체 검토·영상 검토를 섞지 않는다.

## 2026-10-09 재개 첫 단계 완료

사용자 재개 요청에 따라 2026-10-01의 네 구조 비교 정책 평가부터 이어간다.
네 정책의 학습은 모두 완료됐다. 이전 OS T 정지 평가 프로세스는 종료하고 수정된 평가기로 새로 실행했다.
발열 반올림 판정과 회전·교란 평가 실패 누락을 수정, `.venv-ai` 전체 92개(35 생략), `.venv-sim` 집중 29개 검사 통과.
동일 조건 비교와 상위 두 후보 30개 출발 재확인 완료. 후방 골반 8축은 실제 최소 6.166858 m/s,
7 m/s 명령에서 전도·발열 30/30 통과. 기본 8축 6 m/s 명령은 발열 29/30로 전체 통과 실패.
40 km/h·회전 추종·최종 구조 동결은 미완료. 현재 평가 프로세스는 모두 정상 종료했다.
결과·검증·한계는 [증거 87](evidence/87-structure-comparison/README.md), 원본은 `sim/rl/runs/screen_logs/compare_20261009*`.
행별 결과는 `compare_20261009.cases/`, 재현 해시는 `compare_20261009.manifest.json`에 보존한다.
다음 한 단계: 후방 골반 8축 기준선을 보존하고 회전 회피 원인을 분리한다.
별도 정책에서 한 변수씩 변경해 직진·발열 퇴행과 회전 개선을 함께 확인한다. 학습 seed는 아직 0 하나다.
과거 “약 5 m/s 지속”은 철회된 평가이므로 현재 결과로 사용하지 않는다.

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

2026-10-09 다음 비교: 카탈로그 source에서 선택적 lin_err 가중치0/-2, 각503,808스텝학습·회전·열·기립·정지검증완료,미채택. [증거100](evidence/100-dense-speed-reward/README.md).

선형속도오차비교(증거100)양쪽503,808스텝·최종평가완료:회전0/20·전도0/30,7명령추종0/3,기립/전환0/4통과. 비교기립14–15m이동·저속후퇴로미채택. 전체목표active·기준선보존. 다음고관절roll/꼬리yaw한정탐색비교.

회전축한정std0.25 후보 `turn_axes_std25_20261009`학습시작. 대조dense_speed_control보존·독립평가대기. [증거101](evidence/101-targeted-turn-exploration/README.md).

회전구동기1ms계측6조건×on/off=12rollout완료,전체궤적정확일치·전도0. 꼬리yawnear-bound0,고관절roll일부한계도달. 모터부족단일원인미확정. [증거102](evidence/102-turn-actuator-telemetry/README.md).

회전축탐색503,808·최종평가완료:회전0/20·발열0/6·기립/전환0/4,미채택. 배우상쇄clip가설6rollout에서미관측,정책clamp미변경. 다음관측에없는스프링상태의영향을반사실진단한다. [증거103](evidence/103-actor-clipping-diagnosis/README.md).

loaded[left]만바꾼합성반사실3조건:같은관측·명령인데스프링연결과다음상태차이확인. 도달가능성·실패단일원인미확정. 다음좌우loaded2개선택적관측(46→48)후보를기준선보존하며구현·대조평가한다. 아직미구현. [증거104](evidence/104-hidden-spring-state/README.md).

loaded 관측 옵션 구현·초기화 검증 완료: 기본46/후보48,행동8 유지. 초기9rollout 행동·qpos·qvel 정확일치. source 카탈로그에서 새대조/후보 각각500k 요청 학습 시작, 실제평가대기. [증거105](evidence/105-contact-observation-initialization/README.md).

loaded 관측 비교의중간250k 모델을동결해대조/후보각30회회전·직진평가와seed48001기립/전환4회평가실행중. 최종학습은원프로세스에서계속된다.

loaded 관측중간평가완료:양쪽전도0/30·회전0/20,기립/전환전체0/4. 최종학습계속·정책미채택. [증거106](evidence/106-contact-observation-evaluation/README.md).

loaded 관측 비교 최종완료: 양쪽503,808스텝, 새출발각30회전도0·회전0/20, 열6/6·7속도추종0/3, 기립/전환0/4. 후보기립10.642–10.700m 이동·정지13초내미달로미채택. 전체goal active, 기준선보존. 다음관측 활용과 명령별분포 진단. [증거106](evidence/106-contact-observation-evaluation/README.md).

추가loaded활용진단6rollout계측on/off실제궤적정확일치. 새입력가중치와플래그반전의행동차이비영, 입력완전미사용가설기각. 목표미달원인미확정. 다음명령별보상성분/접촉분포진단. [증거107](evidence/107-contact-observation-use/README.md).

명령별보상계측4rollout×43초 종료·on/off행동/상태/보상정확일치. 영점대부분단발지지·stand항flight만벌점. yaw구간ang_mom평균벌점이yaw이득압도가설미지지. 다음완전영점명령에서unloaded발비율선택항만가중치0/음수대조. 아직구현/학습없음. [증거108](evidence/108-mode-reward-distribution/README.md).

완전영점명령 stand_support 선택항 구현. 집중17pass, 전체133중66pass67skip. source카탈로그에서가중치0/−5 각500k요청 학습진행: handles95374/52740. 초기24tensor정확일치. 종료뒤54xxx회전/직진·55xxx60초열·56xxx기립/전환기준평가. 미채택·goal active. [증거109](evidence/109-stand-support-ablation/README.md).

양발지지중간250k복사보존·검증. 평가진행handles11085/87552/57903/87717,원학습95374/52740계속. 중간57001/2회전/직진·58001기립/전환,최종540xx/550xx/560xx는예약만·미실행. 자막전체31/43완료·영상완시청0구분. [증거110](evidence/110-stand-support-evaluation/README.md).

양발지지중간평가4개 프로세스 모두exit0확인:양쪽회전0/8·전도0/12,기립/전환0/2통과. 원 학습계속·최종미평가·미채택. [증거110](evidence/110-stand-support-evaluation/README.md).

양발지지학습양쪽503,808추가·종료exit0확인. 최종평가handles83145/45318 진행: 새출발540xx회전/직진·550xx열·560xx기립/전환. legacy heat.tracked하한만이므로±10%속도기준별도판정. 자막32/43완료·영상완시청0,미채택·goal active. [증거110](evidence/110-stand-support-evaluation/README.md).

양발지지최종검증exit0:회전양쪽0/20·전도0/30·열6/6·7속도0/3·기립/전환0/4. 후보정지13초내미달·기립3.85m로미채택. 다음위상입력고정진단. [증거110](evidence/110-stand-support-evaluation/README.md).

위상입력고정진단5조건완료:기존60초무전도/기립실패,고정4조건약0.9–1.06초모두전도. 기존입력동결안미채택·전체목표미완료. 다음학습행동분포/클리핑·정차데이터분포계측,기존std변경실패93/101보존. [증거111](evidence/111-stand-phase-input/README.md).

행동분포계측4rollout×20초완료·on/off궤적정확일치. 실제pairedstd0.03–0.119·yaw축clip낮음. 학습정차42.8–42.9%/yaw명령5.1%확인. 다음turn지속2→6초대조(미구현),평가원래2초및전체게이트유지.  [증거112](evidence/112-action-distribution/README.md).

turn2초/6초선택옵션구현·집중19pass·전체135중66pass69skip. source카탈로그초기24tensor일치·보상동일. 각500k요청학습handles52713/45690진행. 최종610xx원래2초회전/직진·620xx열·630xx기립/정지평가예정(미실행). 미채택·goal active. [증거113](evidence/113-turn-command-exposure/README.md).

turn노출2/6학습handles52713/45690실제poll진행확인·최종검증driver준비(미실행). 자막전체33/43검토·완시청0. [증거113](evidence/113-turn-command-exposure/README.md).

turn노출2/6중간250k보존·평가시작 handles4470/54563/74991/39594,원학습52713/45690계속. 640xx원래2초회전/직진·65001기립/전환은중간진단이다. 최종610/620/630xx미실행. 자막전체34/43·완시청0구분. [증거114](evidence/114-turn-duration-evaluation/README.md).

turn노출중간평가4프로세스exit0. 양쪽0/8회전·0/12전도·0/2기립전환기준통과. 학습handles52713/45690계속·최종미실행·미채택. [증거114](evidence/114-turn-duration-evaluation/README.md).

turn노출2/6양쪽503,808추가·학습exit0. 최종평가handles78445/39828실행중·후보미채택·goal active. 자막35/43·완시청0구분. [증거114](evidence/114-turn-duration-evaluation/README.md).

turn노출2/6최종모두exit0. 회전양쪽0/20·전도대조0/30후보1/30·열6/6·7속도0/3·기립전환0/4. 후보기립17.4–17.75m·정지13초미달로미채택. 다음좋았던초기반사정책의value/advantage와첫PPO업데이트진단. [증거114](evidence/114-turn-duration-evaluation/README.md).

초기반사배우첫4096PPO업데이트진단완료·24초기tensor일치. 전후새2출발12회각도8/8·전도0/12. 첫업데이트즉시퇴행가설미지지,통합목표달성아님. 다음16k/64k퇴행시점진단(미실행). [증거115](evidence/115-first-ppo-update/README.md).

## 2026-10-09 추가학습 퇴행 진단

16k/64k 두학습·각12회 진단 완료: 회전량6/8→0/8, 전도각0/12. 같은개발출발을 재사용했으며 최종독립검증이 아니다. 최소속도조건도16k0/8·64k3/8로 통합목표미달. [증거116](evidence/116-progressive-ppo-update/README.md). 다음은 업데이트폭 영향분리, 기준선보존·goal active.

## 2026-10-09 PPO KL 제한 비교완료

선택형target_kl 기본None계약검증·전체135검사(66통과69생략)·집중19통과. .005후보64k학습완료. 새출발회전18/20 vs대조1/20, 각전도0/30이나후보최소속도2/20미달. 직진/열6회·기립/전환4회도완료·실패보존, 미채택. [증거117](evidence/117-ppo-kl-guard/README.md). 다음은같은KL제한에서선형속도오차벌점효과분리, goal active.

후속`kl005_linerr2_update64k_20261009` 실행시작: 같은sym_init·seed0·65536스텝·target_kl=.005에서lin_err=-2만추가. 학습로그 `/tmp/raptor-kl005-linerr2.log`, 마지막확인실행handle3652. 완료파일/프로세스를확인한뒤모델평가하며실패/원기준선을보존한다. 이문구는완료증거가아니다.

## 2026-10-09 KL·속도벌점 비교완료

후속학습3652정상종료. 선형속도벌점-2는공통조건회전12/20·최소속도10/20(대조18/20·2/20)로절충, 전도0/30. 열/직진6/6·기립/전환4회완료이나통합미달·미채택. 초기24tensor일치·보상계약2pass. [증거118](evidence/118-kl-speed-reward/README.md). 다음은감속시점/속도명령반응계측.

속도반응계측8rollout완료·on/off4쌍궤적정확일치. [증거119](evidence/119-turn-speed-sensitivity/README.md). 현재회전중정책속도입력+0.8격리진단시작(원목표4m/s유지), 완료검증전·기준선미교체.

속도feedforward첫진단은관측0.1배환산검증문오류로회전전차단(exit1). 실패보존·수정v2handle87169실행중, 다음실제poll/20개행확인. [증거120](evidence/120-turn-speed-feedforward/README.md). 성과미확정·goal active.

120v2실제exit0·20행완료: offset0/.8각회전량8/8·전도0/10, 최소속도0/8→3/8. 동시통과3/8미달·미채택. 다음회전구간유계속도피드백의실행가능성검토, 기존기준선/실패보존·goal active.

121유계속도피드백20회완료exit0: 각회전량8/8·전도0/10, 최소속도3/8→4/8·미채택. [증거121](evidence/121-bounded-speed-feedback/README.md). 다음위상주기와감속관계분리, 전체목표active.

회전위상2.16→2.46Hz진단20회완료: 회전량양쪽8/8·전도양쪽0/10·최소속도0/8→3/8. [증거122](evidence/122-turn-phase-rate/README.md). 미채택·다음유계속도보정과위상조정의상호작용검토, 전체goal active.

123위상/속도조합20회exit0: 둘다회전량8/8·속도4/8·전도0/10, 통과증가없음. 121대조10개결과정확일치·최초import실패보존. [증거123](evidence/123-phase-speed-interaction/README.md). 다음회전중전진충격량/접지력/토크여유계측, 전체미완료·goal active.

124힘/운동량계측6rollout완료·on/off3쌍궤적정확일치. 회전감속은진행/몸방향어긋남과CoM수평속도실감소동반, 횡방향CoM일음수·일수지오차<.1J. [증거124](evidence/124-longitudinal-turn-balance/README.md). 다음제한횡입력보정진단, 목표미완료·goal active.

125제한횡입력20회exit0: 둘다각도8/8·전도0/10이나최소속도0/8, 120대조정확일치·미채택. [증거125](evidence/125-lateral-feedback/README.md). 다음KL제한하횡속도오차보상단독학습검토, 원기준선보존·goal active.

126횡속도보상선택옵션구현·기본궤적/보상계약확인. 전체136중66pass70skip·집중16pass·CLI거부/24초기tensor일치. KL.005+lateral_err=-2 64k완료, 회전/최소속도각0/20·전도0/30·미채택. [증거126](evidence/126-lateral-reward/README.md). 다음초기actor/critic의회전/직진on-policy advantage계측, 전체goal active.

127첫배치random/catalogue각4096샘플저장·각24초기tensor정확일치·optimizer0회. 목표근방회전random0/catalogue200, 전체critic붕괴미확인. [증거127](evidence/127-first-rollout-advantages/README.md). 후속KL.005+catalogue64k 시작·완료미확정, 로그/tmp/raptor-kl005-catalogue64k.log. 원기준선/전체goal유지.

KL.005+catalogue64k handle75557 최종poll로정상종료exit0확인, final10395648. 평가성공은미확정, 모델/완료파일보존·학습중복시작금지.

128KL.005+카탈로그64k및평가모두exit0. 각도17/20·최소속도0/20·전도0/30, 열6/6·직진속도0/6·기립전환0/4로미채택. [증거128](evidence/128-kl-catalogue-evaluation/README.md). 다음첫배치gradient/value/entropy기여분리, 원기준선/전체goal active.

129첫gradient분리완료·optimizer0회. value의actorgradient0·entropy작음, 이전Adamexp_avg와현재actorgradient거의직교(원인확정아님). [증거129](evidence/129-first-gradient/README.md). 다음정책초기화동일·optimizerstate만초기화선택형대조, 원기준선/전체goal유지.

130선택형optimizer리셋구현·default유지/24tensor/100action/저장flag/state검증, 전체136중66pass70skip·집중19pass. freshAdam64k학습exit0·final10395648, 평가시작·결과미확정. [증거130](evidence/130-optimizer-history/README.md).

130freshAdam최종30회exit0: 각도4/20·속도0/20·전도1/30로기각, 추가통합시험미실행. [증거130](evidence/130-optimizer-history/README.md). 다음원optimizer유지/KL.005조건에서학습률한변수비교, 전체goal active. 확보자막43/43검토완료·영상완시청미완료구분.

131학습률선택옵션추가·기본3e-4보존·낮은3e-5/24tensor/100action/실제LR갱신/CLI거부계약통과. 전체136중66pass70skip·집중19pass. 원optimizer/KL.005유지64k학습handle72880진행, 완료후평가·기준선미교체. [증거131](evidence/131-learning-rate/README.md).

131낮은학습률64k학습exit0·final10395648, 모든기록LR3e-5확인. 공통67001–5×6조건평가시작·성공미확정.

131최종모든평가exit0: 회전량15/20·최소속도10/20·동시7/20·전도0/30, 직진/열6/6이나기립/전환0/4·미채택. [증거131](evidence/131-learning-rate/README.md). 다음저학습률후보의yaw보정×속도보정2×2진단시작·goal active.


## 2026-10-09 — 입력 보정 비교 132

131 동결 후보의 회전 배율/속도 피드백 2×2 비교 40회 완료. 두 보정 조합은 각도 8/8·속도 6/8·동시 6/8·전도 0/10이나 −1 rad/s 두 회 속도 미달로 미채택. 직진 결과는 대조와 정확히 같음. 전체 goal active, 기립·정지 미달 유지. 원값·진단 코드: `docs/evidence/132-input-factor-comparison/`. 다음: 남은 좌회전의 접촉 순서와 운동량 방향 비교.


## 2026-10-09 — 회전 접촉 수지 133

132 조합의 seed71001 좌우 ±1을 1ms 계측, off/on 네 회 궤적 정확 일치와 132 최소속도 정확 일치. 좌회전 전진 CoM 일 +6.93J, 우회전 +23.99J; 횡방향 일 −22.26/−24.07J. 좌회전 수직 충격량 좌/우 134.56/92.57Ns로 불균형이나 원인 확정 아님. 에너지 수지 오차 좌 .131J·우 .035J. 전체 goal active·미채택. 다음 한 단계: 회전 시작 다리 위상/접촉 상태 변화로 원인 분리. `docs/evidence/133-turn-contact-balance/`.


## 2026-10-09 — 회전 진입 시점 134

132 조합의 시작4/4.12/4.24/4.36/4.48초 비교40회 완료. 각도40/40·속도24/40·전도0/40, 기본4초8회132 정확일치. 느린회전20/20속도통과·빠른회전4/20. 모든조건을통과하는시점없어미채택. JSON bool 실패를보존후저장형변환수정. 다음한단계: 정차 초기무릎스프링하중의기여를격리진단. `docs/evidence/134-turn-entry-phase/`. 전체goal active.


## 2026-10-09 — 정차 무릎 스프링 하중 135

강성170/0×초기22001/22002 네 회 완료, 모두전도. 초기0.2초 공통창부하와기울기는강성0에서작아지나기립미해결. 실제원강성168.9704(질량스케일),초기qpos/qvel/ctrl·기하·모터배열동일,두텐던강성만차이. 원강성22001모든원행96정확일치. 미채택·전체goal active. 다음: 정차관절변형·COM/지지영역·제한접근으로하중부족과피드백부재분리. `docs/evidence/135-stand-knee-spring-load/`.


## 2026-10-09 — 기립 토크/지지 136

8회계측off/on궤적정확일치·135원행heat/tilt일치.원스프링초기0.2초무릎제한접근86.5–99%,해제0–5%;해제만으로기립미해결.평가reset가초기vx도부여(22001=.379638m/s)하는것확인.다음은초기속도0대조로정지자세/제동을분리하되기존기립실패·최종기준유지. `docs/evidence/136-stand-support-torque/`.goal active.


## 2026-10-09 — 초기속도0 대조137

free-base초기속도0/관절초기perturbation유지대조8회계측일치.원강성0.78/0.82초·강성0 1.20/1.22초에전도,초기속도제거만으로기립미해결.최종기준미변경.다음스프링정지각만기준무릎1.5rad에맞춘후보를진단,기존보행성능승계불가. `docs/evidence/137-zero-initial-speed-stand/`.goal active.


## 2026-10-09 — 스프링정지각 대조138

강성170·정지각1.5rad후보도원속도/무속도×2출발모두기립전도,계측8회일치.잘못기록된강성필드를원본보존후명시수정(물리강성은모두170).미채택·기존보행성능승계불가.다음:고관절±.05rad한구간응답을동일초기상태에서측정해피드백부호확인. `docs/evidence/138-stand-spring-rest-angle/`.goal active.


## 2026-10-09 — 자세응답/피드백139–140

동일상태고관절±.05rad20ms응답10회로피드백부호확인.정지각.956/1.5의PD off/on8회대조원값135/138일치이나기립모두전도·미채택.고관절자세제어만으로지지/정차미해결.다음:정지각1.5에서수동발가락강성만3배대조,능동축증가없고기존보행성능승계불가. `docs/evidence/139-stand-action-response/`, `docs/evidence/140-stand-orientation-feedback/`.goal active.


## 2026-10-09 — 수동 발가락강성 대조141

정지각1.5·강성170에서수동발가락12관절강성1/3배×2출발네회완료·모두전도.기본원행140대조정확일치.기구변경후보미채택·능동축증가없음.다음은현재8축원모델의정차독립학습경로구성(정차에위상달리기보상강제하지않음),성공시에도전체회전/열/전환원기준필요. `docs/evidence/141-stand-toe-stiffness/`.goal active.


## 2026-10-09 — 정차 전용 학습142

원래8축기구/46관측 유지·정차 보상만 교체하는StandEnv와별도PPO훈련/평가경로추가.원source20step물리/관측정확일치·집중19통과·전체138중66통과72생략.4096step smoke 저장/재로딩/평가exit0이나두회기립전도.503808step pilot `stand_skill_pilot_20261009` handle14329 actual running·마지막40960step,완료미확정.같은handlepoll·완료후독립기립평가,통합goal기준유지. `docs/evidence/142-stand-skill-training/`.

142 pilot handle14329의마지막로그 184320step.65536저장본동결평가60초×4개발출발은기립0/4·전도1.80/.90/1.30/1.00초.학습완료후공통조건재평가필요·goal active.같은handlepoll,로그타임아웃만으로재시작하지않음.


## 2026-10-09 — 회전입력 필터143

131보정조합에τ0/.1초40회비교완료.무필터각도20/20·속도11/20·동시11/20,필터16/20·12/20·동시10/20,전도각0/20.원4m/s·2초각도기준유지,필터미채택.정차pilot는동일handle14329계속실행중이며완료후원기립기준평가필요. `docs/evidence/143-turn-yaw-input-filter/`.goal active.


## 2026-10-09 — 정차최종144·terminal대조145

142원학습14329실제exit0·503808step완료.기립최종0/4,종료.44/.46/.78/.36초·꼬리바닥접촉확인.원시/할인진단순위상반해보상악용일반화하지않음.기본10유지하고실패벌점만100의새대조학습시작 `stand_skill_penalty100_20261009`,handle68223실제running·마지막로그139264step.초기13tensor/Adam0정확일치·종료전물리/보상일치·집중20통과.같은handlepoll,완료후기립원기준평가·전체goal active.증거144/145.


## 2026-10-09 — 속도입력/정책시계146

정책시계만속도입력에맞춘40회비교완료,두조건각도20/20·속도11/20·전도0/20으로개선없음·미채택.첫버전float연산차이재현실패보존후복구,기본20회143정확일치. `docs/evidence/146-turn-internal-clock/`.goal active.


## 2026-10-09 — 벌점100 최종평가147

145 handle68223 실제exit0·503808step완료.개발기립60초×4는0/4,종료.78/.88/.94/.76초.열지표는짧은종료구간평균이며정상상태/실물열아님.미채택·현재실행중학습없음.다음한단계:원8축기구의오프라인역동역학평형자세/토크가능성검색후실제출발검증,전체goal기준유지. `docs/evidence/147-stand-penalty-evaluation/`.goal active.


## 2026-10-09 — 정적평형 검색148

원8축기구/구동한계에서정적하중균형후보2개확인(연속무릎토크23.331/24.559Nm),외부힘0초기순방향가속도작음.하지만정확평형6.4–6.5초/미세초기노이즈1.8–2.5초에전도·전체6회기립실패.정적가능과안정성/원자세출발분리·미채택.별도149예정평형주변hip자세PD 진단handle14475실행시작,완료후확인.검색용SciPy1.18.1만venv추가·실패/설치log보존. `docs/evidence/148-static-equilibrium/`.goal active.


## 2026-10-09 — 평형 주변 자세PD149

148정적후보두개×정확평형/미세노이즈두조건6회PD대조완료,기립0/6·미채택.현재실행중학습/진단없음.다음한단계:원평형근처동역학유한차분선형화(이전목표램프상태포함),불안정모드와8구동축제어가능성확인.실제관측가능성/원자세출발/60초/통합goal조건유지. `docs/evidence/149-equilibrium-orientation-feedback/`.goal active.


## 2026-10-09 — 선형화/상태피드백150–151

68동역학상태/8구동입력(축증가아님)로평형선형화,불안정모드λ1.0624와국소입력영향확인.전체상태LQR은새근처seed20×60초전도0·위치/기울기/비행20/20,가상열포함17/20.원기준초기조건/46관측정책의완료아님.토크연속상한소프트웨어cap.995 진단handle94516실행시작·원모터능력미증가,3열실패조건먼저비교.기준선보존·goal active. `docs/evidence/150-local-linearization/`, `docs/evidence/151-local-state-feedback/`.


## 2026-10-09 — 정차전류제한152

원8축/같은평형/K의software연속전류cap.995는평형근처개발20×60초전부통과(전도0·이동최대.02256m·부하최대.990025).원기준reset4회는전도·0/4,전체시뮬레이터상태사용으로운영/원기준/실물카탈로그통과아님.원기준선보존·전체goal active.재현LocalStandModel 추가·집중22통과·전체143중66통과77생략.현재실행중작업없음.다음한단계:기존46관측의관측기/상태추정으로동일근처성능유지가능성검사,원초기자세접근/통합전환기준유지. `docs/evidence/152-stand-current-limit/`.


## 2026-10-09 — 관측 기반 기립153

기존46관측 중41개를 사용하는 추정기+국소제어로 평형근처 개발20×60초 전부 통과. 수동 관절 실제값은 제어 입력에서 제외했지만 몸통속도 관측은 아직 시뮬레이터 값이다. 원기준 시작4회 모두 전도·전체목표 미완료. 집중25통과·전체146중69통과77생략. 현재 실행중 학습 없음. 다음은 원시작 접촉·연동 구속 불일치 계측과 평형 접근 경로 검증, 센서 잡음/지연 평가. [증거153](evidence/153-stand-observer/README.md). goal active.


## 2026-10-09 — 원시작 접촉·접근154

원시작 equality 오차 .019–.055rad, 무지면반력 순간 무릎수동토크77–93Nm 확인. 계측 off/on 전체궤적 정확일치·새모델 원조건 행동153과 정확일치. 종속관절투영/접촉초기화/평형목표접근/양발하중 gate 비교 각각 기립0/4. 관절조건만의 전환은4회 모두 공중에서 진입해 결함 확인. 하중gate만으로 해결되지 않았다. 기본기구/환경/기준선 미변경, 실행중 프로세스 없음. 다음은 실제 접촉·수동 발가락 변형을 포함한 원시작 접근 경로의 오프라인 동역학 최적화/재생 검증. goal active. [증거154](evidence/154-reset-contact-and-approach/README.md).


## 2026-10-09 — 접근 궤적 최적화155

원시작 .6초 직접shooting20시간변수/8모터, 상대미분→절대미분 교정과전환뒤1.6초 평가 총640회 완료. 비용감소는 있었으나 원시작32재생 기립0/32. posthold best80step 행동/위치/기울기 정확재현 후2.42초 전도·관측기만의 문제 아님. 기본환경/기구/기준선 미변경, 원전체목표 미완료. 다음: 같은궤적에서 접근중 원 peak envelope 사용 후 연속cap 전환 대조(모터한계 미상향), 원기립/회전/STOP 기준 유지. 현재 실행중 작업없음·goal active. [증거155](evidence/155-stand-trajectory-shooting/README.md).


## 2026-10-09 — peak envelope/초기제동156

기존궤적상한24비교 실패(8기준선일치/8전환미도달중복). 제동초기값/Powell259평가후16개발재생에서 초기궤적+observer reset seed83002는원시작60초기립통과·3000행정확재현. max이동.03242m·기울기.07606rad·last40가상부하.90250. 고정후새86001–86020 독립조건은전부전도·0/20, 후보미채택. 처음16모두실패라는구두집계는정정했다. 원기구/한계/정책/기준선미변경·goal active. 다음: 관측한초기관절상태로8모터각각제동목표계산/다중원시작평가. 현재실행중작업없음. [증거156](evidence/156-stand-peak-envelope/README.md).


## 2026-10-09 — 관측 제동/접근피드백157

원고정궤적21공통raw필드156과정확일치·83002만통과. 관측초기제동대칭/개별, 접근중자세/자세+속도피드백4변형은각0/21·미채택. 순수관측 BrakingApproach 추가·비구동kp0 입력검사교정·집중29통과·전체150중73통과77생략. 원기구/정책/구동한계/전체기준보존·goal active. 다음: 성공83002와초기vx가가까운기존실패의첫.2초접촉/토크를1ms계측해차이발생시점확인. 현재실행중없음. [증거157](evidence/157-observation-conditioned-braking/README.md).


## 2026-10-09 — 접촉/전환158

83002/86013 실제초기vx.00597/.00173, 첫.8초1ms계측off/on및156raw정확일치. 양발7–8ms에하중·두조건비행0. 실패는.6초전환이후tail75%,이어hiproll90/95%·tail100%포화로확대. .2초혼합은기존성공유지/실패.94초, history유지/초기화변경4회실패. 원기구/한계/정책/원성공보존·goal active. 다음: 관측q/qd와원곡선으로servo target범위제한,실제입력추정반영대조·전체기립/회전/STOP조건유지. 현재실행중없음. [증거158](evidence/158-contact-and-handoff/README.md).


## 2026-10-09 — servo목표경계/예측검증159

관측q/qd+원curve로PD목표범위제한·실행input을prior반영. 전환포화일부감소하지만기존21개발조건1/21(83002만),원성공3000공통필드그대로·미채택. 집중33통과·전체154중77통과77생략. 실제저장상태에서20ms전이를6개복원하니비선형물리모델은작은raw오차,국소선형모델은큰오차;온라인추정정확성/실물증명아님. 다음: 추정posterior를입력으로비선형prior전진대조·truth온라인주입금지·원전체목표유지. 실행중없음·goal active. [증거159](evidence/159-servo-target-bounds/README.md).


## 2026-10-09 — 비선형추정/제어분리160

추정posterior만으로별도물리모델전진,비선형prior/예측관측두변형각개발1/21·기존성공유지/미채택. q/qd/gyro27관측정확투영도쌍1/2. 실제전체상태+servo경계진단0/2:83002는2.64초tail지면접촉(약96N)/기울기10도미달,86013은.86초기울기위반. 추정기만의단일문제아님·원환경/정책/기구/전체기준보존. 다음:접근관측자세에가까운평형후보와지역capture제어검증·tail비발접촉기준유지. 실행중없음·goal active. [증거160](evidence/160-nonlinear-observer/README.md).


## 2026-10-09 — 평형후보/초기구간161

원기구pitch4/5.54/7.08도정적후보3통과·근처10초9/9, 새참조원시작6회는기존83002만4/7도60초통과·86013미해결. 7도성공가상부하.84739·미채택. R입력비용10배도기존성공만. 결정적확인: 원접근21조건중16은.6초전에이동/기울기/비행gate이미위반·후속제어만으로해결불가. 조건제외없이초기capture부터수정. 다음:tail반응측정후hip감속/tail자세대응분리·전체기준유지. 실행중없음·goal active. [증거161](evidence/161-equilibrium-family-and-prefix/README.md).


## 2026-10-09 — 채널 전편 조사 재점검/원 자료 보완

공개목록121·자막전문43·영상음성완시청0·미확보78 유지. 자막3편 재시도모두429·추가확보0, 채널blog403/Scholar접근불가. ANYmal회복·OP3핵심자세·AnyBipe고정평가·Bimo고정트리구동모델·클러치석사초록·중국어수동보행·OK-Robot요구조건9원자료보완. Registry258/188/188·ID/참조검증완료, 다운로드/실패로그보존. 조사문서 `/Users/danny/Documents/Raptor_Engiuniverse_Research_20261009/research-refresh.ko.md`. 원환경/기준선미변경·전체goal active, 개발재개점161의초기capture그대로 유지. 전편시청요청미완료이며 추가조사완료로대체하지않음.


## 2026-10-09 — 초기hip/tailcapture162

원8축/46관측/기준유지,bodypoint속도Jacobian3대조·pulse24·5변형원개발105회완료. 대조21공통raw159정확일치1/21,tail-only1/21기존성공만,roll/paired세변형0/21·미채택. contact1ms4재생정확일치·PDclip오차1.60e-13Nm. paired의.24–.32초양발무하중34/43ms(대조0),첫.6초hip/tail포화0으로포화단일가설배제. 다음:무릎/스프링/수직지지반응분리·원전체기립/회전/STOP기준유지. 현재실행중없음·goal active. [증거162](evidence/162-early-hip-tail-capture/README.md).


## 2026-10-09 — 무릎수직지지163

원reset행동재생pulse48·무변경16전이정확일치. .1/.2와.24초무릎→수직속도반응부호반대로전구간고정gain부적합. 작은무릎구간보정42개발:대조1/21기존성공유지,paired0/21·새성공없음/미채택. 42첫10행원조건정확일치;paired초기flight4→3/9→7이나실패. NumPybool직렬화실패보존·cast교정후동일48재실행/독립근거중복계산안함. 다음:hip/knee/tail결합응답행렬·원전체기준유지. 실행중없음·goal active. [증거163](evidence/163-knee-vertical-support/README.md).
