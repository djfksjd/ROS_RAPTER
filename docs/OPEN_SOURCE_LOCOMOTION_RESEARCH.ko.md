# 2족 Raptor 오픈소스 재사용 조사

## 조사 요약

의사결정용 조사 · 2026.09.27 · 코드 및 문서 확인 / 실행 재현은 하지 않음

### 핵심 판단

활용할 자료는 충분하다. 가장 실용적인 조합은 Open Duck의 모델·학습 정책으로 비교 기준을 만들고, PlaCo로 우리 로봇의 발 위치와 무게중심 궤적을 생성하며, MuJoCo 기반 학습 환경의 관측·보상·정책 실행 구조를 선별해 가져오는 것이다. 세 프로젝트는 각각 기준 로봇, 운동학 도구, 학습 기반을 제공하므로 같은 역할을 중복 구현할 필요를 줄인다. 이는 문서와 코드에 근거한 추천이며 Raptor에서의 성공을 확인한 결과는 아니다. [^1][^2][^3][^4]

지금의 10축 Raptor와 완전히 일치하는, 이미 학습되어 즉시 실행 가능한 정책은 이번 조사에서 확인하지 못했다. Open Duck XML은 다리당 5축과 머리·목 4축을 포함한 14개 actuator를 정의한다. 우리 로봇은 다리당 4축과 꼬리 2축이다. 행동 차원뿐 아니라 질량·발 형상·관측 순서도 달라, 체크포인트를 그대로 꽂는 방법은 적합하지 않다. 기존 10축 설계를 14축으로 바꿀 이유도 없다. [^18][^17]

Disney 계열에도 재사용할 수 있는 실제 자료가 있다. Newton 자산 저장소의 DR Legs는 USD 모델과 보행 정책을 제공한다. 다만 BDX 캐릭터 완제품과 동일하지 않고, 연구·소프트웨어 개발 목적의 별도 이용 조건이 있다. 최신 Microduck은 정책 실행과 학습 방법을 더 넓게 공개하지만 학습 환경은 CUDA를 요구한다. [^7][^9][^10][^23]

Mac에서는 우선 일반 MuJoCo와 ONNX CPU 추론으로 작은 기준 실험을 구성하는 편이 현실적이다. MJX의 Apple Silicon 지원 문구만으로 M5 GPU 대규모 학습이 검증됐다고 해석하면 안 된다. JAX Apple GPU는 실험적이고, 조사한 Open Duck 기본 설치 설정에도 CUDA 의존성이 있다. [^12][^13][^19][^22]

권고 순서: 공개 기준 보행 재현 → 10축 Raptor 운동학·접촉 검증 → 별도 MuJoCo 모델 대조 → 필요할 때 Raptor 전용 정책 학습 → Gazebo 통합. 오늘은 조사만 수행했다.

## 조사 범위와 후보 우선순위

### 무엇을 절약할 것인가

조사 질문은 “랩터와 닮았는가”보다 “실제 모델·코드·정책을 받아 검증의 출발점을 줄일 수 있는가”로 정했다. 사용자 제공 Growbotics 조립 안내는 발견 경로로 검토했으나 웹 수집에서 본문이 충분히 추출되지 않아 핵심 기술 판단은 제작자 저장소를 우선했다. 검색 결과·소개 문구와 실제 파일 트리, 의존성, 모델 actuator 및 라이선스를 구분했다. [^21]

| 순위 | 후보 | 줄일 수 있는 작업 / 주요 제한 |
| --- | --- | --- |
| 1 | Open Duck Mini v2 | 기준 로봇·공개 정책·MuJoCo 경로 / 모델-정책 버전 맞춤 필요 [^1][^18] |
| 1 | PlaCo + motion generator | 발·COM 궤적 및 IK 기반 / 4축 다리의 제약 설정 필요 [^3][^4] |
| 2 | Microduck + RL | 관측 계약·보상·ONNX·구동기 모델 / 최신 학습 CUDA [^6][^7] |
| 2 | MuJoCo Playground | 학습 구조 재사용 / Raptor 환경·보상은 별도 [^2][^12] |
| 3 | Disney DR Legs | 공식 2족 모델·보행 정책 / 폐루프 기구·별도 라이선스 [^9][^10] |
| 3 | BDX-R | MJCF·URDF·RL 예제 / 미완성 문서·배포 단계 [^14][^24] |
| 참고 | Menagerie / BAM | 접촉 모델·수동 기구 / 모터 동정 방법 [^5][^16] |
| 보류 | Mesozoic Labs | 공룡형 MJCF·학습 실험 / 현재 성능·권리 범위 불확실 [^15] |

평가는 우리 프로젝트의 우선순위에 맞춘 공학적 판단이다. “공개 파일 존재”, “제작자 실행·영상 보고”, “이 Mac에서 재현”, “우리 Raptor에서 검증”을 각각 다른 단계로 보았다. 이번 조사에서 마지막 두 단계는 수행하지 않았다.

가정: 현재 10 active DOF 유지, M5 Mac 사용, ROS 2 Jazzy/Gazebo 유지, 유료 연산 자원 생성 없음. 외부 코드는 원본 보관 후 비교 대상으로만 읽었으며 프로젝트에 통합하지 않았다.

## 발견 1 · Open Duck 보행 기준선

Open Duck Mini v2 저장소에는 BEST_WALK_ONNX.onnx와 BEST_WALK_ONNX_2.onnx가 실제로 존재하고, README는 실물 보행·외란 영상과 실행 안내를 연결한다. 공개 정책이 있으므로 전혀 새로 학습하기 전에 원래 로봇의 동작을 재현하는 시도를 할 수 있다. 단, 파일 존재와 이 Mac에서의 동작 성공은 별개이며 바이너리 실행 검증은 하지 않았다. [^1]

현재 브랜치의 실제 파일 트리에서는 README에 적힌 v2_rl_walk_mujoco.py 경로가 발견되지 않았다. experiments/v2/onnx_AWD_mujoco.py 등 다른 실행 스크립트가 있고, 이 파일은 runtime 패키지와 BAM에 의존한다. 따라서 오래된 명령을 그대로 복사하는 대신 정책과 모델, 추론 코드가 맞았던 revision을 먼저 맞추는 것이 필요하다. 이 준비 비용을 무시하고 “즉시 실행”이라고 추천하지 않는다. [^1]

별도 Open Duck Playground에는 새 로봇 추가 절차가 있고 base.py, constants.py, joystick.py, runner.py, XML 및 추론 코드를 나눠 놓았다. ONNX wrapper는 CPUExecutionProvider를 명시한다. 반면 pyproject.toml은 jax[cuda12]를 요구한다. CPU 추론 경로와 CUDA 학습 의존성을 분리해야 하며, 최신 Playground가 예전 Mini 정책과 자동 호환된다는 근거는 확인하지 못했다. [^2][^19][^22]

### 우리 모델과의 구체적 차이

| 대상 | 관절 구성 | 적용 판단 |
| --- | --- | --- |
| Raptor | 다리 4+4 / 꼬리 2 | 총 10 active 유지 |
| Open Duck XML | 다리 5+5 / 목·머리 4 | 총 14 actuator |
| 추가 leg 축 | 각 다리 hip yaw | 우리 관절에 해당 축 없음 |
| 재사용 단위 | 모델-정책 원본 묶음 | 별도 기준 로봇으로 우선 재현 |

14차원 행동에서 머리 4개만 제거해도 남은 10개는 “다리 10축”이므로 우리 “다리 8축+꼬리 2축”과 의미가 다르다. 각도 부호·영점·제어 주기·관측 이력까지 맞춰야 하므로 출력의 잘라내기나 이름 교체로 해결할 수 없다. 재사용 목표는 완성된 Raptor 정책이 아니라 시험 절차와 학습 설계다. [^18][^22]

권리 확인: Mini 본체는 Apache-2.0 표시를 확인했다. Playground와 reference motion generator는 조사한 루트 트리에서 명시적 LICENSE를 찾지 못했다. 본체의 라이선스를 다른 저장소에 자동 적용하지 않고, 코드 복사·재배포 전 확인 대상으로 남긴다. [^1][^2][^3]

## 발견 2 · PlaCo가 현재 병목에 가깝다

PlaCo는 작업 공간의 목표와 제약을 QP로 풀어 역운동학·역동역학을 구성하는 라이브러리다. 발 프레임, 무게중심, 몸체 자세와 관절 제약을 다룰 수 있고, macOS arm64 배포와 MIT 라이선스를 문서에서 확인했다. “발을 이 위치에 두면서 몸통을 옮긴다”를 표현할 도구라는 점에서 관절 각도열을 수작업으로 반복하는 방식보다 재사용 가치가 높다. [^4]

Open Duck reference motion generator는 PlaCo로 참조 보행을 만들고, 이를 imitation reward에 사용할 다항식 계수로 변환하는 구조다. 생성-피팅-재생의 흐름 자체는 우리 보행 준비에 참고할 수 있다. 다만 이 저장소에는 이식 후 학습 검증이 필요하다는 TODO도 있어, 출력 궤적을 동역학적으로 검증된 보행으로 받아들이면 안 된다. [^3]

현재 Raptor의 마지막 기록에서는 하중 이동 중 발목 속도 제한 도달과 tracking error가 함께 관측되었다. 원인은 아직 확정하지 않았다. 이 상황에서 RL 학습을 먼저 길게 실행하기보다, 목표 발 위치와 COM 이동이 4축 다리의 기구학적 가동 범위 및 관절 속도 한도 안에 있는지 확인하는 비교 실험이 우선이라는 판단이다. [^17]

### 제안하는 재사용 경계

첫 단계는 기존 URDF의 관절축·관절 제한·발 기준 좌표를 읽어 IK 모델을 구성하고, 양발 접촉 상태의 작은 COM 이동 목표만 푸는 것이다. 다음은 단일 지지에서 발끝 위치와 발 pitch에 우선순위를 주는 것이다. 다리당 4축으로 독립적인 발 6차원 pose와 몸체 목표를 모두 강제하면 과제 자체가 불가능할 수 있어, 이 경우 관절을 늘리기보다 목표의 자유도와 가중치를 줄여야 한다. 이는 적용 설계 제안이며 실제 계산 결과는 아니다. [^4][^17]

운동학 해가 나와도 물리적 균형은 별도다. 기존 Gazebo에서 접촉 해제, 미끄러짐, 몸체 기울기, 전진량을 관측하며 궤적을 검증해야 한다. 이 과정을 거친 궤적은 후속 imitation learning의 참조 후보가 될 수 있다. 현재 안정성 문제의 원인을 확인하지 않고 외부 gait 각도를 그대로 가져오는 방법은 추천하지 않는다. [^3][^17]

절약 가능한 부분: 일반 QP/IK 구현, 발·COM 목표 관리, 참조 궤적 생성 구조. 남는 작업: Raptor 프레임 매핑, 제한 조건, 접촉·동역학 검증. 소요 일수·성공률은 실제 적용 전이므로 추정치로 제시하지 않았다.

## 발견 3 · Microduck과 Disney의 공개 범위

### Microduck: 최신 운영·학습 구조 참고

Microduck의 runtime과 microduck_rl은 역할이 나뉘어 있다. 학습 저장소는 MuJoCo Warp 기반 mjlab/PPO, BAM 구동기 모델, 랜덤화, ONNX export를 제공하며 CPU MuJoCo 추론 경로도 문서화한다. 최신 정책 계약은 관측 61차원과 행동 14차원이다. 이 구조와 로그 설계는 유용하지만 우리 10축 정책과는 호환되지 않는다. [^6][^7]

학습 quickstart는 CUDA GPU를 요구하므로 M5에서 같은 학습 명령을 실행하는 계획은 잡지 않는다. 코드에는 Apache-2.0을 명시하지만 README는 3D 모델을 Creative Commons BY-SA-NC로 별도 표기한다. 모델 파일별 구체적 조건을 확인하기 전 코드와 동일한 재사용 범위로 취급하지 않는다. [^7]

### Disney DR Legs: 실제 정책이 있는 별도 로봇

Newton assets의 disneyresearch/dr_legs에는 여러 USD 모델, 100 Hz 관절 공간 애니메이션, drlegs_walk.onnx 및 .pt, YAML 설정이 있다. 애니메이션 재생과 폐루프 정책 실행은 구분해야 한다. 모델은 수동·능동 관절 및 여러 기구학적 폐루프를 가진 직렬-병렬 혼합 다리다. [^9]

Newton의 DR Legs 예제는 정책 행동 12차원, 관측 94차원을 요구한다. 공개된 모델·정책·실행 코드가 존재한다는 의미는 분명하지만, 우리 10축 Raptor에 직접 이식할 수 있다는 의미는 아니다. 또 이 로봇을 Disney BDX 캐릭터 완제품의 전체 설계 공개로 소개하지 않는다. [^11][^23]

DR Legs 자산은 연구 또는 소프트웨어 개발·벤치마킹 목적에 제한되고 재배포 시 고지 보존 조건이 있다. Newton 엔진의 Apache-2.0을 개별 자산에 그대로 적용하면 안 된다. Newton은 macOS CPU 지원을 명시하지만 이 Kamino 예제의 Mac 실행과 성능은 확인하지 않았다. [^8][^10]

판단: Microduck은 학습·배포 설계 참고 우선, DR Legs는 공식 2족 정책 비교와 폐루프 기구 연구용 후순위. 둘 다 현재 Raptor 보행 완료를 대체하는 산출물은 아니다.

## 발견 4 · 추가 후보의 활용 범위

### BDX-R: 실제 코드가 있으나 문서 완성도 확인 필요

BDX-R은 커뮤니티의 BDX 영감 프로젝트다. MjLab 저장소는 MJCF와 속도 추종 학습 작업을 제공하고 IsaacLab 저장소에는 URDF·USD가 있다. 다만 MjLab README의 실물 배포, IMU 관측, rough terrain 등은 미완료 roadmap으로 남아 있다. 별도 문서 사이트의 MjLab 안내는 placeholder 수준이고 sim2real 페이지는 실질 절차가 비어 있어, 소개만 보고 완성된 튜토리얼로 추천하지 않는다. [^14][^20][^24]

### Mesozoic Labs: 공룡 모델 연구 참고

Velociraptor MJCF와 MuJoCo/Gymnasium 및 JAX/MJX 학습 기반이 공개되어 있다. 현재 README의 랩터는 action/actuator 22개이며, 대표 GIF도 현 모델의 증거가 아닌 과거 미검증 산출물로 표시한다. 일부 기립 gate의 한계 역시 명시되어 있다. 외형상 유사하지만 우리 10축 로봇의 검증된 보행 정책 공급원으로 선정하지 않는다. 루트 LICENSE도 확인하지 못해 재사용 권리 확인이 필요하다. [^15]

### Menagerie와 BAM: 작은 단위로 쓰면 유용

MuJoCo Menagerie는 Cassie, Berkeley Humanoid 및 IIT SoftFoot 등을 제공한다. Cassie의 수동·폐루프 기구, SoftFoot의 발 모델은 관절·접촉 표현을 읽어볼 후보지만, 이 저장소가 곧 해당 로봇의 학습된 보행 controller 묶음인 것은 아니다. 모델별 LICENSE를 별도로 제공하므로 필요한 모델만 선택해 검토한다. [^16]

BAM은 마찰·하중 의존성 등 servo actuator 동정과 시뮬레이션 연결에 초점을 둔다. 향후 실제 감속기·서보를 선정하고 측정할 때 model mismatch를 줄일 참고 자료다. 다른 로봇의 서보 파라미터를 아직 정해지지 않은 Raptor 모터의 값처럼 사용하는 것은 피한다. [^5]

이번 검색에서 KAIST Raptor와 현재 10축 구조에 맞는 공식 재현 패키지·체크포인트 조합은 찾지 못했다. 이는 비공개 또는 부재를 증명한 결과가 아니다. 이름이 RAPTOR인 드론 제어 저장소는 대상에서 제외했다.

## 발견 5 · Mac 실행과 Gazebo 복귀 조건

### Apple Silicon과 Apple GPU를 분리한다

MuJoCo MJX 문서는 Apple Silicon을 지원 대상으로 적고 있다. 그러나 JAX 설치 표의 Apple GPU는 experimental이며 Mac GPU 안내는 표준 CPU 설치를 권한다. 두 설명을 함께 읽으면 “Mac에서 작은 CPU 실험 가능”과 “M5 GPU 학습 stack이 이 프로젝트에서 검증됨”을 구분해야 한다. 후자는 이번에 확인하지 못했다. [^12][^13]

| 경로 | Mac에서의 조사 판단 |
| --- | --- |
| 일반 MuJoCo + ONNX CPU | 공개 추론 코드 확인. 로컬 최소 재현 후보, 아직 실행 안 함 [^22] |
| PlaCo | macOS arm64 wheel 안내 확인. Raptor 적합성은 별도 [^4] |
| MJX / JAX CPU | 작은 모델 검증 후보. 대규모 학습 속도 미측정 [^12][^13] |
| Apple GPU / Metal | 실험적 지원. M5에서 버전·연산 호환성 검증 필요 [^13] |
| Microduck RL / mjlab | 학습은 CUDA 요구. Mac 로컬 GPU 경로로 추천 안 함 [^7] |
| Newton | 엔진 macOS CPU 지원. DR Legs 예제는 미재현 [^8][^11] |

### sim-to-sim 전환에서 확인할 것

MuJoCo에서 걷는 정책을 얻더라도 Gazebo로 복귀할 때는 질량·관성·관절축·부호·영점·actuator 제한·제어 주기·센서 좌표·마찰과 접촉 설정을 맞춰야 한다. Raptor의 기존 position interface 기록과 외부 모델의 PD actuator 설정은 이름이 position이라는 이유만으로 같은 동역학이라고 가정할 수 없다. [^17][^18]

권고하는 완료 기준은 원본 정책의 정상 재생, 10축 모델 일치 검사, 발 접촉이 실제로 번갈아 해제되는 전진, 명령 속도 추종, STOP 복귀 기록이다. 먼저 평지에서 통과한 뒤 수동 발가락과 분절 꼬리의 물리적 복잡도를 단계적으로 비교한다. 이는 검증 계획이며 실행 결과가 아니다.

클라우드 학습은 필요 시 비용과 장비를 별도 판단한다. 이번 조사에서 패키지 설치, 모델 다운로드, 유료 job, 로봇 제어·학습 실행은 하지 않았다. 수집한 것은 텍스트 소스와 메타데이터다.

## 종합 판단과 실행 전 점검

### 종합 판단과 권고

재사용의 이익은 세 층에서 발생한다. 원본 Open Duck 기준선은 “동작하는 로봇과 정책의 짝”을 찾는 비용을 줄인다. PlaCo는 운동학 solver를 새로 작성하는 비용을 줄인다. Microduck·Playground는 관측과 보상, export의 구성 방법을 제공한다. 세 층 모두 우리 모델의 물리적 검증 자체를 대신하지는 않는다. [^1][^2][^4][^7]

가장 작은 다음 작업은 개발 재개가 허용될 때 Open Duck 한 revision의 모델·정책·추론 스크립트 계약을 읽고, 소규모 CPU 실행으로 원본 보행을 확인하는 것이다. 여기에 짧은 시간 상한을 두고 의존성 문제로 막히면 즉시 기록한다. 원본 재현만 오래 고치는 것을 프로젝트의 새로운 목표로 삼지 않는다.

그와 별도로 Raptor의 기존 Gazebo 문제를 한 변수씩 해결한다. PlaCo의 발·COM 목표로 기구학적 가능성을 확인한 뒤 현재 관절 궤적 방식과 비교한다. 10축 제한은 유지하며, open-loop reference가 실패할 때의 접촉·기울기·속도 기록을 확보한다. 이 결과가 있어야 정책 학습의 보상과 종료 조건도 설계할 수 있다. [^4][^17]

이후 Raptor 전용 MJCF를 만들 경우 원본 URDF를 보존하고 모델 일치 검사부터 수행한다. 공개 정책의 관절 숫자를 맞추기 위해 로봇을 변경하지 않는다. 꼬리의 질량을 포함한 10축 시스템을 학습 대상으로 유지하되, 처음에는 꼬리 목표를 고정한 실험과 움직이는 실험을 나눌 수 있다. 이는 액추에이터 추가와 구분되는 실험 제안이다.

### 한계와 미검증 항목

이번 결론은 원문·코드 정적 조사에 근거한다. 정책 바이너리 유효성, 실제 다운로드 경로, ONNX 입출력 shape 검사, Mac 재생, gait 성공률, 학습 시간과 Gazebo 전이는 실행하지 않았다. 제공자 영상도 독립 재현 증거로 취급하지 않았다. 불명확한 LICENSE는 허용으로 간주하지 않았다.

같은 생태계의 README 여러 개는 독립적인 재현 사례가 아니다. 주요 추천은 파일 존재·의존성·모델 구조·실행 계약을 교차 검토했지만 각 로봇의 성능을 세 개 독립 연구가 검증했다는 뜻은 아니다. 절약 시간은 적용 전 측정할 수 없어 임의의 일수나 백분율로 제시하지 않았다.

개발 상태: 기존 goal은 일시정지 유지. 코드 변경, git push, 학습 job 또는 로봇 프로세스 재시작 없음. 조사 산출물은 Documents의 별도 폴더에 저장했다.

## 방법론과 확인 기록

### 조사 방법

범위를 현재 10축 Raptor의 개발 시간 절약으로 고정했다. 검색어는 Open Duck Mini v2, BDX/Disney/Newton, biped locomotion, PlaCo, MJX Apple Silicon, dinosaur robot open source 등으로 구성했다. 사용자 링크 외에 제작자 GitHub, 공식 MuJoCo/JAX 문서, Disney 공식 페이지를 수집했다.

GitHub REST API로 기본 branch와 tree SHA를 기록하고 README·파일 목록·라이선스 메타데이터를 보존했다. 필요한 XML과 pyproject, 추론 파일은 원문을 받아 actuator 수와 dependency, CPU provider를 확인했다. 메타데이터의 license=null은 법적 판정이 아니라 자동 식별 및 루트 파일 확인 결과로 취급했다.

결론을 초안으로 만든 뒤 반례를 확인했다. 그 과정에서 Disney 자산의 실제 보행 정책 존재, Open Duck README의 사라진 실행 경로, 14 actuator 구조, CUDA 의존성, BDX-R 문서 placeholder, Mesozoic의 미검증 GIF 표시를 확인하여 추천 강도를 조정했다.

조사는 원문 수집, 파일 구조·설정 대조, 반례 확인, 추천 강도 조정 순서로 수행했다. 저장소의 실행 코드를 수정하거나 외부 정책을 실행하지 않았다.

원문 스냅샷과 출처·근거·주장 기록은 로컬 조사 폴더에 보존했다. 이 문서는 GitHub에서 읽을 수 있도록 정리한 조사본이다. 모든 링크의 열람 기준일: 2026-09-27.

## 확인한 저장소 revision

조사 시점의 파일 구조를 다시 확인할 수 있도록 Git tree SHA를 기록했다. 저장소별 최신 상태는 이후 달라질 수 있다.

| 저장소 | 브랜치 | 조사 revision |
|---|---|---|
| apirrone/Open_Duck_Mini | `v2` | [`b23317a485b3`](https://github.com/apirrone/Open_Duck_Mini/tree/b23317a485b3cec7d8417f352478778b3475173c) |
| apirrone/Open_Duck_Playground | `main` | [`b9be205ac644`](https://github.com/apirrone/Open_Duck_Playground/tree/b9be205ac64488c23504ca42e5ec790337adeec3) |
| apirrone/Open_Duck_reference_motion_generator | `main` | [`3d7bc6fd80e6`](https://github.com/apirrone/Open_Duck_reference_motion_generator/tree/3d7bc6fd80e6c35587b2a5279314d946cf63456b) |
| Rhoban/placo | `master` | [`f353a634ed07`](https://github.com/Rhoban/placo/tree/f353a634ed07d2a46971b737d9863ead09720f28) |
| pollen-robotics/microduck_rl | `develop` | [`cb70b792312d`](https://github.com/pollen-robotics/microduck_rl/tree/cb70b792312d559a4da09064d92009079671815f) |
| newton-physics/newton | `main` | [`3b7c3c848b9f`](https://github.com/newton-physics/newton/tree/3b7c3c848b9ff012ab28273d09dc00523dcd961b) |
| newton-physics/newton-assets | `main` | [`a0547548eaa9`](https://github.com/newton-physics/newton-assets/tree/a0547548eaa966c2f5478bee496c3cfba1fa98fc) |
| BDX-R/BDX-R-MjLab | `main` | [`079acd294650`](https://github.com/BDX-R/BDX-R-MjLab/tree/079acd294650aa149644828723da517eabd808c9) |
| kuds/mesozoic-labs | `main` | [`04d107aa4817`](https://github.com/kuds/mesozoic-labs/tree/04d107aa48171e59ce233c150d6eb8b6d2a6ff88) |

## 출처

[^1]: [Open Duck Mini v2](https://github.com/apirrone/Open_Duck_Mini/tree/v2). 확인 위치: README / tree.

[^2]: [Open Duck Playground](https://github.com/apirrone/Open_Duck_Playground). 확인 위치: README / tree.

[^3]: [Open Duck reference motion generator](https://github.com/apirrone/Open_Duck_reference_motion_generator). 확인 위치: README.

[^4]: [PlaCo](https://github.com/Rhoban/placo). 확인 위치: README / License.

[^5]: [BAM actuator identification](https://github.com/Rhoban/bam). 확인 위치: README.

[^6]: [Microduck runtime](https://github.com/pollen-robotics/microduck). 확인 위치: README.

[^7]: [Microduck RL](https://github.com/pollen-robotics/microduck_rl/tree/develop). 확인 위치: README / Quickstart / License.

[^8]: [Newton](https://github.com/newton-physics/newton). 확인 위치: README / Requirements.

[^9]: [Disney DR Legs assets](https://github.com/newton-physics/newton-assets/tree/main/disneyresearch/dr_legs). 확인 위치: Assets.

[^10]: [DR Legs license](https://github.com/newton-physics/newton-assets/blob/main/disneyresearch/dr_legs/LICENSE). 확인 위치: Condition 1.

[^11]: [DR Legs policy example](https://github.com/newton-physics/newton/blob/main/newton/_src/solvers/kamino/examples/rl/example_rl_drlegs.py). 확인 위치: lines 58-59.

[^12]: [MuJoCo MJX documentation](https://mujoco.readthedocs.io/en/latest/mjx.html). 확인 위치: MuJoCo XLA.

[^13]: [JAX installation](https://docs.jax.dev/en/latest/installation.html). 확인 위치: Supported platforms / Mac GPU.

[^14]: [BDX-R MjLab](https://github.com/BDX-R/BDX-R-MjLab). 확인 위치: README / Roadmap.

[^15]: [Mesozoic Labs](https://github.com/kuds/mesozoic-labs). 확인 위치: README / Velociraptor.

[^16]: [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie). 확인 위치: Models / License.

[^17]: [Raptor pause checkpoint](https://github.com/djfksjd/ROS_RAPTER/blob/471c8fe/docs/PAUSE_CHECKPOINT.ko.md). 확인 위치: Local repository: docs/PAUSE_CHECKPOINT.ko.md.

[^18]: [Open Duck actuator XML](https://github.com/apirrone/Open_Duck_Playground/blob/main/playground/open_duck_mini_v2/xmls/open_duck_mini_v2.xml). 확인 위치: actuator.

[^19]: [Open Duck dependency manifest](https://github.com/apirrone/Open_Duck_Playground/blob/main/pyproject.toml). 확인 위치: dependencies.

[^20]: [BDX-R MjLab docs](https://bdx-r.github.io/source/rl/mjlab.html). 확인 위치: How to train.

[^21]: [User supplied assembly guide](https://robotics.growbotics.ai/ko/projects/hardware/open-duck-mini-v2/assembly). 확인 위치: Web retrieval.

[^22]: [Open Duck ONNX runtime](https://github.com/apirrone/Open_Duck_Playground/blob/main/playground/common/onnx_infer.py). 확인 위치: OnnxInfer.

[^23]: [Disney BDX official page](https://la.disneyresearch.com/bdx-droids/). 확인 위치: BDX Droids.

[^24]: [BDX-R IsaacLab](https://github.com/BDX-R/BDX-R-IsaacLab). 확인 위치: README / tree.

[^25]: [Microduck simulation guide](https://github.com/pollen-robotics/microduck/blob/main/docs/robot/simulation.md). 확인 위치: What it is / What you need.

