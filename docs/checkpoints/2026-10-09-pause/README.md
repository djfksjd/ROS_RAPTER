# 일시정지 체크포인트 — 2026-10-09

사용자 요청으로goal paused. 새 로봇 실험을 시작하지 않음. 마지막 증거163. 커밋/push는 체크포인트 보존 작업이다.

## 완료와 미완료

- 기준 `scr_a8_h2` 보존·미교체. 실험 작업 정책131 `kl005_lr3e5_update64k_20261009`도 함께보존; 채택정책 아님.
- 원4m/s회전 보정143: 각도20/20, 속도11/20·통합미달. 기립/가속/감속/STOP 전체카탈로그 미완료.
- 평형근처 관측제어153은20×60초통과하나 원시작의 일반 기립은미달. 원시작 개발83002만60초성공,새86001–86020전부실패·후보미채택.
- 마지막163: 무릎pulse48저장/무변경16정확재생. 구간보정42는대조1/21기존성공유지,paired0/21. 실패·직렬화수정로그 보존.
- 기구8축/원구동한계/46관측 유지. 태스크원기준을쉽게바꾸거나실물성능을달성한것으로보고하지않음.

## 다시 시작할 때

먼저 docs/WORK_STATE.ko.md, docs/PAUSE_CHECKPOINT.ko.md, docs/local-development.md, docs/UNI_AI_WORKFLOW.ko.md와AGENTS.md를 읽고 git status를 확인한다. 기존 Mac에서는 .venv-sim과1.2GiB runs가 남아 있다. 새 checkout에서는 환경문서에 따라의존성을복원하고 다음 명령으로필수파일을검증한다.

```bash
python3 docs/checkpoints/2026-10-09-pause/restore.py
python3 docs/checkpoints/2026-10-09-pause/restore.py --apply
.venv-sim/bin/python -m unittest discover -s tests -v
```

restore.py는SHA256확인후누락파일만복사하며다른기존파일이있으면거부한다. 보존목록 artifacts.json에원래경로/크기/hash가있다. model.zip/vecnorm.pkl은이저장소의신뢰할수있는checkpoint만로드한다. 전체실패훈련모델은Mac의기존sim/rl/runs에그대로있고전부공개한것은아니다. 핵심원시근거87–163과진단소스는docs/evidence에보존된다.

다음 한 단계:163 knee-response의접촉상태별부호변화를바탕으로원대조/paired의.1/.2/.24초원상태를재생하고hip/knee/tail입력의bodypoint vx·세계수직속도·pitch각속도응답행렬을측정한다. 접촉이탈구간의입력가능영역과수직지지/감속상충을분리한다. 원reset·원4m/s좌우±.5/±1회전·직진/가상열퇴행없음·60초기립·실제주행속도에서STOP기준유지. 진단용실제상태를온라인관측기에주입하지않는다.

## 조사와 실행 상태

공개채널목록121·자막전문43·영상음성완시청0·미검토78. 자체분석/출처/주장ledger는docs/research/engiuniverse-20261009에포함. 자동자막원문/영상/외부논문PDF/사용자노트원본은공개하지않았고Documents연구폴더에남아있다. HTTP429/403실패도보존. 전편시청완료아님.

일시정지전검사:AI환경154개77통과/77환경생략; sim환경154개모두통과. 두로그포함. ps목록에서관련학습/실험프로세스없음. 검사는시뮬레이션규약/코드에대한것이며goal성능통과또는실물검증아님.
