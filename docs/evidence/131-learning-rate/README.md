# 131 — 학습률 단일 변수 비교

train_symmetric에--learning-rate를선택형으로추가했다. 기본3e-4유지, 유한양수만허용하며실제PPOschedule와args에저장한다. CLI0/-1/nan/inf거부·default/3e-5저장/실제optimizerLR갱신·24초기tensor/100입력행동정확일치검증. 문서전체136검사66pass70skip·집중19pass.

source는sym_init, 원Adam이력유지·KL.005·random명령/보상/기계·seed0·65536step, 학습률만3e-5로변경했다. frozen117대조3e-4를보존한다. 학습handle72880진행중, 완료/평가성공은미확정이다. 이전sourceoptimizerstate의lr는첫PPOtrain에서schedule로갱신되므로초기저장paramgroup값만을실효학습률증거로사용하지않고progress.csv도확인한다. 기존기준선미교체·전체goal active.

후속확인: 학습72880실제exit0·final10395648, progress의모든기록train/learning_rate=3e-5확인. 공통개발30회평가시작·결과미확정.

최종평가모두exit0: 각도15/20·최소속도10/20·동시7/20·전도0/30. 60초직진/열6/6이나기립/전환전체0/4: 기립이동19.2703–19.5821m로미달. 117과같은공통개발조건의한훈련seed비교이며최종독립합격이아니다. 실패/원값보존·미채택. 다음동결후보에서yaw입력1/1.1와속도피드백off/on의2×2격리진단을시작한다.
