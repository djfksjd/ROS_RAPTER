# Laya 검토 — NanoJev 대체 후보

**조사일: 2026-09-29 · 저장소·README·BENCHMARKS 정적 검토, 패키지 설치와 백엔드 연결까지 완료 · 가중치 다운로드와 Raptor 명령 평가는 미실행.**

[Laya](https://github.com/NandhaKishorM/laya)는 문장을 생성하지 않고 한 번의 forward로 `choice`·`score`·`noul`(예/아니오)
질문에 답하는 인코더 기반 결정 모델입니다. Raptor가 NanoJev에 묻는 질문(허용 행동 12개 중 하나 선택)과
**입력 형식이 같아서**, 행동 목록·STOP 우선 처리·Safety Gate를 그대로 두고 백엔드만 바꿀 수 있습니다.

## 비교

| 항목 | NanoJev + Raptor head (현재) | Laya `laya-multilingual` |
|---|---|---|
| 라이선스 | 코드 MIT, 모델 카드 라이선스 미기재 | 코드·가중치 Apache-2.0 |
| 구조 | 고정 backbone + 학습한 decision head | mmBERT-base 인코더 322M, 1,024 토큰 |
| 한국어 | Raptor 명령 38개 홀드아웃 24/38 (63.2%) | 공개 MASSIVE 20지선다 `ko` 0.450, zero-shot (Raptor 명령 미측정) |
| 지연 | MPS FP32 median 312.7ms (e2e) | README: MPS 32.8ms/질문 (Raptor 환경 미측정) |
| 신뢰도 | softmax 확률 | proper scoring rule 학습, `answer_confidence`로 보류 판단 가능 |
| 다운로드 | 받아 둔 상태 | 678MB (다국어 단일 체크포인트) |
| 유지 상태 | 소규모 재현 저장소 | 2026-09-18 생성, 9-27 마지막 push, 활발 |

## 판단

- **적합성:** 옵션 12개는 Laya가 약한 고차원 라벨(20개 초과) 영역이 아닙니다. `REJECT`·`STOP` 같은 라벨은
  README가 피하라는 `true/false` 류 단어가 아닙니다.
- **주의:** 베이스 체크포인트는 전문 과제에서 zero-shot이 우연 수준에 가깝다고 README가 직접 밝힙니다
  (typed-decisions 0.36). 같은 다국어 체크포인트의 MASSIVE 20지선다 점수는 한국어 0.45, 영어 0.68입니다. 교체 결정은 같은
  38개 홀드아웃으로 측정한 뒤 합니다. 필요하면 기존 108개 학습 예제로 fine-tuning을 검토하되,
  Laya의 공식 경로는 CUDA GPU(Kaggle 2×T4, 4~5시간)라 비용 확인이 먼저입니다.
- **안전:** 백엔드가 바뀌어도 모델은 허용 행동 ID만 고릅니다. 모터 명령은 만들지 않고 `--stop`은 모델을
  거치지 않습니다. 목록 밖 답은 거부합니다.

## 실행 (가중치 다운로드 승인 후)

```bash
.venv-ai/bin/python -m pip install laya==0.3.21
cd ai && ../.venv-ai/bin/python evaluate.py --backend laya --holdout \
  --output ../docs/evidence/laya-holdout.json
```

백엔드: [`ai/laya_backend.py`](../ai/laya_backend.py). 리비전 고정: [`ai/models.json`](../ai/models.json).
