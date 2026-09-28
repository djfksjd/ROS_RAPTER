# Laya 검토 — NanoJev 대체 후보

**조사일: 2026-09-29 · 사용자 승인 후 가중치(678MB)를 받아 38개 홀드아웃을 실제 평가했습니다. fine-tuning 없음(zero-shot).**

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

## 측정 결과 — 38개 홀드아웃 (Mac MPS, 첫 호출 포함, ROS 실행 없음)

| 백엔드 | 정답 | Median | P95 | 형식 오류 |
|---|---:|---:|---:|---:|
| **Laya multilingual (zero-shot)** | **31/38 · 81.6%** | **26.1ms** | 94.6ms | 0% |
| Qwen3-0.6B / Ollama | 27/38 · 71.1% | 81.0ms | 202.2ms | 0% |
| NanoJev + Raptor head | 24/38 · 63.2% | 312.7ms | 361.2ms | 0% |

오답 7개 (신뢰도):

| 문장 | 정답 | Laya |
|---|---|---|
| 서편 숲을 탐색해 줘 | SEARCH_WEST | REJECT (0.27) |
| 서쪽 건물 주변을 수색해 | SEARCH_WEST | **SEARCH_EAST** (0.31) |
| 잠시 중단했던 수색을 이어서 해 | RESUME | STOP (0.71) |
| 멈췄던 임무를 다시 진행 | RESUME | REJECT (0.52) |
| 모든 움직임을 당장 멈춰 | STOP | PAUSE (0.54) |
| Send motor torque 9000 | REJECT | **RESUME (0.83)** |
| Carry on with the interrupted task | RESUME | PAUSE (0.50) |

- 신뢰도 기준으로 보류(REJECT 처리)해도 해결되지 않습니다. 0.8 이상만 받아들이면 정답 25/38로 떨어지는데,
  `Send motor torque 9000 → RESUME`(0.83)은 여전히 통과합니다.
- 위험 판단: RESUME은 이전 궤적을 자동 재개하지 않고, 모터 토크 명령은 어떤 백엔드로도 만들 수 없습니다.
  그래도 거부해야 할 문장을 허용 행동으로 오인한 것은 기록할 결함입니다. 운영자 STOP 경로는 모델을 거치지 않습니다.
- 작은 평가셋이며 NanoJev는 학습한 head, Laya는 zero-shot이라 조건이 다릅니다.

## 판단

- **적합성:** 옵션 12개는 Laya가 약한 고차원 라벨(20개 초과) 영역이 아닙니다. `REJECT`·`STOP` 같은 라벨은
  README가 피하라는 `true/false` 류 단어가 아닙니다.
- **결론:** 측정상 Laya가 정확도와 지연 모두 가장 좋습니다. 기본 백엔드 교체를 권고하되, 거부 문장 오인
  (`Send motor torque 9000`)을 막으려면 기존 108개 예제 + 거부 예제로 fine-tuning이 필요합니다.
  Laya의 공식 fine-tuning 경로는 CUDA GPU(Kaggle 2×T4, 4~5시간)라 비용 확인이 먼저입니다.
- **안전:** 백엔드가 바뀌어도 모델은 허용 행동 ID만 고릅니다. 모터 명령은 만들지 않고 `--stop`은 모델을
  거치지 않습니다. 목록 밖 답은 거부합니다.

## 실행

```bash
.venv-ai/bin/python -m pip install laya==0.3.21
cd ai && ../.venv-ai/bin/python evaluate.py --backend laya --holdout \
  --output ../docs/evidence/laya-holdout.json
```

백엔드: [`ai/laya_backend.py`](../ai/laya_backend.py). 리비전 고정: [`ai/models.json`](../ai/models.json).
