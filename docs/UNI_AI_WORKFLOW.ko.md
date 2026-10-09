# UNI_AI API로 개발 작업 진행하기

문서 기준일: 2026-09-27. 다른 컴퓨터에서 저장소를 clone/pull한 작업 에이전트가
이 문서와 로컬 `.env`만으로 API 사용 절차를 찾을 수 있도록 정리했다.
루트 [AGENTS.md](../AGENTS.md)가 이 문서로 연결한다.

## 1. 사용자가 준비할 것

Python 3와 HTTPS 네트워크 접근이 가능한 환경에서 저장소를 준비한다.
새 환경에서는 다음과 같이 clone한다. 이미 clone했다면 작업 트리를 확인한 뒤
`git pull --ff-only`로 갱신한다. 충돌이나 로컬 변경을 강제로 덮어쓰지 않는다.

```bash
git clone https://github.com/djfksjd/ROS_RAPTER.git
cd ROS_RAPTER
# .env가 없을 때만 템플릿 복사
if [ ! -e .env ]; then cp .env.example .env; fi
chmod 600 .env
```

편집기에서 `.env`에 다음 항목의 값만 채운다. `YOUR_GATEWAY_API_KEY`는 예시 문자열이다.
기존 `HF_TOKEN` 등 다른 항목은 보존한다.

```dotenv
UNI_AI=YOUR_GATEWAY_API_KEY
```

그다음 저장소를 작업 폴더로 연 에이전트에게 **“API로 작업 진행해”**라고 요청한다.
매번 API 문서 URL이나 모델 목록을 다시 전달할 필요가 없다. `AGENTS.md`를 자동으로
읽지 않는 에이전트에는 최초 한 번 루트 지침을 읽도록 설정해야 한다.

키는 Gateway API 접근이 허용된 조직에서 발급한 유효한 키여야 하며, 호출에는 계정의
크레딧이 사용된다. 키 입력만으로 ROS·Gazebo·Docker·모델 가중치까지 설치되는 것은 아니다.
로봇 실행에 필요한 준비는 [로컬 개발 가이드](local-development.md)를 따른다.

## 2. 어떤 작업을 API에 맡기는가

UNI_AI는 **개발 보조 모델을 호출하는 gateway**다. 현재 로봇의 Qwen/Ollama·NanoJev
mission 경로를 대체하거나 로봇 controller에 직접 연결하는 기능이 아니다.

| 담당 | 작업 |
|---|---|
| 로컬 작업 에이전트 | 현재 파일·git 상태 확인, 필요한 문맥 선별, API 호출, 응답 검토, 최소 수정과 테스트 |
| UNI_AI 모델 | 원인 가설, 구현 초안, 코드 리뷰, 다음 검증 절차 제안 |
| 실제 실행 도구 | 로컬/컨테이너 명령 실행과 결과 수집. 원격 API 응답과 별도 |

이 저장소는 문서와 호출 예제를 제공한다. 상주 자율 실행기나 API 응답을 자동 적용하는
프로그램은 설치하지 않는다. 터미널·네트워크 권한이 없는 채팅 모델만으로는 로컬 파일을
수정할 수 없다. 현재 Codex 대화의 실행 모델이나 과금 경로도 이 키로 전환되지 않는다.
호스트 에이전트의 사용량을 0으로 요구하면 그 환경 밖의 실행 방식이 필요하다.

## 3. 공식 endpoint와 인증

사용자가 처음 제공한 페이지는 **챗봇 호출** API다. 해당 endpoint는 챗봇 ID가 필요하고
`model` 값을 무시한다. 개발용 모델을 직접 선택하려면 아래 **일반 모델 endpoint**를 사용한다.

| 항목 | 값 |
|---|---|
| Base URL | `https://factchat-cloud.mindlogic.ai/v1/gateway` |
| 모델 목록 | `GET /models/` |
| 모델 직접 호출 | `POST /chat/completions/` |
| 인증 | `Authorization: Bearer <UNI_AI>` |
| 요청 형식 | `Content-Type: application/json` |
| 환경 변수 이름 | `.env`의 `UNI_AI` |

경로 끝의 `/`까지 사용한다. 별도 organization ID나 chatbot ID는 일반 모델 호출에 필요하지 않다.
SDK를 사용한다면 OpenAI 호환 `base_url`에 위 값을 넣을 수 있다. 아래 예제는 SDK 설치 없이
Python 표준 라이브러리만 사용한다.

공식 근거:

- [일반 모델 Chat Completions](https://docs.mindlogic.ai/docs/general/api-gateway/reference/chat-completions)
- [Gateway 인증](https://docs.mindlogic.ai/docs/general/api-gateway/getting-started/authentication)
- [챗봇 호출과의 차이](https://docs.mindlogic.ai/docs/general/baze/product/api-access)

웹 문서 본문을 읽기 어렵다면 각 URL 뒤에 `.md`를 붙인 공식 Markdown 문서를 확인한다.
문서 설명과 실제 API 응답이 다르면 확인한 차이를 기록하고 endpoint를 임의로 추측하지 않는다.

## 4. 모델 선택과 호출 제한

기본 분석·코드 작성 모델은 `gpt-6-sol`이다. 짧은 요약·분류는 `gpt-6-luna`, 복잡한 원인 분석이나
독립 검토가 필요한 경우 `gpt-6-astra`를 후보로 둔다. 이 구분은 작업 역할 제안이며,
가격이나 품질 순위를 보장하는 측정 결과가 아니다.

매 환경에서 `/models/` 응답으로 접근 가능한 ID를 확인한다. 기본 모델이 없다면
`gpt-5.6-sol` → `claude-sonnet-5` 순서 중 접근 가능한 모델을 선택하고 선택 사실을 보고한다.
모두 없으면 사용자에게 모델 선택만 묻는다. 조직이 제공하지 않는 ID로 반복 호출하지 않는다.

첫 검증은 **목록 조회 1회 + 짧은 completion 1회**로 제한한다. 이후에는 한 작업 단위씩
필요한 파일 조각만 전달하고 사용량을 기록한다. 동시 다수 모델 호출과 무한 자동 반복은 하지 않는다.
`max_completion_tokens`는 호출당 상한이지 계정 전체 지출 한도가 아니다. API 가격을 확인하지
않은 상태에서 금액을 추정하거나 유료 연산 자원을 생성하지 않는다.

GPT 계열에서는 `reasoning_effort: low`로 짧은 확인을 시작할 수 있다. 추론 토큰도 출력 예산에
포함되므로 낮은 상한에서 본문 없이 종료될 수 있다. 긴 분석은 공식 문서가 16,000 이상을
안내하지만, 빈 응답을 이유로 무조건 예산을 올리거나 같은 요청을 반복하지 않는다.
`finish_reason`, `usage`를 보고 문맥을 줄이거나 필요한 상한을 명시해 다음 요청을 결정한다.

## 5. 복사해 실행할 수 있는 최소 예제

**프로젝트 루트에서 실행한다.** 이 예제는 모델 목록을 확인하고 선택된 모델에 짧은 연결 확인을
한 번 요청한다. 로봇을 시작하거나 파일을 수정하지 않는다. 모델 호출에는 크레딧이 사용된다.
키는 셸 인자·로그에 넣지 않으며 `.env`를 Python으로 필요한 항목만 읽는다.
리다이렉트에 인증 정보가 전달되지 않도록 자동 리다이렉트를 거부한다.

```bash
python3 - <<'PY'
import json
import re
import urllib.error
import urllib.request
from pathlib import Path

class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None

path = Path('.env')
if not path.is_file():
    raise SystemExit('.env가 없습니다. .env.example을 복사하고 UNI_AI를 설정하세요.')
values = []
for line in path.read_text(encoding='utf-8').splitlines():
    match = re.match(r'^\s*(?:export\s+)?UNI_AI\s*=\s*(.*?)\s*$', line)
    if match:
        value = match.group(1)
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        values.append(value)
if len(values) != 1 or not values[0] or values[0] == 'YOUR_GATEWAY_API_KEY':
    raise SystemExit('UNI_AI를 비어 있지 않은 값으로 한 번만 설정하세요.')
key = values[0]
base = 'https://factchat-cloud.mindlogic.ai/v1/gateway'
opener = urllib.request.build_opener(NoRedirect())

def request(path, payload=None):
    body = None if payload is None else json.dumps(payload).encode('utf-8')
    req = urllib.request.Request(
        base + path, data=body,
        headers={
            'Authorization': 'Bearer ' + key,
            'User-Agent': 'raptor-uniai/1.0',
            'Accept': 'application/json',
            'Content-Type': 'application/json',
        },
    )
    try:
        with opener.open(req, timeout=55) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        # 오류 본문에 인증·개인 정보가 섞일 수 있어 그대로 출력하지 않는다.
        raise SystemExit(f'UNI_AI HTTP {exc.code}; 문서의 오류 처리 절차를 확인하세요.')
    except (urllib.error.URLError, TimeoutError, ValueError):
        raise SystemExit('네트워크·시간 초과·JSON 오류. 자동 재시도하지 않습니다.')

models = request('/models/')
ids = {item.get('id') for item in models.get('data', [])}
model = next((name for name in (
    'gpt-6-sol', 'gpt-5.6-sol', 'claude-sonnet-5'
) if name in ids), None)
if model is None:
    raise SystemExit('기본/대체 모델이 없습니다. 조직에서 허용한 모델을 선택하세요.')
print('선택 모델:', model)
payload = {
    'model': model,
    'messages': [{'role': 'user', 'content': '연결 확인입니다. API 연결 확인이라고만 답하세요.'}],
    'max_completion_tokens': 4096,
}
if model.startswith('gpt-'):
    payload['reasoning_effort'] = 'low'
result = request('/chat/completions/', payload)
choice = (result.get('choices') or [{}])[0]
content = (choice.get('message') or {}).get('content') or ''
print('응답 모델:', result.get('model'))
print('종료 이유:', choice.get('finish_reason'))
print('토큰 사용량:', json.dumps(result.get('usage'), ensure_ascii=False))
print('응답:', str(content).replace(key, '[REDACTED]'))
if not content:
    print('본문이 비어 있습니다. 호출 성공과 유효한 작업 결과를 구분하세요.')
PY
```

`.env` 항목에는 키 뒤에 인라인 주석을 붙이지 않는다. 키에 대한 `print`, `repr`, 일부 문자 확인,
인증 헤더 dump, `curl -v`, `set -x`를 사용하지 않는다. 전체 `.env`를 API 문맥으로 보내지 않는다.

## 6. 에이전트가 “API로 작업 진행해”를 받았을 때

1. `git status`, `AGENTS.md`, 현재 사용자 요청을 확인한다. 필요한 경우
   [작업 상태](WORK_STATE.ko.md)와 [체크포인트](PAUSE_CHECKPOINT.ko.md)를 읽는다.
   문서 작업만 요청받았다면 로봇 goal은 그대로 둔다.
2. `.env`의 `UNI_AI` 설정 여부와 파일 권한을 확인한다. 실제 값을 출력하지 않는다.
   키가 없다면 필요한 항목명만 알려주고 사용자에게 로컬 편집을 요청한다.
3. 공식 gateway의 모델 목록을 조회한다. 실제 사용할 모델 ID를 선택한다.
4. 사용자 목표·불변 조건·현재 오류·관련 파일 일부·검증 기준으로 짧은 문맥을 구성한다.
   외부 모델이 파일을 직접 읽었다거나 도구를 실행했다고 주장하게 하지 않는다.
5. 원인 분석 또는 최소 변경 제안을 API로 요청한다. 요청 모델, 응답 모델, 토큰 사용량,
   종료 이유와 선택한 결론을 기록한다. 키·인증 정보는 기록하지 않는다.
6. 응답을 실제 코드와 대조한다. 원격 모델의 명령·패치를 그대로 실행하지 않는다.
   승인된 범위의 한 문제만 수정하고 필요한 빌드·검증을 수행한다.
7. 실제 결과와 실패를 구분해 보고한다. 모델의 설명·영상·예상 결과는 성공 증거가 아니다.
   재검토가 필요할 때만 축약된 결과를 다음 API 호출에 전달한다.
8. 사용자에게 허가된 Git/백업 범위만 반영한다. API 연결 확인만으로 전체 goal 완료를 선언하지 않는다.

이 흐름은 의사결정을 API에 위임하는 가이드이며, 호스트 에이전트의 자체 실행·검증 비용까지
대체하지 않는다. 목표 관리 도구에 재개 기능이 없으면 goal 상태를 임의로 다시 만들지 않는다.

## 7. 오류와 실제 검증 기록

| 증상 | 처리 |
|---|---|
| `401` | 키가 비었거나 유효하지 않은지 확인. 키를 출력하거나 새 키를 채팅으로 요청하지 않음 |
| `403` | 조직 Gateway 권한·접근 정책과 비 JSON 차단 응답 여부 확인. 인증 실패로 단정하지 않음 |
| `404` | 모델 목록·endpoint 확인. 챗봇 경로와 일반 모델 경로를 혼동하지 않음 |
| `429` | rate limit 또는 크레딧 상태를 확인. 연속 재시도 중단 |
| `5xx` / timeout | 성공 여부·과금 여부가 불명확할 수 있어 자동 재시도하지 않음 |
| 빈 본문 / `length` | 추론 토큰 소진 여부 확인. 유효한 답변으로 처리하지 않음 |
| 응답 모델 ID 차이 | 요청·응답 ID 모두 기록. 자체 문자열만으로 실제 backend를 확정하지 않음 |

2026-09-27 이 Mac에서 확인한 기록:

- `.env`의 `UNI_AI` 설정과 권한 `600`을 확인했다. 값은 기록하지 않았다.
- 처음 Python 기본 헤더로 모델 조회 시 `403`이었고, 명시적인 User-Agent와 Accept로 다시 조회해
  목록 89개를 받았다. 헤더 변경이 유일한 원인이었다고 확정한 것은 아니다.
- `gpt-6-sol`의 실제 completion 응답을 받았다. 응답 usage는 prompt 1,352,
  completion 294, total 1,646 tokens였다. `credits` 값으로 금액을 확인하지 못했다.
- 체크포인트와 world 파일을 바탕으로 읽기 전용 진단 제안을 받았으며, 로봇 실험은 재시작하지 않았다.
- **위 기록은 당시 환경의 결과다. 다른 계정·컴퓨터의 접근 및 전체 모델 목록은 재확인해야 한다.**

현재 문서 추가 작업에서는 API를 다시 호출하지 않았다.

## 8. 사용자가 제공한 모델 ID 후보

아래는 사용자가 제공한 선택 후보이며, 전체 모델의 호출 성공을 검증한 목록이 아니다.
현재 가용 여부는 항상 `/models/` 응답을 기준으로 한다.

```text
gpt-6-sol
gpt-6-luna
gpt-6-astra
gpt-5.6-sol
gpt-5.6-terra
gpt-5.6-luna
gpt-5.5
accounts/fireworks/models/gpt-oss-120b
claude-opus-5-5
claude-sonnet-5
claude-opus-5
claude-fable-5-1
claude-fable-5
claude-opus-4-8
claude-haiku-4-5-20251001
gemini-3.8-flash
gemini-3.7-flash
gemini-3.6-flash
gemini-3.5-flash
gemini-3.5-flash-lite
gemini-3.1-pro-preview
grok-4.6
grok-4.5
grok-4-1-fast
google/gemma-4-31B-it
muse-spark-1.3
meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8
sonar-pro
sonar-reasoning-pro
solar-pro4
qwen3.8-max
qwen3.7-plus
qwen3.7-max
glm-5.3-flash
glm-5.3
glm-5.2
kimi-k3
seed-2-0-pro-260328
seed-2-0-lite-260428
deepseek-v4-pro
deepseek-v4-flash
```

## 2026-10-03 Gateway 재연결

공식 https://docs.mindlogic.ai/agent-setup/prompt.md, llms.txt, AGENTS.md, openapi.json 확인. 기존 .env UNI_AI 키 그대로, 명시적 User-Agent/Accept + x-api-key로 /models/ HTTP200. 기본헤더 bearer/x-api-key403 후 정상헤더 성공이며 키 오류로 단정하지 않는다. gpt-6-sol chat completion HTTP200(4214tokens), Codex용 /responses/도 Bearer+문서의 명시적헤더로 HTTP200(15tokens), 실제 OK 텍스트를 확인했다. 키·인증 본문·원본 사진 외부 전송 없음.

이 Mac ~/.codex/config.toml에 선택적 profiles.uni_ai/model_providers.factchat 추가: 기본 설정 불변, model gpt-6-sol, 공식 base_url, env_key UNI_AI, wire_api responses. 백업은 사용자 .codex 내부이며 키를 설정파일/쉘프로필에 복사하지 않았다. python3 scripts/uni_ai_codex.py 로 별도 Codex CLI를 시작하면 .env를 shell평가 없이 읽어 자식 프로세스에만 전달한다. --help 로 launcher 동작을 검증했으며 실제 Codex 대화 생성은 별도 실행이다. 현재 호스트 대화가 Gateway로 전환됐거나 호스트 과금이0이라고 주장하지 않는다.

### 2026-10-03 사용자 연결 요청 재확인

공식 설정 문서와 llms.txt → AGENTS.md → openapi.json을 다시 확인했다.
기존 선택적 Codex `uni_ai` 프로필과 안전한 launcher를 유지했다.
현재 `/models/` HTTP 200, 모델 ID 127개; `gpt-6-sol`, `gpt-6-luna`,
`gpt-6-astra`가 목록에 있다. `gpt-6-sol` `/responses/` HTTP 200,
상태 completed, 응답 `API connection verified.`, input 12 / output 8 /
total 20 tokens. launcher `--help`도 exit 0. 전체 모델의 호출 성공을 의미하지 않는다.
키는 출력하거나 설정파일·셸 프로필에 복사하지 않았다. 현재 호스트 대화 모델은
전환되지 않았다. 연결 확인 요청으로 로봇 실험을 재개하거나 GitHub에 push하지 않았다.
