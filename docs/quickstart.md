# Nemori Quickstart

> Updated on 2026-07-18

Nemori는 PostgreSQL + Qdrant 기반의 async memory system입니다. 시작 전에 인프라를 먼저 띄웁니다 (자세한 내용은 [postgresql_setup.md](postgresql_setup.md) 참고):

```bash
docker compose up -d
```

이 명령은 PostgreSQL, Qdrant와 함께 Phoenix 18.1.0을 시작합니다. Phoenix UI는
<http://localhost:6006>에서 확인할 수 있습니다.

## 기본 사용

```python
import asyncio
from nemori import NemoriMemory, MemoryConfig


async def main():
    # DSN, API key, base URL, embedding model은 환경변수에서 resolve됩니다.
    # 명시적 flush가 background processing과 경쟁하지 않도록 threshold를 높입니다.
    config = MemoryConfig(buffer_size_min=10)

    async with NemoriMemory(config=config) as memory:
        health = await memory.health()
        print(f"System healthy: {health.healthy}")

        await memory.add_messages("alice", [
            {"role": "user", "content": "I just moved to Tokyo last month"},
            {"role": "assistant", "content": "How exciting! How are you finding life in Tokyo?"},
            {"role": "user", "content": "Love it! The food is amazing, especially ramen"},
        ])

        # buffer에 쌓인 메시지를 episode로 즉시 변환
        episodes = await memory.flush("alice")
        print(f"Created {len(episodes)} episodes")

        results = await memory.search("alice", "Where does Alice live?")
        print(results)


if __name__ == "__main__":
    asyncio.run(main())
```

## 주요 환경변수

| 환경변수 | 용도 |
|---|---|
| `DATABASE_URL` (또는 `DSN`) | PostgreSQL DSN. 기본값 `postgresql://nemori:nemori@localhost:5432/nemori` |
| `LLM_API_KEY` / `OPENROUTER_API_KEY` / `OPENAI_API_KEY` | LLM API key (우선순위 순) |
| `EMBEDDING_API_KEY` / `OPENAI_API_KEY` | embedding API key |
| `EMBEDDING_MODEL` | embedding model override |
| `LLM_BASE_URL` / `EMBEDDING_BASE_URL` | OpenAI 호환 endpoint override (예: OpenRouter) |
| `QDRANT_URL` / `QDRANT_PORT` / `QDRANT_API_KEY` | Qdrant 접속 정보 (기본 `localhost:6333`) |

## Phoenix LLM tracing

Tracing 의존성을 설치하고 명시적으로 활성화합니다.

```bash
uv sync --extra tracing

export NEMORI_ENABLE_LLM_TRACING=true
export PHOENIX_COLLECTOR_ENDPOINT=http://localhost:6006/v1/traces
export PHOENIX_PROJECT_NAME=nemori
export OPENINFERENCE_HIDE_INPUTS=true
export OPENINFERENCE_HIDE_OUTPUTS=true
export OPENINFERENCE_HIDE_EMBEDDINGS_VECTORS=true
export OPENINFERENCE_HIDE_EMBEDDINGS_TEXT=true

uv run python examples/quickstart.py
```

`nemori.process` 아래에 `nemori.llm.episode`, `nemori.llm.semantic_direct` 등의
phase span과 자동 계측된 OpenAI-compatible LLM span이 생성됩니다. 각 trace에는
`agent_id`와 `user_id`가 함께 기록됩니다. 애플리케이션 자체가 Compose 컨테이너
안에서 실행된다면 collector endpoint는
`http://phoenix:6006/v1/traces`를 사용하세요.

Nemori는 기본적으로 대화·이미지·응답 본문, embedding 입력과 vector를 trace에서
마스킹합니다. 신뢰할 수 있는 로컬 디버깅에서 원문이 필요할 때만 해당 환경변수를
`false`로 설정하세요. Compose가 공개하는 PostgreSQL/Qdrant/Phoenix 포트도 기본적으로
loopback 인터페이스에만 바인딩됩니다.

실행 직전 UTC 시각을 기록했다면 REST API 기반 검증 도구로 phase와 tenant 전파를
확인할 수 있습니다.

```bash
uv run python scripts/verify_phoenix_traces.py \
  --start-time 2026-07-17T17:03:06Z \
  --require-phase episode \
  --require-phase semantic_direct \
  --agent-id default \
  --user-id alice
```

## 참고 사항

- 모든 API는 async입니다. `async with NemoriMemory(...)` context manager로 lifecycle(DB pool, Qdrant client)을 관리하세요.
- 기본 설정상 episode 생성에는 최소 2개의 메시지가 필요합니다 (`episode_min_messages=2`).
- semantic memory는 episode 생성 후 background에서 추출됩니다. context manager 종료(`__aexit__`) 시 진행 중인 background 작업이 정리됩니다. 현황은 `await memory.stats(user_id)`로 확인할 수 있습니다.
- 전체 예제는 [`examples/quickstart.py`](../examples/quickstart.py) 참고.
- `docker compose down`은 Phoenix trace volume을 보존하고,
  `docker compose down -v`는 PostgreSQL/Qdrant/Phoenix 데이터를 모두 삭제합니다.
