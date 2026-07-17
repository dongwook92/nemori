# Nemori Quickstart

> Updated on 2026-07-18

Nemori는 PostgreSQL + Qdrant 기반의 async memory system입니다. 시작 전에 인프라를 먼저 띄웁니다 (자세한 내용은 [postgresql_setup.md](postgresql_setup.md) 참고):

```bash
docker compose up -d
```

## 기본 사용

```python
import asyncio
from nemori import NemoriMemory, MemoryConfig


async def main():
    # DSN, API key, base URL은 환경변수에서 자동 resolve됩니다.
    config = MemoryConfig(
        llm_model="openai/gpt-4.1-mini",
        embedding_model="google/gemini-embedding-001",
    )

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
| `LLM_BASE_URL` / `EMBEDDING_BASE_URL` | OpenAI 호환 endpoint override (예: OpenRouter) |
| `QDRANT_URL` / `QDRANT_PORT` / `QDRANT_API_KEY` | Qdrant 접속 정보 (기본 `localhost:6333`) |

## 참고 사항

- 모든 API는 async입니다. `async with NemoriMemory(...)` context manager로 lifecycle(DB pool, Qdrant client)을 관리하세요.
- 기본 설정상 episode 생성에는 최소 2개의 메시지가 필요합니다 (`episode_min_messages=2`).
- semantic memory는 episode 생성 후 background에서 추출됩니다. context manager 종료(`__aexit__`) 시 진행 중인 background 작업이 정리됩니다. 현황은 `await memory.stats(user_id)`로 확인할 수 있습니다.
- 전체 예제는 [`examples/quickstart.py`](../examples/quickstart.py) 참고.
