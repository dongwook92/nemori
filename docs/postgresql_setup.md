# PostgreSQL & Qdrant Setup

> Updated on 2026-07-18

Nemori는 metadata·text search에 **PostgreSQL**, vector 저장·검색에 **Qdrant**를 사용합니다. 두 backend 모두 필수입니다.

## 권장: Docker Compose

repo root의 `docker-compose.yml`로 한 번에 실행합니다:

```bash
docker compose up -d
```

- **PostgreSQL 16** — port `5432`, user/password/db 모두 `nemori`
- **Qdrant** — port `6333` (HTTP), `6334` (gRPC)
- 데이터는 Docker volume(`nemori_pg_data`, `nemori_qdrant_data`)에 영속화됩니다.

```bash
docker compose down        # 데이터 유지하고 종료
docker compose down -v     # volume까지 삭제
```

## 수동 설치 시

PostgreSQL 12+ 서버에 database와 user를 준비합니다:

```sql
CREATE USER nemori WITH PASSWORD 'nemori';
CREATE DATABASE nemori OWNER nemori;
```

user에게 CREATE, INSERT, UPDATE, DELETE, SELECT 권한이 필요합니다 (schema migration을 Nemori가 직접 수행하기 때문).

## 연결 설정

DSN은 환경변수 또는 `MemoryConfig`로 지정합니다:

```bash
export DATABASE_URL="postgresql://nemori:nemori@localhost:5432/nemori"
```

```python
from nemori import NemoriMemory, MemoryConfig

config = MemoryConfig(
    dsn="postgresql://nemori:nemori@localhost:5432/nemori",
    db_pool_min=5,   # asyncpg pool 최소 크기
    db_pool_max=20,  # asyncpg pool 최대 크기
)
```

resolve 우선순위: `MemoryConfig.dsn` 명시값 → `DATABASE_URL` → `DSN` → 기본값(`postgresql://nemori:nemori@localhost:5432/nemori`).

## Schema Migration

별도 migration 명령이 없습니다. `async with NemoriMemory(...)` 진입 시 `schema_migrations` 테이블을 기준으로 미적용 migration(`nemori/db/migrations.py`)이 자동 적용됩니다.

## Troubleshooting

| 증상 | 원인 | 해결 |
|---|---|---|
| `asyncpg.ConnectionError` | PostgreSQL 미기동 | `docker compose up -d` 후 healthcheck 대기 |
| `DatabaseError: Failed to create connection pool` | DSN 오류 또는 권한 부족 | `DATABASE_URL` 확인, user 권한 확인 |
| Qdrant connection refused | Qdrant container 미준비 | `docker compose ps`로 healthy 상태 확인 |
| Embedding dimension mismatch | collection 생성 후 embedding model 변경 | Qdrant collection 삭제 후 재적재 |
