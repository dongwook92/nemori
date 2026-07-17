# Project Instructions

This file is the canonical source of repository-specific instructions.

- Work directly on `main`. Do not create another branch unless explicitly requested.
- Preserve tenant isolation using both `agent_id` and `user_id`.
- Keep PostgreSQL records and Qdrant vectors consistent when changing write, merge, or delete behavior.
- Add or update relevant tests and run the narrowest offline test. Run live API or E2E tests only when explicitly requested.
