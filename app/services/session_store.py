import asyncpg
from app.core.config import settings
from app.core.logger import logger


class SessionStore:
    def __init__(self):
        self.pool: asyncpg.Pool | None = None

    async def init_pool(self):
        logger.info(f"Connecting to PostgreSQL: {settings.database_url}")
        self.pool = await asyncpg.create_pool(
            dsn=settings.database_url,
            min_size=1,
            max_size=5,
        )
        async with self.pool.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS chat_history (
                    id SERIAL PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                )
            """)
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_session_created
                ON chat_history (session_id, created_at DESC)
            """)
        logger.info("PostgreSQL pool ready")

    async def close_pool(self):
        if self.pool:
            await self.pool.close()
            logger.info("PostgreSQL pool closed")

    async def add_turn(self, session_id: str, role: str, content: str):
        if not self.pool:
            return
        async with self.pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO chat_history (session_id, role, content) VALUES ($1, $2, $3)",
                session_id, role, content,
            )

    async def get_history(self, session_id: str, limit: int = 3) -> list[dict]:
        if not self.pool:
            return []
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                """SELECT role, content FROM chat_history
                   WHERE session_id=$1
                   ORDER BY created_at DESC LIMIT $2""",
                session_id, limit * 2,
            )
        return [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]

    async def cleanup_old(self, hours: int = 24):
        if not self.pool:
            return
        async with self.pool.acquire() as conn:
            result = await conn.execute(
                "DELETE FROM chat_history WHERE created_at < NOW() - INTERVAL '1 hour' * $1",
                hours,
            )
            logger.info(f"Cleaned up old sessions: {result}")


session_store = SessionStore()