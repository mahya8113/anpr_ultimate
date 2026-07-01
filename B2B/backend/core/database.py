"""
database.py - مدیریت اتصال به دیتابیس PostgreSQL
"""

import asyncpg
from contextlib import asynccontextmanager
from typing import Optional, AsyncGenerator
import logging
from .config import settings

logger = logging.getLogger(__name__)

# اتصال سراسری دیتابیس
_pool: Optional[asyncpg.Pool] = None


async def init_db():
    """مقداردهی اولیه اتصال دیتابیس"""
    global _pool
    
    try:
        _pool = await asyncpg.create_pool(
            settings.DATABASE_URL,
            min_size=1,
            max_size=20,
            command_timeout=60,
            max_queries=50000,
            max_inactive_connection_lifetime=300
        )
        logger.info("✅ اتصال به PostgreSQL برقرار شد")
        return _pool
    except Exception as e:
        logger.error(f"❌ خطا در اتصال به PostgreSQL: {e}")
        raise


async def close_db():
    """بستن اتصال دیتابیس"""
    global _pool
    if _pool:
        await _pool.close()
        logger.info("✅ اتصال PostgreSQL بسته شد")


async def get_db_pool() -> asyncpg.Pool:
    """دریافت pool اتصال دیتابیس"""
    if _pool is None:
        await init_db()
    return _pool


@asynccontextmanager
async def get_db_connection() -> AsyncGenerator[asyncpg.Connection, None]:
    """دریافت اتصال از pool (با context manager)"""
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        yield conn


async def execute_query(query: str, *args) -> list:
    """اجرای کوئری SELECT و بازگرداندن نتایج"""
    async with get_db_connection() as conn:
        return await conn.fetch(query, *args)


async def execute_insert(query: str, *args) -> Optional[int]:
    """اجرای کوئری INSERT و بازگرداندن ID رکورد ایجاد شده"""
    async with get_db_connection() as conn:
        return await conn.fetchval(query, *args)


async def execute_update(query: str, *args) -> str:
    """اجرای کوئری UPDATE/INSERT و بازگرداندن وضعیت"""
    async with get_db_connection() as conn:
        return await conn.execute(query, *args)


async def execute_transaction(queries: list) -> bool:
    """اجرای چند کوئری در یک تراکنش"""
    async with get_db_connection() as conn:
        async with conn.transaction():
            for query, args in queries:
                await conn.execute(query, *args)
        return True