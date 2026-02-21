import asyncpg
from loguru import logger
from .config import Config

class DBManager:
    def __init__(self):
        self.pool = None

    async def connect(self):
        if not self.pool:
            self.pool = await asyncpg.create_pool(dsn=Config.DB_DSN)
            logger.info("Database connection established.")

    async def close(self):
        if self.pool:
            await self.pool.close()

    async def init_schema(self):
        async with self.pool.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS tokens (
                    address TEXT PRIMARY KEY,
                    symbol TEXT,
                    name TEXT,
                    decimals INT,
                    chain TEXT,
                    liquidity DOUBLE PRECISION DEFAULT 0,
                    fdv DOUBLE PRECISION DEFAULT 0,
                    last_updated TIMESTAMP DEFAULT NOW()
                );
            """)

            await conn.execute("""
                CREATE TABLE IF NOT EXISTS ohlcv (
                    time TIMESTAMP NOT NULL,
                    address TEXT NOT NULL,
                    open DOUBLE PRECISION,
                    high DOUBLE PRECISION,
                    low DOUBLE PRECISION,
                    close DOUBLE PRECISION,
                    volume DOUBLE PRECISION,
                    liquidity DOUBLE PRECISION, 
                    fdv DOUBLE PRECISION,
                    source TEXT,
                    PRIMARY KEY (time, address)
                );
            """)
            
            try:
                await conn.execute("SELECT create_hypertable('ohlcv', 'time', if_not_exists => TRUE);")
                logger.info("Converted ohlcv to Hypertable.")
            except Exception:
                logger.warning("TimescaleDB extension not found, using standard Postgres.")

            await conn.execute("CREATE INDEX IF NOT EXISTS idx_ohlcv_address ON ohlcv (address);")

            await conn.execute("ALTER TABLE tokens ADD COLUMN IF NOT EXISTS liquidity DOUBLE PRECISION DEFAULT 0;")
            await conn.execute("ALTER TABLE tokens ADD COLUMN IF NOT EXISTS fdv DOUBLE PRECISION DEFAULT 0;")

    async def upsert_tokens(self, tokens):
        if not tokens: return
        async with self.pool.acquire() as conn:
            # tokens: list of (address, symbol, name, decimals, chain, liquidity, fdv)
            await conn.executemany("""
                INSERT INTO tokens (address, symbol, name, decimals, chain, liquidity, fdv, last_updated)
                VALUES ($1, $2, $3, $4, $5, $6, $7, NOW())
                ON CONFLICT (address) DO UPDATE
                SET symbol = EXCLUDED.symbol, liquidity = EXCLUDED.liquidity,
                    fdv = EXCLUDED.fdv, last_updated = NOW();
            """, tokens)

    async def batch_insert_ohlcv(self, records):
        if not records: return
        async with self.pool.acquire() as conn:
            try:
                await conn.executemany("""
                    INSERT INTO ohlcv (time, address, open, high, low, close, volume, liquidity, fdv, source)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                    ON CONFLICT (time, address) DO NOTHING;
                """, records)
            except Exception as e:
                logger.error(f"Batch insert error: {e}")

    async def update_token_market_data(self, records):
        """批量更新 token 的 liquidity/fdv。records: list of (liquidity, fdv, address)"""
        if not records:
            return
        async with self.pool.acquire() as conn:
            await conn.executemany("""
                UPDATE tokens SET liquidity = $1, fdv = $2, last_updated = NOW()
                WHERE address = $3;
            """, records)

    async def get_all_token_addresses(self):
        """返回数据库中所有 token 地址列表"""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch("SELECT address FROM tokens;")
            return [r['address'] for r in rows]