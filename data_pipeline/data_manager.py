import asyncio
import aiohttp
from loguru import logger
from .config import Config
from .db_manager import DBManager
from .providers.birdeye import BirdeyeProvider
from .providers.dexscreener import DexScreenerProvider

class DataManager:
    def __init__(self):
        self.db = DBManager()
        self.birdeye = BirdeyeProvider()
        self.dexscreener = DexScreenerProvider()
        
    async def initialize(self):
        await self.db.connect()
        await self.db.init_schema()

    async def close(self):
        await self.db.close()

    async def pipeline_sync_daily(self):
        # Step 0: Refresh OHLCV for existing tokens
        logger.info("Step 0: Refreshing OHLCV for existing tokens...")
        existing_addresses = await self.db.get_all_token_addresses()
        if existing_addresses:
            async with aiohttp.ClientSession(headers=self.birdeye.headers) as session:
                tasks = []
                for addr in existing_addresses:
                    tasks.append(self.birdeye.get_token_history(session, addr))
                batch_size = 20
                refresh_candles = 0
                for i in range(0, len(tasks), batch_size):
                    batch = tasks[i:i+batch_size]
                    results = await asyncio.gather(*batch)
                    records = [item for sublist in results if sublist for item in sublist]
                    await self.db.batch_insert_ohlcv(records)
                    refresh_candles += len(records)
                logger.info(f"Refresh complete. {refresh_candles} candles added.")

        logger.info("Step 1: Discovering tokens from multiple sources...")
        limit = 20
        trending = await self.birdeye.get_trending_tokens(limit=limit)
        await asyncio.sleep(1.5)
        top_vol = await self.birdeye.get_top_traders(limit=limit)

        # 合并去重（按 address）
        seen = set()
        candidates = []
        for t in trending + top_vol:
            if t['address'] not in seen:
                seen.add(t['address'])
                candidates.append(t)

        logger.info(f"Raw candidates found: {len(candidates)} (trending={len(trending)}, volume={len(top_vol)})")

        selected_tokens = []
        for t in candidates:
            liq = t.get('liquidity', 0)
            fdv = t.get('fdv', 0)

            if liq < Config.MIN_LIQUIDITY_USD: continue
            if fdv < Config.MIN_FDV: continue
            if fdv > Config.MAX_FDV: continue # 剔除像 WIF/BONK 这种巨无霸，专注于早期高成长

            selected_tokens.append(t)

        logger.info(f"Tokens selected after filtering: {len(selected_tokens)}")

        if not selected_tokens:
            logger.warning("No tokens passed the filter. Relax constraints in Config.")
        else:
            db_tokens = [(t['address'], t['symbol'], t['name'], t['decimals'], Config.CHAIN,
                          t.get('liquidity', 0), t.get('fdv', 0)) for t in selected_tokens]
            await self.db.upsert_tokens(db_tokens)

            logger.info(f"Step 4: Fetching OHLCV for {len(selected_tokens)} tokens...")

            async with aiohttp.ClientSession(headers=self.birdeye.headers) as session:
                tasks = []
                for t in selected_tokens:
                    tasks.append(self.birdeye.get_token_history(session, t['address']))

                batch_size = 20
                total_candles = 0

                for i in range(0, len(tasks), batch_size):
                    batch = tasks[i:i+batch_size]
                    results = await asyncio.gather(*batch)

                    records = [item for sublist in results if sublist for item in sublist]

                    # 批量写入
                    await self.db.batch_insert_ohlcv(records)
                    total_candles += len(records)
                    logger.info(f"Processed batch {i}/{len(tasks)}. Inserted {len(records)} candles.")

            logger.success(f"Pipeline complete. Total candles stored: {total_candles}")

        # Step 5: Backfill liquidity/fdv from Birdeye token_overview
        logger.info("Step 5: Backfilling liquidity/fdv from Birdeye...")
        all_addresses = await self.db.get_all_token_addresses()
        if all_addresses:
            logger.info(f"Querying {len(all_addresses)} tokens...")
            async with aiohttp.ClientSession(headers=self.birdeye.headers) as session:
                update_records = []
                for i, addr in enumerate(all_addresses):
                    overview = await self.birdeye.get_token_overview(session, addr)
                    if overview and overview['liquidity'] > 0:
                        update_records.append((overview['liquidity'], overview['fdv'], addr))
                    if (i + 1) % 5 == 0:
                        logger.info(f"  Overview progress: {i+1}/{len(all_addresses)}")
                    await asyncio.sleep(1.2)  # 免费 API 限速 ~1 req/s
                if update_records:
                    await self.db.update_token_market_data(update_records)
                logger.info(f"Updated liquidity/fdv for {len(update_records)}/{len(all_addresses)} tokens.")