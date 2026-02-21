import aiohttp
import asyncio
from datetime import datetime, timedelta
from loguru import logger
from ..config import Config
from .base import DataProvider

class BirdeyeProvider(DataProvider):
    def __init__(self):
        self.base_url = "https://public-api.birdeye.so"
        self.headers = {
            "X-API-KEY": Config.BIRDEYE_API_KEY,
            "accept": "application/json",
            "x-chain": Config.CHAIN
        }
        self.semaphore = asyncio.Semaphore(Config.CONCURRENCY)
        
    async def get_trending_tokens(self, limit=100):
        url = f"{self.base_url}/defi/token_trending"
        params = {
            "sort_by": "rank",
            "sort_type": "asc",
            "offset": "0",
            "limit": str(limit)
        }

        async with aiohttp.ClientSession(headers=self.headers) as session:
            try:
                async with session.get(url, params=params) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        raw_list = data.get('data', {}).get('tokens', [])

                        results = []
                        for t in raw_list:
                            results.append({
                                'address': t['address'],
                                'symbol': t.get('symbol', 'UNKNOWN'),
                                'name': t.get('name', 'UNKNOWN'),
                                'decimals': t.get('decimals', 6),
                                'liquidity': t.get('liquidity', 0),
                                'fdv': t.get('fdv', 0)
                            })
                        return results
                    else:
                        logger.error(f"Birdeye Trending Error: {resp.status}")
                        return []
            except Exception as e:
                logger.error(f"Birdeye Trending Exception: {e}")
                return []

    async def get_top_traders(self, limit=20):
        """按 24h 成交量排序发现高活跃 token"""
        url = f"{self.base_url}/defi/token_trending"
        params = {
            "sort_by": "volume24hUSD",
            "sort_type": "desc",
            "offset": "0",
            "limit": str(limit)
        }
        async with aiohttp.ClientSession(headers=self.headers) as session:
            try:
                async with session.get(url, params=params) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        raw_list = data.get('data', {}).get('tokens', [])
                        return [
                            {
                                'address': t['address'],
                                'symbol': t.get('symbol', 'UNKNOWN'),
                                'name': t.get('name', 'UNKNOWN'),
                                'decimals': t.get('decimals', 6),
                                'liquidity': t.get('liquidity', 0),
                                'fdv': t.get('fdv', 0)
                            }
                            for t in raw_list
                        ]
                    else:
                        logger.warning(f"Birdeye top_traders: HTTP {resp.status}")
                        return []
            except Exception as e:
                logger.warning(f"Birdeye top_traders exception: {e}")
                return []

    async def get_token_history(self, session, address, days=Config.HISTORY_DAYS, _retries=0):
        time_to = int(datetime.now().timestamp())
        time_from = int((datetime.now() - timedelta(days=days)).timestamp())

        url = f"{self.base_url}/defi/ohlcv"
        params = {
            "address": address,
            "type": Config.TIMEFRAME,
            "time_from": time_from,
            "time_to": time_to
        }

        async with self.semaphore:
            try:
                async with session.get(url, params=params) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        items = data.get('data', {}).get('items', [])
                        if not items: return []

                        formatted = []
                        for item in items:
                            formatted.append((
                                datetime.fromtimestamp(item['unixTime']), # time
                                address,                                  # address
                                float(item['o']),                         # open
                                float(item['h']),                         # high
                                float(item['l']),                         # low
                                float(item['c']),                         # close
                                float(item['v']),                         # volume
                                0.0,                                      # liquidity
                                0.0,                                      # fdv
                                'birdeye'                                 # source
                            ))
                        return formatted
                    elif resp.status == 429 and _retries < 3:
                        logger.warning(f"Birdeye 429 for {address}, retry {_retries+1}/3...")
                        await asyncio.sleep(3)
                        return await self.get_token_history(session, address, days, _retries + 1)
                    else:
                        if resp.status != 200:
                            logger.warning(f"Birdeye OHLCV {address}: HTTP {resp.status}")
                        return []
            except Exception as e:
                logger.error(f"Birdeye Fetch Error {address}: {e}")
                return []

    async def get_token_overview(self, session, address, _retries=0):
        """查询单 token 的 liquidity/fdv，最多重试 3 次"""
        url = f"{self.base_url}/defi/token_overview"
        params = {"address": address}
        async with self.semaphore:
            try:
                async with session.get(url, params=params) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        d = data.get('data', {})
                        return {
                            'address': address,
                            'liquidity': float(d.get('liquidity', 0) or 0),
                            'fdv': float(d.get('fdv', 0) or 0),
                        }
                    elif resp.status == 429 and _retries < 3:
                        await asyncio.sleep(3)
                        return await self.get_token_overview(session, address, _retries + 1)
                    else:
                        logger.warning(f"Birdeye overview {address}: HTTP {resp.status}")
                        return None
            except Exception as e:
                logger.warning(f"Birdeye overview error {address}: {e}")
                return None