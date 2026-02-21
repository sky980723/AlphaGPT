import aiohttp
import asyncio
from loguru import logger
from .base import DataProvider
from ..config import Config

class DexScreenerProvider(DataProvider):
    def __init__(self):
        self.base_url = "https://api.dexscreener.com/latest/dex"

    async def get_trending_tokens(self, limit=50):
        url = f"https://api.dexscreener.com/latest/dex/tokens/solana"
        return []

    async def get_token_details_batch(self, session, addresses):
        valid_data = []
        chunk_size = 10
        timeout = aiohttp.ClientTimeout(total=30)

        for i in range(0, len(addresses), chunk_size):
            chunk = addresses[i:i+chunk_size]
            addr_str = ",".join(chunk)
            url = f"{self.base_url}/tokens/{addr_str}"

            for attempt in range(3):
                try:
                    async with session.get(url, timeout=timeout) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            pairs = data.get('pairs', []) or []

                            best_pairs = {}
                            for p in pairs:
                                if p['chainId'] != Config.CHAIN: continue
                                base_addr = p['baseToken']['address']
                                liq = float(p.get('liquidity', {}).get('usd', 0))

                                if base_addr not in best_pairs or liq > best_pairs[base_addr]['liquidity']:
                                    best_pairs[base_addr] = {
                                        'address': base_addr,
                                        'symbol': p['baseToken']['symbol'],
                                        'name': p['baseToken']['name'],
                                        'liquidity': liq,
                                        'fdv': float(p.get('fdv', 0) or 0),
                                        'decimals': 6
                                    }
                            valid_data.extend(best_pairs.values())
                            break
                        else:
                            logger.warning(f"DexScreener {resp.status} for chunk {i}, attempt {attempt+1}")
                except Exception as e:
                    logger.warning(f"DexScreener attempt {attempt+1} failed: {e}")
                    if attempt < 2:
                        await asyncio.sleep(2)

        return valid_data

    async def get_token_history(self, session, address, days):
        return []