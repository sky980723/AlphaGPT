import os
from dotenv import load_dotenv
import base58

load_dotenv()

class ExecutionConfig:
    RPC_URL = os.getenv("QUICKNODE_RPC_URL", "")
    _PRIV_KEY_STR = os.getenv("SOLANA_PRIVATE_KEY", "")

    PAYER_KEYPAIR = None
    WALLET_ADDRESS = ""

    if _PRIV_KEY_STR:
        try:
            from solders.keypair import Keypair
            try:
                PAYER_KEYPAIR = Keypair.from_base58_string(_PRIV_KEY_STR)
            except Exception:
                import json
                PAYER_KEYPAIR = Keypair.from_bytes(json.loads(_PRIV_KEY_STR))
            WALLET_ADDRESS = str(PAYER_KEYPAIR.pubkey())
        except Exception as e:
            import warnings
            warnings.warn(f"Failed to load Solana keypair: {e}")

    DEFAULT_SLIPPAGE_BPS = 200  # bps

    PRIORITY_LEVEL = "High"

    SOL_MINT = "So11111111111111111111111111111111111111112"
    USDC_MINT = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"
