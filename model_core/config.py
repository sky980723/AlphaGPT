import torch
import os
from urllib.parse import quote_plus
from dotenv import load_dotenv

load_dotenv()

class ModelConfig:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    DB_URL = f"postgresql://{os.getenv('DB_USER','postgres')}:{quote_plus(os.getenv('DB_PASSWORD','password'))}@{os.getenv('DB_HOST','localhost')}:{os.getenv('DB_PORT','5432')}/{os.getenv('DB_NAME','crypto_quant')}"
    BATCH_SIZE = 512
    TRAIN_STEPS = 1000
    ENTROPY_COEFF_START = 0.05
    ENTROPY_COEFF_END = 0.01
    GRAD_CLIP_NORM = 1.0
    EARLY_STOP_PATIENCE = 100
    MAX_FORMULA_LEN = 8
    TRADE_SIZE_USD = 1000.0
    MIN_LIQUIDITY = 5000.0 # Treat lower liquidity as zero / non-tradable.
    BASE_FEE = 0.005 # Base fee rate 0.5% (Swap + Gas + Jito Tip).
    INPUT_DIM = 9
