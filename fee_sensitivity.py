"""BASE_FEE sensitivity analysis for best formula."""
import json
import torch
from model_core.config import ModelConfig
from model_core.data_loader import CryptoDataLoader
from model_core.vm import StackVM
from model_core.backtest import MemeBacktest

def main():
    loader = CryptoDataLoader()
    loader.load_data()
    vm = StackVM()

    with open("best_meme_strategy.json") as f:
        formula = json.load(f)

    # Decode formula for display
    feat_names = ['RET', 'LIQ_HEALTH', 'PRESSURE', 'PUMP_DEV',
                  'VOL_CLUSTER', 'MOM_REV', 'REL_STR', 'HL_RANGE', 'CLOSE_POS']
    from model_core.ops import OPS_CONFIG
    ops_names = [c[0] for c in OPS_CONFIG]
    decoded = [feat_names[t] if t < 9 else ops_names[t - 9] for t in formula]
    print(f"Formula: {formula}")
    print(f"Decoded: {decoded}\n")

    res = vm.execute(formula, loader.feat_tensor)
    if res is None:
        print("ERROR: formula invalid")
        return

    fee_levels = [0.000, 0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.008, 0.010]

    print(f"{'BASE_FEE':>10} {'Score':>10} {'Avg CumRet':>12} {'Profitable?':>12}")
    print("-" * 48)

    for fee in fee_levels:
        bt = MemeBacktest()
        bt.base_fee = fee
        score, avg_ret = bt.evaluate(res, loader.raw_data_cache, loader.target_ret)
        tag = "YES" if score.item() > 0 else "no"
        print(f"{fee:>10.3%} {score.item():>10.4f} {avg_ret:>12.2%} {tag:>12}")

    # Find breakeven fee
    lo, hi = 0.0, 0.02
    for _ in range(50):
        mid = (lo + hi) / 2
        bt = MemeBacktest()
        bt.base_fee = mid
        s, _ = bt.evaluate(res, loader.raw_data_cache, loader.target_ret)
        if s.item() > 0:
            lo = mid
        else:
            hi = mid
    breakeven = (lo + hi) / 2
    print(f"\nBreakeven BASE_FEE: {breakeven:.4%}")
    print(f"Current BASE_FEE:   {ModelConfig.BASE_FEE:.4%}")
    print(f"Margin to breakeven: {(ModelConfig.BASE_FEE - breakeven):.4%}")

if __name__ == "__main__":
    main()
