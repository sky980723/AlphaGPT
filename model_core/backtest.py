import torch
from .config import ModelConfig

class MemeBacktest:
    def __init__(self):
        self.trade_size = ModelConfig.TRADE_SIZE_USD
        self.min_liq = ModelConfig.MIN_LIQUIDITY
        self.base_fee = ModelConfig.BASE_FEE

    def evaluate(self, factors, raw_data, target_ret):
        liquidity = raw_data['liquidity']
        signal = torch.sigmoid(factors)
        has_liquidity_data = liquidity.sum() > 0
        if has_liquidity_data:
            is_safe = (liquidity > self.min_liq).float()
            impact_slippage = self.trade_size / (liquidity + 1e-9)
            impact_slippage = torch.clamp(impact_slippage, 0.0, 0.05)
        else:
            is_safe = torch.ones_like(signal)
            impact_slippage = torch.zeros_like(signal)
        position = (signal > 0.85).float() * is_safe
        total_slippage_one_way = self.base_fee + impact_slippage
        prev_pos = torch.roll(position, 1, dims=1)
        prev_pos[:, 0] = 0
        turnover = torch.abs(position - prev_pos)
        tx_cost = turnover * total_slippage_one_way
        gross_pnl = position * target_ret
        net_pnl = gross_pnl - tx_cost
        cum_ret = net_pnl.sum(dim=1)
        big_drawdowns = (net_pnl < -0.05).float().sum(dim=1)
        score = cum_ret - (big_drawdowns * 2.0)
        activity = position.sum(dim=1)
        score = torch.where(activity < 5, torch.tensor(-10.0, device=score.device), score)
        final_fitness = torch.median(score)
        return final_fitness, cum_ret.mean().item()