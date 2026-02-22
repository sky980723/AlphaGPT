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
        activity = position.sum(dim=1)

        # Sharpe-like fitness: continuous, differentiable, no arbitrary thresholds
        mean_ret = net_pnl.mean(dim=1)
        std_ret = net_pnl.std(dim=1) + 1e-6
        sharpe = mean_ret / std_ret

        # Penalize inactive strategies
        sharpe = torch.where(activity < 5, torch.tensor(-10.0, device=sharpe.device), sharpe)

        final_fitness = torch.median(sharpe)
        return final_fitness, cum_ret.mean().item()