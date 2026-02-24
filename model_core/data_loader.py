import pandas as pd
import torch
import sqlalchemy
from .config import ModelConfig
from .factors import FeatureEngineer

RESAMPLE_MINUTES = 15  # 1m K线降采样到 15m，减少噪声提高覆盖率

class CryptoDataLoader:
    def __init__(self):
        self.engine = sqlalchemy.create_engine(ModelConfig.DB_URL)
        self.feat_tensor = None
        self.raw_data_cache = None
        self.target_ret = None
        self.feat_tensor_train = None
        self.feat_tensor_test = None
        self.target_ret_train = None
        self.target_ret_test = None
        self.raw_data_train = None
        self.raw_data_test = None

    def load_data(self, limit_tokens=500, min_data_ratio=0.05):
        print("Loading data from SQL...")
        top_query = f"""
        SELECT address FROM tokens
        LIMIT {limit_tokens}
        """
        addrs = pd.read_sql(top_query, self.engine)['address'].tolist()
        if not addrs: raise ValueError("No tokens found.")
        addr_str = "'" + "','".join(addrs) + "'"
        data_query = f"""
        SELECT time, address, open, high, low, close, volume, liquidity, fdv
        FROM ohlcv
        WHERE address IN ({addr_str})
        ORDER BY time ASC
        """
        df = pd.read_sql(data_query, self.engine)
        df['time'] = pd.to_datetime(df['time'])

        # 降采样 1m → 15m（正确的 OHLCV 聚合方式）
        if RESAMPLE_MINUTES > 1:
            print(f"Resampling {len(df)} rows from 1m to {RESAMPLE_MINUTES}m...")
            df = df.set_index('time').groupby('address').resample(f'{RESAMPLE_MINUTES}min').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum',
                'liquidity': 'last',
                'fdv': 'last',
            }).dropna(subset=['close']).reset_index()
            print(f"After resample: {len(df)} rows")

        # pivot close 用于计算数据覆盖率
        pivot_close = df.pivot(index='time', columns='address', values='close')
        actual_addrs = pivot_close.columns.tolist()

        # 过滤数据量不足的 token
        valid_addrs = []
        for addr in actual_addrs:
            non_null_ratio = pivot_close[addr].notna().mean()
            if non_null_ratio >= min_data_ratio:
                valid_addrs.append(addr)

        dropped = len(actual_addrs) - len(valid_addrs)
        if dropped > 0:
            print(f"Dropped {dropped} tokens with <{min_data_ratio*100:.0f}% data coverage.")
            df = df[df['address'].isin(valid_addrs)]
            actual_addrs = valid_addrs

        if not actual_addrs:
            raise ValueError("No tokens with sufficient data.")

        def to_tensor(col):
            pivot = df.pivot(index='time', columns='address', values=col)
            pivot = pivot.ffill().bfill().fillna(0.0)
            return torch.tensor(pivot.values.T, dtype=torch.float32, device=ModelConfig.DEVICE)

        self.raw_data_cache = {
            'open': to_tensor('open'),
            'high': to_tensor('high'),
            'low': to_tensor('low'),
            'close': to_tensor('close'),
            'volume': to_tensor('volume'),
            'liquidity': to_tensor('liquidity'),
            'fdv': to_tensor('fdv')
        }
        # 如果 ohlcv 中 liquidity 全为 0，从 tokens 表读取快照值回填
        if self.raw_data_cache['liquidity'].sum() == 0:
            actual_addr_str = "'" + "','".join(actual_addrs) + "'"
            liq_query = f"SELECT address, liquidity, fdv FROM tokens WHERE address IN ({actual_addr_str})"
            liq_df = pd.read_sql(liq_query, self.engine)
            if not liq_df.empty:
                for idx, addr in enumerate(actual_addrs):
                    row = liq_df[liq_df['address'] == addr]
                    if not row.empty and row.iloc[0]['liquidity'] > 0:
                        self.raw_data_cache['liquidity'][idx, :] = row.iloc[0]['liquidity']
                        self.raw_data_cache['fdv'][idx, :] = row.iloc[0]['fdv']
                print(f"Backfilled liquidity/fdv from tokens table for {len(liq_df)} tokens.")
        self.feat_tensor = FeatureEngineer.compute_features(self.raw_data_cache)
        # NaN 兜底
        self.feat_tensor = torch.nan_to_num(self.feat_tensor, nan=0.0, posinf=0.0, neginf=0.0)
        op = self.raw_data_cache['open']
        t1 = torch.roll(op, -1, dims=1)
        t2 = torch.roll(op, -2, dims=1)
        self.target_ret = torch.log(t2 / (t1 + 1e-9))
        self.target_ret = torch.nan_to_num(self.target_ret, nan=0.0, posinf=0.0, neginf=0.0)
        self.target_ret[:, -2:] = 0.0
        # Full data for training (no split — direction B)
        self.feat_tensor_train = self.feat_tensor
        self.feat_tensor_test = self.feat_tensor
        self.target_ret_train = self.target_ret
        self.target_ret_test = self.target_ret
        self.raw_data_train = self.raw_data_cache
        self.raw_data_test = self.raw_data_cache
        print(f"Full data mode: {self.feat_tensor.shape[2]} time steps (no train/test split)")
        nan_count = torch.isnan(self.feat_tensor).sum().item()
        print(f"Data Ready. Shape: {self.feat_tensor.shape} ({len(actual_addrs)} tokens, NaN={nan_count})")
