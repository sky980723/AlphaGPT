import torch
from .ops import OPS_CONFIG
from .factors import FeatureEngineer

class StackVM:
    def __init__(self):
        self.feat_offset = FeatureEngineer.INPUT_DIM
        self.op_map = {i + self.feat_offset: cfg[1] for i, cfg in enumerate(OPS_CONFIG)}
        self.arity_map = {i + self.feat_offset: cfg[2] for i, cfg in enumerate(OPS_CONFIG)}
        self.vocab_size = self.feat_offset + len(OPS_CONFIG)
        self._arity = torch.zeros(self.vocab_size, dtype=torch.long)
        self._delta = torch.zeros(self.vocab_size, dtype=torch.long)
        for i in range(self.feat_offset):
            self._arity[i] = 0
            self._delta[i] = 1
        for i, cfg in enumerate(OPS_CONFIG):
            idx = i + self.feat_offset
            self._arity[idx] = cfg[2]
            self._delta[idx] = 1 - cfg[2]
        self._device_cached = None

    def _ensure_device(self, device):
        if self._device_cached != device:
            self._arity_d = self._arity.to(device)
            self._delta_d = self._delta.to(device)
            self._device_cached = device

    def compute_syntax_mask(self, stack_sizes, step, max_len):
        """
        Compute valid token mask based on stack state.
        
        Args:
            stack_sizes: [B] current stack size per batch item
            step: current step (0-indexed)
            max_len: total formula length (8)
        Returns:
            mask: [B, vocab_size] bool tensor, True = allowed
        """
        device = stack_sizes.device
        self._ensure_device(device)

        remaining = max_len - step

        S = stack_sizes.unsqueeze(1)
        a = self._arity_d.unsqueeze(0)
        d = self._delta_d.unsqueeze(0)
        new_S = S + d

        c1 = (a <= S)

        if remaining == 1:
            mask = c1 & (new_S == 1)
        else:
            c2 = (new_S >= 2 - (remaining - 1))
            c3 = (new_S <= 2 * (remaining - 1) + 1)
            mask = c1 & c2 & c3

        empty_rows = ~mask.any(dim=1)
        if empty_rows.any():
            mask[empty_rows] = True

        return mask

    def execute(self, formula_tokens, feat_tensor):
        stack = []
        try:
            for token in formula_tokens:
                token = int(token)
                if token < self.feat_offset:
                    stack.append(feat_tensor[:, token, :])
                elif token in self.op_map:
                    arity = self.arity_map[token]
                    if len(stack) < arity: return None
                    args = []
                    for _ in range(arity):
                        args.append(stack.pop())
                    args.reverse()
                    func = self.op_map[token]
                    res = func(*args)
                    if torch.isnan(res).any() or torch.isinf(res).any():
                        res = torch.nan_to_num(res, nan=0.0, posinf=1.0, neginf=-1.0)
                    stack.append(res)
                else:
                    return None
            if len(stack) == 1:
                return stack[0]
            else:
                return None
        except Exception:
            return None
