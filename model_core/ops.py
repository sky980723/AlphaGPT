import torch

@torch.jit.script
def _ts_delay(x: torch.Tensor, d: int) -> torch.Tensor:
    if d == 0: return x
    pad = torch.zeros((x.shape[0], d), device=x.device)
    return torch.cat([pad, x[:, :-d]], dim=1)

@torch.jit.script
def _ts_mean(x: torch.Tensor, w: int) -> torch.Tensor:
    pad = torch.zeros((x.shape[0], w - 1), device=x.device)
    xp = torch.cat([pad, x], dim=1)
    return xp.unfold(1, w, 1).mean(dim=-1)

@torch.jit.script
def _ts_std(x: torch.Tensor, w: int) -> torch.Tensor:
    pad = torch.zeros((x.shape[0], w - 1), device=x.device)
    xp = torch.cat([pad, x], dim=1)
    return xp.unfold(1, w, 1).std(dim=-1) + 1e-6

@torch.jit.script
def _ts_min(x: torch.Tensor, w: int) -> torch.Tensor:
    pad = torch.full((x.shape[0], w - 1), 1e9, device=x.device)
    xp = torch.cat([pad, x], dim=1)
    return xp.unfold(1, w, 1).min(dim=-1)[0]

@torch.jit.script
def _ts_max(x: torch.Tensor, w: int) -> torch.Tensor:
    pad = torch.full((x.shape[0], w - 1), -1e9, device=x.device)
    xp = torch.cat([pad, x], dim=1)
    return xp.unfold(1, w, 1).max(dim=-1)[0]

@torch.jit.script
def _op_gate(condition: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    mask = (condition > 0).float()
    return mask * x + (1.0 - mask) * y

@torch.jit.script
def _op_jump(x: torch.Tensor) -> torch.Tensor:
    mean = x.mean(dim=1, keepdim=True)
    std = x.std(dim=1, keepdim=True) + 1e-6
    z = (x - mean) / std
    return torch.relu(z - 3.0)

@torch.jit.script
def _op_decay(x: torch.Tensor) -> torch.Tensor:
    return x + 0.8 * _ts_delay(x, 1) + 0.6 * _ts_delay(x, 2)

OPS_CONFIG = [
    ('ADD', lambda x, y: x + y, 2),
    ('SUB', lambda x, y: x - y, 2),
    ('MUL', lambda x, y: x * y, 2),
    ('DIV', lambda x, y: x / (y + 1e-6), 2),
    ('NEG', lambda x: -x, 1),
    ('ABS', torch.abs, 1),
    ('SIGN', torch.sign, 1),
    ('GATE', _op_gate, 3),
    ('JUMP', _op_jump, 1),
    ('DECAY', _op_decay, 1),
    ('DELAY1', lambda x: _ts_delay(x, 1), 1),
    ('MAX3', lambda x: torch.max(x, torch.max(_ts_delay(x,1), _ts_delay(x,2))), 1),
    ('TS_MEAN5', lambda x: _ts_mean(x, 5), 1),
    ('TS_STD5', lambda x: _ts_std(x, 5), 1),
    ('TS_MIN5', lambda x: _ts_min(x, 5), 1),
    ('TS_MAX5', lambda x: _ts_max(x, 5), 1),
    ('RANK', lambda x: x.argsort(dim=1).argsort(dim=1).float() / (x.shape[1] - 1 + 1e-9), 1),
    ('DELAY3', lambda x: _ts_delay(x, 3), 1),
    ('CLAMP', lambda x: torch.clamp(x, -1.0, 1.0), 1),
    ('SQ', lambda x: x * x, 1),
]
