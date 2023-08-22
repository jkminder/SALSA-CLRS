import torch
from collections import defaultdict

def stack_dicts(dicts):
    out = defaultdict(list)
    for d in dicts:
        for k, v in d.items():
            if isinstance(v, torch.Tensor):
                v = v.detach().cpu()
            else:
                v = torch.tensor(v)
            if v.dim() == 0:
                v = v.unsqueeze(0)
            out[k].append(v)
    return {k: torch.cat(v) for k, v in out.items()}

def stack_hidden(input_hidden, hidden, last_hidden, use_last_hidden):
    if use_last_hidden:
        return torch.cat([input_hidden, hidden, last_hidden], dim=-1)
    else:
        return torch.cat([input_hidden, hidden], dim=-1)


class NaNException(Exception):
    pass