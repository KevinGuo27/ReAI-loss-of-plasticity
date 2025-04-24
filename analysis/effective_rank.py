
import torch

def compute_effective_rank(activation: torch.Tensor, eps=1e-10):
    if activation.ndim == 4:
        activation = activation.flatten(start_dim=1)  # (N, C*H*W)
    u, s, v = torch.svd(activation)
    s = s[s > eps]
    p = s / s.sum()
    entropy = -(p * torch.log(p)).sum()
    return torch.exp(entropy).item()