
import torch

def track_dead_neurons(activation: torch.Tensor):
    # Handles both conv (N, C, H, W) and linear (N, F)
    with torch.no_grad():
        if activation.ndim == 4:
            # Conv: mean over H and W
            act_per_channel = activation.mean(dim=(0, 2, 3))
        elif activation.ndim == 2:
            # Linear: mean over batch
            act_per_channel = activation.mean(dim=0)
        else:
            raise ValueError("Unsupported activation shape")

        dead = (act_per_channel == 0).sum().item()
        total = act_per_channel.numel()
    return dead, total