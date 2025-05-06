import torch

def compute_effective_rank(activations: torch.Tensor) -> float:
    """
    Compute effective rank using entropy of normalized singular values.
    
    Args:
        activations: Tensor of shape (N, C, H, W) for conv or (N, F) for linear
        
    Returns:
        float: Effective rank
    """
    # Reshape activations to 2D matrix
    if activations.ndim == 4:
        # For conv layers: (N, C, H, W) -> (N, C*H*W)
        batch_size = activations.size(0)
        flattened = activations.reshape(batch_size, -1)
    else:
        # For linear layers: already in (N, F) format
        flattened = activations
    
    # Compute SVD
    try:
        _, s, _ = torch.svd(flattened)
        # Normalize singular values
        s_norm = s / s.sum()
        # Remove zeros to avoid log(0)
        s_norm = s_norm[s_norm > 1e-12]
        # Compute entropy
        entropy = -(s_norm * torch.log(s_norm)).sum()
        # Convert entropy to effective rank
        eff_rank = torch.exp(entropy).item()
    except:
        # If SVD fails, return 1.0 as default
        eff_rank = 1.0
    
    return eff_rank