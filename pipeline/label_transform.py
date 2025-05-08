import torch

def one_hot(labels, num_classes=10):
    """Converts integer labels to one-hot encoded vectors.
    
    This is the standard way to transform class labels into vectors where only
    one element is 1 (hot) and the rest are 0.
    
    Example:
    Label 3 with num_classes=5 becomes:
    [0,0,0,1,0]  # Only index 3 is hot
    
    """
    return torch.nn.functional.one_hot(labels, num_classes).float()

def multi_hot(labels, num_classes=10):
    """Creates a 'soft' label encoding where each label activates two positions.
    
    This acts as a form of label smoothing regularization by making the target
    distribution less sparse. For each label, it activates both the label's position
    and the next adjacent class position (with wraparound).
    
    Example:
    Label 3 with num_classes=5 becomes:
    [0,0,0,1,1]  # Both position 3 and 4 are hot
    Label 4 with num_classes=5 becomes:
    [1,0,0,0,1]  # Both position 4 and 0 are hot (wraparound)
    
    """
    # Example: multi-hot by flipping one adjacent class to 1 as well
    onehot = torch.nn.functional.one_hot(labels, num_classes).float()
    shifted = torch.nn.functional.one_hot((labels + 1) % num_classes, num_classes).float()
    return torch.clamp(onehot + shifted, max=1)


def one_hot_self_concat(labels, num_classes=10):
    """Creates a concatenated version of one-hot encoding with itself.
    This doubles the dimension size while maintaining the same information,
    which can be interesting for studying representation redundancy.
    
    Example:
    Label 3 with num_classes=5 becomes:
    [0,0,0,1,0, 0,0,0,1,0]  # Same one-hot repeated twice
    """
    onehot = torch.nn.functional.one_hot(labels, num_classes).float()
    return torch.cat([onehot, onehot], dim=-1)  # Concatenate with itself

# Cache for random embeddings to ensure consistency
_RANDOM_EMBEDDINGS_CACHE = {}

def random_labels(labels, num_classes=10, output_dim=None):
    """Generates random label embeddings of specified dimension.
    If output_dim is None, defaults to num_classes.
    
    This can be used to study how different label space dimensionalities
    affect learning, even when the number of actual classes remains constant.
    
    Uses a cache to ensure the same label always maps to the same embedding.
    """
    # global _RANDOM_EMBEDDINGS_CACHE # Not strictly needed if only reading/modifying dict contents
    
    if output_dim is None:
        output_dim = num_classes
        
    # Create a unique key for this configuration
    cache_key = f"{num_classes}_{output_dim}"
    
    # Generate random vectors once and reuse them
    if cache_key not in _RANDOM_EMBEDDINGS_CACHE:
        # Set fixed seed for reproducibility
        torch.manual_seed(42)
        _RANDOM_EMBEDDINGS_CACHE[cache_key] = torch.randn(num_classes, output_dim)
        print(f"Created random embeddings cache for {num_classes} classes with dimension {output_dim}. Cache ID: {id(_RANDOM_EMBEDDINGS_CACHE)}")
    
    # Use these as a lookup table for the labels
    return _RANDOM_EMBEDDINGS_CACHE[cache_key][labels]