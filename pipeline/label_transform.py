import torch

def one_hot(labels, num_classes=10):
    return torch.nn.functional.one_hot(labels, num_classes).float()

def multi_hot(labels, num_classes=10):
    # Example: multi-hot by flipping one adjacent class to 1 as well
    onehot = torch.nn.functional.one_hot(labels, num_classes).float()
    shifted = torch.nn.functional.one_hot((labels + 1) % num_classes, num_classes).float()
    return torch.clamp(onehot + shifted, max=1)

def continuous_embedding(labels, num_classes=10):
    return torch.randn(labels.size(0), 5)  # dummy: fixed-size vector