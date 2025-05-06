import torch
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
from typing import Dict, List
from analysis.dead_relu_tracker import count_dead_neurons
from analysis.effective_rank import compute_effective_rank

class ActivationTracker:
    def __init__(self, model, layer_names: List[str]):
        """
        Initialize activation tracker for specified layers.
        
        Args:
            model: PyTorch model
            layer_names: List of layer names to track
        """
        self.layer_names = layer_names
        self.activations = {name: [] for name in layer_names}
        self.hooks = []
        
        # Register hooks
        for name, module in model.named_modules():
            if name in layer_names:
                hook = module.register_forward_hook(self._create_hook(name))
                self.hooks.append(hook)
    
    def _create_hook(self, name):
        def hook(module, input, output):
            self.activations[name].append(output.detach().cpu())
        return hook
    
    def clear_activations(self):
        """Clear stored activations."""
        for name in self.layer_names:
            self.activations[name] = []
    
    def get_concatenated_activations(self):
        """Get concatenated activations for each layer."""
        return {name: torch.cat(acts, dim=0) 
                for name, acts in self.activations.items()}
    
    def remove_hooks(self):
        """Remove all hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

def evaluate_model(model, 
                  dataset,
                  num_samples: int = 2000,
                  batch_size: int = 128,
                  tracked_layers: List[str] = None):
    """
    Evaluate model's layer behavior on a fixed dataset.
    
    Args:
        model: PyTorch model
        dataset: Dataset to evaluate on
        num_samples: Number of samples to evaluate
        batch_size: Batch size for evaluation
        tracked_layers: List of layer names to track
    """
    if tracked_layers is None:
        tracked_layers = [
            'layer1.1.relu',  # Second ReLU in first residual block
            'layer2.1.relu',  # Second ReLU in second residual block
            'layer3.1.relu',  # Second ReLU in third residual block
            'layer4.0.relu',  # First ReLU in fourth residual block
            'layer4.1.relu'   # Second ReLU in fourth block
        ]
    
    # Create subset of dataset
    indices = torch.randperm(len(dataset))[:num_samples]
    eval_dataset = Subset(dataset, indices)
    eval_loader = DataLoader(eval_dataset, 
                           batch_size=batch_size,
                           shuffle=False,
                           num_workers=2)
    
    # Setup activation tracking
    tracker = ActivationTracker(model, tracked_layers)
    device = next(model.parameters()).device
    
    # Collect activations
    model.eval()
    with torch.no_grad():
        for inputs, _ in eval_loader:
            inputs = inputs.to(device)
            _ = model(inputs)
    
    # Get concatenated activations
    all_activations = tracker.get_concatenated_activations()
    
    # Compute metrics
    results = {}
    for layer_name, activations in all_activations.items():
        dead_percent = count_dead_neurons(activations)
        eff_rank = compute_effective_rank(activations)
        
        results[layer_name] = {
            'dead_neurons_percent': dead_percent,
            'effective_rank': eff_rank
        }
        
        print(f"\nLayer: {layer_name}")
        print(f"Dead Neurons: {dead_percent:.2f}%")
        print(f"Effective Rank: {eff_rank:.2f}")
    
    # Cleanup
    tracker.remove_hooks()
    tracker.clear_activations()
    
    return results 