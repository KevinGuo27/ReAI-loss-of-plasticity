import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List
import torch
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
from analysis.dead_relu_tracker import track_dead_neurons
from analysis.effective_rank import compute_effective_rank
from analysis.activations import register_layer_hooks
from pipeline.dataset import LabelTransformedDataset
import os

class ExperimentTracker:
    # Static dictionary to store metrics across all label modes
    all_metrics = {}
    
    def __init__(self, eval_samples=2000, batch_size=90, label_mode='raw'):
        self.eval_samples = eval_samples
        self.batch_size = batch_size
        self.label_mode = label_mode
        
        # Define the 5 ReLU layers to track
        self.tracked_layers = [
            'resnet.layer1.1.relu',  # Second ReLU in first residual block
            'resnet.layer2.1.relu',  # Second ReLU in second residual block
            'resnet.layer3.1.relu',  # Second ReLU in third residual block
            'resnet.layer4.0.relu',  # First ReLU in fourth residual block
            'resnet.layer4.1.relu'   # Second ReLU in fourth block
        ]
        
        # Initialize metrics dictionary
        self.metrics = {
            'incremental': {
                'accuracy': [],
                'dead_neurons': {layer: [] for layer in self.tracked_layers},
                'effective_rank': {layer: [] for layer in self.tracked_layers}
            }
        }
        self.num_classes_history = []
        self.activation_hooks = {}
        
        # Add this label mode's metrics to the static dictionary
        ExperimentTracker.all_metrics[label_mode] = self.metrics
        
        # Setup evaluation data
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5070751592371323, 0.48654887331495095, 0.4409178433670343],
                std=[0.2673342858792401, 0.2564384629170883, 0.25742285334015846]
            )
        ])
        
        self.base_dataset = torchvision.datasets.CIFAR100(
            root='./data', train=True, download=True, transform=transform)
        self.eval_dataset = LabelTransformedDataset(self.base_dataset, label_mode)

    def setup_hooks(self, model):
        """Register hooks for specific layers of ResNet"""
        self.activation_hooks = register_layer_hooks(model, self.tracked_layers)

    def get_evaluation_loader(self, num_classes):
        """Create a DataLoader with exactly eval_samples examples balanced across classes"""
        samples_per_class = self.eval_samples // num_classes
        indices = []
        
        # Get indices for each class
        for class_idx in range(num_classes):
            class_indices = [i for i in range(len(self.base_dataset)) 
                           if self.base_dataset[i][1] == class_idx]
            indices.extend(class_indices[:samples_per_class])
        
        print(f"Number of samples: {len(indices)}")
        # If we don't have enough samples, raise an error
        # if len(indices) < self.eval_samples:
        #     raise ValueError(f"Not enough samples to evaluate. Need {self.eval_samples}, got {len(indices)}")
        
        # Create subset and dataloader with reduced workers
        eval_subset = Subset(self.eval_dataset, indices[:self.eval_samples])
        return DataLoader(
            eval_subset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=0,  # Run in main process to avoid memory issues
            pin_memory=True  # More efficient GPU transfer
        )

    def compute_metrics(self, model, num_classes) -> Dict:
        """Compute metrics for the current model state"""
        metrics = {}
        device = next(model.parameters()).device
        
        # Get evaluation loader
        eval_loader = self.get_evaluation_loader(num_classes)
        
        # Reset activation storage for each layer
        layer_activations = {layer: None for layer in self.tracked_layers}
        
        # Collect activations from a single batch
        model.eval()
        with torch.no_grad():
            # Just use the first batch
            inputs, _, _ = next(iter(eval_loader))
            inputs = inputs.to(device)
            _ = model(inputs)  # Forward pass to trigger hooks
            
            # Store activations from this batch
            for layer_name in self.tracked_layers:
                layer_activations[layer_name] = self.activation_hooks[layer_name]
        
        # Compute metrics for each tracked layer
        for layer_name in self.tracked_layers:
            # Compute metrics on current batch activations
            dead_count, total = track_dead_neurons(layer_activations[layer_name])
            eff_rank = compute_effective_rank(layer_activations[layer_name])
            
            metrics[f'dead_neurons_{layer_name}'] = dead_count / total
            metrics[f'effective_rank_{layer_name}'] = eff_rank
        
        return metrics

    def update(self, phase: str, metrics_dict: Dict, num_classes: int):
        """Update metrics for a given phase"""
        # Update accuracy
        if 'accuracy' in metrics_dict:
            self.metrics[phase]['accuracy'].append(metrics_dict['accuracy'])
        
        # Update layer-specific metrics
        for layer_name in self.tracked_layers:
            dead_key = f'dead_neurons_{layer_name}'
            rank_key = f'effective_rank_{layer_name}'
            
            if dead_key in metrics_dict:
                self.metrics[phase]['dead_neurons'][layer_name].append(metrics_dict[dead_key])
            if rank_key in metrics_dict:
                self.metrics[phase]['effective_rank'][layer_name].append(metrics_dict[rank_key])
        
        if (not self.num_classes_history or self.num_classes_history[-1] != num_classes):
            self.num_classes_history.append(num_classes)

    def plot_results(self, save_dir: str = 'results'):
        """Generate comparative plots for all label modes"""
        os.makedirs(save_dir, exist_ok=True)
        
        plt.style.use('seaborn-v0_8-darkgrid')
        colors = ['b', 'g', 'r', 'c', 'm']  # Colors for different label modes
        
        # Plot accuracy comparison
        plt.figure(figsize=(12, 8))
        for idx, (mode, metrics) in enumerate(ExperimentTracker.all_metrics.items()):
            plt.plot(self.num_classes_history, 
                    metrics['incremental']['accuracy'],
                    marker='o', 
                    color=colors[idx % len(colors)],
                    label=mode.replace('_', ' ').title(),
                    linewidth=2)
        plt.xlabel('Number of Classes')
        plt.ylabel('Accuracy (%)')
        plt.title('Accuracy Comparison Across Label Representations')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'{save_dir}/accuracy_comparison.png')
        plt.close()

        # Plot dead neurons comparison for each layer
        for layer_name in self.tracked_layers:
            plt.figure(figsize=(12, 8))
            for idx, (mode, metrics) in enumerate(ExperimentTracker.all_metrics.items()):
                plt.plot(self.num_classes_history,
                        metrics['incremental']['dead_neurons'][layer_name],
                        marker='o',
                        color=colors[idx % len(colors)],
                        label=mode.replace('_', ' ').title(),
                        linewidth=2)
            plt.xlabel('Number of Classes')
            plt.ylabel('Dead Neurons Ratio')
            plt.title(f'Dead Neurons Comparison\n{layer_name}')
            plt.legend()
            plt.grid(True)
            plt.savefig(f'{save_dir}/dead_neurons_comparison_{layer_name}.png')
            plt.close()

        # Plot effective rank comparison for each layer
        for layer_name in self.tracked_layers:
            plt.figure(figsize=(12, 8))
            for idx, (mode, metrics) in enumerate(ExperimentTracker.all_metrics.items()):
                plt.plot(self.num_classes_history,
                        metrics['incremental']['effective_rank'][layer_name],
                        marker='o',
                        color=colors[idx % len(colors)],
                        label=mode.replace('_', ' ').title(),
                        linewidth=2)
            plt.xlabel('Number of Classes')
            plt.ylabel('Effective Rank')
            plt.title(f'Effective Rank Comparison\n{layer_name}')
            plt.legend()
            plt.grid(True)
            plt.savefig(f'{save_dir}/effective_rank_comparison_{layer_name}.png')
            plt.close()

        # Save individual mode results in subdirectories
        mode_dir = f'{save_dir}/{self.label_mode}'
        os.makedirs(mode_dir, exist_ok=True)
        
        # Plot individual mode metrics
        plt.figure(figsize=(10, 6))
        plt.plot(self.num_classes_history, 
                self.metrics['incremental']['accuracy'],
                marker='o')
        plt.xlabel('Number of Classes')
        plt.ylabel('Accuracy (%)')
        plt.title(f'Incremental Learning Accuracy\n{self.label_mode.upper()} Labels')
        plt.grid(True)
        plt.savefig(f'{mode_dir}/accuracy.png')
        plt.close()

        for layer_name in self.tracked_layers:
            # Dead neurons for this mode
            plt.figure(figsize=(10, 6))
            plt.plot(self.num_classes_history,
                    self.metrics['incremental']['dead_neurons'][layer_name],
                    marker='o')
            plt.xlabel('Number of Classes')
            plt.ylabel('Dead Neurons Ratio')
            plt.title(f'Dead Neurons vs Number of Classes\n{layer_name} ({self.label_mode.upper()})')
            plt.grid(True)
            plt.savefig(f'{mode_dir}/dead_neurons_{layer_name}.png')
            plt.close()

            # Effective rank for this mode
            plt.figure(figsize=(10, 6))
            plt.plot(self.num_classes_history,
                    self.metrics['incremental']['effective_rank'][layer_name],
                    marker='o')
            plt.xlabel('Number of Classes')
            plt.ylabel('Effective Rank')
            plt.title(f'Effective Rank vs Number of Classes\n{layer_name} ({self.label_mode.upper()})')
            plt.grid(True)
            plt.savefig(f'{mode_dir}/effective_rank_{layer_name}.png')
            plt.close() 