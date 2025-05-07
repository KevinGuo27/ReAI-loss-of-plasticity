import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from pipeline.resnet import IncrementalResNet18
from pipeline.experiment_tracker import ExperimentTracker
from pipeline.dataset import LabelTransformedDataset
import os
import numpy as np

def get_transforms(train=True):
    normalize = transforms.Normalize(
        mean=[0.5070751592371323, 0.48654887331495095, 0.4409178433670343],
        std=[0.2673342858792401, 0.2564384629170883, 0.25742285334015846]
    )
    
    if train:
        return transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            normalize
        ])
    else:
        return transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])

class Trainer:
    def __init__(self, device, label_mode='raw'):
        self.device = device
        self.batch_size = 90
        self.label_mode = label_mode
        
        # Setup loss function based on label mode
        if label_mode == 'raw':
            self.criterion = nn.CrossEntropyLoss()
        elif label_mode in ['one_hot', 'multi_hot']:
            self.criterion = nn.BCEWithLogitsLoss()  # Better for binary vectors
        else:
            self.criterion = nn.MSELoss()  # Use MSE for other transformed labels
        
        # Setup data
        train_transform = get_transforms(train=True)
        test_transform = get_transforms(train=False)
        
        # Load datasets with label transformation
        base_trainset = torchvision.datasets.CIFAR100(
            root='./data', train=True, download=True, transform=train_transform)
        base_testset = torchvision.datasets.CIFAR100(
            root='./data', train=False, download=True, transform=test_transform)
            
        self.full_trainset = LabelTransformedDataset(base_trainset, label_mode)
        self.full_testset = LabelTransformedDataset(base_testset, label_mode)
        
        # Define tasks (20 tasks, 5 classes each)
        self.tasks = [list(range(i, i+5)) for i in range(0, 100, 5)]
        
        # Initialize experiment tracker with label mode
        self.tracker = ExperimentTracker(label_mode=label_mode)
        
        # Create results directory
        os.makedirs(f'results/{label_mode}', exist_ok=True)
        
    def get_output_size(self):
        """Get output size based on label mode"""
        if self.label_mode == 'raw':
            return 100
        elif self.label_mode == 'one_hot':
            return 100
        elif self.label_mode == 'multi_hot':
            return 100
        elif self.label_mode == 'one_hot_self_concat':
            return 200  # Double size due to concatenation
        elif self.label_mode == 'random_labels':
            return 100  # Can be modified if different dimension is desired
            
    def get_subset_indices(self, dataset, classes):
        """Get indices of samples belonging to specified classes"""
        indices = []
        for idx in range(len(dataset)):
            original_label = dataset.get_original_label(idx)
            if original_label in classes:
                indices.append(idx)
        return indices
    
    def setup_data_for_phase(self, phase_num):
        """Setup datasets for current phase using task-specific classes"""
        # Get current task classes
        current_task_classes = self.tasks[phase_num]
        
        # Get indices for current task classes only
        train_indices = self.get_subset_indices(self.full_trainset, current_task_classes)
        test_indices = self.get_subset_indices(self.full_testset, current_task_classes)
        
        # Create subsets
        trainset = Subset(self.full_trainset, train_indices)
        testset = Subset(self.full_testset, test_indices)
        
        # Create dataloaders with optimized settings for GPU
        self.trainloader = DataLoader(
            trainset, 
            batch_size=self.batch_size, 
            shuffle=True, 
            num_workers=2,  # Increased for better performance
            pin_memory=True,  # Faster data transfer to GPU
            persistent_workers=True  # Keep workers alive between epochs
        )
        self.testloader = DataLoader(
            testset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=2,  # Increased for better performance
            pin_memory=True,  # Faster data transfer to GPU
            persistent_workers=True  # Keep workers alive between epochs
        )
        
        return len(current_task_classes)
    
    def adjust_learning_rate(self, optimizer, epoch):
        """Adjust learning rate according to schedule"""
        if epoch < 60:  # First 60 epochs
            lr = 0.1
        elif epoch < 120:  # Next 60 epochs
            lr = 0.02
        elif epoch < 160:  # Next 40 epochs
            lr = 0.004
        else:  # Final 40 epochs
            lr = 0.0008
        
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    
    def train_model(self, model, phase_num, num_epochs=10):
        """Train model for one phase"""
        # Verify model is on correct device
        model = model.to(self.device)
        print(f"Model device: {next(model.parameters()).device}")
        
        optimizer = optim.SGD(
            model.parameters(),
            lr=0.1,
            momentum=0.9,
            weight_decay=0.0005
        )
        
        best_acc = 0
        best_weights = None
        
        # Setup activation hooks for metrics
        self.tracker.setup_hooks(model)
        
        for epoch in range(num_epochs):
            self.adjust_learning_rate(optimizer, epoch)
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            
            for inputs, targets, original_targets in self.trainloader:
                # Move data to device and verify
                inputs = inputs.to(self.device)
                if self.label_mode == 'raw':
                    targets = targets.to(self.device)
                else:
                    targets = targets.to(self.device).float()
                original_targets = original_targets.to(self.device)
                
                # Verify data is on correct device
                if epoch == 0 and total == 0:  # Print only once at start
                    print(f"Input device: {inputs.device}")
                    print(f"Target device: {targets.device}")
                    print(f"Original target device: {original_targets.device}")
                
                optimizer.zero_grad()
                outputs = model(inputs)
                
                if self.label_mode == 'raw':
                    loss = self.criterion(outputs, targets)
                    _, predicted = outputs.max(1)
                    correct += predicted.eq(original_targets).sum().item()
                else:
                    loss = self.criterion(outputs, targets)
                    # For transformed labels, convert back to class predictions
                    if self.label_mode == 'one_hot_self_concat':
                        outputs = outputs[:, :100]  # Use first half for prediction
                    _, predicted = outputs.max(1)
                    correct += predicted.eq(original_targets).sum().item()
                
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                total += targets.size(0)
            
            # Evaluate
            test_acc = self.evaluate(model)
            
            # Save best model
            if test_acc > best_acc:
                best_acc = test_acc
                best_weights = model.state_dict().copy()
                checkpoint = {
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'phase': phase_num,
                    'epoch': epoch,
                    'accuracy': test_acc,
                    'label_mode': self.label_mode
                }
                torch.save(checkpoint, f'results/{self.label_mode}/phase_{phase_num}_best.pth')
            
            print(f'Phase {phase_num} | Epoch {epoch+1}/{num_epochs}')
            print(f'Train Loss: {running_loss/len(self.trainloader):.3f}')
            print(f'Train Acc: {100.*correct/total:.2f}%')
            print(f'Test Acc: {test_acc:.2f}%')
        
        # Load best weights
        model.load_state_dict(best_weights)
        return best_acc
    
    def evaluate(self, model):
        """Evaluate model on test set"""
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets, original_targets in self.testloader:
                inputs = inputs.to(self.device)
                if self.label_mode == 'raw':
                    targets = targets.to(self.device)
                    outputs = model(inputs)
                    _, predicted = outputs.max(1)
                else:
                    targets = targets.to(self.device).float()
                    outputs = model(inputs)
                    if self.label_mode == 'one_hot_self_concat':
                        outputs = outputs[:, :100]  # Use first half for prediction
                    _, predicted = outputs.max(1)
                
                original_targets = original_targets.to(self.device)
                total += original_targets.size(0)
                correct += predicted.eq(original_targets).sum().item()
        
        return 100. * correct / total
    
    def evaluate_all_seen_classes(self, model, current_phase):
        """Evaluate model on all classes seen so far"""
        seen_classes = []
        for i in range(current_phase + 1):
            seen_classes.extend(self.tasks[i])
            
        test_indices = self.get_subset_indices(self.full_testset, seen_classes)
        testset = Subset(self.full_testset, test_indices)
        testloader = DataLoader(testset, batch_size=self.batch_size, shuffle=False, num_workers=2)
        
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets, original_targets in testloader:
                inputs = inputs.to(self.device)
                original_targets = original_targets.to(self.device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += original_targets.size(0)
                correct += predicted.eq(original_targets).sum().item()
        
        return 100. * correct / total
    
    def run_experiment(self, num_phases=20, classes_per_phase=5):
        """Run incremental learning experiment"""
        # Initialize model with correct output dimension based on label mode
        output_dim = 100 if self.label_mode in ['one_hot', 'multi_hot', 'random_labels'] else classes_per_phase
        if self.label_mode == 'one_hot_self_concat':
            output_dim = 200
            
        incremental_model = IncrementalResNet18(
            initial_classes=output_dim
        ).to(self.device)
        
        for phase in range(num_phases):
            print(f"\n=== Phase {phase} ({self.label_mode} mode) ===")
            _ = self.setup_data_for_phase(phase)
            
            # Train on current task
            print("\nTraining incremental model...")
            inc_acc = self.train_model(incremental_model, phase)
            
            # Compute metrics on all seen classes
            all_classes_acc = self.evaluate_all_seen_classes(incremental_model, phase)
            inc_metrics = self.tracker.compute_metrics(incremental_model, (phase + 1) * classes_per_phase)
            inc_metrics['accuracy'] = all_classes_acc
            
            # Update metrics
            self.tracker.update('incremental', inc_metrics, (phase + 1) * classes_per_phase)
            
            # Expand model for next phase if needed
            if phase < num_phases - 1:
                incremental_model.expand_output(
                    additional_classes=classes_per_phase
                )
            
            # Plot results
            self.tracker.plot_results()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer = Trainer(device)
    trainer.run_experiment()

if __name__ == "__main__":
    main()