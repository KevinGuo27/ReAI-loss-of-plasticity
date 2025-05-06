import torch
from torch.utils.data import Dataset
from pipeline.label_transform import one_hot, multi_hot, one_hot_self_concat, random_labels

class LabelTransformedDataset(Dataset):
    """Wrapper dataset that applies label transformations while maintaining access to original labels"""
    def __init__(self, dataset, label_mode='raw', num_classes=100):
        self.dataset = dataset
        self.label_mode = label_mode
        self.num_classes = num_classes
        
    def __getitem__(self, index):
        data, original_label = self.dataset[index]
        
        # Transform the label based on mode
        if self.label_mode == 'raw':
            transformed_label = original_label
        elif self.label_mode == 'one_hot':
            transformed_label = one_hot(torch.tensor(original_label), self.num_classes)
        elif self.label_mode == 'multi_hot':
            transformed_label = multi_hot(torch.tensor(original_label), self.num_classes)
        elif self.label_mode == 'one_hot_self_concat':
            transformed_label = one_hot_self_concat(torch.tensor(original_label), self.num_classes)
        elif self.label_mode == 'random_labels':
            transformed_label = random_labels(torch.tensor(original_label), self.num_classes)
        else:
            raise ValueError(f"Unknown label mode: {self.label_mode}")
            
        return data, transformed_label, original_label
    
    def __len__(self):
        return len(self.dataset)
    
    def get_original_label(self, index):
        """Helper method to get original label directly"""
        return self.dataset[index][1] 