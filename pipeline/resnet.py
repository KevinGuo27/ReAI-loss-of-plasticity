import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.init as init

class IncrementalResNet18(nn.Module):
    def __init__(self, initial_classes=5):
        super(IncrementalResNet18, self).__init__()
        # Load pretrained ResNet-18 without final layer
        self.resnet = models.resnet18(pretrained=False)
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()  # Remove final layer
        
        # Create expandable classification head
        self.classifier = nn.Linear(in_features, initial_classes)
        
        # Initialize weights
        self._initialize_weights()
        
        # Debug: print model structure
        print("\nModel structure:")
        print(self)
    
    def _initialize_weights(self):
        # Kaiming initialization for Conv and Linear layers
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
    
    def expand_output(self, additional_classes=5):
        """Expand the output layer by adding new classes"""
        current_classes = self.classifier.out_features
        new_classifier = nn.Linear(self.classifier.in_features, 
                                 current_classes + additional_classes)
        
        # Copy existing weights
        with torch.no_grad():
            new_classifier.weight[:current_classes] = self.classifier.weight
            new_classifier.bias[:current_classes] = self.classifier.bias
            
            # Initialize new weights
            init.kaiming_normal_(new_classifier.weight[current_classes:], 
                               mode='fan_out', nonlinearity='relu')
            init.constant_(new_classifier.bias[current_classes:], 0)
        
        self.classifier = new_classifier
    
    def forward(self, x):
        features = self.resnet(x)
        return self.classifier(features) 