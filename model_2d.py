import torch
import torch.nn as nn
from typing import List, Dict
import os
from torchvision.models import resnet18, ResNet18_Weights

# Import your existing 3D model components
from monai.networks.nets import resnet as monai_resnet
from x_transformers.x_transformers import CrossAttender
import logging

class MultiSequenceResNet2D(nn.Module):
    """
    A 2D fusion model that uses pre-trained torchvision ResNets as backbones.
    """
    def __init__(
        self,
        ct_types: List[str],
        num_classes: int = 2,
        fusion_method: str = 'concat',
        dropout_rate: float = 0.3
    ):
        super().__init__()
        self.ct_types = ct_types
        self.fusion_method = fusion_method
        num_sequences = len(ct_types)

        self.backbones = nn.ModuleList()
        for _ in range(num_sequences):
            # Load a standard ResNet-18 pre-trained on ImageNet
            backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
            
            # Modify the first convolutional layer to accept 1-channel (grayscale) input
            original_conv1 = backbone.conv1
            backbone.conv1 = nn.Conv2d(1, original_conv1.out_channels, 
                                       kernel_size=original_conv1.kernel_size, 
                                       stride=original_conv1.stride, 
                                       padding=original_conv1.padding, bias=False)
            
            # Remove the final fully connected layer to use it as a feature extractor
            backbone.fc = nn.Identity()
            self.backbones.append(backbone)
            
        num_features_per_seq = 512 # Output features of ResNet-18
        f_dim = 256 # Intermediate dimension for the classifier head

        if self.fusion_method == 'concat':
            classifier_input_features = num_features_per_seq * num_sequences
            self.classifier = nn.Sequential(
                nn.Linear(classifier_input_features, f_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout_rate),
                nn.Linear(f_dim, f_dim // 2),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout_rate),
                nn.Linear(f_dim // 2, num_classes)
            )
        # Note: Attention fusion could be added here as well if needed
        else:
            raise NotImplementedError("Only 'concat' fusion is implemented for the 2D model.")

    def forward(self, data_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        sequence_vectors = [
            backbone(data_dict[ct_type]) 
            for backbone, ct_type in zip(self.backbones, self.ct_types)
        ]
        
        fused = torch.cat(sequence_vectors, dim=1)
        logits = self.classifier(fused)
        return logits