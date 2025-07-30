import torch
import torch.nn as nn
from typing import List, Dict
import os

# Ensure you have x-transformers installed: pip install x-transformers
from x_transformers.x_transformers import CrossAttender

from monai.networks.nets import resnet, SwinUNETR
import logging
from config import Config

# --- Setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# --- Setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MultiSequenceResNet(nn.Module):
    """
    A robust 3D ResNet model for binary classification that can leverage
    pre-trained weights from MedicalNet and fuses features from multiple
    CT sequences.

    Args:
        ct_types (List[str]): List of CT sequence keys (e.g., ['A', 'V']).
        num_classes (int): The number of output classes.
        model_depth (int): The depth of the ResNet model (18, 34, or 50).
        fusion_method (str): Fusion method, 'concat' or 'attention'.
        dropout_rate (float): Dropout rate for the final classifier.
        pretrained (bool): If True, loads pre-trained MedicalNet weights.
        pretrained_path (str): The file path to the pre-trained model weights.
    """
    def __init__(
        self,
        ct_types: List[str],
        num_classes: int = 2,
        model_depth: int = 18,
        fusion_method: str = 'concat',
        dropout_rate: float = 0.5,
        pretrained: bool = True,
        pretrained_path: str = None
    ):
        super().__init__()
        
        if fusion_method not in ['concat', 'attention']:
            raise ValueError("fusion_method must be 'concat' or 'attention'")

        self.ct_types = ct_types
        self.fusion_method = fusion_method
        num_sequences = len(ct_types)

        # 1. Create a ResNet backbone for each CT sequence.
        self.backbones = nn.ModuleList()
        for _ in range(num_sequences):
            backbone = resnet.ResNet(
                spatial_dims=3,
                n_input_channels=1,
                block='basic' if model_depth <= 34 else 'bottleneck',
                layers=self._get_layer_config(model_depth),
                block_inplanes=[64, 128, 256, 512],
                feed_forward=False # Output feature vector, not logits
            )
            
            if pretrained:
                if pretrained_path is None or not os.path.exists(pretrained_path):
                    raise FileNotFoundError(f"Pre-trained model file not found at: {pretrained_path}")
                
                logger.info(f"Loading pre-trained weights for ResNet-{model_depth} from local path: {pretrained_path}")
                
                # Load the state dictionary from the local file
                state_dict = torch.load(pretrained_path)
                
                # Remove the final classification layer from the pre-trained weights
                # as we will be training our own.
                if 'fc.weight' in state_dict: del state_dict['fc.weight']
                if 'fc.bias' in state_dict: del state_dict['fc.bias']
                
                # Load the weights, ignoring mismatches (like the missing fc layer)
                backbone.load_state_dict(state_dict, strict=False)

            self.backbones.append(backbone)
        
        # 2. Define the fusion and classification layers.
        block_expansion = 1 if model_depth <= 34 else 4
        num_features_per_seq = 512 * block_expansion

        if self.fusion_method == 'concat':
            classifier_input_features = num_features_per_seq * num_sequences
            self.classifier = nn.Sequential(
                nn.Dropout(p=dropout_rate),
                nn.Linear(classifier_input_features, num_classes)
            )
        
        elif self.fusion_method == 'attention':
            self.class_token = nn.Parameter(torch.randn(1, 1, num_features_per_seq))
            self.cross_attention = CrossAttender(dim=num_features_per_seq, heads=8, depth=1)
            self.classifier = nn.Sequential(
                nn.Dropout(p=dropout_rate),
                nn.Linear(num_features_per_seq, num_classes)
            )

    def forward(self, data_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Defines the forward pass of the model."""
        sequence_vectors = [
            backbone(data_dict[ct_type]) 
            for backbone, ct_type in zip(self.backbones, self.ct_types)
        ]
        
        if self.fusion_method == 'concat':
            fused = torch.cat(sequence_vectors, dim=1)
        
        elif self.fusion_method == 'attention':
            context = torch.stack(sequence_vectors, dim=1)
            b = context.shape[0]
            query = self.class_token.expand(b, -1, -1)
            fused = self.cross_attention(query, context).squeeze(1)

        logits = self.classifier(fused)
        return logits

    def _get_layer_config(self, model_depth: int):
        """Helper function to get ResNet layer configurations."""
        configs = {
            18: (2, 2, 2, 2), 34: (3, 4, 6, 3), 50: (3, 4, 6, 3)
        }
        if model_depth not in configs:
            logger.warning(f"Model depth {model_depth} not supported for pre-training, defaulting to 18.")
            return configs[18]
        return configs[model_depth]


class MultiSequenceSwinUNETR(nn.Module):
    """
    A powerful 3D fusion model for binary classification that uses pre-trained
    Swin UNETR encoders as backbones. The model fuses features from multiple
    CT sequences using either concatenation or cross-attention.

    Args:
        ct_types (List[str]): List of CT sequence keys (e.g., ['A', 'V']).
        num_classes (int): The number of output classes.
        fusion_method (str): Fusion method, 'concat' or 'attention'.
        dropout_rate (float): Dropout rate for the final classifier.
    """
    def __init__(
        self,
        ct_types: List[str],
        num_classes: int = 2,
        fusion_method: str = 'concat',
        dropout_rate: float = 0.5
    ):
        super().__init__()
        
        if fusion_method not in ['concat', 'attention']:
            raise ValueError("fusion_method must be 'concat' or 'attention'")

        self.ct_types = ct_types
        self.fusion_method = fusion_method
        num_sequences = len(ct_types)

        # 1. Create a Swin UNETR backbone for each CT sequence.
        #    We will use the powerful encoder part of this model.
        self.backbones = nn.ModuleList()
        for _ in range(num_sequences):
            # Load the pre-trained Swin UNETR model for BTCV multi-organ segmentation.
            # This will download the weights automatically on first use.
            swin_unetr_model = SwinUNETR(
                img_size=(96, 96, 96), # The model was pre-trained on this patch size
                in_channels=1,
                out_channels=14, # Dummy out_channels, we only use the encoder
                feature_size=48,
                use_checkpoint=True,
            )
            
            # Load pre-trained weights into the model
            # You may need to install 'gdown' for this: pip install gdown
            weights = SwinUNETR.get_pretrain_dict("swin_unetr.base_5000ep_f48_fe_exp.pth")
            swin_unetr_model.load_from(weights=weights)
            
            # We only need the encoder part for feature extraction
            self.backbones.append(swin_unetr_model.swinViT)

        # 2. Define the fusion and classification layers.
        #    The feature dimension from the Swin UNETR encoder is 768.
        num_features_per_seq = 768

        if self.fusion_method == 'concat':
            classifier_input_features = num_features_per_seq * num_sequences
            self.classifier = nn.Sequential(
                nn.Dropout(p=dropout_rate),
                nn.Linear(classifier_input_features, num_classes)
            )
        
        elif self.fusion_method == 'attention':
            self.class_token = nn.Parameter(torch.randn(1, 1, num_features_per_seq))
            self.cross_attention = CrossAttender(dim=num_features_per_seq, heads=8, depth=1)
            self.classifier = nn.Sequential(
                nn.Dropout(p=dropout_rate),
                nn.Linear(num_features_per_seq, num_classes)
            )

    def forward(self, data_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Defines the forward pass of the model."""
        
        sequence_vectors = []
        for i, ct_type in enumerate(self.ct_types):
            # The SwinViT backbone outputs a tuple; the first element is the feature vector
            # Shape: (Batch, Num_Patches, Feature_Dim)
            feature_sequence, _ = self.backbones[i](data_dict[ct_type])
            
            # We take the feature of the class token (the first token in the sequence)
            # as the representative feature vector for the entire image.
            feature_vector = feature_sequence[:, 0]
            sequence_vectors.append(feature_vector)
        
        if self.fusion_method == 'concat':
            fused = torch.cat(sequence_vectors, dim=1)
        
        elif self.fusion_method == 'attention':
            context = torch.stack(sequence_vectors, dim=1)
            b = context.shape[0]
            query = self.class_token.expand(b, -1, -1)
            fused = self.cross_attention(query, context).squeeze(1)

        logits = self.classifier(fused)
        return logits