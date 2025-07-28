import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Dict
import monai
from monai.networks.nets import resnet
from monai.networks.blocks import Convolution, UpSample
import logging

logger = logging.getLogger(__name__)

# Early fusion removed - CT modalities are not registered
# class EarlyFusionResNet(nn.Module):
#     """
#     Early fusion ResNet model that takes concatenated CT sequences as input.
#     Input: (B, C, D, H, W) where C = number of CT sequences (e.g., 4 for A, D, N, V)
#     NOTE: This is not appropriate for unregistered CT modalities
#     """


class IntermediateFusionResNet(nn.Module):
    """
    Intermediate fusion ResNet model that processes each CT sequence separately
    and fuses features at intermediate layers.
    """
    
    def __init__(
        self,
        num_sequences: int = 4,
        num_classes: int = 2,
        spatial_dims: int = 3,
        model_depth: int = 18,
        fusion_layer: str = 'late',  # 'early', 'middle', 'late'
        fusion_method: str = 'concat',  # 'concat', 'attention', 'weighted_sum'
        dropout_rate: float = 0.5
    ):
        super().__init__()
        
        self.num_sequences = num_sequences
        self.fusion_layer = fusion_layer
        self.fusion_method = fusion_method
        
        # Individual ResNet backbones for each sequence
        self.backbones = nn.ModuleList([
            resnet.ResNet(
                spatial_dims=spatial_dims,
                n_input_channels=1,  # Single channel per sequence
                num_classes=num_classes,
                block='basic' if model_depth <= 34 else 'bottleneck',
                layers=self._get_layer_config(model_depth),
                block_inplanes=[64, 128, 256, 512]
            ) for _ in range(num_sequences)
        ])
        
        # Fusion layers
        if fusion_method == 'concat':
            self.fusion_linear = nn.Linear(num_sequences * num_classes, num_classes)
        elif fusion_method == 'attention':
            self.attention_weights = nn.Parameter(torch.ones(num_sequences))
            self.fusion_linear = nn.Linear(num_classes, num_classes)
        elif fusion_method == 'weighted_sum':
            self.sequence_weights = nn.Parameter(torch.ones(num_sequences))
            self.fusion_linear = nn.Linear(num_classes, num_classes)
        
        logger.info(f"IntermediateFusionResNet initialized with {num_sequences} sequences")
    
    def _get_layer_config(self, model_depth: int) -> Tuple[int, int, int, int]:
        """Get layer configuration based on model depth."""
        configs = {
            18: (2, 2, 2, 2),
            34: (3, 4, 6, 3),
            50: (3, 4, 6, 3),
            101: (3, 4, 23, 3),
            152: (3, 8, 36, 3)
        }
        return configs.get(model_depth, (2, 2, 2, 2))
    
    def forward(self, sequences: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            sequences: List of tensors, each of shape (B, 1, D, H, W)
        Returns:
            logits: Output logits of shape (B, num_classes)
        """
        # Process each sequence separately
        sequence_outputs = []
        for i, (sequence, backbone) in enumerate(zip(sequences, self.backbones)):
            output = backbone(sequence)
            sequence_outputs.append(output)
        
        # Stack outputs
        stacked_outputs = torch.stack(sequence_outputs, dim=1)  # (B, num_sequences, num_classes)
        
        # Apply fusion method
        if self.fusion_method == 'concat':
            # Concatenate all sequence outputs
            fused = stacked_outputs.view(stacked_outputs.size(0), -1)  # (B, num_sequences * num_classes)
            logits = self.fusion_linear(fused)
        
        elif self.fusion_method == 'attention':
            # Weighted sum using attention weights
            attention_weights = F.softmax(self.attention_weights, dim=0)  # (num_sequences,)
            weighted_outputs = stacked_outputs * attention_weights.unsqueeze(0).unsqueeze(-1)  # (B, num_sequences, num_classes)
            fused = weighted_outputs.sum(dim=1)  # (B, num_classes)
            logits = self.fusion_linear(fused)
        
        elif self.fusion_method == 'weighted_sum':
            # Weighted sum using learnable weights
            sequence_weights = F.softmax(self.sequence_weights, dim=0)  # (num_sequences,)
            weighted_outputs = stacked_outputs * sequence_weights.unsqueeze(0).unsqueeze(-1)  # (B, num_sequences, num_classes)
            fused = weighted_outputs.sum(dim=1)  # (B, num_classes)
            logits = self.fusion_linear(fused)
        
        return logits
    
class CorrectedIntermediateFusionResNet(nn.Module):
    """
    Corrected ResNet model that processes each CT sequence, pools features,
    and fuses the resulting feature vectors for classification.
    """
    def __init__(
        self,
        num_sequences: int = 4,
        num_classes: int = 2,
        spatial_dims: int = 3,
        model_depth: int = 18,
        fusion_method: str = 'concat',  # 'concat', 'attention', 'weighted_sum'
        dropout_rate: float = 0.5
    ):
        super().__init__()

        self.num_sequences = num_sequences
        self.fusion_method = fusion_method

        # --- CORRECTED INITIALIZATION ---
        # 1. Individual ResNet backbones for each sequence
        # Note: num_classes and dropout_prob are removed, block_type is fixed.
        self.backbones = nn.ModuleList([
            resnet.ResNet(
                spatial_dims=spatial_dims,
                n_input_channels=1,
                block='basic' if model_depth <= 34 else 'bottleneck',
                layers=self._get_layer_config(model_depth),
                block_inplanes=[64, 128, 256, 512],
                feed_forward=False, # Ensures output is a feature map
            ) for _ in range(num_sequences)
        ])
        
        # 2. ADDED: Pooling layer to convert feature maps to vectors
        self.pooling = nn.AdaptiveAvgPool3d(1)

        # 3. CORRECTED: Fusion layers based on actual feature vector sizes
        # Determine the number of output features from a single backbone
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, 32, 32, 32) # Small dummy tensor
            features = self.backbones[0](dummy_input)
            num_features_per_sequence = features.shape[1]

        if fusion_method == 'concat':
            total_features = num_sequences * num_features_per_sequence
            self.classifier = nn.Linear(total_features, num_classes)
        elif fusion_method in ['attention', 'weighted_sum']:
            if fusion_method == 'attention':
                self.attention_weights = nn.Parameter(torch.ones(num_sequences))
            else: # weighted_sum
                self.sequence_weights = nn.Parameter(torch.ones(num_sequences))
            self.classifier = nn.Linear(num_features_per_sequence, num_classes)
        
        self.dropout = nn.Dropout(dropout_rate)

    def _get_layer_config(self, model_depth: int) -> Tuple[int, int, int, int]:
        configs = {
            18: (2, 2, 2, 2), 34: (3, 4, 6, 3), 50: (3, 4, 6, 3),
            101: (3, 4, 23, 3), 152: (3, 8, 36, 3)
        }
        return configs.get(model_depth, (2, 2, 2, 2))

    def forward(self, sequences: List[torch.Tensor]) -> torch.Tensor:
        # 1. Get feature vectors from each sequence
        sequence_vectors = []
        for sequence, backbone in zip(sequences, self.backbones):
            feature_map = backbone(sequence)       # (B, C, D, H, W)
            pooled_vector = self.pooling(feature_map) # (B, C, 1, 1, 1)
            sequence_vectors.append(pooled_vector.flatten(start_dim=1)) # (B, C)
        
        # Stack vectors: (B, num_sequences, num_features_per_sequence)
        stacked_vectors = torch.stack(sequence_vectors, dim=1)

        # 2. Apply fusion method on feature vectors
        if self.fusion_method == 'concat':
            # Flatten into (B, num_sequences * num_features)
            fused = stacked_vectors.view(stacked_vectors.size(0), -1)
        
        elif self.fusion_method in ['attention', 'weighted_sum']:
            weights_source = self.attention_weights if self.fusion_method == 'attention' else self.sequence_weights
            weights = F.softmax(weights_source, dim=0) # (num_sequences,)
            
            # Weighted sum: (B, num_seq, C) * (1, num_seq, 1) -> sum -> (B, C)
            fused = (stacked_vectors * weights.unsqueeze(0).unsqueeze(-1)).sum(dim=1)

        # 3. Apply dropout and final classification
        fused = self.dropout(fused)
        logits = self.classifier(fused)
        
        return logits


class EnsembleResNet(nn.Module):
    """
    Ensemble model that trains separate models for each CT sequence
    and combines their predictions.
    """
    
    def __init__(
        self,
        num_sequences: int = 4,
        num_classes: int = 2,
        spatial_dims: int = 3,
        model_depth: int = 18,
        ensemble_method: str = 'voting',  # 'voting', 'weighted_voting', 'stacking'
        dropout_rate: float = 0.5
    ):
        super().__init__()
        
        self.num_sequences = num_sequences
        self.ensemble_method = ensemble_method
        
        # Individual ResNet models for each sequence
        self.models = nn.ModuleList([
            resnet.ResNet(
                spatial_dims=spatial_dims,
                in_channels=1,
                num_classes=num_classes,
                block_type='basic' if model_depth <= 34 else 'bottleneck',
                layers=self._get_layer_config(model_depth),
                dropout_prob=dropout_rate
            ) for _ in range(num_sequences)
        ])
        
        # Ensemble combination layer
        if ensemble_method == 'weighted_voting':
            self.ensemble_weights = nn.Parameter(torch.ones(num_sequences))
        elif ensemble_method == 'stacking':
            self.meta_classifier = nn.Linear(num_sequences * num_classes, num_classes)
        
        logger.info(f"EnsembleResNet initialized with {num_sequences} models")
    
    def _get_layer_config(self, model_depth: int) -> Tuple[int, int, int, int]:
        """Get layer configuration based on model depth."""
        configs = {
            18: (2, 2, 2, 2),
            34: (3, 4, 6, 3),
            50: (3, 4, 6, 3),
            101: (3, 4, 23, 3),
            152: (3, 8, 36, 3)
        }
        return configs.get(model_depth, (2, 2, 2, 2))
    
    def forward(self, sequences: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            sequences: List of tensors, each of shape (B, 1, D, H, W)
        Returns:
            logits: Output logits of shape (B, num_classes)
        """
        # Get predictions from each model
        predictions = []
        for sequence, model in zip(sequences, self.models):
            pred = model(sequence)
            predictions.append(pred)
        
        # Stack predictions
        stacked_preds = torch.stack(predictions, dim=1)  # (B, num_sequences, num_classes)
        
        # Apply ensemble method
        if self.ensemble_method == 'voting':
            # Simple averaging
            logits = stacked_preds.mean(dim=1)  # (B, num_classes)
        
        elif self.ensemble_method == 'weighted_voting':
            # Weighted averaging
            weights = F.softmax(self.ensemble_weights, dim=0)  # (num_sequences,)
            weighted_preds = stacked_preds * weights.unsqueeze(0).unsqueeze(-1)  # (B, num_sequences, num_classes)
            logits = weighted_preds.sum(dim=1)  # (B, num_classes)
        
        elif self.ensemble_method == 'stacking':
            # Stack predictions and use meta-classifier
            flattened = stacked_preds.view(stacked_preds.size(0), -1)  # (B, num_sequences * num_classes)
            logits = self.meta_classifier(flattened)
        
        return logits


class TumorClassificationModel(nn.Module):
    """
    Main model class that can handle different fusion strategies.
    Note: Early fusion removed since CT modalities are not registered.
    """
    
    def __init__(
        self,
        fusion_strategy: str = 'intermediate',
        num_sequences: int = 4,
        num_classes: int = 2,
        spatial_dims: int = 3,
        model_depth: int = 18,
        fusion_method: str = 'concat',
        ensemble_method: str = 'voting',
        dropout_rate: float = 0.5,
        pretrained: bool = False
    ):
        super().__init__()
        
        self.fusion_strategy = fusion_strategy
        
        if fusion_strategy == 'intermediate':
            self.model = IntermediateFusionResNet(
                num_sequences=num_sequences,
                num_classes=num_classes,
                spatial_dims=spatial_dims,
                model_depth=model_depth,
                fusion_method=fusion_method,
                dropout_rate=dropout_rate
            )
        
        elif fusion_strategy == 'ensemble':
            self.model = EnsembleResNet(
                num_sequences=num_sequences,
                num_classes=num_classes,
                spatial_dims=spatial_dims,
                model_depth=model_depth,
                ensemble_method=ensemble_method,
                dropout_rate=dropout_rate
            )
        
        else:
            raise ValueError(f"Unsupported fusion strategy: {fusion_strategy}. Use 'intermediate' or 'ensemble'")
        
        logger.info(f"TumorClassificationModel initialized with {fusion_strategy} fusion strategy")
    
    def forward(self, x):
        """
        Forward pass that handles different input formats based on fusion strategy.
        
        Args:
            x: Input data format depends on fusion strategy:
                - 'intermediate'/'ensemble': List of (B, 1, D, H, W) tensors
        """
        return self.model(x)
    
    def get_model_info(self) -> Dict:
        """Get information about the model configuration."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'fusion_strategy': self.fusion_strategy,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_type': type(self.model).__name__
        }


# Example usage and testing
if __name__ == "__main__":
    # Test different model configurations (early fusion removed for unregistered modalities)
    batch_size = 2
    spatial_dims = 3
    target_size = (64, 64, 64)
    
    # Test Intermediate Fusion
    print("\nTesting Intermediate Fusion Model:")
    intermediate_model = TumorClassificationModel(
        fusion_strategy='intermediate',
        num_sequences=4,
        num_classes=2,
        model_depth=18,
        fusion_method='attention'
    )
    
    intermediate_input = [torch.randn(batch_size, 1, *target_size) for _ in range(4)]  # List of (B, 1, D, H, W)
    intermediate_output = intermediate_model(intermediate_input)
    print(f"Intermediate fusion input: {len(intermediate_input)} sequences of shape {intermediate_input[0].shape}")
    print(f"Intermediate fusion output shape: {intermediate_output.shape}")
    print(f"Model info: {intermediate_model.get_model_info()}")
    
    # Test Ensemble Model
    print("\nTesting Ensemble Model:")
    ensemble_model = TumorClassificationModel(
        fusion_strategy='ensemble',
        num_sequences=4,
        num_classes=2,
        model_depth=18,
        ensemble_method='weighted_voting'
    )
    
    ensemble_input = [torch.randn(batch_size, 1, *target_size) for _ in range(4)]  # List of (B, 1, D, H, W)
    ensemble_output = ensemble_model(ensemble_input)
    print(f"Ensemble input: {len(ensemble_input)} sequences of shape {ensemble_input[0].shape}")
    print(f"Ensemble output shape: {ensemble_output.shape}")
    print(f"Model info: {ensemble_model.get_model_info()}") 