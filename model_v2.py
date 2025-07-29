import torch
import torch.nn as nn
from typing import List, Dict

# Ensure you have x-transformers installed: pip install x-transformers
from x_transformers.x_transformers import CrossAttender

from monai.networks.nets import resnet
import logging

# --- Setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MultiSequenceResNet(nn.Module):
    """
    CORRECTED: A simple and robust 3D ResNet model for binary classification that
    fuses features from multiple CT sequences. This version removes the redundant
    pooling layer, fixing the dimension error.
    """
    def __init__(
        self,
        ct_types: List[str],
        num_classes: int = 2,
        model_depth: int = 18,
        fusion_method: str = 'concat',
        dropout_rate: float = 0.5
    ):
        super().__init__()
        
        if fusion_method not in ['concat', 'attention']:
            raise ValueError("fusion_method must be 'concat' or 'attention'")

        self.ct_types = ct_types
        self.fusion_method = fusion_method
        num_sequences = len(ct_types)

        # 1. Create ResNet backbones as feature extractors.
        #    `feed_forward=False` makes the model output a feature vector (B, C)
        #    instead of final classification logits.
        self.backbones = nn.ModuleList([
            resnet.ResNet(
                spatial_dims=3,
                n_input_channels=1,
                block='basic' if model_depth <= 34 else 'bottleneck',
                layers=self._get_layer_config(model_depth),
                block_inplanes=[64, 128, 256, 512],
                feed_forward=False
            ) for _ in range(num_sequences)
        ])
        
        # 2. REMOVED: The redundant self.pooling layer is gone.

        # 3. Define the fusion and classification layers.
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
            
            # CORRECTED: Added depth=1 to specify a single layer of cross-attention.
            self.cross_attention = CrossAttender(dim=num_features_per_seq, heads=8, depth=1)
            
            self.classifier = nn.Sequential(
                nn.Dropout(p=dropout_rate),
                nn.Linear(num_features_per_seq, num_classes)
            )

    def forward(self, data_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Defines the forward pass of the model."""
        
        # 1. Get the feature vector directly from each backbone.
        sequence_vectors = []
        for i, ct_type in enumerate(self.ct_types):
            # The output here is already a feature vector (B, C)
            feature_vector = self.backbones[i](data_dict[ct_type])
            sequence_vectors.append(feature_vector)
        
        # 2. Fuse the feature vectors.
        if self.fusion_method == 'concat':
            fused = torch.cat(sequence_vectors, dim=1)
        
        elif self.fusion_method == 'attention':
            context = torch.stack(sequence_vectors, dim=1)
            b = context.shape[0]
            query = self.class_token.expand(b, -1, -1)
            fused = self.cross_attention(query, context).squeeze(1)

        # 3. Pass the fused vector through the final classifier.
        logits = self.classifier(fused)
        return logits

    def _get_layer_config(self, model_depth: int):
        """Helper function to get ResNet layer configurations."""
        configs = {
            18: (2, 2, 2, 2), 34: (3, 4, 6, 3), 50: (3, 4, 6, 3),
            101: (3, 4, 23, 3), 152: (3, 8, 36, 3)
        }
        if model_depth not in configs:
            logger.warning(f"Model depth {model_depth} not in config, defaulting to 18.")
            return configs[18]
        return configs[model_depth]


# --- Example Usage and Testing ---
if __name__ == "__main__":
    batch_size = 4
    target_size = (96, 96, 96)
    ct_keys_to_use = ['A', 'V']

    # --- Test Concatenation Fusion ---
    print("\n--- Testing Model with 'concat' fusion ---")
    concat_model = MultiSequenceResNet(
        ct_types=ct_keys_to_use,
        fusion_method='concat',
        model_depth=18
    )
    
    dummy_input = {key: torch.randn(batch_size, 1, *target_size) for key in ct_keys_to_use}
    output_concat = concat_model(dummy_input)
    
    print(f"Input keys: {list(dummy_input.keys())}")
    print(f"Output shape (logits): {output_concat.shape}")
    assert output_concat.shape == (batch_size, 2)
    print("Concat model test PASSED.")

    # --- Test Attention Fusion ---
    print("\n--- Testing Model with 'attention' fusion ---")
    attention_model = MultiSequenceResNet(
        ct_types=ct_keys_to_use,
        fusion_method='attention',
        model_depth=18
    )
    
    output_attention = attention_model(dummy_input)
    
    print(f"Input keys: {list(dummy_input.keys())}")
    print(f"Output shape (logits): {output_attention.shape}")
    assert output_attention.shape == (batch_size, 2)
    print("Attention model test PASSED.")
