import json
import os
import torch
from torch.utils.data import Dataset
import monai
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    Spacingd,
    ScaleIntensityRanged,
    Resized,
    ToTensord,
    MaskIntensityd,
)
from typing import Dict, List, Tuple, Union
import logging

logger = logging.getLogger(__name__)

class MONAITumorDataset(Dataset):
    """
    Simple MONAI dataset for tumor aggressiveness classification.
    """
    
    def __init__(
        self,
        json_path: str,
        ct_types: List[str] = ['A', 'D', 'N', 'V'],
        fusion_strategy: str = 'intermediate',
        target_size: Tuple[int, int, int] = (128, 128, 128),
        normalize: bool = True,
        apply_voi_mask: bool = True,
        is_training: bool = False,
        verbose: bool = True
    ):
        """
        Args:
            json_path: Path to JSON manifest file
            ct_types: List of CT sequence types to use
            fusion_strategy: 'intermediate', 'single', or 'ensemble'
            target_size: Target size for resampling (D, H, W)
            normalize: Whether to normalize CT values
            apply_voi_mask: Whether to apply VOI mask to CT
            is_training: Whether this is for training (enables augmentation)
            verbose: Whether to print progress information
        """
        self.json_path = json_path
        self.ct_types = ct_types
        self.fusion_strategy = fusion_strategy
        self.target_size = target_size
        self.normalize = normalize
        self.apply_voi_mask = apply_voi_mask
        self.is_training = is_training
        self.verbose = verbose
        
        # Load JSON manifest
        with open(json_path, 'r') as f:
            self.data_list = json.load(f)
        
        if verbose:
            logger.info(f"Loaded {len(self.data_list)} samples from {json_path}")
            logger.info(f"CT types: {ct_types}")
            logger.info(f"Fusion strategy: {fusion_strategy}")
            logger.info(f"Training mode: {is_training}")
        
        # Create transforms
        self.transforms = self._create_transforms()
        
        # Create MONAI dataset
        self.monai_dataset = monai.data.Dataset(
            data=self.data_list,
            transform=self.transforms
        )
    
    def _create_transforms(self) -> Compose:
        """Create MONAI transforms with proper handling for different data types."""
        # Separate keys for different data types
        image_keys = self.ct_types.copy()
        mask_keys = ['mask'] if self.apply_voi_mask else []
        all_keys = image_keys + mask_keys
        
        transforms = [
            LoadImaged(keys=all_keys),
            EnsureChannelFirstd(keys=all_keys),
            Orientationd(keys=all_keys, axcodes="RAS"),
        ]
        
        # Apply spacing with appropriate modes for each data type
        if image_keys:
            transforms.append(Spacingd(keys=image_keys, pixdim=(1.0, 1.0, 1.0), mode="bilinear"))
        if mask_keys:
            transforms.append(Spacingd(keys=mask_keys, pixdim=(1.0, 1.0, 1.0), mode="nearest"))
        
        # Add normalization for CT images
        if self.normalize and image_keys:
            transforms.append(
                ScaleIntensityRanged(
                    keys=image_keys,
                    a_min=-1000,
                    a_max=1000,
                    b_min=0.0,
                    b_max=1.0,
                    clip=True,
                )
            )
        
        # Add VOI masking if requested
        if self.apply_voi_mask and image_keys:
            transforms.append(
                MaskIntensityd(keys=image_keys, mask_key="mask")
            )
        
        # Add foreground cropping to remove empty space
        if image_keys:
            transforms.append(
                monai.transforms.CropForegroundd(
                    keys=image_keys,
                    source_key=image_keys[0],  # Use first CT as reference
                    margin=10,  # Keep some margin around the foreground
                )
            )
        
        # Add resize with padding/cropping to maintain aspect ratio
        transforms.append(
            monai.transforms.ResizeWithPadOrCropd(
                keys=all_keys,
                spatial_size=self.target_size,
                mode="constant",
                constant_values=0,
            )
        )
        
        # Add data augmentation for training
        if hasattr(self, 'is_training') and self.is_training:
            transforms.extend([
                monai.transforms.RandFlipd(keys=all_keys, prob=0.5, spatial_axis=0),
                monai.transforms.RandFlipd(keys=all_keys, prob=0.5, spatial_axis=1),
                monai.transforms.RandFlipd(keys=all_keys, prob=0.5, spatial_axis=2),
                monai.transforms.RandRotate90d(keys=all_keys, prob=0.5, max_k=3),
                monai.transforms.RandScaleIntensityd(keys=image_keys, prob=0.5, factors=0.1),
                monai.transforms.RandShiftIntensityd(keys=image_keys, prob=0.5, offsets=0.1),
            ])
        
        transforms.append(ToTensord(keys=all_keys))
        
        return Compose(transforms)
    
    def __len__(self) -> int:
        return len(self.monai_dataset)
    
    def __getitem__(self, idx: int) -> Dict:
        """Get a sample from the dataset."""
        # Get transformed data from MONAI dataset
        data_dict = self.monai_dataset[idx]
        
        # Always return consistent format: all CT sequences + label
        # The model will handle which sequences to use based on fusion strategy
        result = {'label': data_dict['label']}
        for ct_type in self.ct_types:
            result[ct_type] = data_dict[ct_type]
        
        return result
    
    def get_label_distribution(self) -> Dict[int, int]:
        """Get distribution of labels in the dataset."""
        label_counts = {}
        for item in self.data_list:
            label = item['label']
            label_counts[label] = label_counts.get(label, 0) + 1
        return label_counts


class MONAITumorDataLoader:
    """
    Simple data loader for MONAI dataset.
    """
    
    def __init__(
        self,
        data_root: str,
        batch_size: int = 4,
        num_workers: int = 4,
        target_size: Tuple[int, int, int] = (128, 128, 128),
        fusion_strategy: str = 'intermediate',
        ct_types: List[str] = ['A', 'D', 'N', 'V'],
        **dataset_kwargs
    ):
        self.data_root = data_root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.target_size = target_size
        self.fusion_strategy = fusion_strategy
        self.ct_types = ct_types
        self.dataset_kwargs = dataset_kwargs
    
    def create_train_loader(self) -> torch.utils.data.DataLoader:
        """Create training data loader."""
        json_path = os.path.join(self.data_root, 'training_manifest.json')
        
        dataset = MONAITumorDataset(
            json_path=json_path,
            ct_types=self.ct_types,
            fusion_strategy=self.fusion_strategy,
            target_size=self.target_size,
            is_training=True,  # Enable augmentation for training
            **self.dataset_kwargs
        )
        
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def create_internal_val_loader(self) -> torch.utils.data.DataLoader:
        """Create internal validation data loader."""
        json_path = os.path.join(self.data_root, 'internal_test_manifest.json')
        
        dataset = MONAITumorDataset(
            json_path=json_path,
            ct_types=self.ct_types,
            fusion_strategy=self.fusion_strategy,
            target_size=self.target_size,
            is_training=False,  # No augmentation for validation
            **self.dataset_kwargs
        )
        
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def create_external_test_loader(self) -> torch.utils.data.DataLoader:
        """Create external test data loader."""
        json_path = os.path.join(self.data_root, 'external_test_manifest.json')
        
        dataset = MONAITumorDataset(
            json_path=json_path,
            ct_types=self.ct_types,
            fusion_strategy=self.fusion_strategy,
            target_size=self.target_size,
            is_training=False,  # No augmentation for testing
            **self.dataset_kwargs
        )
        
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )


# Simple test
if __name__ == '__main__':
    data_root = "Data"
    
    # Test dataset with consistent format
    print("\nTesting MONAI Dataset:")
    
    try:
        dataset = MONAITumorDataset(
            json_path=os.path.join(data_root, 'training_manifest.json'),
            ct_types=['A', 'D', 'N', 'V'],
            fusion_strategy='intermediate',
            target_size=(64, 64, 64),  # Smaller for testing
            normalize=False,
            is_training=False,  # No augmentation for testing
            verbose=True
        )
        
        print(f"Dataset length: {len(dataset)}")
        
        # Test first sample
        sample_dict = dataset[0]
        print(f"Label: {sample_dict['label']}")
        print(f"Keys: {list(sample_dict.keys())}")
        print(f"Number of sequences: {len([k for k in sample_dict.keys() if k in ['A', 'D', 'N', 'V']])}")
        print(f"Each sequence shape: {sample_dict['A'].shape}")
        
        # Test training mode with augmentation
        print("\nTesting training mode with augmentation:")
        train_dataset = MONAITumorDataset(
            json_path=os.path.join(data_root, 'training_manifest.json'),
            ct_types=['A', 'D', 'N', 'V'],
            fusion_strategy='intermediate',
            target_size=(64, 64, 64),
            normalize=False,
            is_training=True,  # Enable augmentation
            verbose=True
        )
        
        train_sample = train_dataset[0]
        print(f"Training sample keys: {list(train_sample.keys())}")
        print(f"Training sample shape: {train_sample['A'].shape}")
                
    except Exception as e:
        print(f"Error testing dataset: {str(e)}")
    
    # Test data loader
    print("\nTesting DataLoader:")
    loader = MONAITumorDataLoader(
        data_root=data_root,
        batch_size=16,
        target_size=(64, 64, 64),
        fusion_strategy='intermediate'
    )
    
    try:
        train_loader = loader.create_train_loader()
        print(f"Train loader created with {len(train_loader)} batches")
        
        # Test one batch
        for batch_idx, batch_dict in enumerate(train_loader):
            print(f"Batch {batch_idx}:")
            print(f"  - Keys: {list(batch_dict.keys())}")
            print(f"  - Labels: {batch_dict['label']}")
            print(f"  - Number of sequences: {len([k for k in batch_dict.keys() if k in ['A', 'D', 'N', 'V']])}")
            print(f"  - Each sequence shape: {batch_dict['A'].shape}")
            print(f"  - Label distribution: {batch_dict['label'].bincount()}")
            break
    except Exception as e:
        print(f"Error testing data loader: {str(e)}") 