import json
import os
import torch
from torch.utils.data import Dataset
import monai
from monai.transforms import (
    Compose,
    LoadImaged,
    AddChanneld,
    Orientationd,
    Spacingd,
    ScaleIntensityRanged,
    CropForegroundd,
    SpatialPadd,
    Resized,
    ToTensord,
    NormalizeIntensityd,
    MaskIntensityd,
)
from typing import Dict, List, Optional, Tuple, Union
import logging

logger = logging.getLogger(__name__)

class MONAITumorDataset(Dataset):
    """
    MONAI-compatible dataset for tumor aggressiveness classification.
    Uses JSON manifest files for efficient data loading.
    """
    
    def __init__(
        self,
        json_path: str,
        ct_types: List[str] = ['A', 'D', 'N', 'V'],
        fusion_strategy: str = 'intermediate',
        target_size: Tuple[int, int, int] = (128, 128, 128),
        normalize: bool = True,
        apply_voi_mask: bool = True,
        cache_rate: float = 0.0,
        num_workers: int = 4,
        verbose: bool = True
    ):
        """
        Args:
            json_path: Path to JSON manifest file
            ct_types: List of CT sequence types to use
            fusion_strategy: 'intermediate', 'single', or 'ensemble' (early fusion removed for unregistered modalities)
            target_size: Target size for resampling (D, H, W)
            normalize: Whether to normalize CT values
            apply_voi_mask: Whether to apply VOI mask to CT
            cache_rate: Fraction of data to cache in memory
            num_workers: Number of workers for data loading
            verbose: Whether to print progress information
        """
        self.json_path = json_path
        self.ct_types = ct_types
        self.fusion_strategy = fusion_strategy
        self.target_size = target_size
        self.normalize = normalize
        self.apply_voi_mask = apply_voi_mask
        self.cache_rate = cache_rate
        self.num_workers = num_workers
        self.verbose = verbose
        
        # Load JSON manifest
        with open(json_path, 'r') as f:
            self.data_list = json.load(f)
        
        if verbose:
            logger.info(f"Loaded {len(self.data_list)} samples from {json_path}")
            logger.info(f"CT types: {ct_types}")
            logger.info(f"Fusion strategy: {fusion_strategy}")
        
        # Create MONAI transforms
        self.transforms = self._create_transforms()
        
        # Create MONAI dataset
        self.monai_dataset = monai.data.Dataset(
            data=self.data_list,
            transform=self.transforms,
            cache_rate=cache_rate,
            num_workers=num_workers
        )
    
    def _create_transforms(self) -> Compose:
        """Create MONAI transforms for preprocessing."""
        keys = []
        
        # Add CT sequence keys
        for ct_type in self.ct_types:
            keys.append(ct_type)
        
        # Add mask key if using VOI masking
        if self.apply_voi_mask:
            keys.append('mask')
        
        # Define transforms
        transforms = [
            LoadImaged(keys=keys),
            AddChanneld(keys=keys),
            Orientationd(keys=keys, axcodes="RAS"),
            Spacingd(
                keys=keys,
                pixdim=(1.0, 1.0, 1.0),
                mode=("bilinear", "bilinear", "bilinear", "nearest") if self.apply_voi_mask else ("bilinear", "bilinear", "bilinear")
            ),
        ]
        
        # Add intensity normalization for CT sequences
        if self.normalize:
            for ct_type in self.ct_types:
                transforms.extend([
                    ScaleIntensityRanged(
                        keys=[ct_type],
                        a_min=-1000,
                        a_max=1000,
                        b_min=0.0,
                        b_max=1.0,
                        clip=True,
                    ),
                    NormalizeIntensityd(keys=[ct_type], nonzero=True, channel_wise=True),
                ])
        
        # Add VOI masking if requested
        if self.apply_voi_mask:
            for ct_type in self.ct_types:
                transforms.append(
                    MaskIntensityd(keys=[ct_type], mask_key="mask")
                )
        
        # Add spatial transforms
        transforms.extend([
            CropForegroundd(keys=keys, source_key=keys[0]),
            SpatialPadd(keys=keys, spatial_size=self.target_size),
            Resized(keys=keys, spatial_size=self.target_size),
            ToTensord(keys=keys),
        ])
        
        return Compose(transforms)
    
    def __len__(self) -> int:
        return len(self.monai_dataset)
    
    def __getitem__(self, idx: int) -> Union[torch.Tensor, Tuple[torch.Tensor, int], Tuple[List[torch.Tensor], int]]:
        """Get a sample from the dataset."""
        # Get transformed data from MONAI dataset
        data_dict = self.monai_dataset[idx]
        
        # Extract label
        label = data_dict['label']
        
        # Apply fusion strategy
        if self.fusion_strategy == 'intermediate':
            # Return as list for custom fusion in model
            tensors = []
            for ct_type in self.ct_types:
                tensors.append(data_dict[ct_type])
            
            return tensors, label
        
        elif self.fusion_strategy == 'single':
            # Use only first CT type
            ct_type = self.ct_types[0]
            return data_dict[ct_type], label
        
        elif self.fusion_strategy == 'ensemble':
            # Return all sequences separately for ensemble training
            tensors = []
            for ct_type in self.ct_types:
                tensors.append(data_dict[ct_type])
            
            return tensors, label
        
        else:
            raise ValueError(f"Unsupported fusion strategy: {self.fusion_strategy}. Use 'intermediate', 'single', or 'ensemble'")
    
    def get_label_distribution(self) -> Dict[int, int]:
        """Get distribution of labels in the dataset."""
        label_counts = {}
        for item in self.data_list:
            label = item['label']
            label_counts[label] = label_counts.get(label, 0) + 1
        return label_counts


class MONAITumorDataLoader:
    """
    Convenience class for creating MONAI-based train/val/test data loaders.
    """
    
    def __init__(
        self,
        data_root: str,
        batch_size: int = 4,
        num_workers: int = 4,
        target_size: Tuple[int, int, int] = (128, 128, 128),
        fusion_strategy: str = 'intermediate',
        ct_types: List[str] = ['A', 'D', 'N', 'V'],
        cache_rate: float = 0.0,
        **dataset_kwargs
    ):
        """
        Args:
            data_root: Root directory containing all data
            batch_size: Batch size for data loaders
            num_workers: Number of workers for data loading
            target_size: Target size for resampling
            fusion_strategy: Fusion strategy for CT sequences ('intermediate', 'single', or 'ensemble')
            ct_types: CT sequence types to use
            cache_rate: Fraction of data to cache in memory
            **dataset_kwargs: Additional arguments for dataset
        """
        self.data_root = data_root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.target_size = target_size
        self.fusion_strategy = fusion_strategy
        self.ct_types = ct_types
        self.cache_rate = cache_rate
        self.dataset_kwargs = dataset_kwargs
    
    def create_train_loader(self) -> torch.utils.data.DataLoader:
        """Create training data loader."""
        json_path = os.path.join(self.data_root, 'training_manifest.json')
        
        dataset = MONAITumorDataset(
            json_path=json_path,
            ct_types=self.ct_types,
            fusion_strategy=self.fusion_strategy,
            target_size=self.target_size,
            cache_rate=self.cache_rate,
            num_workers=self.num_workers,
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
            cache_rate=self.cache_rate,
            num_workers=self.num_workers,
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
            cache_rate=self.cache_rate,
            num_workers=self.num_workers,
            **self.dataset_kwargs
        )
        
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )


# Example usage
if __name__ == '__main__':
    # Test the MONAI dataset
    data_root = "Data"
    
    # Test different fusion strategies (removed early fusion for unregistered modalities)
    strategies = ['intermediate', 'single', 'ensemble']
    
    for strategy in strategies:
        print(f"\nTesting {strategy} fusion strategy:")
        
        try:
            dataset = MONAITumorDataset(
                json_path=os.path.join(data_root, 'training_manifest.json'),
                ct_types=['A', 'D', 'N', 'V'],
                fusion_strategy=strategy,
                target_size=(64, 64, 64),  # Smaller size for testing
                verbose=True
            )
            
            print(f"Dataset length: {len(dataset)}")
            
            # Test first sample
            sample, label = dataset[0]
            print(f"Label: {label}")
            
            if strategy == 'intermediate':
                print(f"Number of sequences: {len(sample)}")
                print(f"Each sequence shape: {sample[0].shape}")  # Should be (1, 64, 64, 64)
            elif strategy == 'single':
                print(f"Sample shape: {sample.shape}")  # Should be (1, 64, 64, 64)
            elif strategy == 'ensemble':
                print(f"Number of sequences: {len(sample)}")
                print(f"Each sequence shape: {sample[0].shape}")  # Should be (1, 64, 64, 64)
                
        except Exception as e:
            print(f"Error testing {strategy}: {str(e)}")
    
    # Test data loader
    print("\nTesting MONAI DataLoader:")
    loader = MONAITumorDataLoader(
        data_root=data_root,
        batch_size=2,
        target_size=(64, 64, 64),
        fusion_strategy='intermediate'
    )
    
    try:
        train_loader = loader.create_train_loader()
        print(f"Train loader created with {len(train_loader)} batches")
        
        # Test one batch
        for batch_idx, (data, labels) in enumerate(train_loader):
            print(f"Batch {batch_idx}: data shape {data.shape}, labels {labels}")
            break
    except Exception as e:
        print(f"Error testing data loader: {str(e)}") 