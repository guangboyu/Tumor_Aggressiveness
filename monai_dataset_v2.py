import json
import torch
import logging
import os
from typing import List, Tuple, Dict, Any

from torch.utils.data import Dataset
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    Spacingd,
    ScaleIntensityRanged,
    MaskIntensityd,
    CropForegroundd,
    ResizeWithPadOrCropd,
    RandFlipd,
    RandRotate90d,
    ToTensord,
)

# Set up a logger for clean informational messages
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MONAITumorDataset(Dataset):
    """
    A robust MONAI-based dataset for 3D CT classification.

    This class inherits from torch.utils.data.Dataset and implements a standard,
    augmented preprocessing pipeline for training and validation.

    Args:
        json_path (str): Path to the JSON manifest file.
        ct_types (List[str]): List of CT sequence keys to load (e.g., ['A', 'V']).
        target_size (Tuple[int, int, int]): The target spatial size for model input.
        apply_voi_mask (bool): Whether to apply the VOI mask to the CT images.
        is_train (bool): If True, applies data augmentations. Set to False for validation/testing.
    """

    def __init__(
        self,
        json_path: str,
        ct_types: List[str] = ['A', 'D', 'N', 'V'],
        target_size: Tuple[int, int, int] = (128, 128, 128),
        apply_voi_mask: bool = True,
        is_train: bool = True,
    ):
        self.ct_types = ct_types
        self.target_size = target_size
        self.apply_voi_mask = apply_voi_mask
        self.is_train = is_train

        # Load the list of data dictionaries from the JSON manifest
        with open(json_path, 'r') as f:
            self.data_list = json.load(f)

        logger.info(f"Loaded {len(self.data_list)} samples from {json_path}")

        # Create the appropriate transform pipeline (training vs. validation)
        self.transforms = self._create_transforms()

    def __len__(self) -> int:
        """Returns the total number of samples in the dataset."""
        return len(self.data_list)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Gets a single data item from the dataset.

        Args:
            idx (int): The index of the data item.

        Returns:
            A dictionary containing the transformed image tensors and the label.
        """
        # The MONAI transforms will be applied to the dictionary from data_list
        return self.transforms(self.data_list[idx])

    def _create_transforms(self) -> Compose:
        """Creates the MONAI transform pipeline."""
        image_keys = self.ct_types.copy()
        all_keys = image_keys.copy()
        if self.apply_voi_mask:
            all_keys.append('mask')

        # Define separate interpolation modes for images ('bilinear') and masks ('nearest')
        resample_modes = ['bilinear'] * len(image_keys)
        if self.apply_voi_mask:
            resample_modes.append('nearest')

        # ---- Define the base transforms for both training and validation ----
        base_transforms = [
            LoadImaged(keys=all_keys),
            EnsureChannelFirstd(keys=all_keys),
            Orientationd(keys=all_keys, axcodes="RAS"),
            Spacingd(keys=all_keys, pixdim=(1.0, 1.0, 1.0), mode=resample_modes),
            ScaleIntensityRanged(
                keys=image_keys, a_min=-1000, a_max=1000,
                b_min=0.0, b_max=1.0, clip=True
            ),
            CropForegroundd(keys=all_keys, source_key=image_keys[0]),
        ]

        if self.apply_voi_mask:
            base_transforms.append(MaskIntensityd(keys=image_keys, mask_key="mask"))

        # ---- Define augmentations for the training set ----
        if self.is_train:
            augmentations = [
                ResizeWithPadOrCropd(keys=all_keys, spatial_size=self.target_size, method="symmetric"),
                RandFlipd(keys=all_keys, prob=0.5, spatial_axis=0), # Sagittal
                RandFlipd(keys=all_keys, prob=0.5, spatial_axis=1), # Coronal
                RandRotate90d(keys=all_keys, prob=0.5, max_k=3, spatial_axes=(0, 1)), # Axial
            ]
        else:
            # For validation, just resize without random transformations
            augmentations = [
                ResizeWithPadOrCropd(keys=all_keys, spatial_size=self.target_size, method="symmetric"),
            ]

        # ---- Combine all transforms and convert to tensor ----
        final_transforms = base_transforms + augmentations + [ToTensord(keys=all_keys)]
        
        return Compose(final_transforms)

    def get_label_distribution(self) -> Dict[int, int]:
        """Calculates and returns the distribution of labels in the dataset."""
        label_counts = {}
        for item in self.data_list:
            label = item['label']
            label_counts[label] = label_counts.get(label, 0) + 1
        return label_counts


class MONAITumorDataLoader:
    """
    A refactored, efficient data loader factory for the MONAITumorDataset.
    It creates data loaders for training, validation, and testing.
    """
    def __init__(
        self,
        data_root: str,
        batch_size: int = 4,
        num_workers: int = 4,
        target_size: Tuple[int, int, int] = (128, 128, 128),
        ct_types: List[str] = ['A', 'D', 'N', 'V'],
        **dataset_kwargs
    ):
        """
        Args:
            data_root (str): The root directory containing the JSON manifest files.
            batch_size (int): Number of samples per batch.
            num_workers (int): Number of subprocesses to use for data loading.
            target_size (Tuple[int, int, int]): Target spatial size for model input.
            ct_types (List[str]): List of CT sequence keys to load.
            **dataset_kwargs: Additional keyword arguments to pass to MONAITumorDataset.
                               Example: apply_voi_mask=True
        """
        self.data_root = data_root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset_params = {
            "ct_types": ct_types,
            "target_size": target_size,
            **dataset_kwargs
        }

    def _create_loader(self, manifest_name: str, is_train: bool) -> torch.utils.data.DataLoader:
        """Private helper method to create a DataLoader, removing code duplication."""
        json_path = os.path.join(self.data_root, manifest_name)
        
        dataset = MONAITumorDataset(
            json_path=json_path,
            is_train=is_train,
            **self.dataset_params
        )
        print(f"{json_path} dataset length: ", len(dataset))
        
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=is_train, # Shuffle only if it's a training loader
            num_workers=self.num_workers,
            pin_memory=True
        )

    def get_train_loader(self) -> torch.utils.data.DataLoader:
        """Creates the training data loader."""
        return self._create_loader("training_manifest.json", is_train=True)

    def get_val_loader(self) -> torch.utils.data.DataLoader:
        """Creates the internal validation data loader."""
        return self._create_loader("internal_test_manifest.json", is_train=False)

    def get_test_loader(self) -> torch.utils.data.DataLoader:
        """Creates the external test data loader."""
        return self._create_loader("external_test_manifest.json", is_train=False)


# # --- UPDATED TEST SCRIPT ---
# if __name__ == '__main__':
#     # This block now uses the revised MONAITumorDataset and the refactored DataLoader
#     # Note: This test requires the revised MONAITumorDataset class to be defined in the same file
#     # or imported correctly. It will fail if dummy paths don't exist.

#     data_root = "Data" 
#     print("\n--- Testing MONAI DataLoader Factory ---")
    
#     # Instantiate the refactored loader factory
#     loader_factory = MONAITumorDataLoader(
#         data_root=data_root,
#         batch_size=16, # Use batch size of 1 for simple testing
#         target_size=(64, 64, 64),
#         ct_types=['A', 'D', 'N', 'V'],
#         apply_voi_mask=True # Example of passing a dataset_kwarg
#     )
#     print(len)
    
#     try:
#         # Get the training loader
#         train_loader = loader_factory.get_train_loader()
#         print(f"Train loader created successfully.")
#         # Test one batch
#         for batch_idx, batch_dict in enumerate(train_loader):
#             print(f"Batch {batch_idx}:")
#             print(f"  - Keys: {list(batch_dict.keys())}")
#             print(f"  - Labels: {batch_dict['label']}")
#             print(f"  - Number of sequences: {len([k for k in batch_dict.keys() if k in ['A', 'D', 'N', 'V']])}")
#             print(f"  - Each sequence shape: {batch_dict['A'].shape}")
#             print(f"  - Label distribution: {batch_dict['label'].bincount()}")
#             break

#     except Exception as e:
#         logger.error(f"\nCould not run example. This is expected if dummy file paths do not exist.")
#         logger.error(f"Error: {e}")

#     try:
#         # Get the validation loader
#         val_loader = loader_factory.get_val_loader()
#         print(f"Validation loader created successfully.")
#         # Test one batch
#         for batch_idx, batch_dict in enumerate(val_loader):
#             print(f"Batch {batch_idx}:")
#             print(f"  - Keys: {list(batch_dict.keys())}")
#             print(f"  - Labels: {batch_dict['label']}")
#             print(f"  - Number of sequences: {len([k for k in batch_dict.keys() if k in ['A', 'D', 'N', 'V']])}")
#             print(f"  - Each sequence shape: {batch_dict['A'].shape}")
#             print(f"  - Label distribution: {batch_dict['label'].bincount()}")
#             break

#     except Exception as e:
#         logger.error(f"\nCould not run example. This is expected if dummy file paths do not exist.")
#         logger.error(f"Error: {e}")