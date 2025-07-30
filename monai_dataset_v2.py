import json
import torch
import logging
import os
from typing import List, Tuple, Dict, Any

from torch.utils.data import Dataset
from monai.data import CacheDataset, DataLoader, SmartCacheDataset
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
        print(f"Label distribution: {dataset.get_label_distribution()}")
        
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
    
class CacheMONAITumorDataLoader:
    """
    A data loader factory that creates data loaders for training and validation,
    with built-in support for caching the training set.
    """
    def __init__(
        self,
        data_root: str,
        batch_size: int = 4,
        num_workers: int = 4,
        cache_rate: float = 0.0, # Added cache_rate
        **dataset_kwargs
    ):
        self.data_root = data_root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.cache_rate = cache_rate
        self.dataset_params = dataset_kwargs

    def get_train_loader(self) -> DataLoader:
        """Creates the training data loader with optional caching."""
        json_path = os.path.join(self.data_root, "training_manifest.json")
        
        # Create the base dataset instance
        train_ds = MONAITumorDataset(
            json_path=json_path,
            is_train=True,
            **self.dataset_params
        )
        
        # Wrap with CacheDataset if a cache rate is specified
        if self.cache_rate > 0.0:
            logger.info(f"Using CacheDataset for training with cache_rate={self.cache_rate}. The first epoch will be slow.")
            train_ds = CacheDataset(
                data=train_ds,
                cache_rate=self.cache_rate,
                num_workers=self.num_workers
            )

        return DataLoader(
            train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available()
        )

    def get_val_loader(self) -> DataLoader:
        """Creates the validation data loader (no caching)."""
        json_path = os.path.join(self.data_root, "internal_test_manifest.json")
        
        val_ds = MONAITumorDataset(
            json_path=json_path,
            is_train=False,
            **self.dataset_params
        )
        
        return DataLoader(
            val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available()
        )