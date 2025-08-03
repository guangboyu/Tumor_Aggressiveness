import os
import json
import torch
import logging
from typing import List, Tuple, Dict, Any

from torch.utils.data import Dataset
from monai.data import SmartCacheDataset, DataLoader, CacheDataset
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    Spacingd,
    ScaleIntensityRanged,
    ScaleIntensity,
    MaskIntensityd,
    CropForegroundd,
    ResizeWithPadOrCropd,
    RandFlipd,
    RandRotate90d,
    ToTensord,
    MapTransform,
)

# --- Setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SelectLargestMaskSlice(MapTransform):
    """
    A MONAI transform to select the 2D slice with the largest mask area from a 3D volume.
    This is applied to all image keys in the dictionary.
    """
    def __init__(self, keys: List[str], mask_key: str):
        super().__init__(keys)
        self.mask_key = mask_key

    def __call__(self, data):
        d = dict(data)
        mask = d[self.mask_key]
        
        # Sum the mask over the height and width to find the slice with the most pixels
        # Assuming shape is (C, H, W, D)
        slice_sums = torch.sum(mask, dim=(1, 2))
        largest_slice_idx = torch.argmax(slice_sums)
        
        # Select this slice for all specified keys
        for key in self.keys:
            if key in d:
                # Squeeze the channel dimension for 2D processing
                d[key] = d[key][:, :, :, largest_slice_idx].squeeze(0)
        return d


class MONAITumorDataset(Dataset):
    """
    A simple MONAI-based dataset for 3D CT classification.
    This class is a pure data source. All transforms, caching, and batching
    are handled by the MONAITumorDataLoader.
    """
    def __init__(self, json_path: str):
        with open(json_path, 'r') as f:
            self.data_list = json.load(f)
        logger.info(f"Loaded {len(self.data_list)} data items from {json_path}")

    def __len__(self) -> int:
        return len(self.data_list)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        # This dataset simply returns the dictionary of file paths
        return self.data_list[idx]


class MONAITumorDataLoader:
    """
    A data loader factory that creates data loaders for training and validation,
    using SmartCacheDataset for efficient, augmented training.
    This version is compatible with older MONAI APIs.
    """
    def __init__(
        self,
        data_root: str,
        batch_size: int = 4,
        num_workers: int = 4,
        replace_rate: float = 1.0, # Use replace_rate for older MONAI versions
        use_2d_slices: bool = False,
        **dataset_kwargs
    ):
        self.data_root = data_root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.replace_rate = replace_rate
        self.use_2d_slices = use_2d_slices
        self.dataset_params = dataset_kwargs

    def _get_transforms(self, is_train: bool):
        """Defines the full transform pipeline."""
        ct_types = self.dataset_params.get('ct_types', [])
        apply_voi_mask = self.dataset_params.get('apply_voi_mask', False)
        target_size = self.dataset_params.get('target_size', (96, 96, 96))

        image_keys = ct_types.copy()
        all_keys = image_keys.copy()
        if apply_voi_mask:
            all_keys.append('mask')

        resample_modes = ['bilinear'] * len(image_keys)
        if apply_voi_mask:
            resample_modes.append('nearest')

        # --- Define the full pipeline ---
        transforms = [
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
        if apply_voi_mask:
            transforms.append(MaskIntensityd(keys=image_keys, mask_key="mask"))

        if is_train:
            # Add random augmentations for training
            transforms.extend([
                ResizeWithPadOrCropd(keys=all_keys, spatial_size=target_size, method="symmetric"),
                # RandFlipd(keys=all_keys, prob=0.5, spatial_axis=0),
                # RandFlipd(keys=all_keys, prob=0.5, spatial_axis=1),
                # RandRotate90d(keys=all_keys, prob=0.5, max_k=3, spatial_axes=(0, 1)),
            ])
        else:
            # For validation, just resize
            transforms.append(
                ResizeWithPadOrCropd(keys=all_keys, spatial_size=target_size, method="symmetric")
            )

        transforms.append(ToTensord(keys=all_keys))
        return Compose(transforms)


    def get_train_loader(self) -> DataLoader:
        """Creates the training data loader with Smart Caching."""
        json_path = os.path.join(self.data_root, "training_manifest.json")
        
        train_ds = MONAITumorDataset(json_path=json_path)
        train_transforms = self._get_transforms(is_train=True)
        
        # CORRECTED: Removed the 'num_workers' argument from the constructor
        # for compatibility with older MONAI versions.
        smart_cache_ds = SmartCacheDataset(
            data=train_ds.data_list,
            transform=train_transforms,
            replace_rate=self.replace_rate,
            cache_num=len(train_ds)
        )

        # The DataLoader's num_workers will handle parallel processing.
        return DataLoader(
            smart_cache_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers= 0,
            pin_memory=torch.cuda.is_available()
        )

    def get_val_loader(self) -> DataLoader:
        """Creates the validation data loader with caching."""
        json_path = os.path.join(self.data_root, "internal_test_manifest.json")
        
        val_ds = MONAITumorDataset(json_path=json_path)
        val_transforms = self._get_transforms(is_train=False)
        
        # CORRECTED: Removed the 'num_workers' argument from the constructor
        val_cache_ds = CacheDataset(
            data=val_ds.data_list,
            transform=val_transforms,
            cache_rate=1.0
        )
        
        return DataLoader(
            val_cache_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers = 0,
            pin_memory=torch.cuda.is_available()
        )

