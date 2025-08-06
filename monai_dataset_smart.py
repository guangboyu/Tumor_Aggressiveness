import os
import json
import torch
import logging
from typing import List, Tuple, Dict, Any
import numpy as np
from scipy.ndimage import binary_dilation

from torch.utils.data import Dataset
from monai.data import SmartCacheDataset, DataLoader, CacheDataset, PersistentDataset
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
    MapTransform,
    Resized,
)

# --- Setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DilateMaskd(MapTransform):
    """
    A MONAI transform to dilate a binary mask using a spherical structuring element.
    This is useful for expanding the ROI to include surrounding tissue.
    """
    def __init__(self, keys: List[str], kernel_size: int):
        super().__init__(keys)
        self.kernel_size = kernel_size
        # Create a 3D spherical structuring element for dilation
        self.structuring_element = self._create_spherical_kernel(kernel_size)

    def _create_spherical_kernel(self, size: int):
        """Creates a 3D spherical structuring element for morphological operations."""
        if size % 2 == 0:
            size += 1  # Kernel size must be odd
        radius = size // 2
        x, y, z = np.ogrid[-radius:radius+1, -radius:radius+1, -radius:radius+1]
        return x**2 + y**2 + z**2 <= radius**2

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            if key in d:
                mask_tensor = d[key]
                # Assuming shape is (C, H, W, D), squeeze channel dim for processing
                mask_np = mask_tensor.squeeze(0).numpy()
                
                # Apply binary dilation
                dilated_mask_np = binary_dilation(mask_np, structure=self.structuring_element)
                
                # Convert back to tensor and add channel dimension back
                d[key] = torch.from_numpy(dilated_mask_np).unsqueeze(0)
        return d


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
        slice_sums = torch.sum(mask, dim=(1, 2))
        largest_slice_idx = torch.argmax(slice_sums)
        
        # Select this slice for all specified keys
        for key in self.keys:
            if key in d and isinstance(d[key], torch.Tensor):
                # Squeeze the channel dimension for 2D processing
                d[key] = d[key][:, :, :, largest_slice_idx]
        return d


class MONAITumorDataset(Dataset):
    """
    A simple data source class. It only loads the list of file paths.
    All transformations are handled by the DataLoader.
    """
    def __init__(self, json_path: str):
        with open(json_path, 'r') as f:
            self.data_list = json.load(f)
        logger.info(f"Loaded {len(self.data_list)} data items from {json_path}")

    def __len__(self) -> int:
        return len(self.data_list)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.data_list[idx]


class MONAITumorDataLoader:
    """
    A data loader factory that creates data loaders for training and validation,
    with support for phase-specific masks.
    """
    def __init__(
        self,
        data_root: str,
        batch_size: int = 4,
        num_workers: int = 4,
        replace_rate: float = 1.0,
        use_2d_slices: bool = False,
        dilate_roi_size: int = 0,
        **dataset_kwargs
    ):
        self.data_root = data_root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.replace_rate = replace_rate
        self.use_2d_slices = use_2d_slices
        self.dataset_params = dataset_kwargs
        self.dilate_roi_size = dilate_roi_size
        print(f"Dilate ROI size: {self.dilate_roi_size}")

    def _get_transforms(self, is_train: bool):
        """Defines the full transform pipeline."""
        ct_types = self.dataset_params.get('ct_types', [])
        apply_voi_mask = self.dataset_params.get('apply_voi_mask', False)
        target_size = self.dataset_params.get('target_size', (96, 96, 96))

        image_keys = ct_types.copy()
        mask_keys = [f"{key}_mask" for key in ct_types]
        all_keys = image_keys + mask_keys if apply_voi_mask else image_keys

        resample_modes = ['bilinear'] * len(image_keys)
        if apply_voi_mask:
            resample_modes += ['nearest'] * len(mask_keys)

        # --- Define the full pipeline ---
        transforms = [
            LoadImaged(keys=all_keys),
            EnsureChannelFirstd(keys=all_keys),
            Orientationd(keys=all_keys, axcodes="RAS"),
            Spacingd(keys=all_keys, pixdim=(1.0, 1.0, 1.0), mode=resample_modes),
            ScaleIntensityRanged(
                keys=image_keys, a_min=-500, a_max=500,
                b_min=0.0, b_max=1.0, clip=True
            ),
            CropForegroundd(keys=all_keys, source_key=image_keys[0]),
        ]
        
        if apply_voi_mask and self.dilate_roi_size > 0:
            logger.info(f"Applying mask dilation with kernel size: {self.dilate_roi_size}")
            transforms.append(DilateMaskd(keys=mask_keys, kernel_size=self.dilate_roi_size))

        if apply_voi_mask:
            for ct_type in ct_types:
                transforms.append(MaskIntensityd(keys=ct_type, mask_key=f"{ct_type}_mask"))

        if self.use_2d_slices:
            first_mask_key = mask_keys[0] if apply_voi_mask else image_keys[0]
            transforms.append(SelectLargestMaskSlice(keys=all_keys, mask_key=first_mask_key))

        if is_train:
            if self.use_2d_slices:
                transforms.extend([
                    Resized(keys=all_keys, spatial_size=target_size),
                    RandFlipd(keys=all_keys, prob=0.5, spatial_axis=0),
                    RandRotate90d(keys=all_keys, prob=0.5, max_k=3, spatial_axes=(0, 1)), # Explicitly 2D rotation
                ])
            else: # 3D training
                transforms.extend([
                    ResizeWithPadOrCropd(keys=all_keys, spatial_size=target_size, method="symmetric"),
                    RandFlipd(keys=all_keys, prob=0.5, spatial_axis=0),
                    RandFlipd(keys=all_keys, prob=0.5, spatial_axis=1),
                    RandRotate90d(keys=all_keys, prob=0.5, max_k=3, spatial_axes=(0, 1)),
                ])
        else: # Validation (not training)
            if self.use_2d_slices:
                transforms.append(Resized(keys=all_keys, spatial_size=target_size))
            else: # 3D validation
                transforms.append(ResizeWithPadOrCropd(keys=all_keys, spatial_size=target_size, method="symmetric"))


        transforms.append(ToTensord(keys=all_keys))
        return Compose(transforms)


    def get_train_loader(self) -> DataLoader:
        """Creates the training data loader with Smart Caching."""
        json_path = os.path.join(self.data_root, "training_manifest.json")
        
        train_ds = MONAITumorDataset(json_path=json_path)
        train_transforms = self._get_transforms(is_train=True)
        
        smart_cache_ds = SmartCacheDataset(
            data=train_ds.data_list,
            transform=train_transforms,
            replace_rate=self.replace_rate,
            cache_num=len(train_ds)
        )

        return DataLoader(
            smart_cache_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0, # Keep at 0 for Windows compatibility
            pin_memory=torch.cuda.is_available()
        )

    def get_val_loader(self) -> DataLoader:
        """Creates the validation data loader with caching."""
        json_path = os.path.join(self.data_root, "internal_test_manifest.json")
        
        val_ds = MONAITumorDataset(json_path=json_path)
        val_transforms = self._get_transforms(is_train=False)
        
        val_cache_ds = CacheDataset(
            data=val_ds.data_list,
            transform=val_transforms,
            cache_rate=1.0
        )
        
        return DataLoader(
            val_cache_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0, # Keep at 0 for Windows compatibility
            pin_memory=torch.cuda.is_available()
        )
