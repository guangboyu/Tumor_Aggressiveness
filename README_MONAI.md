# MONAI Dataset Implementation

This document explains how to use the new MONAI-based dataset implementation for tumor aggressiveness classification.

## Overview

The new implementation uses MONAI's efficient data loading and preprocessing pipeline with JSON manifest files for better performance and maintainability.

## Features

- **JSON Manifest Files**: Pre-generated file path lists for efficient data loading
- **MONAI Transforms**: Optimized preprocessing pipeline with caching support
- **Multiple Fusion Strategies**: Intermediate, single, and ensemble fusion (early fusion removed for unregistered modalities)
- **VOI Masking**: Automatic application of volume of interest masks
- **Caching**: Optional in-memory caching for faster training

## Quick Start

### 1. Generate JSON Manifests

First, generate JSON manifest files that contain all file paths and labels:

```bash
# Generate manifests for all datasets
python data_process.py --generate_json_only --dataset all

# Or generate for specific dataset
python data_process.py --generate_json_only --dataset training
```

This creates:
- `Data/training_manifest.json`
- `Data/internal_test_manifest.json` 
- `Data/external_test_manifest.json`

### 2. Use MONAI Dataset

```python
from monai_dataset import MONAITumorDataset, MONAITumorDataLoader

# Create dataset
dataset = MONAITumorDataset(
    json_path="Data/training_manifest.json",
    ct_types=['A', 'D', 'N', 'V'],
    fusion_strategy='intermediate',
    target_size=(128, 128, 128),
    normalize=True,
    apply_voi_mask=True,  # Apply VOI masking
    verbose=True
)

# Create data loader
loader = MONAITumorDataLoader(
    data_root="Data",
    batch_size=4,
    target_size=(128, 128, 128),
    fusion_strategy='intermediate',
    cache_rate=0.1
)

train_loader = loader.create_train_loader()
val_loader = loader.create_internal_val_loader()
```

## JSON Manifest Structure

Each JSON file contains a list of dictionaries with the following structure:

```json
[
  {
    "label": 0,
    "patient_id": "P1548715",
    "pinyin": "AI_JIU_GEN",
    "A": "Data/data_nifty/1.Training_DICOM_603/AI_JIU_GEN_P1548715/CT/A_8/A_8_02_shenAgioRoutine_20170728165801_6.nii.gz",
    "D": "Data/data_nifty/1.Training_DICOM_603/AI_JIU_GEN_P1548715/CT/D_8/D_8_02_shenAgioRoutine_20170728165801_6.nii.gz",
    "N": "Data/data_nifty/1.Training_DICOM_603/AI_JIU_GEN_P1548715/CT/N_8/N_8_02_shenAgioRoutine_20170728165801_6.nii.gz",
    "V": "Data/data_nifty/1.Training_DICOM_603/AI_JIU_GEN_P1548715/CT/V_8/V_8_02_shenAgioRoutine_20170728165801_6.nii.gz",
    "mask": "Data/VOI_nifty/1.Training_VOI_603/AI_JIU_GEN_P1548715/VOI/A.nii.gz"
  }
]
```

## Fusion Strategies

**Note**: Early fusion has been removed since CT modalities (A, D, N, V) are not registered to each other. Concatenating unregistered modalities as channels would be incorrect.

### Intermediate Fusion
```python
# Returns list of separate tensors for custom fusion in the model
dataset = MONAITumorDataset(
    json_path="Data/training_manifest.json",
    fusion_strategy='intermediate'
)
# Output: [(1, D, H, W), (1, D, H, W), (1, D, H, W), (1, D, H, W)]
# Each tensor represents one CT modality (A, D, N, V)
```

### Single Sequence
```python
# Uses only one CT sequence (first in the list)
dataset = MONAITumorDataset(
    json_path="Data/training_manifest.json",
    fusion_strategy='single'
)
# Output: (1, D, H, W) - only first CT type (A)
```

### Ensemble
```python
# Returns all sequences separately for ensemble training
dataset = MONAITumorDataset(
    json_path="Data/training_manifest.json",
    fusion_strategy='ensemble'
)
# Output: [(1, D, H, W), (1, D, H, W), (1, D, H, W), (1, D, H, W)]
# Train separate models for each modality and combine predictions
```

## MONAI Transforms

The dataset applies simple, essential transforms:

1. **LoadImaged**: Load medical images and VOI masks
2. **EnsureChannelFirstd**: Ensure channel dimension is first
3. **Orientationd**: Standardize orientation to RAS
4. **Spacingd**: Resample to uniform spacing (1mm isotropic)
5. **ScaleIntensityRanged**: Clip HU values to [-1000, 1000] and normalize to [0, 1]
6. **MaskIntensityd**: Apply VOI mask to CT images (if enabled)
7. **Resized**: Resize to target dimensions
8. **ToTensord**: Convert to PyTorch tensors

## Performance Optimization

### Multi-worker Loading
```python
# Use multiple workers for data loading
loader = MONAITumorDataLoader(
    data_root="Data",
    num_workers=8,  # Adjust based on your system
    batch_size=4
)
```

## Testing

Run the test script to verify everything works:

```bash
python test_json_generation.py
```

This will:
1. Generate JSON manifests
2. Test MONAI dataset loading
3. Show sample data structures
4. Verify label distributions

## Migration from Old Dataset

To migrate from the old `dataset.py` to the new MONAI implementation:

1. **Generate JSON manifests**:
   ```bash
   python data_process.py --generate_json_only --dataset all
   ```

2. **Update imports**:
   ```python
   # Old
   from dataset import TumorAggressivenessDataset, TumorAggressivenessDataLoader
   
   # New
   from monai_dataset import MONAITumorDataset, MONAITumorDataLoader
   ```

3. **Update dataset creation**:
   ```python
   # Old
   dataset = TumorAggressivenessDataset(
       csv_path="Data/ccRCC_Survival_Analysis_Dataset_english/training_set_603_cases.csv",
       ct_root="Data/data_nifty/1.Training_DICOM_603",
       voi_root="Data/ROI/1.Training_ROI_603"
   )
   
   # New
   dataset = MONAITumorDataset(
       json_path="Data/training_manifest.json"
   )
   ```

## Troubleshooting

### NumPy Version Issues
If you encounter NumPy compatibility issues:
```bash
pip install "numpy<2"
```

### Missing JSON Files
If JSON manifests don't exist:
```bash
python data_process.py --generate_json_only --dataset all
```

### MONAI Installation
If MONAI is not installed:
```bash
pip install monai
```

## Benefits

1. **Faster Training**: Optimized data loading with caching
2. **Better Memory Management**: Efficient preprocessing pipeline
3. **Standardized Transforms**: MONAI's battle-tested medical image transforms
4. **Easy Maintenance**: JSON manifests make data organization clear
5. **Flexible Fusion**: Support for all fusion strategies
6. **Robust Error Handling**: Better error messages and validation 