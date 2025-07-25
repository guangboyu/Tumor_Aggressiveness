import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import nibabel as nib
import nrrd
from typing import List, Dict, Optional, Tuple, Union
import warnings
from scipy import ndimage
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TumorAggressivenessDataset(Dataset):
    """
    Universal dataset for tumor aggressiveness classification supporting multiple fusion strategies.
    
    Fusion Strategies:
    - 'early': Concatenate all CT sequences as channels (4 channels: A, D, N, V)
    - 'intermediate': Load each sequence separately, return as list for custom fusion
    - 'single': Use only one specified CT sequence
    - 'ensemble': Return all sequences separately for ensemble training
    """
    
    def __init__(
        self,
        csv_path: str,
        ct_root: str,
        voi_root: str,
        ct_types: List[str] = ['A', 'D', 'N', 'V'],
        fusion_strategy: str = 'early',
        target_size: Tuple[int, int, int] = (128, 128, 128),
        normalize: bool = True,
        apply_voi_mask: bool = True,
        transform: Optional[callable] = None,
        cache_data: bool = False,
        verbose: bool = True
    ):
        """
        Args:
            csv_path: Path to CSV file with labels
            ct_root: Root directory for CT data
            voi_root: Root directory for VOI segmentation data
            ct_types: List of CT sequence types to use ['A', 'D', 'N', 'V']
            fusion_strategy: 'early', 'intermediate', 'single', or 'ensemble'
            target_size: Target size for resampling (D, H, W)
            normalize: Whether to normalize CT values
            apply_voi_mask: Whether to apply VOI mask to CT
            transform: Additional transforms to apply
            cache_data: Whether to cache loaded data in memory
            verbose: Whether to print progress information
        """
        self.data = pd.read_csv(csv_path)
        self.ct_root = ct_root
        self.voi_root = voi_root
        self.ct_types = ct_types
        self.fusion_strategy = fusion_strategy
        self.target_size = target_size
        self.normalize = normalize
        self.apply_voi_mask = apply_voi_mask
        self.transform = transform
        self.cache_data = cache_data
        self.verbose = verbose
        
        # Validate fusion strategy
        valid_strategies = ['early', 'intermediate', 'single', 'ensemble']
        if fusion_strategy not in valid_strategies:
            raise ValueError(f"fusion_strategy must be one of {valid_strategies}")
        
        # For single strategy, use first CT type
        if fusion_strategy == 'single' and len(ct_types) > 1:
            self.ct_types = [ct_types[0]]
            if verbose:
                logger.info(f"Single fusion strategy: using only {self.ct_types[0]} sequence")
        
        # Cache for storing loaded data
        self.data_cache = {} if cache_data else None
        
        # Statistics for normalization
        self.ct_stats = None
        
        if verbose:
            logger.info(f"Dataset initialized with {len(self.data)} samples")
            logger.info(f"CT types: {self.ct_types}")
            logger.info(f"Fusion strategy: {fusion_strategy}")
            logger.info(f"Target size: {target_size}")
    
    def _load_ct_sequence(self, patient_folder: str, ct_type: str) -> np.ndarray:
        """Load a single CT sequence for a given patient and sequence type."""
        if '1.Training' in self.ct_root:
            parent_folder = '1.Training_DICOM_603'
        elif '2.Internal' in self.ct_root:
            parent_folder = '2.Internal Test_DICOM_259'
        elif '3.External' in self.ct_root:
            parent_folder = '3.External Test_DICOM_308'
        else:
            parent_folder = self.ct_root  # fallback
        ct_base = os.path.join('Data', 'data_nifty', parent_folder, patient_folder, 'CT')
        if not os.path.exists(ct_base):
            raise FileNotFoundError(f"CT base path not found: {ct_base}")
        # Find any folder that matches ct_type_*
        ct_folders = [f for f in os.listdir(ct_base) if f.startswith(f"{ct_type}_") and os.path.isdir(os.path.join(ct_base, f))]
        if not ct_folders:
            raise FileNotFoundError(f"No CT folder found for type {ct_type} in {ct_base}")
        ct_path = os.path.join(ct_base, ct_folders[0])

        nii_files = [f for f in os.listdir(ct_path) if f.endswith('.nii.gz')]
        if not nii_files:
            raise FileNotFoundError(f"No NIfTI file found in {ct_path}")
        ct_file = os.path.join(ct_path, nii_files[0])
        ct_img = nib.load(ct_file).get_fdata()
        return np.array(ct_img)
    
    def _load_voi_mask(self, patient_folder: str, ct_type: str) -> np.ndarray:
        """Load VOI mask for a given patient and sequence type."""
        # Use the correct parent folder for ROI
        if '1.Training' in self.voi_root:
            parent_folder = '1.Training_ROI_603'
        elif '2.Internal' in self.voi_root:
            parent_folder = '2.Internal Test_ROI_259'
        elif '3.External' in self.voi_root:
            parent_folder = '3.External Test_ROI_308'
        else:
            parent_folder = self.voi_root  # fallback
        voi_path = os.path.join('Data', 'ROI', parent_folder, patient_folder, 'ROI')
        if not os.path.exists(voi_path):
            raise FileNotFoundError(f"VOI path not found: {voi_path}")
        # Find any file that matches ct_type_*.nrrd
        voi_files = [f for f in os.listdir(voi_path) if f.startswith(f"{ct_type}_") and f.endswith('.nrrd')]
        if not voi_files:
            # Try alternative naming pattern (just ct_type.nrrd)
            voi_files = [f for f in os.listdir(voi_path) if f == f"{ct_type}.nrrd"]
        if not voi_files:
            raise FileNotFoundError(f"VOI file not found for type {ct_type} in {voi_path}")
        voi_file = os.path.join(voi_path, voi_files[0])
        voi_img, _ = nrrd.read(voi_file)
        return np.array(voi_img)
    
    def _resample_volume(self, volume: np.ndarray, target_size: Tuple[int, int, int]) -> np.ndarray:
        """Resample volume to target size using trilinear interpolation."""
        if volume.shape == target_size:
            return volume
        
        # Calculate zoom factors
        zoom_factors = [target_size[i] / volume.shape[i] for i in range(3)]
        
        # Resample using scipy
        resampled = ndimage.zoom(volume, zoom_factors, order=1)
        
        return resampled
    
    def _normalize_ct(self, ct_volume: np.ndarray) -> np.ndarray:
        """Normalize CT values to [0, 1] range."""
        # Clip to reasonable HU range
        ct_volume = np.clip(ct_volume, -1000, 1000)
        
        # Normalize to [0, 1]
        ct_volume = (ct_volume + 1000) / 2000
        
        return ct_volume
    
    def _apply_voi_mask(self, ct_volume: np.ndarray, voi_mask: np.ndarray) -> np.ndarray:
        """Apply VOI mask to CT volume."""
        # Ensure same shape
        if ct_volume.shape != voi_mask.shape:
            voi_mask = self._resample_volume(voi_mask, ct_volume.shape)
        
        # Apply mask
        masked_ct = ct_volume * (voi_mask > 0)
        
        return masked_ct
    
    def _load_and_preprocess_sample(self, name_id: str, pinyin: str) -> Dict[str, np.ndarray]:
        """Load and preprocess all CT sequences and VOI masks for a sample."""
        sample_data = {}
        patient_folder = f"{pinyin}_{name_id}"
        for ct_type in self.ct_types:
            try:
                ct_volume = self._load_ct_sequence(patient_folder, ct_type)
                voi_mask = self._load_voi_mask(patient_folder, ct_type)
                if self.apply_voi_mask:
                    ct_volume = self._apply_voi_mask(ct_volume, voi_mask)
                ct_volume = self._resample_volume(ct_volume, self.target_size)
                if self.normalize:
                    ct_volume = self._normalize_ct(ct_volume)
                sample_data[ct_type] = ct_volume
            except Exception as e:
                if self.verbose:
                    logger.warning(f"Error loading {ct_type} for {patient_folder}: {str(e)}")
                sample_data[ct_type] = np.zeros(self.target_size)
        return sample_data
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Union[torch.Tensor, Tuple[torch.Tensor, int], Tuple[List[torch.Tensor], int]]:
        """Get a sample from the dataset."""
        row = self.data.iloc[idx]
        name_id = row['serial_number']
        pinyin = row['pinyin']
        label = int(row['aggressive_pathology_1_indolent_2_aggressive']) - 1  # Convert to 0/1
        
        # Check cache first
        if self.cache_data and name_id in self.data_cache:
            sample_data = self.data_cache[name_id]
        else:
            sample_data = self._load_and_preprocess_sample(name_id, pinyin)
            if self.cache_data:
                self.data_cache[name_id] = sample_data
        
        # Apply fusion strategy
        if self.fusion_strategy == 'early':
            # Concatenate all sequences as channels
            channels = []
            for ct_type in self.ct_types:
                channels.append(sample_data[ct_type])
            
            # Stack as channels (C, D, H, W)
            combined = np.stack(channels, axis=0)
            tensor_data = torch.tensor(combined, dtype=torch.float32)
            
            if self.transform:
                tensor_data = self.transform(tensor_data)
            
            return tensor_data, label
        
        elif self.fusion_strategy == 'intermediate':
            # Return as list for custom fusion in model
            tensors = []
            for ct_type in self.ct_types:
                tensor = torch.tensor(sample_data[ct_type], dtype=torch.float32).unsqueeze(0)  # (1, D, H, W)
                if self.transform:
                    tensor = self.transform(tensor)
                tensors.append(tensor)
            
            return tensors, label
        
        elif self.fusion_strategy == 'single':
            # Use only first CT type
            ct_type = self.ct_types[0]
            tensor_data = torch.tensor(sample_data[ct_type], dtype=torch.float32).unsqueeze(0)  # (1, D, H, W)
            
            if self.transform:
                tensor_data = self.transform(tensor_data)
            
            return tensor_data, label
        
        elif self.fusion_strategy == 'ensemble':
            # Return all sequences separately for ensemble training
            tensors = []
            for ct_type in self.ct_types:
                tensor = torch.tensor(sample_data[ct_type], dtype=torch.float32).unsqueeze(0)  # (1, D, H, W)
                if self.transform:
                    tensor = self.transform(tensor)
                tensors.append(tensor)
            
            return tensors, label
    
    def get_sample_info(self, idx: int) -> Dict:
        """Get information about a sample without loading the data."""
        row = self.data.iloc[idx]
        return {
            'name_id': row['serial_number'],
            'pinyin': row['pinyin'],
            'name': row['name'],
            'age': row['age'],
            'gender': row['gender'],
            'tumor_length_cm': row['tumor_length_cm'],
            'label': int(row['aggressive_pathology_1_indolent_2_aggressive']) - 1,
            'ct_types': self.ct_types
        }
    
    def get_label_distribution(self) -> Dict:
        """Get distribution of labels in the dataset."""
        labels = self.data['aggressive_pathology_1_indolent_2_aggressive'].values - 1
        unique, counts = np.unique(labels, return_counts=True)
        return dict(zip(unique, counts))
    
    def analyze_data_completeness(self) -> Dict:
        """Analyze the completeness of CT sequences across all patients."""
        completeness_stats = {
            'total_patients': len(self.data),
            'sequence_availability': {ct: 0 for ct in self.ct_types},
            'patients_with_all_sequences': 0,
            'patients_with_no_sequences': 0,
            'missing_sequences_by_patient': {}
        }
        
        for idx in range(len(self.data)):
            name_id = self.data.iloc[idx]['serial_number']
            pinyin = self.data.iloc[idx]['pinyin']
            patient_folder = f"{pinyin}_{name_id}"
            available_sequences = []
            
            for ct_type in self.ct_types:
                ct_path = os.path.join(self.ct_root, patient_folder, 'CT', f"{ct_type}_8")
                if os.path.exists(ct_path):
                    completeness_stats['sequence_availability'][ct_type] += 1
                    available_sequences.append(ct_type)
            
            if len(available_sequences) == len(self.ct_types):
                completeness_stats['patients_with_all_sequences'] += 1
            elif len(available_sequences) == 0:
                completeness_stats['patients_with_no_sequences'] += 1
                completeness_stats['missing_sequences_by_patient'][name_id] = 'ALL'
            else:
                missing = [ct for ct in self.ct_types if ct not in available_sequences]
                completeness_stats['missing_sequences_by_patient'][name_id] = missing
        
        return completeness_stats


class TumorAggressivenessDataLoader:
    """
    Convenience class for creating train/val/test data loaders.
    """
    
    def __init__(
        self,
        data_root: str,
        batch_size: int = 4,
        num_workers: int = 4,
        target_size: Tuple[int, int, int] = (128, 128, 128),
        fusion_strategy: str = 'early',
        ct_types: List[str] = ['A', 'D', 'N', 'V'],
        **dataset_kwargs
    ):
        """
        Args:
            data_root: Root directory containing all data
            batch_size: Batch size for data loaders
            num_workers: Number of workers for data loading
            target_size: Target size for resampling
            fusion_strategy: Fusion strategy for CT sequences
            ct_types: CT sequence types to use
            **dataset_kwargs: Additional arguments for dataset
        """
        self.data_root = data_root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.target_size = target_size
        self.fusion_strategy = fusion_strategy
        self.ct_types = ct_types
        self.dataset_kwargs = dataset_kwargs
    
    def create_train_loader(self) -> torch.utils.data.DataLoader:
        """Create training data loader."""
        dataset = TumorAggressivenessDataset(
            csv_path=os.path.join(self.data_root, 'ccRCC_Survival_Analysis_Dataset_english', 'training_set_603_cases.csv'),
            ct_root=os.path.join(self.data_root, 'data_nifty', '1.Training_DICOM_603'),
            voi_root=os.path.join(self.data_root, 'ROI', '1.Training_DICOM_603'),
            ct_types=self.ct_types,
            fusion_strategy=self.fusion_strategy,
            target_size=self.target_size,
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
        dataset = TumorAggressivenessDataset(
            csv_path=os.path.join(self.data_root, 'ccRCC_Survival_Analysis_Dataset_english', 'internal_test_set_259_cases.csv'),
            ct_root=os.path.join(self.data_root, 'data_nifty', '2.Internal Test_DICOM_259'),
            voi_root=os.path.join(self.data_root, 'ROI', '2.Internal Test_DICOM_259'),
            ct_types=self.ct_types,
            fusion_strategy=self.fusion_strategy,
            target_size=self.target_size,
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
        dataset = TumorAggressivenessDataset(
            csv_path=os.path.join(self.data_root, 'ccRCC_Survival_Analysis_Dataset_english', 'external_verification_set_308_cases.csv'),
            ct_root=os.path.join(self.data_root, 'data_nifty', '3.External Test_DICOM_308'),
            voi_root=os.path.join(self.data_root, 'ROI', '3.External Test_DICOM_308'),
            ct_types=self.ct_types,
            fusion_strategy=self.fusion_strategy,
            target_size=self.target_size,
            **self.dataset_kwargs
        )
        
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )


# # Example usage and testing
# if __name__ == "__main__":
#     # Test the dataset
#     data_root = "Data"
    
#     # Test different fusion strategies
#     strategies = ['early', 'intermediate', 'single', 'ensemble']
    
#     for strategy in strategies:
#         print(f"\nTesting {strategy} fusion strategy:")
        
#         dataset = TumorAggressivenessDataset(
#             csv_path=os.path.join(data_root, 'ccRCC_Survival_Analysis_Dataset_english', 'training_set_603_cases.csv'),
#             ct_root=os.path.join(data_root, 'data_nifty', '1.Training_DICOM_603'),
#             voi_root=os.path.join(data_root, 'ROI', '1.Training_DICOM_603'),
#             ct_types=['A', 'D', 'N', 'V'],
#             fusion_strategy=strategy,
#             target_size=(64, 64, 64),  # Smaller size for testing
#             verbose=True
#         )
        
#         # Test first sample
#         try:
#             sample, label = dataset[0]
#             sample_info = dataset.get_sample_info(0)
            
#             print(f"Sample info: {sample_info}")
#             print(f"Label: {label}")
            
#             if strategy == 'early':
#                 print(f"Sample shape: {sample.shape}")  # Should be (4, 64, 64, 64)
#             elif strategy == 'intermediate':
#                 print(f"Number of sequences: {len(sample)}")
#                 print(f"Each sequence shape: {sample[0].shape}")  # Should be (1, 64, 64, 64)
#             elif strategy == 'single':
#                 print(f"Sample shape: {sample.shape}")  # Should be (1, 64, 64, 64)
#             elif strategy == 'ensemble':
#                 print(f"Number of sequences: {len(sample)}")
#                 print(f"Each sequence shape: {sample[0].shape}")  # Should be (1, 64, 64, 64)
                
#         except Exception as e:
#             print(f"Error testing {strategy}: {str(e)}")
    
#     # Test data loader
#     print("\nTesting DataLoader:")
#     loader = TumorAggressivenessDataLoader(
#         data_root=data_root,
#         batch_size=2,
#         target_size=(64, 64, 64),
#         fusion_strategy='early'
#     )
    
#     train_loader = loader.create_train_loader()
#     print(f"Train loader created with {len(train_loader)} batches")
    
#     # Test one batch
#     try:
#         for batch_idx, (data, labels) in enumerate(train_loader):
#             print(f"Batch {batch_idx}: data shape {data.shape}, labels {labels}")
#             break
#     except Exception as e:
#         print(f"Error testing data loader: {str(e)}") 

# # Simple test: Load and print shape of a single CT file
# if __name__ == "__main__":
#     import nibabel as nib
#     example_ct_path = r"Data\data_nifty\1.Training_DICOM_603\AI_JIU_GEN_P1548715\CT\A_8\A_8_02_shenAgioRoutine_20170728165801_6.nii.gz"
#     if os.path.exists(example_ct_path):
#         ct_img = nib.load(example_ct_path)
#         ct_data = ct_img.get_fdata()
#         print(f"Loaded CT shape: {ct_data.shape}")
#     else:
#         print(f"File not found: {example_ct_path}") 