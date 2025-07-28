import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import torchio as tio
from typing import List, Dict, Optional, Tuple, Union
import warnings
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TumorAggressivenessDataset(Dataset):
    """
    Simplified tumor aggressiveness classification dataset using TorchIO.
    
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
        
        # Create file path manifest during initialization
        self.file_paths = self._create_file_path_manifest()
        
        # Create TorchIO transforms
        self.tio_transforms = self._create_tio_transforms()
        
        if verbose:
            logger.info(f"Dataset initialized with {len(self.data)} samples")
            logger.info(f"CT types: {self.ct_types}")
            logger.info(f"Fusion strategy: {fusion_strategy}")
            logger.info(f"Target size: {target_size}")
            logger.info(f"File path manifest created with {len(self.file_paths)} valid samples")
    
    def _create_file_path_manifest(self) -> Dict[str, Dict[str, Dict[str, str]]]:
        """
        Create a manifest of all file paths during initialization.
        Returns a dictionary mapping patient_id to file paths for each CT type.
        """
        file_paths = {}
        
        if self.verbose:
            logger.info("Creating file path manifest...")
        
        # Determine parent folder names based on data root
        if '1.Training' in self.ct_root:
            ct_parent_folder = '1.Training_DICOM_603'
            voi_parent_folder = '1.Training_VOI_603'
        elif '2.Internal' in self.ct_root:
            ct_parent_folder = '2.Internal Test_DICOM_259'
            voi_parent_folder = '2.Internal Test_VOI_259'
        elif '3.External' in self.ct_root:
            ct_parent_folder = '3.External Test_DICOM_308'
            voi_parent_folder = '3.External Test_VOI_308'
        else:
            ct_parent_folder = self.ct_root
            voi_parent_folder = self.voi_root
        
        valid_samples = 0
        
        for idx in range(len(self.data)):
            row = self.data.iloc[idx]
            name_id = row['serial_number']
            pinyin = row['pinyin']
            patient_folder = f"{pinyin}_{name_id}"
            
            patient_paths = {}
            all_sequences_available = True
            
            for ct_type in self.ct_types:
                # CT file path
                ct_base = os.path.join('Data', 'data_nifty', ct_parent_folder, patient_folder, 'CT')
                ct_path = None
                
                if os.path.exists(ct_base):
                    # Find any folder that matches ct_type_*
                    ct_folders = [f for f in os.listdir(ct_base) if f.startswith(f"{ct_type}_") and os.path.isdir(os.path.join(ct_base, f))]
                    if ct_folders:
                        ct_folder_path = os.path.join(ct_base, ct_folders[0])
                        nii_files = [f for f in os.listdir(ct_folder_path) if f.endswith('.nii.gz')]
                        if nii_files:
                            ct_path = os.path.join(ct_folder_path, nii_files[0])
                
                # VOI file path
                voi_path = os.path.join('Data', 'VOI_nifty', voi_parent_folder, patient_folder, 'VOI', f"{ct_type}.nii.gz")
                
                # Check if both CT and VOI files exist
                if ct_path and os.path.exists(ct_path) and os.path.exists(voi_path):
                    patient_paths[ct_type] = {
                        'ct_path': ct_path,
                        'voi_path': voi_path
                    }
                else:
                    all_sequences_available = False
                    if self.verbose:
                        logger.warning(f"Missing files for {patient_folder} - {ct_type}: CT={ct_path}, VOI={voi_path}")
            
            # Only include patients with all required sequences
            if all_sequences_available:
                file_paths[name_id] = patient_paths
                valid_samples += 1
            else:
                if self.verbose:
                    logger.warning(f"Skipping {patient_folder} - missing required sequences")
        
        if self.verbose:
            logger.info(f"File path manifest created: {valid_samples}/{len(self.data)} samples have all required sequences")
        
        return file_paths
    
    def _create_tio_transforms(self) -> tio.Transform:
        """
        Create TorchIO transforms for preprocessing.
        """
        transforms = []
        
        # Resize to target size
        transforms.append(tio.Resize(self.target_size))
        
        # Normalize CT values if requested
        if self.normalize:
            transforms.append(tio.Clamp(out_min=-1000, out_max=1000))
            transforms.append(tio.ZNormalization())
        
        # Apply additional transforms if provided
        if self.transform:
            transforms.append(self.transform)
        
        return tio.Compose(transforms)
    
    def _load_sample_with_tio(self, name_id: str) -> Dict[str, torch.Tensor]:
        """
        Load a sample using TorchIO.
        """
        # Get pre-computed file paths for this patient
        if name_id not in self.file_paths:
            raise ValueError(f"No file paths found for patient {name_id}")
        
        patient_paths = self.file_paths[name_id]
        
        # Create TorchIO subject
        subject_dict = {}
        
        for ct_type in self.ct_types:
            ct_path = patient_paths[ct_type]['ct_path']
            voi_path = patient_paths[ct_type]['voi_path']
            
            # Load CT image
            ct_image = tio.ScalarImage(ct_path)
            subject_dict[f'ct_{ct_type}'] = ct_image
            
            # Load VOI mask
            voi_mask = tio.LabelMap(voi_path)
            subject_dict[f'voi_{ct_type}'] = voi_mask
        
        # Create TorchIO subject
        subject = tio.Subject(subject_dict)
        
        # Apply transforms
        transformed_subject = self.tio_transforms(subject)
        
        # Extract data
        sample_data = {}
        for ct_type in self.ct_types:
            ct_tensor = transformed_subject[f'ct_{ct_type}'].data  # Shape: (1, D, H, W)
            voi_tensor = transformed_subject[f'voi_{ct_type}'].data  # Shape: (1, D, H, W)
            
            # Apply VOI mask if requested
            if self.apply_voi_mask:
                ct_tensor = ct_tensor * (voi_tensor > 0)
            
            sample_data[ct_type] = ct_tensor.squeeze(0)  # Remove channel dimension: (D, H, W)
        
        return sample_data
    
    def __len__(self) -> int:
        return len(self.file_paths)
    
    def __getitem__(self, idx: int) -> Union[torch.Tensor, Tuple[torch.Tensor, int], Tuple[List[torch.Tensor], int]]:
        """Get a sample from the dataset using TorchIO."""
        # Get patient info from the file paths dictionary
        patient_ids = list(self.file_paths.keys())
        name_id = patient_ids[idx]
        
        # Find corresponding row in data
        row = self.data[self.data['serial_number'] == name_id]
        if len(row) == 0:
            raise ValueError(f"No data found for patient {name_id}")
        
        row = row.iloc[0]
        label = int(row['aggressive_pathology_1_indolent_2_aggressive']) - 1  # Convert to 0/1
        
        # Check cache first
        if self.cache_data and name_id in self.data_cache:
            sample_data = self.data_cache[name_id]
        else:
            sample_data = self._load_sample_with_tio(name_id)
            if self.cache_data:
                self.data_cache[name_id] = sample_data
        
        # Apply fusion strategy
        if self.fusion_strategy == 'early':
            # Concatenate all sequences as channels
            channels = []
            for ct_type in self.ct_types:
                channels.append(sample_data[ct_type])
            
            # Stack as channels (C, D, H, W)
            combined = torch.stack(channels, dim=0)
            
            return combined, label
        
        elif self.fusion_strategy == 'intermediate':
            # Return as list for custom fusion in model
            tensors = []
            for ct_type in self.ct_types:
                tensor = sample_data[ct_type].unsqueeze(0)  # (1, D, H, W)
                tensors.append(tensor)
            
            return tensors, label
        
        elif self.fusion_strategy == 'single':
            # Use only first CT type
            ct_type = self.ct_types[0]
            tensor_data = sample_data[ct_type].unsqueeze(0)  # (1, D, H, W)
            
            return tensor_data, label
        
        elif self.fusion_strategy == 'ensemble':
            # Return all sequences separately for ensemble training
            tensors = []
            for ct_type in self.ct_types:
                tensor = sample_data[ct_type].unsqueeze(0)  # (1, D, H, W)
                tensors.append(tensor)
            
            return tensors, label
    
    def get_sample_info(self, idx: int) -> Dict:
        """Get information about a sample without loading the data."""
        patient_ids = list(self.file_paths.keys())
        name_id = patient_ids[idx]
        
        row = self.data[self.data['serial_number'] == name_id]
        if len(row) == 0:
            raise ValueError(f"No data found for patient {name_id}")
        
        row = row.iloc[0]
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
        # Only count samples that have all required sequences
        valid_labels = []
        for name_id in self.file_paths.keys():
            row = self.data[self.data['serial_number'] == name_id]
            if len(row) > 0:
                label = int(row.iloc[0]['aggressive_pathology_1_indolent_2_aggressive']) - 1
                valid_labels.append(label)
        
        unique, counts = np.unique(valid_labels, return_counts=True)
        return dict(zip(unique, counts))
    
    def analyze_data_completeness(self) -> Dict:
        """Analyze the completeness of CT sequences across all patients."""
        completeness_stats = {
            'total_patients_in_csv': len(self.data),
            'total_patients_with_all_sequences': len(self.file_paths),
            'sequence_availability': {ct: 0 for ct in self.ct_types},
            'patients_with_all_sequences': len(self.file_paths),
            'patients_with_no_sequences': 0,
            'missing_sequences_by_patient': {}
        }
        
        # Count sequence availability from file paths
        for name_id, patient_paths in self.file_paths.items():
            for ct_type in self.ct_types:
                if ct_type in patient_paths:
                    completeness_stats['sequence_availability'][ct_type] += 1
        
        # Count patients with no sequences
        for idx in range(len(self.data)):
            name_id = self.data.iloc[idx]['serial_number']
            if name_id not in self.file_paths:
                completeness_stats['patients_with_no_sequences'] += 1
                completeness_stats['missing_sequences_by_patient'][name_id] = 'ALL'
        
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


# Example usage and testing
if __name__ == '__main__':
    # Test the dataset
    data_root = "Data"
    
    # Test different fusion strategies
    strategies = ['early', 'intermediate', 'single', 'ensemble']
    
    for strategy in strategies:
        print(f"\nTesting {strategy} fusion strategy:")
        
        dataset = TumorAggressivenessDataset(
            csv_path=os.path.join(data_root, 'ccRCC_Survival_Analysis_Dataset_english', 'training_set_603_cases.csv'),
            ct_root=os.path.join(data_root, 'data_nifty', '1.Training_DICOM_603'),
            voi_root=os.path.join(data_root, 'ROI', '1.Training_DICOM_603'),
            ct_types=['A', 'D', 'N', 'V'],
            fusion_strategy=strategy,
            target_size=(64, 64, 64),  # Smaller size for testing
            verbose=True
        )

        print("length of dataset", len(dataset))
        
        # Test first sample
        try:
            sample, label = dataset[0]
            sample_info = dataset.get_sample_info(0)
            
            print(f"Sample info: {sample_info}")
            print(f"Label: {label}")
            
            if strategy == 'early':
                print(f"Sample shape: {sample.shape}")  # Should be (4, 64, 64, 64)
            elif strategy == 'intermediate':
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
    print("\nTesting DataLoader:")
    loader = TumorAggressivenessDataLoader(
        data_root=data_root,
        batch_size=2,
        target_size=(64, 64, 64),
        fusion_strategy='early'
    )
    
    train_loader = loader.create_train_loader()
    print(f"Train loader created with {len(train_loader)} batches")
    
    # Test one batch
    try:
        for batch_idx, (data, labels) in enumerate(train_loader):
            print(f"Batch {batch_idx}: data shape {data.shape}, labels {labels}")
            break
    except Exception as e:
        print(f"Error testing data loader: {str(e)}") 