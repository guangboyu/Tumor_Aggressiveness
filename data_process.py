import os
import SimpleITK as sitk
import numpy as np
import pandas as pd
from tqdm import tqdm
import logging
from typing import List, Dict, Tuple
import argparse
import json

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VOIPreprocessor:
    """
    Preprocess VOI masks to align with CT coordinate systems and save as NIfTI files.
    This significantly speeds up training by avoiding real-time resampling.
    """
    
    def __init__(self, data_root: str = "Data"):
        self.data_root = data_root
        self.voi_nifty_root = os.path.join(data_root, "VOI_nifty")
        
        # Create output directory
        os.makedirs(self.voi_nifty_root, exist_ok=True)
        
        # Define dataset configurations
        self.datasets = {
            'training': {
                'csv': 'ccRCC_Survival_Analysis_Dataset_english/training_set_603_cases.csv',
                'ct_root': 'data_nifty/1.Training_DICOM_603',
                'voi_root': 'ROI/1.Training_ROI_603',
                'output_root': '1.Training_VOI_603'
            },
            'internal_test': {
                'csv': 'ccRCC_Survival_Analysis_Dataset_english/internal_test_set_259_cases.csv',
                'ct_root': 'data_nifty/2.Internal Test_DICOM_259',
                'voi_root': 'ROI/2.Internal Test_ROI_259',
                'output_root': '2.Internal Test_VOI_259'
            },
            'external_test': {
                'csv': 'ccRCC_Survival_Analysis_Dataset_english/external_verification_set_308_cases.csv',
                'ct_root': 'data_nifty/3.External Test_DICOM_308',
                'voi_root': 'ROI/3.External Test_ROI_308',
                'output_root': '3.External Test_VOI_308'
            }
        }
    
    def _find_ct_file(self, patient_folder: str, ct_type: str, ct_root: str) -> str:
        """Find the CT file for a given patient and sequence type."""
        ct_base = os.path.join(self.data_root, ct_root, patient_folder, 'CT')
        
        if not os.path.exists(ct_base):
            raise FileNotFoundError(f"CT base path not found: {ct_base}")
        
        # Find any folder that matches ct_type_*
        ct_folders = [f for f in os.listdir(ct_base) 
                     if f.startswith(f"{ct_type}_") and os.path.isdir(os.path.join(ct_base, f))]
        
        if not ct_folders:
            raise FileNotFoundError(f"No CT folder found for type {ct_type} in {ct_base}")
        
        ct_path = os.path.join(ct_base, ct_folders[0])
        nii_files = [f for f in os.listdir(ct_path) if f.endswith('.nii.gz')]
        
        if not nii_files:
            raise FileNotFoundError(f"No NIfTI file found in {ct_path}")
        
        return os.path.join(ct_path, nii_files[0])
    
    def _find_voi_file(self, patient_folder: str, ct_type: str, voi_root: str) -> str:
        """Find the VOI file for a given patient and sequence type."""
        voi_path = os.path.join(self.data_root, voi_root, patient_folder, 'ROI')
        
        if not os.path.exists(voi_path):
            raise FileNotFoundError(f"VOI path not found: {voi_path}")
        
        # Find any file that matches ct_type_*.nrrd
        voi_files = [f for f in os.listdir(voi_path) 
                    if f.startswith(f"{ct_type}_") and f.endswith('.nrrd')]
        
        if not voi_files:
            # Try alternative naming pattern (just ct_type.nrrd)
            voi_files = [f for f in os.listdir(voi_path) if f == f"{ct_type}.nrrd"]
        
        if not voi_files:
            raise FileNotFoundError(f"VOI file not found for type {ct_type} in {voi_path}")
        
        return os.path.join(voi_path, voi_files[0])
    
    def _resample_voi_to_ct(self, voi_img: sitk.Image, ct_img: sitk.Image) -> sitk.Image:
        """Resample VOI to match CT coordinate system."""
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(ct_img)
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)  # Use nearest neighbor for masks
        resampler.SetDefaultPixelValue(0)
        
        return resampler.Execute(voi_img)
    
    def _process_patient(self, patient_folder: str, ct_types: List[str], 
                        ct_root: str, voi_root: str, output_root: str) -> Dict[str, str]:
        """Process a single patient's VOI files."""
        results = {}
        
        for ct_type in ct_types:
            try:
                # Find CT and VOI files
                ct_file = self._find_ct_file(patient_folder, ct_type, ct_root)
                voi_file = self._find_voi_file(patient_folder, ct_type, voi_root)
                
                # Load images
                ct_img = sitk.ReadImage(ct_file, sitk.sitkFloat64)
                voi_img = sitk.ReadImage(voi_file)
                
                # Resample VOI to match CT
                voi_resampled = self._resample_voi_to_ct(voi_img, ct_img)
                
                # Create output directory
                output_dir = os.path.join(self.voi_nifty_root, output_root, patient_folder, 'VOI')
                os.makedirs(output_dir, exist_ok=True)
                
                # Save resampled VOI as NIfTI
                output_file = os.path.join(output_dir, f"{ct_type}.nii.gz")
                sitk.WriteImage(voi_resampled, output_file)
                
                results[ct_type] = output_file
                
            except Exception as e:
                logger.warning(f"Error processing {ct_type} for {patient_folder}: {str(e)}")
                results[ct_type] = None
        
        return results
    
    def process_dataset(self, dataset_name: str, ct_types: List[str] = ['A', 'D', 'N', 'V']) -> Dict:
        """Process an entire dataset."""
        if dataset_name not in self.datasets:
            raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(self.datasets.keys())}")
        
        config = self.datasets[dataset_name]
        csv_path = os.path.join(self.data_root, config['csv'])
        
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        
        # Load patient list
        df = pd.read_csv(csv_path)
        logger.info(f"Processing {dataset_name} dataset with {len(df)} patients")
        
        # Statistics
        stats = {
            'total_patients': len(df),
            'processed_patients': 0,
            'failed_patients': 0,
            'processed_sequences': 0,
            'failed_sequences': 0,
            'errors': []
        }
        
        # Process each patient
        for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {dataset_name}"):
            patient_folder = f"{row['pinyin']}_{row['serial_number']}"
            
            try:
                results = self._process_patient(
                    patient_folder=patient_folder,
                    ct_types=ct_types,
                    ct_root=config['ct_root'],
                    voi_root=config['voi_root'],
                    output_root=config['output_root']
                )
                
                # Count successful sequences
                successful_sequences = sum(1 for result in results.values() if result is not None)
                stats['processed_sequences'] += successful_sequences
                stats['failed_sequences'] += len(ct_types) - successful_sequences
                
                if successful_sequences > 0:
                    stats['processed_patients'] += 1
                else:
                    stats['failed_patients'] += 1
                    stats['errors'].append(f"Patient {patient_folder}: All sequences failed")
                
            except Exception as e:
                stats['failed_patients'] += 1
                stats['failed_sequences'] += len(ct_types)
                stats['errors'].append(f"Patient {patient_folder}: {str(e)}")
                logger.error(f"Error processing patient {patient_folder}: {str(e)}")
        
        return stats
    
    def process_all_datasets(self, ct_types: List[str] = ['A', 'D', 'N', 'V']) -> Dict:
        """Process all datasets."""
        all_stats = {}
        
        for dataset_name in self.datasets.keys():
            logger.info(f"\n{'='*50}")
            logger.info(f"Processing {dataset_name} dataset")
            logger.info(f"{'='*50}")
            
            try:
                stats = self.process_dataset(dataset_name, ct_types)
                all_stats[dataset_name] = stats
                
                # Print summary
                logger.info(f"\n{dataset_name} Summary:")
                logger.info(f"  Total patients: {stats['total_patients']}")
                logger.info(f"  Processed patients: {stats['processed_patients']}")
                logger.info(f"  Failed patients: {stats['failed_patients']}")
                logger.info(f"  Processed sequences: {stats['processed_sequences']}")
                logger.info(f"  Failed sequences: {stats['failed_sequences']}")
                
            except Exception as e:
                logger.error(f"Error processing {dataset_name} dataset: {str(e)}")
                all_stats[dataset_name] = {'error': str(e)}
        
        return all_stats
    
    def verify_processing(self, dataset_name: str, ct_types: List[str] = ['A', 'D', 'N', 'V']) -> Dict:
        """Verify that processing was successful by checking file existence."""
        if dataset_name not in self.datasets:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        config = self.datasets[dataset_name]
        csv_path = os.path.join(self.data_root, config['csv'])
        df = pd.read_csv(csv_path)
        
        verification_stats = {
            'total_patients': len(df),
            'verified_patients': 0,
            'missing_patients': 0,
            'verified_sequences': 0,
            'missing_sequences': 0,
            'missing_files': []
        }
        
        for idx, row in df.iterrows():
            patient_folder = f"{row['pinyin']}_{row['serial_number']}"
            output_dir = os.path.join(self.voi_nifty_root, config['output_root'], patient_folder, 'VOI')
            
            patient_verified = True
            for ct_type in ct_types:
                expected_file = os.path.join(output_dir, f"{ct_type}.nii.gz")
                
                if os.path.exists(expected_file):
                    verification_stats['verified_sequences'] += 1
                else:
                    verification_stats['missing_sequences'] += 1
                    verification_stats['missing_files'].append(expected_file)
                    patient_verified = False
            
            if patient_verified:
                verification_stats['verified_patients'] += 1
            else:
                verification_stats['missing_patients'] += 1
        
        return verification_stats

    def generate_json_manifest(self, dataset_name: str, ct_types: List[str] = ['A', 'D', 'N', 'V']) -> Dict:
        """
        Generate a JSON manifest file for MONAI dataset loading.
        Returns a list of dictionaries with file paths and labels.
        """
        if dataset_name not in self.datasets:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        config = self.datasets[dataset_name]
        csv_path = os.path.join(self.data_root, config['csv'])
        
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        
        # Load patient list
        df = pd.read_csv(csv_path)
        logger.info(f"Generating JSON manifest for {dataset_name} dataset with {len(df)} patients")
        
        data_list = []
        stats = {
            'total_patients': len(df),
            'processed_patients': 0,
            'failed_patients': 0,
            'errors': []
        }
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Generating manifest for {dataset_name}"):
            patient_folder = f"{row['pinyin']}_{row['serial_number']}"
            label = int(row['aggressive_pathology_1_indolent_2_aggressive']) - 1  # Convert to 0/1
            
            try:
                item = {
                    'label': label,
                    'patient_id': row['serial_number'],
                    'pinyin': row['pinyin']
                }
                
                # Add CT file paths
                for ct_type in ct_types:
                    ct_file = self._find_ct_file(patient_folder, ct_type, config['ct_root'])
                    item[ct_type] = ct_file
                
                # Add mask file path (using preprocessed VOI)
                mask_file = os.path.join(self.voi_nifty_root, config['output_root'], 
                                       patient_folder, 'VOI', f"{ct_types[0]}.nii.gz")
                item['mask'] = mask_file
                
                # Verify all files exist
                all_files_exist = True
                for key, file_path in item.items():
                    if key in ['label', 'patient_id', 'pinyin']:
                        continue
                    if not os.path.exists(file_path):
                        all_files_exist = False
                        stats['errors'].append(f"Missing file for {patient_folder} - {key}: {file_path}")
                        break
                
                if all_files_exist:
                    data_list.append(item)
                    stats['processed_patients'] += 1
                else:
                    stats['failed_patients'] += 1
                    
            except Exception as e:
                stats['failed_patients'] += 1
                stats['errors'].append(f"Error processing {patient_folder}: {str(e)}")
                logger.warning(f"Error processing {patient_folder}: {str(e)}")
        
        return {
            'data_list': data_list,
            'stats': stats
        }
    
    def save_json_manifest(self, dataset_name: str, ct_types: List[str] = ['A', 'D', 'N', 'V']) -> str:
        """
        Generate and save JSON manifest file.
        Returns the path to the saved JSON file.
        """
        result = self.generate_json_manifest(dataset_name, ct_types)
        
        # Create output filename
        output_file = os.path.join(self.data_root, f"{dataset_name}_manifest.json")
        
        # Save JSON file
        with open(output_file, 'w') as f:
            json.dump(result['data_list'], f, indent=2)
        
        logger.info(f"JSON manifest saved to: {output_file}")
        logger.info(f"Processed: {result['stats']['processed_patients']}/{result['stats']['total_patients']} patients")
        logger.info(f"Failed: {result['stats']['failed_patients']} patients")
        
        if result['stats']['errors']:
            logger.warning(f"Found {len(result['stats']['errors'])} errors during processing")
        
        return output_file
    
    def generate_all_json_manifests(self, ct_types: List[str] = ['A', 'D', 'N', 'V']) -> Dict[str, str]:
        """
        Generate JSON manifests for all datasets.
        Returns a dictionary mapping dataset names to JSON file paths.
        """
        manifest_files = {}
        
        for dataset_name in self.datasets.keys():
            logger.info(f"\n{'='*50}")
            logger.info(f"Generating JSON manifest for {dataset_name} dataset")
            logger.info(f"{'='*50}")
            
            try:
                json_file = self.save_json_manifest(dataset_name, ct_types)
                manifest_files[dataset_name] = json_file
                
            except Exception as e:
                logger.error(f"Error generating manifest for {dataset_name}: {str(e)}")
                manifest_files[dataset_name] = None
        
        return manifest_files


def main():
    parser = argparse.ArgumentParser(description='Preprocess VOI masks and generate JSON manifests for MONAI')
    parser.add_argument('--data_root', type=str, default='Data', help='Root directory for data')
    parser.add_argument('--ct_types', type=str, nargs='+', default=['A', 'D', 'N', 'V'], 
                       help='CT sequence types to process')
    parser.add_argument('--dataset', type=str, choices=['training', 'internal_test', 'external_test', 'all'],
                       default='all', help='Which dataset to process')
    parser.add_argument('--verify_only', action='store_true', help='Only verify existing processing')
    parser.add_argument('--generate_json_only', action='store_true', help='Only generate JSON manifests')
    parser.add_argument('--skip_voi_processing', action='store_true', help='Skip VOI preprocessing, only generate JSON')
    
    args = parser.parse_args()
    
    # Initialize preprocessor
    preprocessor = VOIPreprocessor(args.data_root)
    
    if args.generate_json_only:
        logger.info("JSON generation mode - creating MONAI-compatible manifests")
        
        if args.dataset == 'all':
            manifest_files = preprocessor.generate_all_json_manifests(args.ct_types)
            
            logger.info(f"\n{'='*60}")
            logger.info("JSON MANIFEST GENERATION SUMMARY")
            logger.info(f"{'='*60}")
            
            for dataset_name, json_file in manifest_files.items():
                if json_file:
                    logger.info(f"{dataset_name}: {json_file}")
                else:
                    logger.error(f"{dataset_name}: Failed to generate manifest")
        
        else:
            json_file = preprocessor.save_json_manifest(args.dataset, args.ct_types)
            logger.info(f"Generated manifest for {args.dataset}: {json_file}")
    
    elif args.verify_only:
        logger.info("Verification mode - checking existing processed files")
        
        if args.dataset == 'all':
            for dataset_name in preprocessor.datasets.keys():
                logger.info(f"\nVerifying {dataset_name} dataset...")
                stats = preprocessor.verify_processing(dataset_name, args.ct_types)
                logger.info(f"Verified: {stats['verified_patients']}/{stats['total_patients']} patients")
                logger.info(f"Verified: {stats['verified_sequences']}/{stats['total_patients'] * len(args.ct_types)} sequences")
        else:
            stats = preprocessor.verify_processing(args.dataset, args.ct_types)
            logger.info(f"Verified: {stats['verified_patients']}/{stats['total_patients']} patients")
            logger.info(f"Verified: {stats['verified_sequences']}/{stats['total_patients'] * len(args.ct_types)} sequences")
    
    else:
        # Process datasets
        if not args.skip_voi_processing:
            if args.dataset == 'all':
                all_stats = preprocessor.process_all_datasets(args.ct_types)
                
                # Print final summary
                logger.info(f"\n{'='*60}")
                logger.info("VOI PROCESSING SUMMARY")
                logger.info(f"{'='*60}")
                
                for dataset_name, stats in all_stats.items():
                    if 'error' not in stats:
                        logger.info(f"\n{dataset_name}:")
                        logger.info(f"  Processed: {stats['processed_patients']}/{stats['total_patients']} patients")
                        logger.info(f"  Success rate: {stats['processed_patients']/stats['total_patients']*100:.1f}%")
                    else:
                        logger.error(f"{dataset_name}: {stats['error']}")
            
            else:
                stats = preprocessor.process_dataset(args.dataset, args.ct_types)
                logger.info(f"\nFinal Summary for {args.dataset}:")
                logger.info(f"  Processed: {stats['processed_patients']}/{stats['total_patients']} patients")
                logger.info(f"  Success rate: {stats['processed_patients']/stats['total_patients']*100:.1f}%")
        
        # Generate JSON manifests
        logger.info(f"\n{'='*60}")
        logger.info("GENERATING JSON MANIFESTS")
        logger.info(f"{'='*60}")
        
        if args.dataset == 'all':
            manifest_files = preprocessor.generate_all_json_manifests(args.ct_types)
            
            for dataset_name, json_file in manifest_files.items():
                if json_file:
                    logger.info(f"{dataset_name}: {json_file}")
                else:
                    logger.error(f"{dataset_name}: Failed to generate manifest")
        
        else:
            json_file = preprocessor.save_json_manifest(args.dataset, args.ct_types)
            logger.info(f"Generated manifest for {args.dataset}: {json_file}")


if __name__ == "__main__":
    main()
