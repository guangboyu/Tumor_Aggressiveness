import os
import SimpleITK as sitk
import pandas as pd
from tqdm import tqdm
import logging
import json
from pathlib import Path

# --- Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    A refactored class to first preprocess all VOI masks and then separately
    generate MONAI-compatible JSON manifests for multi-sequence CT data.
    """
    def __init__(self, data_root: str = "Data"):
        self.data_root = Path(data_root)
        self.voi_nifty_root = self.data_root / "VOI_nifty"
        os.makedirs(self.voi_nifty_root, exist_ok=True)

        self.datasets_config = {
            'training': {
                'csv': 'ccRCC_Survival_Analysis_Dataset_english/training_set_603_cases.csv',
                'ct_root': 'data_nifty/1.Training_DICOM_603',
                'voi_root': 'ROI/1.Training_ROI_603',
                'output_voi_root': '1.Training_VOI_603',
                'manifest_name': 'training_manifest.json'
            },
            'internal_test': {
                'csv': 'ccRCC_Survival_Analysis_Dataset_english/internal_test_set_259_cases.csv',
                'ct_root': 'data_nifty/2.Internal Test_DICOM_259',
                'voi_root': 'ROI/2.Internal Test_ROI_259',
                'output_voi_root': '2.Internal Test_VOI_259',
                'manifest_name': 'internal_test_manifest.json'
            },
            'external_test': {
                'csv': 'ccRCC_Survival_Analysis_Dataset_english/external_verification_set_308_cases.csv',
                'ct_root': 'data_nifty/3.External Test_DICOM_308',
                'voi_root': 'ROI/3.External Test_ROI_308',
                'output_voi_root': '3.External Test_VOI_308',
                'manifest_name': 'external_test_manifest.json'
            }
        }

    def _find_file(self, base_path: Path, prefix: str, extension: str) -> Path:
        """A generic and more robust file finder."""
        if not base_path.exists():
            return None
        
        for item in base_path.iterdir():
            if item.name.startswith(prefix) and item.name.endswith(extension):
                if item.is_file():
                    return item
                elif item.is_dir():
                    nii_files = list(item.glob('*.nii.gz'))
                    if nii_files:
                        return nii_files[0]
        return None

    def resample_all_masks(self, ct_types: list = ['A', 'D', 'N', 'V']):
        """
        Iterates through all datasets and resamples the .nrrd VOI masks to match
        their corresponding CT scans, saving them as .nii.gz files.
        """
        for name, config in self.datasets_config.items():
            logger.info(f"\n{'='*50}\nResampling masks for dataset: {name}\n{'='*50}")
            
            csv_path = self.data_root / config['csv']
            if not csv_path.exists():
                logger.error(f"CSV file not found, skipping: {csv_path}")
                continue

            df = pd.read_csv(csv_path)
            for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Resampling {name}"):
                patient_folder = f"{row['pinyin']}_{row['serial_number']}"
                for ct_type in ct_types:
                    try:
                        ct_base = self.data_root / config['ct_root'] / patient_folder / 'CT'
                        ct_file = self._find_file(ct_base, f"{ct_type}_", ".nii.gz")

                        voi_base = self.data_root / config['voi_root'] / patient_folder / 'ROI'
                        voi_file = self._find_file(voi_base, f"{ct_type}_", ".nrrd")
                        if not voi_file: voi_file = self._find_file(voi_base, ct_type, ".nrrd")

                        if not ct_file or not voi_file:
                            continue

                        output_dir = self.voi_nifty_root / config['output_voi_root'] / patient_folder / 'VOI'
                        output_mask_path = output_dir / f"{ct_type}.nii.gz"
                        
                        # Skip if the resampled file already exists
                        if output_mask_path.exists():
                            continue

                        ct_img = sitk.ReadImage(str(ct_file))
                        voi_img = sitk.ReadImage(str(voi_file))

                        resampler = sitk.ResampleImageFilter()
                        resampler.SetReferenceImage(ct_img)
                        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
                        voi_resampled = resampler.Execute(voi_img)

                        output_dir.mkdir(parents=True, exist_ok=True)
                        sitk.WriteImage(voi_resampled, str(output_mask_path))

                    except Exception as e:
                        logger.warning(f"Skipping resampling for {ct_type} in {patient_folder} due to error: {e}")

    def generate_all_manifests(self, ct_types: list = ['A', 'D', 'N', 'V']):
        """
        Generates the JSON manifest files by finding the already-resampled
        .nii.gz masks and their corresponding CT scans.
        """
        for name, config in self.datasets_config.items():
            logger.info(f"\n{'='*50}\nGenerating manifest for dataset: {name}\n{'='*50}")
            
            csv_path = self.data_root / config['csv']
            if not csv_path.exists():
                logger.error(f"CSV file not found, skipping: {csv_path}")
                continue

            df = pd.read_csv(csv_path)
            manifest_data = []

            for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Generating manifest for {name}"):
                patient_folder = f"{row['pinyin']}_{row['serial_number']}"
                patient_entry = {
                    'label': int(row['aggressive_pathology_1_indolent_2_aggressive']) - 1,
                    'patient_id': row['serial_number']
                }
                
                has_valid_sequence = False
                for ct_type in ct_types:
                    ct_base = self.data_root / config['ct_root'] / patient_folder / 'CT'
                    ct_file = self._find_file(ct_base, f"{ct_type}_", ".nii.gz")

                    output_dir = self.voi_nifty_root / config['output_voi_root'] / patient_folder / 'VOI'
                    mask_file = output_dir / f"{ct_type}.nii.gz"

                    if ct_file and ct_file.exists() and mask_file and mask_file.exists():
                        patient_entry[ct_type] = str(ct_file.relative_to(self.data_root.parent))
                        patient_entry[f"mask_{ct_type}"] = str(mask_file.relative_to(self.data_root.parent))
                        has_valid_sequence = True

                if has_valid_sequence:
                    manifest_data.append(patient_entry)

            manifest_path = self.data_root / config['manifest_name']
            with open(manifest_path, 'w') as f:
                json.dump(manifest_data, f, indent=2)
            logger.info(f"Successfully generated manifest for {len(manifest_data)} patients at: {manifest_path}")


if __name__ == "__main__":
    preprocessor = DataPreprocessor(data_root="Data")
    
    # # Step 1: Resample all masks first. This can be run once.
    # logger.info("--- Starting Step 1: Resampling all NRRD masks to NIfTI format ---")
    # preprocessor.resample_all_masks(ct_types=['A', 'D', 'N', 'V'])
    
    # Step 2: Generate the JSON manifests from the pre-processed files.
    logger.info("\n--- Starting Step 2: Generating JSON manifests ---")
    preprocessor.generate_all_manifests(ct_types=['A', 'D', 'N', 'V'])
    
    logger.info("\n--- Data processing complete. ---")
