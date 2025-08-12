import os
import json
import pandas as pd
from radiomics import featureextractor
import logging
from tqdm import tqdm

# --- Setup ---
# Set up a logger for pyradiomics to provide detailed information
radiomics_logger = logging.getLogger("radiomics")
radiomics_logger.setLevel(logging.ERROR) # Set to INFO for more detailed output

# Set up a general logger for our script
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def extract_radiomics_features(manifest_path: str, output_csv_path: str, params_yaml_path: str, ct_types: list = ['A', 'D', 'N', 'V']):
    """
    Extracts radiomics features for each CT phase from a JSON manifest file
    using a specified YAML configuration file.

    Args:
        manifest_path (str): Path to the JSON manifest file.
        output_csv_path (str): Path to save the output CSV file.
        params_yaml_path (str): Path to the pyradiomics YAML configuration file.
        ct_types (list): A list of CT phase keys to process (e.g., ['A', 'V']).
    """
    if not os.path.exists(manifest_path):
        logger.error(f"Manifest file not found at: {manifest_path}")
        return
    if not os.path.exists(params_yaml_path):
        logger.error(f"Pyradiomics params file not found at: {params_yaml_path}")
        return

    with open(manifest_path, 'r') as f:
        data_list = json.load(f)

    # 1. Initialize the pyradiomics feature extractor from the YAML file
    extractor = featureextractor.RadiomicsFeatureExtractor(params_yaml_path)
    logger.info(f"Initialized pyradiomics feature extractor from: {params_yaml_path}")

    all_patient_features = []

    # 2. Iterate through each patient in the manifest
    for record in tqdm(data_list, desc="Processing Patients"):
        patient_id = record.get("patient_id", "N/A")
        
        patient_features = {
            'patient_id': patient_id,
            'label': record.get('label')
        }
        
        # 3. Iterate through each specified CT phase for the patient
        for ct_type in ct_types:
            image_key = ct_type
            mask_key = f"{ct_type}_mask"

            if image_key not in record or mask_key not in record:
                logger.warning(f"Skipping {ct_type} for patient {patient_id}: missing image or mask path in JSON.")
                continue

            image_path = record[image_key]
            mask_path = record[mask_key]

            if not os.path.exists(image_path) or not os.path.exists(mask_path):
                logger.warning(f"Skipping {ct_type} for patient {patient_id}: file not found at {image_path} or {mask_path}.")
                continue

            try:
                # 4. Execute the feature extraction
                result = extractor.execute(image_path, mask_path)
                
                # 5. Add a prefix to each feature name to identify its CT phase
                for feature_name, feature_value in result.items():
                    if feature_name.startswith('original_') or feature_name.startswith('log-') or feature_name.startswith('wavelet-'):
                        new_feature_name = f"{ct_type}_{feature_name}"
                        patient_features[new_feature_name] = feature_value

            except Exception as e:
                logger.error(f"Failed to extract features for patient {patient_id}, phase {ct_type}. Error: {e}")

        all_patient_features.append(patient_features)

    # 6. Convert the list of dictionaries to a pandas DataFrame and save
    if not all_patient_features:
        logger.warning("No features were extracted. The output CSV will be empty.")
        return

    df = pd.DataFrame(all_patient_features)
    df.to_csv(output_csv_path, index=False)
    logger.info(f"Successfully extracted features for {len(df)} patients.")
    logger.info(f"Results saved to: {output_csv_path}")


if __name__ == "__main__":
    # --- Configuration ---
    # Define the path to your manifest file and where to save the results
    json_file = r"Data\internal_test_manifest.json"
    output_csv = r"Data\radiomics_features_validation.csv"
    params_file = r"radiomics\radiomics_params.yaml" # Path to your new YAML config file
    
    # Define which CT phases you want to extract features from
    phases_to_process = ['A', 'D', 'N', 'V']

    extract_radiomics_features(
        manifest_path=json_file,
        output_csv_path=output_csv,
        params_yaml_path=params_file,
        ct_types=phases_to_process
    )
