import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
import logging

# --- Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def select_features(
    input_csv_path: str, 
    output_csv_path: str, 
    correlation_threshold: float = 0.9, 
    k_best_features: int = 50
):
    """
    Performs a two-stage feature selection process on a radiomics feature set.

    Stage 1: Removes highly correlated features.
    Stage 2: Selects the top k features using a univariate statistical test.

    Args:
        input_csv_path (str): Path to the input CSV file with all radiomics features.
        output_csv_path (str): Path to save the final selected features.
        correlation_threshold (float): The threshold above which to remove correlated features.
        k_best_features (int): The number of top features to select in the final step.
    """
    logger.info(f"Loading radiomics features from: {input_csv_path}")
    try:
        df = pd.read_csv(input_csv_path)
    except FileNotFoundError:
        logger.error(f"Error: Input file not found at {input_csv_path}")
        return

    # Separate identifiers, target label, and features
    if 'patient_id' not in df.columns or 'label' not in df.columns:
        logger.error("Error: CSV must contain 'patient_id' and 'label' columns.")
        return
        
    labels = df['label']
    patient_ids = df['patient_id']
    features = df.drop(columns=['patient_id', 'label'])
    logger.info(f"Initial number of features: {features.shape[1]}")

    # --- 1. Handle Missing Values ---
    # Drop columns with more than 20% missing values
    features = features.loc[:, features.isnull().mean() < 0.2]
    logger.info(f"Features after dropping columns with >20% NaNs: {features.shape[1]}")
    
    # Impute remaining NaNs with the mean
    imputer = SimpleImputer(strategy='mean')
    features_imputed = pd.DataFrame(imputer.fit_transform(features), columns=features.columns)

    # --- 2. Remove Highly Correlated Features ---
    logger.info(f"Removing features with correlation > {correlation_threshold}...")
    corr_matrix = features_imputed.corr().abs()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > correlation_threshold)]
    
    features_uncorrelated = features_imputed.drop(columns=to_drop)
    logger.info(f"Dropped {len(to_drop)} correlated features.")
    logger.info(f"Number of features after correlation removal: {features_uncorrelated.shape[1]}")

    # --- 3. Select K-Best Features ---
    if k_best_features >= features_uncorrelated.shape[1]:
        logger.warning(f"k_best_features ({k_best_features}) is >= number of available features. Skipping SelectKBest.")
        final_features = features_uncorrelated
    else:
        logger.info(f"Selecting the top {k_best_features} features using SelectKBest (ANOVA F-test)...")
        selector = SelectKBest(f_classif, k=k_best_features)
        selector.fit(features_uncorrelated, labels)
        
        # Get the columns to keep and create a new dataframe
        cols_to_keep = selector.get_support(indices=True)
        final_features = features_uncorrelated.iloc[:, cols_to_keep]
        logger.info(f"Final number of selected features: {final_features.shape[1]}")

    # --- 4. Combine and Save Results ---
    final_df = pd.concat([patient_ids, labels, final_features], axis=1)
    
    final_df.to_csv(output_csv_path, index=False)
    logger.info(f"Final feature set saved to: {output_csv_path}")


if __name__ == "__main__":
    # --- Configuration ---
    # Path to your raw, unfiltered radiomics features
    INPUT_CSV = r"Data/radiomics_features_validation.csv"
    
    # Path to save the new, filtered feature set
    OUTPUT_CSV = r"Data/radiomics_features_selected_validation.csv"
    
    # --- Parameters ---
    # Threshold for removing correlated features (0.9 is a common value)
    CORR_THRESHOLD = 0.9
    
    # The final number of features you want to select for your model
    K_FEATURES = 50

    select_features(
        input_csv_path=INPUT_CSV,
        output_csv_path=OUTPUT_CSV,
        correlation_threshold=CORR_THRESHOLD,
        k_best_features=K_FEATURES
    )