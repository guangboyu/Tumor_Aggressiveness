import pandas as pd
import numpy as np
import logging
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.impute import SimpleImputer

# Import the classifier models
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# --- Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def _calculate_metrics(y_true, y_pred, y_prob):
    """A helper function to calculate all required classification metrics."""
    # Calculate confusion matrix to get TN, FP, FN, TP
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    # Sensitivity (Recall or True Positive Rate)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    # Specificity (True Negative Rate)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    return {
        'Accuracy': accuracy_score(y_true, y_pred),
        'F1-Score': f1_score(y_true, y_pred),
        'AUROC': roc_auc_score(y_true, y_prob),
        'Sensitivity': sensitivity,
        'Specificity': specificity
    }


def train_and_evaluate_cv(input_csv_path: str):
    """
    Trains and evaluates models using 5-fold cross-validation.
    """
    logger.info(f"Loading selected features from: {input_csv_path}")
    df = pd.read_csv(input_csv_path)
    y = df['label']
    X = df.drop(columns=['patient_id', 'label'])
    logger.info(f"Loaded {X.shape[0]} samples with {X.shape[1]} features.")

    models = {
        'SVM': SVC(probability=True, random_state=42),
        'RandomForest': RandomForestClassifier(random_state=42),
        'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
        'LightGBM': LGBMClassifier(random_state=42),
        'CatBoost': CatBoostClassifier(random_state=42, verbose=0)
    }

    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    results = {}

    for model_name, model in models.items():
        logger.info(f"\n--- Evaluating {model_name} (Cross-Validation) ---")
        
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('smote', SMOTE(random_state=42)),
            ('classifier', model)
        ])

        fold_metrics = []
        for fold, (train_index, test_index) in enumerate(skf.split(X, y)):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)
            y_prob = pipeline.predict_proba(X_test)[:, 1]

            metrics = _calculate_metrics(y_test, y_pred, y_prob)
            fold_metrics.append(metrics)
            logger.info(f"  Fold {fold+1}/{n_splits} - AUROC: {metrics['AUROC']:.4f}, F1: {metrics['F1-Score']:.4f}")

        # Calculate mean and std for each metric
        mean_metrics = pd.DataFrame(fold_metrics).mean()
        std_metrics = pd.DataFrame(fold_metrics).std()
        
        results[model_name] = {
            metric: f"{mean_metrics[metric]:.4f} ± {std_metrics[metric]:.4f}"
            for metric in mean_metrics.index
        }

    logger.info("\n" + "="*60)
    logger.info("Cross-Validation Results Summary (with SMOTE)")
    logger.info("="*60)
    
    results_df = pd.DataFrame(results).T
    print(results_df[['Accuracy', 'F1-Score', 'AUROC', 'Sensitivity', 'Specificity']])


def train_validate_split(train_csv_path: str, val_csv_path_full: str):
    """
    Trains models on a full training set and evaluates them on both the
    training and a separate validation set.
    """
    logger.info(f"Loading SELECTED training data from: {train_csv_path}")
    logger.info(f"Loading FULL validation data from: {val_csv_path_full}")
    train_df = pd.read_csv(train_csv_path)
    val_df_full = pd.read_csv(val_csv_path_full)

    train_feature_columns = train_df.drop(columns=['patient_id', 'label']).columns
    logger.info(f"Using {len(train_feature_columns)} features selected from the training set.")

    y_train_raw = train_df['label']
    X_train_raw = train_df[train_feature_columns]
    
    y_val = val_df_full['label']
    X_val_raw = val_df_full[train_feature_columns]

    imputer = SimpleImputer(strategy='mean')
    X_train = pd.DataFrame(imputer.fit_transform(X_train_raw), columns=train_feature_columns)
    X_val = pd.DataFrame(imputer.transform(X_val_raw), columns=train_feature_columns)

    logger.info(f"Loaded and imputed {len(X_train)} training samples and {len(X_val)} validation samples.")

    models = {
        'SVM': SVC(probability=True, random_state=42),
        'RandomForest': RandomForestClassifier(random_state=42),
        'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
        'LightGBM': LGBMClassifier(random_state=42),
        'CatBoost': CatBoostClassifier(random_state=42, verbose=0)
    }

    train_results, val_results = {}, {}

    for model_name, model in models.items():
        logger.info(f"\n--- Training and Validating {model_name} ---")
        
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            # ('smote', SMOTE(random_state=42)),
            ('classifier', model)
        ])

        pipeline.fit(X_train, y_train_raw)

        # Evaluate on Training Set
        trained_model = pipeline.named_steps['classifier']
        scaler = pipeline.named_steps['scaler']
        X_train_scaled = scaler.transform(X_train)
        y_pred_train = trained_model.predict(X_train_scaled)
        y_prob_train = trained_model.predict_proba(X_train_scaled)[:, 1]
        train_results[model_name] = _calculate_metrics(y_train_raw, y_pred_train, y_prob_train)
        
        # Evaluate on Validation Set
        y_pred_val = pipeline.predict(X_val)
        y_prob_val = pipeline.predict_proba(X_val)[:, 1]
        val_results[model_name] = _calculate_metrics(y_val, y_pred_val, y_prob_val)

    logger.info("\n" + "="*50)
    logger.info("Training Set Performance (Evaluated on original data)")
    logger.info("="*50)
    train_df_results = pd.DataFrame(train_results).T
    print(train_df_results[['Accuracy', 'F1-Score', 'AUROC', 'Sensitivity', 'Specificity']])
    
    logger.info("\n" + "="*50)
    logger.info("Validation Set Performance")
    logger.info("="*50)
    val_df_results = pd.DataFrame(val_results).T
    print(val_df_results[['Accuracy', 'F1-Score', 'AUROC', 'Sensitivity', 'Specificity']])


if __name__ == "__main__":
    # --- Configuration ---
    INPUT_CSV_TRAIN_SELECTED = r"Data/radiomics_features_selected.csv"
    INPUT_CSV_VALIDATION_FULL = r"Data/radiomics_features_validation.csv"
    
    # --- Run the train/validate split function with the correct files ---
    train_validate_split(
        train_csv_path=INPUT_CSV_TRAIN_SELECTED,
        val_csv_path_full=INPUT_CSV_VALIDATION_FULL
    )

    # --- To run the cross-validation function instead, uncomment the lines below ---
    # logger.info("\n\n--- Starting Cross-Validation ---")
    # train_and_evaluate_cv(input_csv_path=INPUT_CSV_TRAIN_SELECTED)
