import pandas as pd
import numpy as np
from sklearn.feature_selection import (
    SelectKBest, f_classif, mutual_info_classif, 
    RFE, SelectFromModel, VarianceThreshold
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LassoCV, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
import warnings
warnings.filterwarnings('ignore')

class RadiomicsFeatureSelector:
    """
    Comprehensive feature selection for radiomics features
    """
    
    def __init__(self, data_path="Data/radiomics_features.csv", 
                 output_path="Data/radiomics_features_selected.csv",
                 correlation_threshold=0.95,
                 target_column=None):
        """
        Initialize the feature selector
        
        Args:
            data_path: Path to input radiomics features CSV
            output_path: Path to save selected features CSV
            correlation_threshold: Threshold for removing highly correlated features
            target_column: Name of target column (if None, will try to detect)
        """
        self.data_path = data_path
        self.output_path = output_path
        self.correlation_threshold = correlation_threshold
        self.target_column = target_column
        self.data = None
        self.features = None
        self.target = None
        self.selected_features = None
        
    def load_data(self):
        """Load the radiomics features data"""
        print(f"Loading data from {self.data_path}...")
        self.data = pd.read_csv(self.data_path)
        print(f"Data shape: {self.data.shape}")
        print(f"Columns: {list(self.data.columns)}")
        
        # Try to identify target column if not specified
        if self.target_column is None:
            potential_targets = ['survival', 'status', 'label', 'target', 'outcome', 'class']
            for col in potential_targets:
                if col in self.data.columns:
                    self.target_column = col
                    break
            
            if self.target_column is None:
                print("Warning: No target column detected. Using last column as target.")
                self.target_column = self.data.columns[-1]
        
        print(f"Using target column: {self.target_column}")
        
        # Separate features and target
        self.features = self.data.drop(columns=[self.target_column])
        self.target = self.data[self.target_column]
        
        print(f"Features shape: {self.features.shape}")
        print(f"Target shape: {self.target.shape}")
        
        return self.data
    
    def remove_constant_features(self):
        """Remove features with zero variance"""
        print("\n1. Removing constant features...")
        initial_count = len(self.features.columns)
        
        # Remove features with zero variance
        variance_selector = VarianceThreshold(threshold=0.0)
        variance_selector.fit(self.features)
        
        # Get feature names that have variance
        feature_mask = variance_selector.get_support()
        self.features = self.features.loc[:, feature_mask]
        
        removed_count = initial_count - len(self.features.columns)
        print(f"Removed {removed_count} constant features")
        print(f"Remaining features: {len(self.features.columns)}")
        
        return self.features
    
    def remove_highly_correlated_features(self):
        """Remove highly correlated features based on correlation threshold"""
        print(f"\n2. Removing highly correlated features (threshold: {self.correlation_threshold})...")
        initial_count = len(self.features.columns)
        
        # Calculate correlation matrix
        corr_matrix = self.features.corr().abs()
        
        # Create mask for upper triangle
        upper_tri = np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        
        # Find highly correlated feature pairs
        high_corr_pairs = np.where((corr_matrix > self.correlation_threshold) & upper_tri)
        
        # Get features to remove (keep the first occurrence)
        features_to_remove = set()
        for i, j in zip(high_corr_pairs[0], high_corr_pairs[1]):
            features_to_remove.add(self.features.columns[j])
        
        # Remove highly correlated features
        self.features = self.features.drop(columns=list(features_to_remove))
        
        removed_count = initial_count - len(self.features.columns)
        print(f"Removed {removed_count} highly correlated features")
        print(f"Remaining features: {len(self.features.columns)}")
        
        return self.features
    
    def advanced_feature_selection(self, max_features=100):
        """Apply advanced feature selection methods"""
        print(f"\n3. Applying advanced feature selection (target: {max_features} features)...")
        initial_count = len(self.features.columns)
        
        # Standardize features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(self.features)
        
        # Method 1: Univariate feature selection using F-statistic
        print("   - Univariate selection (F-statistic)...")
        f_selector = SelectKBest(score_func=f_classif, k=min(max_features * 2, initial_count))
        f_selector.fit(features_scaled, self.target)
        f_scores = f_selector.scores_
        f_pvalues = f_selector.pvalues_
        
        # Method 2: Mutual information
        print("   - Mutual information selection...")
        mi_selector = SelectKBest(score_func=mutual_info_classif, k=min(max_features * 2, initial_count))
        mi_selector.fit(features_scaled, self.target)
        mi_scores = mi_selector.scores_
        
        # Method 3: Lasso-based selection
        print("   - Lasso-based selection...")
        try:
            lasso = LassoCV(cv=5, random_state=42, max_iter=1000)
            lasso.fit(features_scaled, self.target)
            lasso_coefs = np.abs(lasso.coef_)
        except:
            print("   - Lasso failed, using default coefficients")
            lasso_coefs = np.ones(initial_count)
        
        # Method 4: Random Forest importance
        print("   - Random Forest importance...")
        try:
            rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            rf.fit(features_scaled, self.target)
            rf_importance = rf.feature_importances_
        except:
            print("   - Random Forest failed, using default importance")
            rf_importance = np.ones(initial_count)
        
        # Combine all scores
        feature_scores = pd.DataFrame({
            'feature': self.features.columns,
            'f_score': f_scores,
            'mi_score': mi_scores,
            'lasso_coef': lasso_coefs,
            'rf_importance': rf_importance
        })
        
        # Normalize scores to 0-1 range
        for col in ['f_score', 'mi_score', 'lasso_coef', 'rf_importance']:
            feature_scores[col] = (feature_scores[col] - feature_scores[col].min()) / \
                                 (feature_scores[col].max() - feature_scores[col].min())
        
        # Calculate combined score
        feature_scores['combined_score'] = (
            feature_scores['f_score'] + 
            feature_scores['mi_score'] + 
            feature_scores['lasso_coef'] + 
            feature_scores['rf_importance']
        ) / 4
        
        # Select top features
        feature_scores = feature_scores.sort_values('combined_score', ascending=False)
        top_features = feature_scores.head(max_features)['feature'].tolist()
        
        # Apply selection
        self.features = self.features[top_features]
        
        final_count = len(self.features.columns)
        print(f"Selected {final_count} features using advanced methods")
        
        return self.features, feature_scores
    
    def save_selected_features(self):
        """Save the selected features to CSV"""
        print(f"\n4. Saving selected features to {self.output_path}...")
        
        # Combine features and target
        selected_data = pd.concat([self.features, self.target], axis=1)
        
        # Save to CSV
        selected_data.to_csv(self.output_path, index=False)
        print(f"Saved {len(selected_data.columns)} columns to {self.output_path}")
        
        return selected_data
    
    def create_selection_report(self, feature_scores):
        """Create a comprehensive report of the feature selection process"""
        print("\n5. Creating feature selection report...")
        
        # Create output directory if it doesn't exist
        import os
        os.makedirs('outputs', exist_ok=True)
        
        # Plot feature importance scores
        plt.figure(figsize=(15, 10))
        
        # Top 20 features by combined score
        plt.subplot(2, 2, 1)
        top_20 = feature_scores.head(20)
        plt.barh(range(len(top_20)), top_20['combined_score'])
        plt.yticks(range(len(top_20)), top_20['feature'], fontsize=8)
        plt.xlabel('Combined Score')
        plt.title('Top 20 Features by Combined Score')
        plt.gca().invert_yaxis()
        
        # Score distribution
        plt.subplot(2, 2, 2)
        plt.hist(feature_scores['combined_score'], bins=30, alpha=0.7)
        plt.xlabel('Combined Score')
        plt.ylabel('Frequency')
        plt.title('Distribution of Feature Scores')
        
        # Correlation heatmap of selected features
        plt.subplot(2, 2, 3)
        corr_matrix = self.features.corr()
        sns.heatmap(corr_matrix, cmap='coolwarm', center=0, 
                   square=True, cbar_kws={"shrink": .8})
        plt.title('Correlation Matrix of Selected Features')
        
        # Feature scores comparison
        plt.subplot(2, 2, 4)
        score_cols = ['f_score', 'mi_score', 'lasso_coef', 'rf_importance']
        feature_scores[score_cols].boxplot()
        plt.xticks(rotation=45)
        plt.ylabel('Normalized Score')
        plt.title('Feature Scores by Method')
        
        plt.tight_layout()
        plt.savefig('outputs/feature_selection_report.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Save feature scores
        feature_scores.to_csv('outputs/feature_scores.csv', index=False)
        print("Report saved to outputs/feature_selection_report.png")
        print("Feature scores saved to outputs/feature_scores.csv")
    
    def run_feature_selection(self, max_features=100):
        """Run the complete feature selection pipeline"""
        print("=" * 60)
        print("RADIOMICS FEATURE SELECTION PIPELINE")
        print("=" * 60)
        
        # Load data
        self.load_data()
        
        # Remove constant features
        self.remove_constant_features()
        
        # Remove highly correlated features
        self.remove_highly_correlated_features()
        
        # Apply advanced feature selection
        self.features, feature_scores = self.advanced_feature_selection(max_features)
        
        # Save selected features
        selected_data = self.save_selected_features()
        
        # Create report
        self.create_selection_report(feature_scores)
        
        print("\n" + "=" * 60)
        print("FEATURE SELECTION COMPLETED!")
        print(f"Initial features: {len(self.data.columns) - 1}")
        print(f"Final features: {len(self.features.columns)}")
        print(f"Reduction: {((len(self.data.columns) - 1 - len(self.features.columns)) / (len(self.data.columns) - 1) * 100):.1f}%")
        print("=" * 60)
        
        return selected_data, feature_scores

def main():
    """Main function to run feature selection"""
    # Initialize feature selector
    selector = RadiomicsFeatureSelector(
        data_path="Data/radiomics_features_validation.csv",
        output_path="Data/radiomics_features_selected_validation.csv",
        correlation_threshold=0.95,
        target_column=None  # Will auto-detect
    )
    
    # Run feature selection
    selected_data, feature_scores = selector.run_feature_selection(max_features=100)
    
    print(f"\nSelected features saved to: {selector.output_path}")
    print(f"Feature selection report saved to: outputs/")

if __name__ == "__main__":
    main()
