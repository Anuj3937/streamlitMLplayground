import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

def clean_and_process_data(df, target_column, problem_type):
    """
    Clean and preprocess the dataset for machine learning.
    
    Args:
        df: Raw dataframe
        target_column: Name of target column
        problem_type: 'classification' or 'regression'
    
    Returns:
        cleaned_df: Processed dataframe
        preprocessing_steps: List of applied preprocessing steps
    """
    
    preprocessing_steps = []
    cleaned_df = df.copy()
    
    # 1. Handle missing values
    initial_shape = cleaned_df.shape
    
    # Remove rows with missing target values
    if cleaned_df[target_column].isnull().any():
        cleaned_df = cleaned_df.dropna(subset=[target_column])
        preprocessing_steps.append(f"Removed {initial_shape[0] - len(cleaned_df)} rows with missing target values")
    
    # Handle missing values in features
    numeric_columns = cleaned_df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_columns = cleaned_df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Remove target from feature lists
    if target_column in numeric_columns:
        numeric_columns.remove(target_column)
    if target_column in categorical_columns:
        categorical_columns.remove(target_column)
    
    # Impute numeric columns
    if numeric_columns:
        numeric_imputer = SimpleImputer(strategy='median')
        for col in numeric_columns:
            if cleaned_df[col].isnull().any():
                cleaned_df[col] = numeric_imputer.fit_transform(cleaned_df[[col]]).flatten()
                preprocessing_steps.append(f"Imputed missing values in {col} with median")
    
    # Impute categorical columns
    if categorical_columns:
        for col in categorical_columns:
            if cleaned_df[col].isnull().any():
                mode_value = cleaned_df[col].mode().iloc[0] if len(cleaned_df[col].mode()) > 0 else 'Unknown'
                cleaned_df[col] = cleaned_df[col].fillna(mode_value)
                preprocessing_steps.append(f"Imputed missing values in {col} with mode '{mode_value}'")
    
    # 2. Handle categorical variables
    for col in categorical_columns:
        unique_count = cleaned_df[col].nunique()
        
        if unique_count > 10:
            # High cardinality - keep only top 10 categories
            top_categories = cleaned_df[col].value_counts().head(10).index.tolist()
            cleaned_df[col] = cleaned_df[col].apply(lambda x: x if x in top_categories else 'Other')
            preprocessing_steps.append(f"Reduced {col} cardinality to top 10 categories + 'Other'")
        
        # Label encode categorical variables
        le = LabelEncoder()
        cleaned_df[col] = le.fit_transform(cleaned_df[col].astype(str))
        preprocessing_steps.append(f"Label encoded categorical variable: {col}")
    
    # 3. Handle outliers in numeric columns
    for col in numeric_columns:
        Q1 = cleaned_df[col].quantile(0.25)
        Q3 = cleaned_df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers_count = len(cleaned_df[(cleaned_df[col] < lower_bound) | (cleaned_df[col] > upper_bound)])
        
        if outliers_count > 0:
            # Cap outliers instead of removing them
            cleaned_df[col] = cleaned_df[col].clip(lower=lower_bound, upper=upper_bound)
            preprocessing_steps.append(f"Capped {outliers_count} outliers in {col}")
    
    # 4. Feature scaling for numeric columns (excluding target for regression)
    if problem_type == "regression" and target_column in numeric_columns:
        # Don't scale the target variable for regression
        scale_columns = [col for col in numeric_columns if col != target_column]
    else:
        scale_columns = numeric_columns
    
    if scale_columns:
        scaler = StandardScaler()
        cleaned_df[scale_columns] = scaler.fit_transform(cleaned_df[scale_columns])
        preprocessing_steps.append(f"Standardized {len(scale_columns)} numeric features")
    
    # 5. Handle target variable for classification
    if problem_type == "classification" and target_column in categorical_columns:
        # Target was already label encoded above
        pass
    elif problem_type == "classification" and cleaned_df[target_column].dtype == 'object':
        # Encode string target for classification
        le_target = LabelEncoder()
        cleaned_df[target_column] = le_target.fit_transform(cleaned_df[target_column])
        preprocessing_steps.append(f"Label encoded target variable: {target_column}")
    
    # 6. Remove duplicate rows
    initial_rows = len(cleaned_df)
    cleaned_df = cleaned_df.drop_duplicates()
    duplicates_removed = initial_rows - len(cleaned_df)
    if duplicates_removed > 0:
        preprocessing_steps.append(f"Removed {duplicates_removed} duplicate rows")
    
    # 7. Final data type optimization
    for col in cleaned_df.columns:
        if cleaned_df[col].dtype == 'float64':
            # Check if can be converted to int
            if cleaned_df[col].apply(lambda x: float(x).is_integer()).all():
                cleaned_df[col] = cleaned_df[col].astype('int64')
    
    preprocessing_steps.append(f"Final dataset shape: {cleaned_df.shape}")
    
    return cleaned_df, preprocessing_steps

def get_data_insights(df, target_column, problem_type):
    """
    Generate insights about the processed dataset.
    
    Args:
        df: Processed dataframe
        target_column: Name of target column
        problem_type: 'classification' or 'regression'
    
    Returns:
        insights: List of insight strings
    """
    
    insights = []
    
    # Dataset size insights
    if len(df) > 10000:
        insights.append(f"Large dataset with {len(df):,} samples - excellent for training robust models")
    elif len(df) > 1000:
        insights.append(f"Good dataset size with {len(df):,} samples - sufficient for reliable training")
    else:
        insights.append(f"Moderate dataset size with {len(df):,} samples - may benefit from data augmentation")
    
    # Feature count insights
    feature_count = len(df.columns) - 1
    if feature_count > 50:
        insights.append(f"High-dimensional dataset with {feature_count} features - consider feature selection")
    elif feature_count > 10:
        insights.append(f"Rich feature set with {feature_count} features - good for complex pattern learning")
    else:
        insights.append(f"Compact feature set with {feature_count} features - interpretable model expected")
    
    # Target variable insights
    if problem_type == "classification":
        class_counts = df[target_column].value_counts()
        unique_classes = len(class_counts)
        
        if unique_classes == 2:
            insights.append("Binary classification problem - consider logistic regression and tree-based models")
        else:
            insights.append(f"Multi-class classification with {unique_classes} classes")
        
        # Class balance
        min_class_size = class_counts.min()
        max_class_size = class_counts.max()
        imbalance_ratio = max_class_size / min_class_size
        
        if imbalance_ratio > 10:
            insights.append("Highly imbalanced classes detected - consider SMOTE or class weights")
        elif imbalance_ratio > 3:
            insights.append("Moderate class imbalance - monitor precision/recall metrics")
        else:
            insights.append("Well-balanced classes - accuracy will be a reliable metric")
    
    else:  # Regression
        target_std = df[target_column].std()
        target_mean = df[target_column].mean()
        
        if target_std / abs(target_mean) > 1:
            insights.append("High target variability - ensemble methods may perform well")
        else:
            insights.append("Moderate target variability - linear models may be effective")
    
    # Feature correlation insights
    numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_features) > 2:
        corr_matrix = df[numeric_features].corr()
        high_corr_pairs = []
        
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if abs(corr_matrix.iloc[i, j]) > 0.8:
                    high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j]))
        
        if len(high_corr_pairs) > 0:
            insights.append(f"Found {len(high_corr_pairs)} highly correlated feature pairs - consider dimensionality reduction")
        else:
            insights.append("Low feature correlation - good feature diversity for model training")
    
    # Data quality insights
    missing_percentage = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
    if missing_percentage == 0:
        insights.append("Perfect data quality - no missing values detected")
    elif missing_percentage < 5:
        insights.append("Excellent data quality - minimal missing values")
    else:
        insights.append(f"Data quality concerns - {missing_percentage:.1f}% missing values handled")
    
    return insights
