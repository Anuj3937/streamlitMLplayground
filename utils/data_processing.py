import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
import warnings
from feature_engine.datetime import DatetimeFeatures
from feature_engine.creation import MathFeatures
import re

warnings.filterwarnings('ignore')

def identify_and_remove_irrelevant_features(df, target_column):
    """
    Intelligently identify and remove irrelevant features like IDs, constants, etc.
    """
    features_to_remove = []
    removal_reasons = []
    
    for col in df.columns:
        if col == target_column:
            continue
            
        # Check for ID-like columns
        if _is_id_column(col, df[col]):
            features_to_remove.append(col)
            removal_reasons.append(f"'{col}' identified as ID column")
            continue
            
        # Check for constant features
        if df[col].nunique() <= 1:
            features_to_remove.append(col)
            removal_reasons.append(f"'{col}' is constant (only {df[col].nunique()} unique value)")
            continue
            
        # Check for features with too many unique values (likely IDs)
        if df[col].dtype == 'object' and df[col].nunique() > 0.95 * len(df):
            features_to_remove.append(col)
            removal_reasons.append(f"'{col}' has too many unique values ({df[col].nunique()}/{len(df)}) - likely ID")
            continue
            
        # Check for numeric columns that are likely IDs
        if pd.api.types.is_numeric_dtype(df[col]) and df[col].nunique() > 0.95 * len(df):
            features_to_remove.append(col)
            removal_reasons.append(f"'{col}' is numeric with too many unique values - likely ID")
            continue
    
    # Remove identified features
    cleaned_df = df.drop(columns=features_to_remove)
    
    return cleaned_df, features_to_remove, removal_reasons

def _is_id_column(column_name, column_data):
    """
    Check if a column is likely an ID column based on name and data patterns.
    """
    id_patterns = [
        r'.*id$', r'^id.*', r'.*_id$', r'^.*_id_.*$',
        r'.*customer.*id.*', r'.*user.*id.*', r'.*account.*id.*',
        r'.*reference.*', r'.*ref$', r'^ref.*',
        r'.*key$', r'^key.*', r'.*_key$',
        r'.*index$', r'^index.*', r'.*_index$'
    ]
    
    column_name_lower = column_name.lower()
    
    # Check name patterns
    for pattern in id_patterns:
        if re.match(pattern, column_name_lower):
            return True
    
    # Check if all values are unique (strong ID indicator)
    if column_data.nunique() == len(column_data):
        return True
        
    return False

def remove_highly_correlated_features(df, target_column, correlation_threshold=0.95):
    """
    Remove highly correlated features to reduce multicollinearity.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_column in numeric_cols:
        numeric_cols.remove(target_column)
    
    if len(numeric_cols) < 2:
        return df, []
    
    corr_matrix = df[numeric_cols].corr().abs()
    upper_triangle = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    
    features_to_drop = []
    removal_reasons = []
    
    for column in upper_triangle.columns:
        correlated_features = upper_triangle.index[upper_triangle[column] > correlation_threshold].tolist()
        if correlated_features:
            for corr_feature in correlated_features:
                if corr_feature not in features_to_drop:
                    features_to_drop.append(column)
                    removal_reasons.append(f"'{column}' highly correlated with '{corr_feature}' (r={upper_triangle.loc[corr_feature, column]:.3f})")
                    break
    
    cleaned_df = df.drop(columns=features_to_drop)
    return cleaned_df, removal_reasons

def clean_and_process_data(df, target_column, problem_type):
    """
    Enhanced clean and preprocess the dataset for machine learning.
    """
    preprocessing_steps = []
    cleaned_df = df.copy()
    
    # Step 1: Remove irrelevant features (IDs, constants, etc.)
    cleaned_df, removed_features, removal_reasons = identify_and_remove_irrelevant_features(cleaned_df, target_column)
    for reason in removal_reasons:
        preprocessing_steps.append(f"🗑️ {reason}")
    
    # Step 2: Handle missing values in target
    if cleaned_df[target_column].isnull().any():
        initial_rows = len(cleaned_df)
        cleaned_df.dropna(subset=[target_column], inplace=True)
        rows_removed = initial_rows - len(cleaned_df)
        if rows_removed > 0:
            preprocessing_steps.append(f"🧹 Removed {rows_removed} rows with missing target values")

    # Step 3: Identify feature types
    numeric_columns = cleaned_df.select_dtypes(include=np.number).columns.tolist()
    categorical_columns = cleaned_df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    if target_column in numeric_columns:
        numeric_columns.remove(target_column)
    if target_column in categorical_columns:
        categorical_columns.remove(target_column)

    # Step 4: Handle missing values in features
    if len(numeric_columns) > 0 and cleaned_df[numeric_columns].isnull().sum().sum() > 0:
        numeric_imputer = SimpleImputer(strategy='median')
        cleaned_df[numeric_columns] = numeric_imputer.fit_transform(cleaned_df[numeric_columns])
        preprocessing_steps.append(f"🔧 Imputed missing values in numeric columns with median")

    if len(categorical_columns) > 0:
        for col in categorical_columns:
            if cleaned_df[col].isnull().any():
                mode_value = cleaned_df[col].mode().iloc[0] if not cleaned_df[col].mode().empty else 'Unknown'
                cleaned_df[col].fillna(mode_value, inplace=True)
        if any(cleaned_df[categorical_columns].isnull().sum() > 0):
            preprocessing_steps.append("🔧 Imputed missing values in categorical columns with mode")

    # Step 5: Handle high cardinality categorical variables
    for col in categorical_columns:
        unique_count = cleaned_df[col].nunique()
        if unique_count > 20:  # Reduced threshold for better handling
            top_categories = cleaned_df[col].value_counts().nlargest(15).index
            cleaned_df[col] = np.where(cleaned_df[col].isin(top_categories), cleaned_df[col], 'Other')
            preprocessing_steps.append(f"📊 Reduced cardinality in '{col}': kept top 15 categories + 'Other'")

    # Step 6: Encode categorical variables
    if categorical_columns:
        for col in categorical_columns:
            le = LabelEncoder()
            cleaned_df[col] = le.fit_transform(cleaned_df[col].astype(str))
        preprocessing_steps.append("🔢 Label encoded categorical features")

    # Step 7: Remove low variance features
    if len(numeric_columns) > 0:
        feature_cols = [c for c in cleaned_df.columns if c != target_column]
        variance_selector = VarianceThreshold(threshold=0.001)  # Remove near-constant features
        X_selected = variance_selector.fit_transform(cleaned_df[feature_cols])
        selected_features = [feature_cols[i] for i in range(len(feature_cols)) if variance_selector.get_support()[i]]
        
        removed_low_var = len(feature_cols) - len(selected_features)
        if removed_low_var > 0:
            cleaned_df = cleaned_df[selected_features + [target_column]]
            preprocessing_steps.append(f"📉 Removed {removed_low_var} low-variance features")

    # Step 8: Remove highly correlated features
    cleaned_df, correlation_removal_reasons = remove_highly_correlated_features(cleaned_df, target_column)
    for reason in correlation_removal_reasons:
        preprocessing_steps.append(f"🔗 {reason}")

    # Step 9: Handle outliers for numeric features
    numeric_columns = cleaned_df.select_dtypes(include=np.number).columns.tolist()
    if target_column in numeric_columns:
        numeric_columns.remove(target_column)
    
    total_outliers_capped = 0
    for col in numeric_columns:
        Q1, Q3 = cleaned_df[col].quantile(0.25), cleaned_df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
        outliers = cleaned_df[(cleaned_df[col] < lower) | (cleaned_df[col] > upper)].shape[0]
        if outliers > 0:
            cleaned_df[col] = cleaned_df[col].clip(lower, upper)
            total_outliers_capped += outliers
    
    if total_outliers_capped > 0:
        preprocessing_steps.append(f"✂️ Capped {total_outliers_capped} outliers across numeric features")

    # Step 10: Feature scaling
    if len(numeric_columns) > 0:
        scaler = StandardScaler()
        cleaned_df[numeric_columns] = scaler.fit_transform(cleaned_df[numeric_columns])
        preprocessing_steps.append("⚖️ Standardized numeric features")

    # Step 11: Handle target encoding for classification
    if problem_type == "classification" and cleaned_df[target_column].dtype == 'object':
        le_target = LabelEncoder()
        cleaned_df[target_column] = le_target.fit_transform(cleaned_df[target_column])
        preprocessing_steps.append(f"🎯 Label encoded target variable: {target_column}")

    # Step 12: Remove duplicates
    initial_rows = len(cleaned_df)
    cleaned_df.drop_duplicates(inplace=True)
    duplicates_removed = initial_rows - len(cleaned_df)
    if duplicates_removed > 0:
        preprocessing_steps.append(f"🧹 Removed {duplicates_removed} duplicate rows")

    # Final validation
    final_feature_count = len([c for c in cleaned_df.columns if c != target_column])
    preprocessing_steps.append(f"✅ Final dataset: {len(cleaned_df)} samples, {final_feature_count} features")

    return cleaned_df, preprocessing_steps

def perform_feature_engineering(df: pd.DataFrame):
    """
    Enhanced feature engineering with intelligent feature handling.
    """
    df_transformed = df.copy()
    steps_applied = []
    
    # First, identify and handle datetime columns
    datetime_cols = []
    for col in df_transformed.select_dtypes(include=['object']).columns:
        try:
            # Try to convert to datetime
            temp_series = pd.to_datetime(df_transformed[col], errors='coerce')
            if temp_series.notna().sum() > 0.5 * len(df_transformed):  # If >50% are valid dates
                df_transformed[col] = temp_series
                datetime_cols.append(col)
        except (ValueError, TypeError):
            continue

    # Extract features from datetime columns
    if datetime_cols:
        try:
            dt_creator = DatetimeFeatures(
                variables=datetime_cols,
                features_to_extract=['month', 'year', 'day_of_week', 'day_of_year', 'quarter'],
                drop_original=True
            )
            df_transformed = dt_creator.fit_transform(df_transformed)
            steps_applied.append(f"📅 Extracted datetime features from: {', '.join(datetime_cols)}")
        except Exception as e:
            steps_applied.append(f"⚠️ Failed to extract datetime features: {str(e)}")

    # Mathematical combinations for top variance numeric features
    numeric_cols = df_transformed.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    if len(numeric_cols) >= 2:
        # Get top variance features (but not more than 5 to avoid feature explosion)
        top_variance_cols = df_transformed[numeric_cols].var().nlargest(min(5, len(numeric_cols))).index.tolist()
        
        if len(top_variance_cols) >= 2:
            try:
                # Create mathematical combinations
                combiner = MathFeatures(
                    variables=top_variance_cols,
                    func=['sum', 'mean'],  # Reduced functions to avoid too many features
                    new_variables_names=None  # Let it auto-generate names
                )
                df_transformed = combiner.fit_transform(df_transformed)
                steps_applied.append(f"🧮 Created mathematical combinations from top variance features: {', '.join(top_variance_cols)}")
            except Exception as e:
                steps_applied.append(f"⚠️ Failed to create mathematical combinations: {str(e)}")

    return df_transformed, steps_applied

def get_data_insights(df, target_column, problem_type):
    """
    Generate enhanced insights about the processed dataset.
    """
    insights = []
    
    # Dataset size insights
    if len(df) > 10000:
        insights.append(f"📊 Large dataset with {len(df):,} samples - excellent for training robust models")
    elif len(df) > 1000:
        insights.append(f"📊 Good dataset size with {len(df):,} samples - sufficient for reliable training")
    else:
        insights.append(f"📊 Moderate dataset size with {len(df):,} samples - consider data augmentation techniques")
    
    # Feature count insights
    feature_count = len([c for c in df.columns if c != target_column])
    if feature_count > 50:
        insights.append(f"🔢 High-dimensional dataset ({feature_count} features) - feature selection applied")
    elif feature_count < 5:
        insights.append(f"🔢 Low feature count ({feature_count} features) - consider feature engineering")
    
    # Target distribution insights
    if problem_type == "classification":
        class_distribution = df[target_column].value_counts()
        if len(class_distribution) == 2:
            minority_ratio = class_distribution.min() / class_distribution.sum()
            if minority_ratio < 0.1:
                insights.append("⚠️ Highly imbalanced classes detected - consider resampling techniques")
            elif minority_ratio < 0.3:
                insights.append("⚖️ Moderately imbalanced classes - monitor model performance carefully")
    
    return insights
