# utils/data_processing.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import warnings

# This will handle all existing data cleaning
from feature_engine.datetime import DatetimeFeatures
# Corrected import from feature_engine.creation
from feature_engine.creation import MathFeatures

warnings.filterwarnings('ignore')

def clean_and_process_data(df, target_column, problem_type):
    """
    Clean and preprocess the dataset for machine learning.
    """
    preprocessing_steps = []
    cleaned_df = df.copy()
    
    # 1. Handle missing values
    if cleaned_df[target_column].isnull().any():
        initial_rows = len(cleaned_df)
        cleaned_df.dropna(subset=[target_column], inplace=True)
        rows_removed = initial_rows - len(cleaned_df)
        preprocessing_steps.append(f"Removed {rows_removed} rows with missing target values")

    numeric_columns = cleaned_df.select_dtypes(include=np.number).columns.tolist()
    categorical_columns = cleaned_df.select_dtypes(include=['object', 'category']).columns.tolist()

    if target_column in numeric_columns:
        numeric_columns.remove(target_column)
    if target_column in categorical_columns:
        categorical_columns.remove(target_column)
    
    # Impute numeric columns
    if cleaned_df[numeric_columns].isnull().sum().sum() > 0:
        numeric_imputer = SimpleImputer(strategy='median')
        cleaned_df[numeric_columns] = numeric_imputer.fit_transform(cleaned_df[numeric_columns])
        preprocessing_steps.append(f"Imputed missing values in numeric columns with median")

    # Impute categorical columns
    if cleaned_df[categorical_columns].isnull().sum().sum() > 0:
        for col in categorical_columns:
            if cleaned_df[col].isnull().any():
                mode_value = cleaned_df[col].mode().iloc[0] if not cleaned_df[col].mode().empty else 'Unknown'
                cleaned_df[col].fillna(mode_value, inplace=True)
        preprocessing_steps.append("Imputed missing values in categorical columns with mode")

    # 2. Handle categorical variables
    for col in categorical_columns:
        if cleaned_df[col].nunique() > 10:
            top_10 = cleaned_df[col].value_counts().nlargest(10).index
            cleaned_df[col] = np.where(cleaned_df[col].isin(top_10), cleaned_df[col], 'Other')
            preprocessing_steps.append(f"Reduced cardinality in '{col}' to top 10 categories + 'Other'")
        
        le = LabelEncoder()
        cleaned_df[col] = le.fit_transform(cleaned_df[col].astype(str))
    if categorical_columns:
        preprocessing_steps.append("Label encoded categorical features")

    # 3. Handle outliers
    for col in numeric_columns:
        Q1, Q3 = cleaned_df[col].quantile(0.25), cleaned_df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
        outliers = cleaned_df[(cleaned_df[col] < lower) | (cleaned_df[col] > upper)].shape[0]
        if outliers > 0:
            cleaned_df[col] = cleaned_df[col].clip(lower, upper)
            preprocessing_steps.append(f"Capped {outliers} outliers in '{col}'")

    # 4. Feature scaling
    if numeric_columns:
        scaler = StandardScaler()
        cleaned_df[numeric_columns] = scaler.fit_transform(cleaned_df[numeric_columns])
        preprocessing_steps.append("Standardized numeric features")
    
    # 5. Handle target for classification
    if problem_type == "classification" and df[target_column].dtype == 'object':
        le_target = LabelEncoder()
        cleaned_df[target_column] = le_target.fit_transform(cleaned_df[target_column])
        preprocessing_steps.append(f"Label encoded target variable: {target_column}")

    # 6. Remove duplicates
    initial_rows = len(cleaned_df)
    cleaned_df.drop_duplicates(inplace=True)
    if initial_rows > len(cleaned_df):
        preprocessing_steps.append(f"Removed {initial_rows - len(cleaned_df)} duplicate rows")
    
    return cleaned_df, preprocessing_steps

# This function remains unchanged for now
def get_data_insights(df, target_column, problem_type):
    """
    Generate insights about the processed dataset.
    """
    insights = []
    
    # Dataset size insights
    if len(df) > 10000:
        insights.append(f"Large dataset with {len(df):,} samples - excellent for training robust models")
    elif len(df) > 1000:
        insights.append(f"Good dataset size with {len(df):,} samples - sufficient for reliable training")
    else:
        insights.append(f"Moderate dataset size with {len(df):,} samples - may benefit from data augmentation")
    
    return insights

def perform_feature_engineering(df: pd.DataFrame):
    """
    Performs automated feature engineering using the feature-engine library.
    """
    df_transformed = df.copy()
    steps_applied = []

    # 1. Datetime Feature Engineering
    # Convert object columns that might be dates
    for col in df_transformed.select_dtypes(include=['object']).columns:
        try:
            df_transformed[col] = pd.to_datetime(df_transformed[col])
        except (ValueError, TypeError):
            continue # Not a datetime column

    datetime_cols = df_transformed.select_dtypes(include=['datetime64', 'datetime']).columns.tolist()
    if datetime_cols:
        dt_creator = DatetimeFeatures(
            variables=datetime_cols,
            features_to_extract=['month', 'year', 'day_of_week', 'day_of_year'],
            drop_original=True
        )
        df_transformed = dt_creator.fit_transform(df_transformed)
        steps_applied.append(f"Extracted datetime features from: {', '.join(datetime_cols)}")

    # 2. Mathematical Combination for numeric features
    numeric_cols = df_transformed.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    if len(numeric_cols) >= 2:
        top_variance_cols = df_transformed[numeric_cols].var().nlargest(5).index.tolist()
        
        if len(top_variance_cols) >= 2:
            # CORRECTED CLASS NAME AND PARAMETERS
            combiner = MathFeatures(
                variables=top_variance_cols,
                func=['sum', 'prod', 'mean'],
                new_variables_names=['sum_top_vars', 'prod_top_vars', 'mean_top_vars']
            )
            df_transformed = combiner.fit_transform(df_transformed)
            steps_applied.append(f"Created combined mathematical features from: {', '.join(top_variance_cols)}")
            
    return df_transformed, steps_applied