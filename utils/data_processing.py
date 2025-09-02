import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.impute import SimpleImputer, KNNImputer
from feature_engine.datetime import DatetimeFeatures
from feature_engine.creation import MathFeatures
from feature_engine.outliers import Winsorizer
from typing import Tuple, List, Dict, Any
import warnings
from .error_handler import handle_errors
from .config import CONFIG

warnings.filterwarnings('ignore')

class AdvancedDataProcessor:
    """Enhanced data processing with multiple strategies"""
    
    def __init__(self):
        self.scalers = {}
        self.encoders = {}
        self.imputers = {}
        self.processing_history = []
    
    @handle_errors()
    def clean_and_process_data(self, df: pd.DataFrame, target_column: str, 
                             problem_type: str, processing_options: Dict[str, Any] = None) -> Tuple[pd.DataFrame, List[str]]:
        """
        Enhanced data cleaning and preprocessing
        """
        if processing_options is None:
            processing_options = {}
        
        preprocessing_steps = []
        cleaned_df = df.copy()
        
        # 1. Handle target column missing values
        if cleaned_df[target_column].isnull().any():
            initial_rows = len(cleaned_df)
            cleaned_df.dropna(subset=[target_column], inplace=True)
            rows_removed = initial_rows - len(cleaned_df)
            preprocessing_steps.append(f"✅ Removed {rows_removed} rows with missing target values")
        
        # Separate features and target
        feature_columns = [col for col in cleaned_df.columns if col != target_column]
        numeric_columns = cleaned_df[feature_columns].select_dtypes(include=np.number).columns.tolist()
        categorical_columns = cleaned_df[feature_columns].select_dtypes(include=['object', 'category']).columns.tolist()
        
        # 2. Advanced missing value imputation
        missing_strategy = processing_options.get('missing_strategy', 'auto')
        cleaned_df, missing_steps = self._handle_missing_values(
            cleaned_df, numeric_columns, categorical_columns, missing_strategy
        )
        preprocessing_steps.extend(missing_steps)
        
        # 3. Outlier handling
        outlier_method = processing_options.get('outlier_method', 'iqr')
        cleaned_df, outlier_steps = self._handle_outliers(
            cleaned_df, numeric_columns, outlier_method
        )
        preprocessing_steps.extend(outlier_steps)
        
        # 4. Categorical encoding
        encoding_method = processing_options.get('encoding_method', 'auto')
        cleaned_df, encoding_steps = self._encode_categorical_features(
            cleaned_df, categorical_columns, encoding_method
        )
        preprocessing_steps.extend(encoding_steps)
        
        # 5. Feature scaling
        scaling_method = processing_options.get('scaling_method', 'standard')
        if numeric_columns:
            cleaned_df, scaling_steps = self._scale_numeric_features(
                cleaned_df, numeric_columns, scaling_method
            )
            preprocessing_steps.extend(scaling_steps)
        
        # 6. Handle target encoding for classification
        if problem_type == "classification" and df[target_column].dtype == 'object':
            le_target = LabelEncoder()
            cleaned_df[target_column] = le_target.fit_transform(cleaned_df[target_column])
            self.encoders['target'] = le_target
            preprocessing_steps.append(f"✅ Label encoded target variable: {target_column}")
        
        # 7. Remove duplicates
        initial_rows = len(cleaned_df)
        cleaned_df.drop_duplicates(inplace=True)
        if initial_rows > len(cleaned_df):
            preprocessing_steps.append(f"✅ Removed {initial_rows - len(cleaned_df)} duplicate rows")
        
        # 8. Final validation
        if len(cleaned_df) < 10:
            preprocessing_steps.append("⚠️ Warning: Very small dataset after cleaning")
        
        return cleaned_df, preprocessing_steps
    
    def _handle_missing_values(self, df: pd.DataFrame, numeric_cols: List[str], 
                              categorical_cols: List[str], strategy: str) -> Tuple[pd.DataFrame, List[str]]:
        """Handle missing values with multiple strategies"""
        steps = []
        
        # Numeric columns
        if df[numeric_cols].isnull().sum().sum() > 0:
            if strategy == 'knn':
                imputer = KNNImputer(n_neighbors=5)
                df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
                self.imputers['numeric'] = imputer
                steps.append("✅ Applied KNN imputation for numeric features")
            else:
                # Default: median imputation
                imputer = SimpleImputer(strategy='median')
                df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
                self.imputers['numeric'] = imputer
                steps.append("✅ Applied median imputation for numeric features")
        
        # Categorical columns
        if df[categorical_cols].isnull().sum().sum() > 0:
            for col in categorical_cols:
                if df[col].isnull().any():
                    if strategy == 'mode':
                        mode_value = df[col].mode().iloc[0] if not df[col].mode().empty else 'Unknown'
                    else:
                        mode_value = 'Missing'
                    
                    df[col].fillna(mode_value, inplace=True)
            steps.append("✅ Applied mode/missing imputation for categorical features")
        
        return df, steps
    
    def _handle_outliers(self, df: pd.DataFrame, numeric_cols: List[str], 
                        method: str) -> Tuple[pd.DataFrame, List[str]]:
        """Handle outliers using different methods"""
        steps = []
        outliers_removed = 0
        
        for col in numeric_cols:
            if method == 'iqr':
                Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
                
                outliers = df[(df[col] < lower) | (df[col] > upper)].shape[0]
                if outliers > 0:
                    df[col] = df[col].clip(lower, upper)
                    outliers_removed += outliers
                    
            elif method == 'winsorize':
                winsorizer = Winsorizer(capping_method='iqr', tail='both', fold=1.5)
                df[col] = winsorizer.fit_transform(df[[col]])
                
            elif method == 'zscore':
                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                outliers = (z_scores > 3).sum()
                if outliers > 0:
                    df[col] = df[col].mask(z_scores > 3, df[col].median())
                    outliers_removed += outliers
        
        if outliers_removed > 0:
            steps.append(f"✅ Handled {outliers_removed} outliers using {method} method")
        
        return df, steps
    
    def _encode_categorical_features(self, df: pd.DataFrame, categorical_cols: List[str], 
                                   method: str) -> Tuple[pd.DataFrame, List[str]]:
        """Encode categorical features"""
        steps = []
        
        for col in categorical_cols:
            # Reduce high cardinality
            if df[col].nunique() > 20:
                top_categories = df[col].value_counts().nlargest(19).index
                df[col] = np.where(df[col].isin(top_categories), df[col], 'Other')
                steps.append(f"✅ Reduced cardinality in '{col}' to top 19 + 'Other'")
            
            # Encode
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            self.encoders[col] = le
        
        if categorical_cols:
            steps.append("✅ Label encoded all categorical features")
        
        return df, steps
    
    def _scale_numeric_features(self, df: pd.DataFrame, numeric_cols: List[str], 
                               method: str) -> Tuple[pd.DataFrame, List[str]]:
        """Scale numeric features"""
        steps = []
        
        if method == 'robust':
            scaler = RobustScaler()
            steps.append("✅ Applied robust scaling to numeric features")
        else:
            scaler = StandardScaler()
            steps.append("✅ Applied standard scaling to numeric features")
        
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
        self.scalers['numeric'] = scaler
        
        return df, steps
    
    @handle_errors()
    def perform_feature_engineering(self, df: pd.DataFrame, 
                                  target_column: str = None) -> Tuple[pd.DataFrame, List[str]]:
        """
        Perform automated feature engineering
        """
        df_transformed = df.copy()
        steps_applied = []
        
        # 1. Datetime feature engineering
        datetime_steps = self._create_datetime_features(df_transformed)
        if datetime_steps:
            df_transformed, dt_steps = datetime_steps
            steps_applied.extend(dt_steps)
        
        # 2. Mathematical combinations
        math_steps = self._create_mathematical_features(df_transformed, target_column)
        if math_steps:
            df_transformed, m_steps = math_steps
            steps_applied.extend(m_steps)
        
        # 3. Polynomial features (for small datasets)
        if len(df_transformed) < 5000:
            poly_steps = self._create_polynomial_features(df_transformed, target_column)
            if poly_steps:
                df_transformed, p_steps = poly_steps
                steps_applied.extend(p_steps)
        
        return df_transformed, steps_applied
    
    def _create_datetime_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """Create datetime features"""
        steps = []
        
        # Try to convert object columns to datetime
        for col in df.select_dtypes(include=['object']).columns:
            try:
                df[col] = pd.to_datetime(df[col])
            except (ValueError, TypeError):
                continue
        
        datetime_cols = df.select_dtypes(include=['datetime64', 'datetime']).columns.tolist()
        
        if datetime_cols:
            try:
                dt_creator = DatetimeFeatures(
                    variables=datetime_cols,
                    features_to_extract=['month', 'year', 'day_of_week', 'day_of_year', 'hour'],
                    drop_original=True
                )
                df = dt_creator.fit_transform(df)
                steps.append(f"✅ Created datetime features from: {', '.join(datetime_cols)}")
                return df, steps
            except Exception as e:
                steps.append(f"⚠️ Datetime feature creation failed: {str(e)}")
        
        return None
    
    def _create_mathematical_features(self, df: pd.DataFrame, 
                                    target_column: str = None) -> Tuple[pd.DataFrame, List[str]]:
        """Create mathematical combination features"""
        steps = []
        
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        if target_column and target_column in numeric_cols:
            numeric_cols.remove(target_column)
        
        if len(numeric_cols) >= 2:
            # Select top variance columns
            top_variance_cols = df[numeric_cols].var().nlargest(min(5, len(numeric_cols))).index.tolist()
            
            if len(top_variance_cols) >= 2:
                try:
                    # Create sum and product features
                    df[f'sum_top_{len(top_variance_cols)}'] = df[top_variance_cols].sum(axis=1)
                    df[f'mean_top_{len(top_variance_cols)}'] = df[top_variance_cols].mean(axis=1)
                    df[f'std_top_{len(top_variance_cols)}'] = df[top_variance_cols].std(axis=1)
                    
                    steps.append(f"✅ Created mathematical features from: {', '.join(top_variance_cols[:3])}...")
                    return df, steps
                except Exception as e:
                    steps.append(f"⚠️ Mathematical feature creation failed: {str(e)}")
        
        return None
    
    def _create_polynomial_features(self, df: pd.DataFrame, 
                                  target_column: str = None) -> Tuple[pd.DataFrame, List[str]]:
        """Create polynomial features for small datasets"""
        from sklearn.preprocessing import PolynomialFeatures
        
        steps = []
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        if target_column and target_column in numeric_cols:
            numeric_cols.remove(target_column)
        
        if len(numeric_cols) >= 2 and len(numeric_cols) <= 5:  # Only for 2-5 features
            try:
                poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)
                poly_features = poly.fit_transform(df[numeric_cols])
                
                feature_names = poly.get_feature_names_out(numeric_cols)
                poly_df = pd.DataFrame(poly_features, columns=feature_names, index=df.index)
                
                # Only keep interaction terms (not original features)
                interaction_cols = [col for col in feature_names if ' ' in col][:10]  # Limit to 10
                
                for col in interaction_cols:
                    df[col] = poly_df[col]
                
                steps.append(f"✅ Created {len(interaction_cols)} polynomial interaction features")
                return df, steps
            except Exception as e:
                steps.append(f"⚠️ Polynomial feature creation failed: {str(e)}")
        
        return None

# Global processor instance
processor = AdvancedDataProcessor()
