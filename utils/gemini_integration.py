import streamlit as st
import google.generativeai as genai
import pandas as pd
from typing import List
from .error_handler import handle_errors


def configure_gemini():
    """Configures the Gemini API key."""
    try:
        genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    except Exception as e:
        st.error(f"Failed to configure Gemini API: {e}")
        st.info("Please ensure your GEMINI_API_KEY is set in .streamlit/secrets.toml")
        st.stop()


def get_gemini_model():
    """Returns an instance of the Gemini Pro model."""
    return genai.GenerativeModel('gemini-1.5-flash')


@handle_errors()
def get_insights_with_gemini(df: pd.DataFrame, target_column: str) -> List[str]:
    """Generate deep insights about the dataset using Gemini AI"""
    configure_gemini()
    model = get_gemini_model()
    
    # Create a comprehensive dataset summary
    summary_stats = df.describe(include='all').to_string()
    data_types = df.dtypes.to_string()
    missing_info = df.isnull().sum().to_string()
    correlation_info = ""
    
    # Get correlation info for numeric columns
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    if len(numeric_cols) > 1:
        corr_matrix = df[numeric_cols].corr()
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.7:
                    high_corr_pairs.append(f"{corr_matrix.columns[i]} <-> {corr_matrix.columns[j]}: {corr_val:.3f}")
        
        if high_corr_pairs:
            correlation_info = "High correlations found:\n" + "\n".join(high_corr_pairs)
    
    prompt = f"""
    As an expert data scientist, analyze this dataset and provide 5-7 key insights in bullet points.
    
    Dataset Information:
    - Target column: '{target_column}'
    - Shape: {df.shape}
    - Columns: {list(df.columns)}
    
    Statistical Summary:
    {summary_stats}
    
    Data Types:
    {data_types}
    
    Missing Values:
    {missing_info}
    
    {correlation_info}
    
    Target Analysis:
    - Unique values in target: {df[target_column].nunique()}
    - Target distribution: {df[target_column].value_counts().head().to_dict()}
    
    Provide insights about:
    1. Data quality and completeness
    2. Target variable characteristics
    3. Feature relationships and patterns
    4. Potential data quality issues
    5. Recommendations for model improvement
    
    Format as bullet points starting with "•"
    """
    
    try:
        response = model.generate_content(prompt)
        insights = [line.strip() for line in response.text.split('•') if line.strip()]
        return [insight for insight in insights if len(insight) > 10]  # Filter out short/empty insights
    except Exception as e:
        st.warning(f"AI insights unavailable: {str(e)}")
        return ["Failed to generate AI-powered insights."]


@handle_errors()
def generate_code_with_gemini(df: pd.DataFrame, target_column: str, problem_type: str) -> str:
    """Generate comprehensive ML pipeline code using Gemini"""
    configure_gemini()
    model = get_gemini_model()
    
    # Get dataset characteristics
    feature_info = []
    for col in df.columns:
        if col != target_column:
            dtype = str(df[col].dtype)
            nunique = df[col].nunique()
            missing = df[col].isnull().sum()
            feature_info.append(f"  {col}: {dtype}, {nunique} unique, {missing} missing")
    
    feature_summary = "\n".join(feature_info[:10])  # Limit to first 10 features
    
    prompt = f"""
You are an expert machine learning engineer. Generate a complete, production-ready Python script for training and evaluating ML models.

DATASET INFORMATION:
- Problem Type: {problem_type}
- Target Column: '{target_column}'
- Dataset Shape: {df.shape}
- Target Distribution: {df[target_column].value_counts().head().to_dict()}

FEATURES SUMMARY:
{feature_summary}

REQUIREMENTS:
The script must include:

1. IMPORTS: Import all necessary libraries
2. DATA PREPARATION: Define X (features) and y (target)
3. TRAIN-TEST SPLIT: 80/20 split with random_state=42
4. MODEL DEFINITIONS: At least 6 different algorithms appropriate for {problem_type}
5. HYPERPARAMETER OPTIMIZATION: Use Optuna for each model (20 trials max)
6. CROSS-VALIDATION: 5-fold CV during optimization
7. MODEL TRAINING: Train each optimized model
8. EVALUATION: Comprehensive metrics on test set
9. RESULTS COMPILATION: Store all results in 'results' list
10. BEST MODEL SELECTION: Identify best performing model

SPECIFIC REQUIREMENTS:
- Use appropriate algorithms for {problem_type}
- Include RandomForest, XGBoost, and LightGBM
- Add LogisticRegression/LinearRegression as baseline
- Include at least 2 additional algorithms
- Each model should have hyperparameter optimization
- Store results as list of dictionaries with keys: 'model', performance metrics, 'training_time', 'best_params'
- Create variables: 'results_df' (DataFrame), 'best_model_name' (string), 'best_model' (trained model object)
- Include feature importance extraction where possible
- Add proper error handling

PERFORMANCE METRICS:
- Classification: accuracy, precision, recall, f1_score
- Regression: r2_score, mse, rmse, mae

OUTPUT FORMAT:
- Pure Python code only
- No markdown formatting
- No explanatory text
- Executable code that defines all required variables
- Include print statements showing training progress

The code should be production-ready and handle the specific dataset characteristics mentioned above.
"""
    
    try:
        response = model.generate_content(prompt)
        code = response.text.strip()
        
        # Clean up the code
        if code.startswith('```'):
            code = code[9:]
        elif code.startswith('```'):
            code = code[3:]
        if code.endswith('```'):
            code = code[:-3]
        
        return code.strip()
    except Exception as e:
        st.error(f"Code generation failed: {str(e)}")
        return generate_fallback_code(df, target_column, problem_type)


def generate_fallback_code(df: pd.DataFrame, target_column: str, problem_type: str) -> str:
    """Generate fallback ML code if Gemini fails"""
    
    if problem_type == "classification":
        return f"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import xgboost as xgb
import time

# Prepare data
X = df.drop('{target_column}', axis=1)
y = df['{target_column}']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("Starting model training...")

# Define models
models = {{
    'RandomForest': RandomForestClassifier(random_state=42),
    'XGBoost': xgb.XGBClassifier(random_state=42, eval_metric='logloss'),
    'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000)
}}

results = []

for name, model in models.items():
    print(f"Training {{name}}...")
    start_time = time.time()
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    
    # Train final model
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    training_time = time.time() - start_time
    
    results.append({{
        'model': name,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'cv_score': cv_scores.mean(),
        'training_time': training_time,
        'best_params': {{}}
    }})
    
    print(f"{{name}} accuracy: {{accuracy:.4f}}")

# Create results DataFrame
results_df = pd.DataFrame(results)

# Find best model
best_model_name = results_df.loc[results_df['accuracy'].idxmax(), 'model']
best_model = models[best_model_name]

# Feature importance
feature_importance = None
if hasattr(best_model, 'feature_importances_'):
    feature_importance = {{
        'features': X.columns.tolist(),
        'importance': best_model.feature_importances_.tolist()
    }}

print(f"Best model: {{best_model_name}}")
"""
    else:  # regression
        return f"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import xgboost as xgb
import time

# Prepare data
X = df.drop('{target_column}', axis=1)
y = df['{target_column}']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Starting model training...")

# Define models
models = {{
    'RandomForest': RandomForestRegressor(random_state=42),
    'XGBoost': xgb.XGBRegressor(random_state=42),
    'LinearRegression': LinearRegression()
}}

results = []

for name, model in models.items():
    print(f"Training {{name}}...")
    start_time = time.time()
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
    
    # Train final model
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    
    # Metrics
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    
    training_time = time.time() - start_time
    
    results.append({{
        'model': name,
        'r2_score': r2,
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'cv_score': cv_scores.mean(),
        'training_time': training_time,
        'best_params': {{}}
    }})
    
    print(f"{{name}} R2 score: {{r2:.4f}}")

# Create results DataFrame
results_df = pd.DataFrame(results)

# Find best model
best_model_name = results_df.loc[results_df['r2_score'].idxmax(), 'model']
best_model = models[best_model_name]

# Feature importance
feature_importance = None
if hasattr(best_model, 'feature_importances_'):
    feature_importance = {{
        'features': X.columns.tolist(),
        'importance': best_model.feature_importances_.tolist()
    }}

print(f"Best model: {{best_model_name}}")
"""


@handle_errors()
def explain_code_with_gemini(code: str) -> str:
    """Generate detailed code explanation using Gemini"""
    configure_gemini()
    model = get_gemini_model()
    
    prompt = f"""
As a senior machine learning engineer, provide a clear, structured explanation of this ML pipeline code.

CODE TO EXPLAIN:
{code[:3000]}  # Limit code length

Please explain:
1. **Data Preparation**: How the data is prepared for training
2. **Model Selection**: Which algorithms are used and why
3. **Hyperparameter Optimization**: How parameters are tuned
4. **Training Process**: How models are trained and validated
5. **Evaluation**: How model performance is measured
6. **Results**: How the best model is selected

Use markdown formatting with clear headings and bullet points.
Keep explanations practical and focused on what the code actually does.
"""
    
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"""
## Code Explanation

This ML pipeline performs the following key steps:

### 🔧 Data Preparation
- Separates features (X) and target variable (y)
- Splits data into training (80%) and testing (20%) sets
- Ensures reproducible results with random_state=42

### 🤖 Model Training
- Trains multiple machine learning algorithms
- Uses cross-validation to assess model reliability
- Applies hyperparameter optimization where specified

### 📊 Performance Evaluation
- Calculates comprehensive performance metrics
- Compares models using consistent evaluation criteria
- Selects the best performing model automatically

### 🎯 Results
- Generates detailed performance comparison
- Identifies the optimal model for deployment
- Provides feature importance analysis when available

**Note:** Code explanation generation failed: {str(e)}
"""


# Additional utility functions for better error handling
def validate_gemini_response(response_text: str) -> bool:
    """Validate that Gemini response is meaningful"""
    if not response_text or len(response_text.strip()) < 10:
        return False
    
    # Check for common error indicators
    error_indicators = ['error', 'failed', 'unable', 'cannot generate']
    if any(indicator in response_text.lower() for indicator in error_indicators):
        return False
    
    return True


def format_insights(raw_insights: str) -> List[str]:
    """Format raw Gemini insights into clean bullet points"""
    insights = []
    
    # Split by lines and clean up
    lines = raw_insights.split('\n')
    
    for line in lines:
        line = line.strip()
        
        # Skip empty lines
        if not line:
            continue
            
        # Remove bullet point symbols
        line = line.lstrip('- -*').strip()
        
        # Skip lines that are too short to be meaningful
        if len(line) < 20:
            continue
            
        # Add to insights if it's meaningful
        if line and not line.startswith('#'):
            insights.append(line)
    
    return insights[:7]  # Return max 7 insights


def create_fallback_insights(df: pd.DataFrame, target_column: str) -> List[str]:
    """Create basic insights when Gemini fails"""
    insights = []
    
    # Dataset size insight
    insights.append(f"Dataset contains {len(df):,} samples with {len(df.columns):,} features")
    
    # Missing data insight
    missing_pct = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
    if missing_pct > 0:
        insights.append(f"Dataset has {missing_pct:.1f}% missing values that need attention")
    else:
        insights.append("Dataset is complete with no missing values")
    
    # Target variable insight
    if df[target_column].dtype in ['object', 'category']:
        unique_targets = df[target_column].nunique()
        insights.append(f"Target variable '{target_column}' has {unique_targets} unique classes")
    else:
        target_range = df[target_column].max() - df[target_column].min()
        insights.append(f"Target variable '{target_column}' ranges from {df[target_column].min():.2f} to {df[target_column].max():.2f}")
    
    # Feature types insight
    numeric_features = len(df.select_dtypes(include=['int64', 'float64']).columns)
    categorical_features = len(df.select_dtypes(include=['object', 'category']).columns)
    insights.append(f"Dataset has {numeric_features} numeric and {categorical_features} categorical features")
    
    return insights
