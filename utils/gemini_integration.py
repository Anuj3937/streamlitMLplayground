import re
import streamlit as st
import google.generativeai as genai
import pandas as pd

# -----------------------------
# Gemini configuration
# -----------------------------

def configure_gemini():
    try:
        genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    except Exception as e:
        st.error(f"Failed to configure Gemini API: {e}. Set GEMINI_API_KEY in .streamlit/secrets.toml")
        st.stop()

@st.cache_resource
def get_gemini_model():
    return genai.GenerativeModel("gemini-1.5-flash-latest")

# -----------------------------
# Insights function (exported)
# -----------------------------

def get_insights_with_gemini(df: pd.DataFrame, target_column: str):
    """
    Generate concise insights about the dataset using Gemini.
    """
    configure_gemini()
    model = get_gemini_model()

    prompt = f"""
    As an expert data scientist, analyze the dataset summary and provide concise, actionable insights.
    Use bullet points only. Focus on practical ML considerations.

    Dataset Info:
    - Rows: {len(df)}
    - Features: {len(df.columns) - 1}
    - Target: {target_column}

    Sample Data:
    {df.head(3).to_string()}

    Statistical Summary:
    {df.describe().to_string()}

    Target Distribution:
    {df[target_column].value_counts().head().to_string()}

    Provide 4-6 insights covering:
    - Data quality and preprocessing effectiveness
    - Feature relationships and importance indicators  
    - Potential modeling challenges and recommendations
    - Class balance or target distribution insights
    """

    try:
        response = model.generate_content(prompt)
        insights = [ln.strip("- ").strip() for ln in response.text.splitlines() if ln.strip() and not ln.strip().startswith('#')]
        return [i for i in insights if len(i) > 10][:6]  # Limit to 6 insights
    except Exception as e:
        st.error(f"Error generating insights with Gemini: {e}")
        return ["AI insights temporarily unavailable. Dataset appears ready for modeling."]

# -----------------------------
# Robust training code template
# -----------------------------

def _get_robust_training_template():
    """
    Returns a bulletproof training script template.
    """
    return """
import numpy as np
import pandas as pd
import time
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    r2_score, mean_squared_error, mean_absolute_error, confusion_matrix
)
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
import optuna
from optuna.samplers import TPESampler

# Prepare data
feature_cols = [c for c in df.columns if c != target_column]
X = df[feature_cols]
y = df[target_column]

print(f"Training with {len(X)} samples and {len(feature_cols)} features")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42,
    stratify=y if problem_type == "classification" else None
)

# Define model candidates
if problem_type == "classification":
    model_configs = {
        "LogisticRegression": {
            "model_class": LogisticRegression,
            "params": {
                "C": (1e-4, 100, "log"),
                "max_iter": 2000,
                "random_state": 42,
                "solver": (["lbfgs", "liblinear"], "categorical")
            }
        },
        "RandomForest": {
            "model_class": RandomForestClassifier,
            "params": {
                "n_estimators": (50, 300, "int"),
                "max_depth": (3, 20, "int"),
                "min_samples_split": (2, 10, "int"),
                "min_samples_leaf": (1, 5, "int"),
                "random_state": 42
            }
        },
        "DecisionTree": {
            "model_class": DecisionTreeClassifier,
            "params": {
                "max_depth": (3, 15, "int"),
                "min_samples_split": (2, 10, "int"),
                "min_samples_leaf": (1, 5, "int"),
                "random_state": 42
            }
        }
    }
    scoring_metric = "f1_weighted"
else:
    model_configs = {
        "LinearRegression": {
            "model_class": LinearRegression,
            "params": {}
        },
        "RandomForest": {
            "model_class": RandomForestRegressor,
            "params": {
                "n_estimators": (50, 300, "int"),
                "max_depth": (3, 20, "int"),
                "min_samples_split": (2, 10, "int"),
                "min_samples_leaf": (1, 5, "int"),
                "random_state": 42
            }
        },
        "DecisionTree": {
            "model_class": DecisionTreeRegressor,
            "params": {
                "max_depth": (3, 15, "int"),
                "min_samples_split": (2, 10, "int"),
                "min_samples_leaf": (1, 5, "int"),
                "random_state": 42
            }
        }
    }
    scoring_metric = "r2"

def create_objective(model_name, config):
    def objective(trial):
        # Build parameters for this trial
        trial_params = {}
        for param_name, param_config in config["params"].items():
            if isinstance(param_config, tuple) and len(param_config) == 3:
                if param_config[2] == "log":
                    trial_params[param_name] = trial.suggest_float(param_name, param_config[0], param_config[1], log=True)
                elif param_config[2] == "int":
                    trial_params[param_name] = trial.suggest_int(param_name, param_config[0], param_config[1])
                elif param_config[2] == "categorical":
                    trial_params[param_name] = trial.suggest_categorical(param_name, param_config[0])
            elif isinstance(param_config, tuple) and len(param_config) == 2:
                if param_config[1] == "categorical":
                    trial_params[param_name] = trial.suggest_categorical(param_name, param_config[0])
            else:
                trial_params[param_name] = param_config
        
        # Create model with trial parameters
        model = config["model_class"](**trial_params)
        
        # Cross-validation score
        try:
            scores = cross_val_score(model, X_train, y_train, cv=3, scoring=scoring_metric)
            return float(np.mean(scores))
        except Exception as e:
            print(f"Trial failed for {model_name}: {e}")
            return -999.0  # Very bad score for failed trials
    
    return objective

# Train and optimize each model
results = []
best_model_overall = None
best_score_overall = -np.inf
best_model_name = None

for model_name, config in model_configs.items():
    print(f"\\nOptimizing {model_name}...")
    start_time = time.time()
    
    try:
        # Create and run optimization study
        study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=42))
        
        # Determine number of trials based on parameter complexity
        n_trials = 20 if len(config["params"]) > 0 else 1
        study.optimize(create_objective(model_name, config), n_trials=n_trials, show_progress_bar=False)
        
        # Get best parameters and retrain
        best_params = study.best_params
        
        # Create final model with best parameters
        final_params = {}
        for param_name, param_config in config["params"].items():
            if param_name in best_params:
                final_params[param_name] = best_params[param_name]
            elif not isinstance(param_config, tuple):
                final_params[param_name] = param_config
        
        model = config["model_class"](**final_params)
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        training_time = time.time() - start_time
        
        # Calculate metrics
        if problem_type == "classification":
            accuracy = float(accuracy_score(y_test, y_pred))
            precision = float(precision_score(y_test, y_pred, average='weighted', zero_division=0))
            recall = float(recall_score(y_test, y_pred, average='weighted', zero_division=0))
            f1 = float(f1_score(y_test, y_pred, average='weighted', zero_division=0))
            
            results.append({
                "model": model_name,
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "best_params": best_params,
                "training_time": training_time
            })
            
            score_for_comparison = f1  # Use F1 for model selection
            
        else:
            r2 = float(r2_score(y_test, y_pred))
            mse = float(mean_squared_error(y_test, y_pred))
            mae = float(mean_absolute_error(y_test, y_pred))
            rmse = float(np.sqrt(mse))
            
            results.append({
                "model": model_name,
                "r2_score": r2,
                "mse": mse,
                "mae": mae,
                "rmse": rmse,
                "best_params": best_params,
                "training_time": training_time
            })
            
            score_for_comparison = r2  # Use R2 for model selection
        
        # Track best model
        if score_for_comparison > best_score_overall:
            best_score_overall = score_for_comparison
            best_model_overall = model
            best_model_name = model_name
            
        print(f"{model_name} completed - Score: {score_for_comparison:.4f}")
        
    except Exception as e:
        print(f"Failed to train {model_name}: {e}")
        continue

# Create results DataFrame
results_df = pd.DataFrame(results)

# Prepare best model information
if best_model_overall is not None:
    # Feature importance
    feature_importance = {}
    if hasattr(best_model_overall, "feature_importances_"):
        feature_importance = dict(zip(feature_cols, best_model_overall.feature_importances_.tolist()))
    elif hasattr(best_model_overall, "coef_"):
        coef = np.ravel(best_model_overall.coef_) if best_model_overall.coef_.ndim > 1 else best_model_overall.coef_
        feature_importance = dict(zip(feature_cols, np.abs(coef).tolist()))
    
    # Additional predictions for analysis
    y_pred_final = best_model_overall.predict(X_test)
    
    # For classification, try to get prediction probabilities
    y_pred_proba = None
    if problem_type == "classification" and hasattr(best_model_overall, "predict_proba"):
        try:
            proba = best_model_overall.predict_proba(X_test)
            y_pred_proba = proba[:, -1] if proba.shape[1] == 2 else proba
        except Exception:
            y_pred_proba = None

print(f"\\nTraining completed! Best model: {best_model_name}")
print(f"Results shape: {results_df.shape}")
"""

def generate_code_with_gemini(df: pd.DataFrame, target_column: str, problem_type: str):
    """
    Generate ML training code using Gemini with robust fallback.
    """
    configure_gemini()
    model = get_gemini_model()

    # Analyze dataset characteristics
    feature_count = len([c for c in df.columns if c != target_column])
    sample_count = len(df)
    
    prompt = f"""
Generate a complete, executable Python script for ML training with these requirements:

DATASET INFO:
- Samples: {sample_count}
- Features: {feature_count} 
- Target: {target_column}
- Problem: {problem_type}

REQUIREMENTS:
1. Use EXACTLY these variables that are already defined: df, target_column, problem_type
2. Create feature matrix X from df (excluding target_column)
3. Use train_test_split with test_size=0.2, random_state=42
4. For classification: use stratify=y
5. Use Optuna with TPESampler(seed=42) for hyperparameter tuning
6. Test these models:
   - LogisticRegression/LinearRegression
   - RandomForestClassifier/RandomForestRegressor  
   - DecisionTreeClassifier/DecisionTreeRegressor
7. Use 3-fold cross-validation in objective functions
8. Create results_df with proper column names:
   Classification: model, accuracy, precision, recall, f1_score, best_params, training_time
   Regression: model, r2_score, mse, mae, rmse, best_params, training_time
9. Export these variables: results_df, best_model, best_model_name, feature_importance, y_pred_proba (if available)

CRITICAL RULES:
- NO parameter name conflicts (don't pass 'classifier'/'regressor' to model constructors)
- NO imports beyond sklearn, numpy, pandas, optuna, time
- Handle all exceptions gracefully
- Use proper parameter extraction from Optuna trials
- Return executable code with no markdown formatting
"""

    try:
        response = model.generate_content(prompt)
        generated_code = _sanitize_and_validate_code(response.text)
        
        # Validate the generated code has required components
        if not _validate_generated_code(generated_code):
            print("Generated code validation failed, using robust template")
            return _get_robust_training_template()
            
        return generated_code
        
    except Exception as e:
        st.warning(f"AI code generation failed: {e}. Using robust template.")
        return _get_robust_training_template()

def _sanitize_and_validate_code(text: str) -> str:
    """
    Sanitize and validate generated code.
    """
    if not isinstance(text, str) or len(text.strip()) == 0:
        return _get_robust_training_template()

    # Remove markdown formatting
    code = text.strip()
    # code = re.sub(r'^```
    code = re.sub(r'^```.*\n?', '', code, flags=re.MULTILINE)
    # code = re.sub(r'\n```

    # Security checks - prevent dangerous operations
    forbidden_patterns = [
        r'\b(os|sys|subprocess|shutil|pathlib|socket|requests|urllib)\b',
        r'exec\s*$$', r'eval\s*$$', r'__import__\s*$$',
        r'open\s*$$', r'file\s*$$', r'input\s*$$',
        r'raw_input\s*$$', r'compile\s*$$'
    ]
    
    for pattern in forbidden_patterns:
        if re.search(pattern, code, re.IGNORECASE):
            print(f"Security violation detected: {pattern}")
            return _get_robust_training_template()

    return code

def _validate_generated_code(code: str) -> bool:
    """
    Validate that generated code has required components.
    """
    required_components = [
        'results_df', 'best_model', 'best_model_name', 'feature_importance',
        'train_test_split', 'cross_val_score', 'optuna'
    ]
    
    for component in required_components:
        if component not in code:
            print(f"Missing required component: {component}")
            return False
    
    return True

def explain_code_with_gemini(code: str):
    """
    Generate explanation for ML training code.
    """
    configure_gemini()
    model = get_gemini_model()
    
    prompt = f"""
Explain this machine learning training pipeline in simple terms:

{code[:2000]}...

Provide a clear, concise explanation covering:
1. Data preparation steps
2. Model types being tested
3. Hyperparameter optimization approach
4. Evaluation metrics used
5. Output format

Keep it under 200 words and use simple language.
"""

    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"""
This ML pipeline performs the following steps:

1. **Data Preparation**: Splits the dataset into training (80%) and testing (20%) sets
2. **Model Testing**: Evaluates multiple algorithms (Logistic/Linear Regression, Random Forest, Decision Tree)
3. **Hyperparameter Tuning**: Uses Optuna to automatically find the best settings for each model
4. **Cross-Validation**: Tests each model 3 times to ensure reliable performance estimates
5. **Model Selection**: Chooses the best performing model based on accuracy (classification) or R² score (regression)
6. **Results**: Provides detailed performance metrics and the trained model ready for use

The process is fully automated and will select the optimal model configuration for your specific dataset.
"""
