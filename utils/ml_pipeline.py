import pandas as pd
import numpy as np
import time
import warnings
import io
import sys
import contextlib
from typing import Dict, Any, Callable, Optional
from .error_handler import handle_errors
from .config import CONFIG

warnings.filterwarnings('ignore')

@handle_errors()
def execute_ml_code(code: str, df: pd.DataFrame, target_column: str, problem_type: str,
                   progress_callback: Optional[Callable] = None,
                   logs_callback: Optional[Callable] = None) -> Optional[Dict[str, Any]]:
    """
    Execute ML code with comprehensive monitoring and error handling
    """
    
    # Capture output
    captured_output = io.StringIO()
    
    try:
        if progress_callback:
            progress_callback(5, "🔧 Setting up execution environment...")
        
        # Prepare execution globals with all necessary imports
        exec_globals = {
            # Core libraries
            'pd': pd,
            'np': np,
            'df': df,
            'target_column': target_column,
            'problem_type': problem_type,
            'time': time,
            'warnings': warnings,
            
            # ML libraries
            'train_test_split': None,
            'cross_val_score': None,
            'StratifiedKFold': None,
            'KFold': None,
            'GridSearchCV': None,
            
            # Models
            'LogisticRegression': None,
            'LinearRegression': None,
            'RandomForestClassifier': None,
            'RandomForestRegressor': None,
            'SVC': None,
            'SVR': None,
            'DecisionTreeClassifier': None,
            'DecisionTreeRegressor': None,
            'GradientBoostingClassifier': None,
            'GradientBoostingRegressor': None,
            
            # Metrics
            'accuracy_score': None,
            'precision_score': None,
            'recall_score': None,
            'f1_score': None,
            'r2_score': None,
            'mean_squared_error': None,
            'mean_absolute_error': None,
            'confusion_matrix': None,
            
            # Additional libraries
            'xgb': None,
            'lgb': None,
            'optuna': None,
        }
        
        if progress_callback:
            progress_callback(10, "📚 Loading required libraries...")
        
        # Import libraries dynamically
        try:
            from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, KFold, GridSearchCV
            from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
            from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
            from sklearn.svm import SVC, SVR
            from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
            from sklearn.neural_network import MLPClassifier, MLPRegressor
            from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                                       r2_score, mean_squared_error, mean_absolute_error, confusion_matrix)
            
            # Update globals with imported functions
            exec_globals.update({
                'train_test_split': train_test_split,
                'cross_val_score': cross_val_score,
                'StratifiedKFold': StratifiedKFold,
                'KFold': KFold,
                'GridSearchCV': GridSearchCV,
                'LogisticRegression': LogisticRegression,
                'LinearRegression': LinearRegression,
                'Ridge': Ridge,
                'Lasso': Lasso,
                'RandomForestClassifier': RandomForestClassifier,
                'RandomForestRegressor': RandomForestRegressor,
                'GradientBoostingClassifier': GradientBoostingClassifier,
                'GradientBoostingRegressor': GradientBoostingRegressor,
                'SVC': SVC,
                'SVR': SVR,
                'DecisionTreeClassifier': DecisionTreeClassifier,
                'DecisionTreeRegressor': DecisionTreeRegressor,
                'MLPClassifier': MLPClassifier,
                'MLPRegressor': MLPRegressor,
                'accuracy_score': accuracy_score,
                'precision_score': precision_score,
                'recall_score': recall_score,
                'f1_score': f1_score,
                'r2_score': r2_score,
                'mean_squared_error': mean_squared_error,
                'mean_absolute_error': mean_absolute_error,
                'confusion_matrix': confusion_matrix,
            })
            
        except ImportError as e:
            raise Exception(f"Failed to import required scikit-learn modules: {str(e)}")
        
        # Try to import optional libraries
        try:
            import xgboost as xgb
            exec_globals['xgb'] = xgb
        except ImportError:
            pass
        
        try:
            import lightgbm as lgb
            exec_globals['lgb'] = lgb
        except ImportError:
            pass
        
        try:
            import optuna
            exec_globals['optuna'] = optuna
        except ImportError:
            pass
        
        if progress_callback:
            progress_callback(20, "🚀 Starting model training pipeline...")
        
        # Execute the code with output capture
        with contextlib.redirect_stdout(captured_output), contextlib.redirect_stderr(captured_output):
            exec(code, exec_globals)
        
        if progress_callback:
            progress_callback(80, "📊 Processing results...")
        
        # Update logs
        if logs_callback:
            logs_callback(captured_output.getvalue())
        
        # Extract results from execution
        results_df = exec_globals.get('results_df')
        best_model_name = exec_globals.get('best_model_name')
        best_model_object = exec_globals.get('best_model') or exec_globals.get('best_model_object')
        feature_importance = exec_globals.get('feature_importance')
        
        # Validate required outputs
        if results_df is None:
            raise Exception("Code execution did not produce 'results_df'. Please ensure your code creates this variable.")
        
        if best_model_name is None:
            raise Exception("Code execution did not produce 'best_model_name'. Please ensure your code creates this variable.")
        
        if progress_callback:
            progress_callback(90, "🔧 Compiling final results...")
        
        # Get the best model row from results
        if isinstance(results_df, pd.DataFrame):
            best_model_row = results_df[results_df['model'] == best_model_name]
            if best_model_row.empty:
                # Fallback: get the first row if name doesn't match
                best_model_row = results_df.iloc[0]
            else:
                best_model_row = best_model_row.iloc[0]
        else:
            raise Exception("results_df is not a pandas DataFrame")
        
        # Prepare data info
        feature_cols = [col for col in df.columns if col != target_column]
        X = df[feature_cols]
        y = df[target_column]
        
        # Recreate train-test split to get test data
        try:
            if problem_type == 'classification':
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42, stratify=y
                )
            else:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )
        except Exception:
            # Fallback without stratification
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
        
        # Generate predictions for analysis
        best_predictions = None
        if best_model_object and hasattr(best_model_object, 'predict'):
            try:
                best_predictions = best_model_object.predict(X_test)
            except Exception as e:
                print(f"Warning: Could not generate predictions: {str(e)}")
        
        # Compile comprehensive results
        results = {
            'model_comparison': results_df.to_dict('records'),
            'best_model': {
                'name': best_model_name,
                'model_object': best_model_object,
                'feature_importance': feature_importance,
                'feature_names': X.columns.tolist(),
                'predictions': best_predictions.tolist() if best_predictions is not None else None,
                **best_model_row.to_dict()
            },
            'data_info': {
                'total_samples': len(df),
                'training_samples': len(X_train),
                'test_samples': len(X_test),
                'features_count': len(X.columns),
                'target_column': target_column,
                'problem_type': problem_type,
                'X_test': X_test
            }
        }
        
        # Add problem-specific analysis
        if problem_type == "classification" and best_predictions is not None:
            try:
                cm = confusion_matrix(y_test, best_predictions)
                results['best_model']['confusion_matrix'] = cm.tolist()
            except Exception:
                pass
                
        elif problem_type == "regression" and best_predictions is not None:
            try:
                residuals = y_test.values - best_predictions
                results['best_model']['residuals'] = residuals.tolist()
            except Exception:
                pass
        
        if progress_callback:
            progress_callback(100, "✅ Training completed successfully!")
        
        return results
        
    except Exception as e:
        error_msg = str(e)
        
        if progress_callback:
            progress_callback(0, f"❌ Error: {error_msg}")
        
        if logs_callback:
            error_log = f"EXECUTION ERROR: {error_msg}\n\n=== CAPTURED LOGS ===\n{captured_output.getvalue()}"
            logs_callback(error_log)
        
        # Log the error for debugging
        print(f"ML Pipeline Error: {error_msg}")
        print("Captured output:", captured_output.getvalue())
        
        raise Exception(f"ML pipeline execution failed: {error_msg}")

@handle_errors()
def validate_code_structure(code: str) -> tuple[bool, list[str]]:
    """Validate that the generated code has required structure"""
    issues = []
    
    required_variables = ['results', 'results_df', 'best_model_name', 'best_model']
    required_imports = ['train_test_split', 'cross_val_score']
    
    for var in required_variables:
        if var not in code:
            issues.append(f"Missing required variable: {var}")
    
    for imp in required_imports:
        if imp not in code:
            issues.append(f"Missing required import/usage: {imp}")
    
    # Check for basic ML workflow
    if 'fit(' not in code:
        issues.append("No model training detected (missing .fit() calls)")
    
    if 'predict(' not in code:
        issues.append("No predictions detected (missing .predict() calls)")
    
    return len(issues) == 0, issues

@handle_errors()
def generate_fallback_pipeline(df: pd.DataFrame, target_column: str, problem_type: str) -> str:
    """Generate a simple fallback ML pipeline if AI generation fails"""
    
    feature_cols = [col for col in df.columns if col != target_column]
    
    if problem_type == "classification":
        return f'''
# Fallback Classification Pipeline
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import time

print("🚀 Starting fallback classification pipeline...")

# Prepare data
X = df[{feature_cols}]
y = df['{target_column}']

print(f"📊 Dataset: {{len(X)}} samples, {{len(X.columns)}} features")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("🔄 Training models...")

# Define models
models = {{
    'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
    'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000)
}}

results = []

for name, model in models.items():
    print(f"Training {{name}}...")
    start_time = time.time()
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    
    # Train model
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
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
    
    print(f"✅ {{name}} - Accuracy: {{accuracy:.4f}}")

# Create results DataFrame
results_df = pd.DataFrame(results)

# Find best model
best_idx = results_df['accuracy'].idxmax()
best_model_name = results_df.loc[best_idx, 'model']
best_model = models[best_model_name]

# Feature importance
feature_importance = None
if hasattr(best_model, 'feature_importances_'):
    feature_importance = {{
        'features': X.columns.tolist(),
        'importance': best_model.feature_importances_.tolist()
    }}

print(f"🏆 Best model: {{best_model_name}} (Accuracy: {{results_df.loc[best_idx, 'accuracy']:.4f}})")
'''
    
    else:  # regression
        return f'''
# Fallback Regression Pipeline
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import time

print("🚀 Starting fallback regression pipeline...")

# Prepare data
X = df[{feature_cols}]
y = df['{target_column}']

print(f"📊 Dataset: {{len(X)}} samples, {{len(X.columns)}} features")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("🔄 Training models...")

# Define models
models = {{
    'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
    'LinearRegression': LinearRegression()
}}

results = []

for name, model in models.items():
    print(f"Training {{name}}...")
    start_time = time.time()
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
    
    # Train model
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
    
    print(f"✅ {{name}} - R² Score: {{r2:.4f}}")

# Create results DataFrame
results_df = pd.DataFrame(results)

# Find best model
best_idx = results_df['r2_score'].idxmax()
best_model_name = results_df.loc[best_idx, 'model']
best_model = models[best_model_name]

# Feature importance
feature_importance = None
if hasattr(best_model, 'feature_importances_'):
    feature_importance = {{
        'features': X.columns.tolist(),
        'importance': best_model.feature_importances_.tolist()
    }}

print(f"🏆 Best model: {{best_model_name}} (R² Score: {{results_df.loc[best_idx, 'r2_score']:.4f}})")
'''

def sanitize_execution_environment():
    """Clean up execution environment variables"""
    import gc
    gc.collect()

def log_execution_stats(start_time: float, end_time: float, model_count: int):
    """Log execution statistics"""
    duration = end_time - start_time
    print(f"⏱️ Total execution time: {duration:.2f} seconds")
    print(f"🤖 Models trained: {model_count}")
    print(f"⚡ Average time per model: {duration/model_count:.2f} seconds")

# Global pipeline instance for reuse
_pipeline_cache = {}

def get_cached_pipeline(cache_key: str):
    """Get cached pipeline results if available"""
    return _pipeline_cache.get(cache_key)

def cache_pipeline_results(cache_key: str, results: Dict[str, Any]):
    """Cache pipeline results for reuse"""
    _pipeline_cache[cache_key] = results
    
    # Limit cache size
    if len(_pipeline_cache) > 10:
        # Remove oldest entry
        oldest_key = next(iter(_pipeline_cache))
        del _pipeline_cache[oldest_key]
