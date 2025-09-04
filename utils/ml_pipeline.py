import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, confusion_matrix
import xgboost as xgb
import time
import warnings
import optuna
import io
import contextlib
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

warnings.filterwarnings("ignore")

def _smart_extract_results(exec_globals, df, target_column, problem_type):
    """
    Dynamically extract results from executed code regardless of variable naming.
    """
    results_df = None
    best_model = None
    best_model_name = None
    feature_importance = {}
    
    # Try multiple possible variable names for results DataFrame
    possible_results_names = [
        "results_df", "results", "model_results", "comparison_results",
        "model_comparison", "performance_results"
    ]
    
    for name in possible_results_names:
        candidate = exec_globals.get(name)
        if isinstance(candidate, pd.DataFrame) and len(candidate) > 0:
            results_df = candidate
            print(f"Found results DataFrame as: {name}")
            break
    
    # Try multiple possible variable names for best model
    # **KEY FIX: Added best_model_overall to the search list**
    possible_model_names = [
        "best_model", "best_model_overall", "final_model", "optimal_model", 
        "selected_model", "best_model_", "trained_model", "champion_model"
    ]
    
    for name in possible_model_names:
        candidate = exec_globals.get(name)
        if candidate and _safe_has_predict(candidate):
            best_model = candidate
            print(f"Found best model as: {name}")
            break
    
    # If still no model found, search all variables for anything with predict method
    if best_model is None:
        print("Searching all variables for trained models...")
        for name, candidate in exec_globals.items():
            if _safe_has_predict(candidate):
                best_model = candidate
                print(f"Found model in variable: {name}")
                break
    
    # Try multiple possible variable names for model name
    possible_name_vars = [
        "best_model_name", "best_algorithm", "champion_algorithm",
        "selected_algorithm", "optimal_algorithm", "winning_model"
    ]
    
    for name in possible_name_vars:
        candidate = exec_globals.get(name)
        if isinstance(candidate, str) and len(candidate) > 0:
            best_model_name = candidate
            print(f"Found model name as: {name}")
            break
    
    # Try multiple possible variable names for feature importance
    possible_importance_names = [
        "feature_importance", "feature_importances", "importances",
        "feature_weights", "variable_importance"
    ]
    
    for name in possible_importance_names:
        candidate = exec_globals.get(name)
        if isinstance(candidate, dict) and len(candidate) > 0:
            feature_importance = candidate
            print(f"Found feature importance as: {name}")
            break
        elif hasattr(candidate, 'tolist'):  # numpy array or similar
            try:
                feature_cols = [c for c in df.columns if c != target_column]
                feature_importance = dict(zip(feature_cols, candidate.tolist()))
                print(f"Converted {name} to feature importance dict")
                break
            except:
                continue
    
    # If we still don't have results_df, try to construct it from individual variables
    if results_df is None:
        print("No results DataFrame found, attempting to construct from individual variables...")
        results_df = _construct_results_from_globals(exec_globals, problem_type)
    
    # If we still don't have model name, try to infer it
    if best_model_name is None and best_model is not None:
        best_model_name = best_model.__class__.__name__
        print(f"Inferred model name: {best_model_name}")
    
    # If we still don't have feature importance but have a model, try to extract it
    if len(feature_importance) == 0 and best_model is not None:
        feature_importance = _extract_feature_importance(best_model, df, target_column)
    
    return results_df, best_model, best_model_name, feature_importance

def _safe_has_predict(obj):
    """
    Safely check if object has predict method without triggering sklearn errors.
    """
    try:
        # Check if it's a sklearn-like model by looking for predict method
        predict_method = getattr(obj, 'predict', None)
        if not callable(predict_method):
            return False
        
        # Additional check: sklearn models usually have fit method too
        fit_method = getattr(obj, 'fit', None)
        if not callable(fit_method):
            return False
            
        # Check if it's a trained model by seeing if it has required attributes
        # For ensemble models, check if estimators_ exists (but don't access it)
        if hasattr(obj, '__class__') and 'sklearn' in str(obj.__class__):
            return True
            
        return True
        
    except Exception as e:
        print(f"Error checking predict method: {e}")
        return False

def _construct_results_from_globals(exec_globals, problem_type):
    """
    Attempt to construct results DataFrame from individual performance variables.
    """
    try:
        results = []
        
        # If we found individual metrics in globals
        if problem_type == "classification":
            basic_result = {
                "model": exec_globals.get("best_model_name", "OptimizedModel"),
                "accuracy": exec_globals.get("accuracy", 0.0),
                "precision": exec_globals.get("precision", 0.0),
                "recall": exec_globals.get("recall", 0.0),
                "f1_score": exec_globals.get("f1_score", 0.0),
            }
        else:
            basic_result = {
                "model": exec_globals.get("best_model_name", "OptimizedModel"),
                "r2_score": exec_globals.get("r2_score", 0.0),
                "mse": exec_globals.get("mse", 0.0),
                "mae": exec_globals.get("mae", 0.0),
                "rmse": exec_globals.get("rmse", 0.0),
            }
        
        results.append(basic_result)
        return pd.DataFrame(results)
        
    except Exception as e:
        print(f"Failed to construct results DataFrame: {e}")
        return None

def _extract_feature_importance(model, df, target_column):
    """
    Extract feature importance from model if possible.
    """
    try:
        feature_cols = [c for c in df.columns if c != target_column]
        
        if hasattr(model, "feature_importances_"):
            return dict(zip(feature_cols, model.feature_importances_.tolist()))
        elif hasattr(model, "coef_"):
            coef = np.ravel(model.coef_) if model.coef_.ndim > 1 else model.coef_
            return dict(zip(feature_cols, np.abs(coef).tolist()))
        else:
            print("Model doesn't have feature_importances_ or coef_ attributes")
            return {}
    except Exception as e:
        print(f"Failed to extract feature importance: {e}")
        return {}

def execute_ml_code(code, df, target_column, problem_type, progress_callback=None, logs_callback=None):
    """
    Execute ML code with dynamic output detection.
    """
    captured_output = io.StringIO()
    
    try:
        if progress_callback:
            progress_callback(10, "🔧 Setting up training environment...")

        # Enhanced global namespace
        exec_globals = {
            "pd": pd,
            "np": np,
            "df": df,
            "target_column": target_column,
            "problem_type": problem_type,
            "time": time,
            "optuna": optuna,
            "cross_val_score": cross_val_score,
            "train_test_split": train_test_split,
            "LogisticRegression": LogisticRegression,
            "LinearRegression": LinearRegression,
            "RandomForestClassifier": RandomForestClassifier,
            "RandomForestRegressor": RandomForestRegressor,
            "SVC": SVC,
            "SVR": SVR,
            "DecisionTreeClassifier": DecisionTreeClassifier,
            "DecisionTreeRegressor": DecisionTreeRegressor,
            "XGBClassifier": xgb.XGBClassifier,
            "XGBRegressor": xgb.XGBRegressor,
            "accuracy_score": accuracy_score,
            "precision_score": precision_score,
            "recall_score": recall_score,
            "f1_score": f1_score,
            "r2_score": r2_score,
            "mean_squared_error": mean_squared_error,
            "mean_absolute_error": mean_absolute_error,
            "confusion_matrix": confusion_matrix,
            "print": lambda *args, **kwargs: captured_output.write(" ".join(map(str, args)) + "\n"),
        }

        if progress_callback:
            progress_callback(20, "🚀 Starting model training and optimization...")

        # Execute code with comprehensive output capture
        with contextlib.redirect_stdout(captured_output), contextlib.redirect_stderr(captured_output):
            exec(code, exec_globals)

        if progress_callback:
            progress_callback(80, "📊 Processing results and metrics...")

        # Get logs first for debugging
        log_output = captured_output.getvalue()
        if logs_callback:
            logs_callback(log_output)

        print("Available variables after execution:")
        for name, value in exec_globals.items():
            if not name.startswith('__') and name not in ['pd', 'np', 'df', 'target_column', 'problem_type', 'time', 'optuna']:
                print(f"  {name}: {type(value)}")

        # Use smart extraction instead of hardcoded names
        results_df, best_model_object, best_model_name, feature_importance = _smart_extract_results(
            exec_globals, df, target_column, problem_type
        )

        # Validate extracted results
        if results_df is None or len(results_df) == 0:
            raise Exception("No valid results DataFrame could be found or constructed from the executed code.")
        
        if best_model_object is None:
            raise Exception("No trained model could be found in the executed code.")
        
        if best_model_name is None or len(best_model_name.strip()) == 0:
            raise Exception("No model name could be determined from the executed code.")

        print(f"✅ Successfully extracted: {best_model_name} with {len(results_df)} results")

        # Continue with rest of the pipeline...
        feature_cols = [c for c in df.columns if c != target_column]
        X = df[feature_cols]
        y = df[target_column]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42,
            stratify=y if problem_type == "classification" else None
        )

        # Generate predictions
        best_predictions = None
        try:
            best_predictions = best_model_object.predict(X_test)
            print(f"Successfully generated {len(best_predictions)} predictions")
        except Exception as e:
            print(f"Warning: Could not generate predictions: {e}")

        # Build results package
        results = {
            "model_comparison": results_df.to_dict("records"),
            "best_model": {
                "name": best_model_name,
                "model_object": best_model_object,
                "feature_importance": feature_importance,
                "feature_names": X.columns.tolist(),
                "predictions": best_predictions.tolist() if best_predictions is not None else None,
            },
            "data_info": {
                "total_samples": len(df),
                "training_samples": len(X_train),
                "test_samples": len(X_test),
                "features_count": len(X.columns),
                "target_column": target_column,
                "problem_type": problem_type,
                "X_test": X_test,
                "y_test": y_test.tolist(),
            },
        }

        # Add problem-specific metrics
        if best_predictions is not None:
            if problem_type == "classification":
                try:
                    cm = confusion_matrix(y_test, best_predictions)
                    results["best_model"]["confusion_matrix"] = cm.tolist()
                    results["best_model"]["accuracy"] = float(accuracy_score(y_test, best_predictions))
                    results["best_model"]["precision"] = float(precision_score(y_test, best_predictions, average='weighted', zero_division=0))
                    results["best_model"]["recall"] = float(recall_score(y_test, best_predictions, average='weighted', zero_division=0))
                    results["best_model"]["f1_score"] = float(f1_score(y_test, best_predictions, average='weighted', zero_division=0))
                except Exception as e:
                    print(f"Warning: Could not calculate classification metrics: {e}")
                    
            else:  # Regression
                try:
                    residuals = y_test - best_predictions
                    results["best_model"]["residuals"] = residuals.tolist()
                    results["best_model"]["r2_score"] = float(r2_score(y_test, best_predictions))
                    results["best_model"]["mse"] = float(mean_squared_error(y_test, best_predictions))
                    results["best_model"]["mae"] = float(mean_absolute_error(y_test, best_predictions))
                    results["best_model"]["rmse"] = float(np.sqrt(mean_squared_error(y_test, best_predictions)))
                except Exception as e:
                    print(f"Warning: Could not calculate regression metrics: {e}")

        if progress_callback:
            progress_callback(100, "✅ Training completed successfully!")

        return results

    except Exception as e:
        error_msg = str(e)
        
        if progress_callback:
            progress_callback(0, f"❌ Error: {error_msg}")
        
        if logs_callback:
            logs_callback(f"EXECUTION ERROR: {error_msg}\n\n--- CAPTURED LOGS ---\n{captured_output.getvalue()}\n\n--- AVAILABLE VARIABLES ---\n" + 
                         "\n".join([f"{k}: {type(v)}" for k, v in exec_globals.items() if not k.startswith('__')]))
        
        print(f"ML Pipeline execution failed: {error_msg}")
        raise e
