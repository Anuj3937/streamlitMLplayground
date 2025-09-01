# utils/ml_pipeline.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, confusion_matrix
import xgboost as xgb
import time
import warnings
import optuna # Add this import
import io
import sys
import contextlib

warnings.filterwarnings('ignore')

# The generate_ml_code function is no longer needed, you can delete it.

def execute_ml_code(code, df, target_column, problem_type, progress_callback=None, logs_callback=None):
    """
    Execute the generated ML code and return results.
    """
    try:
        captured_output = io.StringIO()
        
        if progress_callback:
            progress_callback(10, "Initializing training environment...")

        exec_globals = {
            'pd': pd,
            'np': np,
            'df': df,
            'target_column': target_column,
            'problem_type': problem_type,
            'time': time,
            'optuna': optuna,
            'train_test_split': train_test_split,
            'GridSearchCV': GridSearchCV,
            'cross_val_score': cross_val_score,
            'LogisticRegression': LogisticRegression,
            'LinearRegression': LinearRegression,
            'RandomForestClassifier': RandomForestClassifier,
            'RandomForestRegressor': RandomForestRegressor,
            'SVC': SVC,
            'SVR': SVR,
            'DecisionTreeClassifier': DecisionTreeClassifier,
            'DecisionTreeRegressor': DecisionTreeRegressor,
            'xgb': xgb,
            'accuracy_score': accuracy_score,
            'precision_score': precision_score,
            'recall_score': recall_score,
            'f1_score': f1_score,
            'r2_score': r2_score,
            'mean_squared_error': mean_squared_error,
            'mean_absolute_error': mean_absolute_error,
            'confusion_matrix': confusion_matrix,
            'warnings': warnings
        }
        
        if progress_callback:
            progress_callback(20, "Starting model training and tuning...")

        with contextlib.redirect_stdout(captured_output):
            exec(code, exec_globals)
        
        if progress_callback:
            progress_callback(80, "Processing results...")

        if logs_callback:
            logs_callback(captured_output.getvalue())
        
        results_df = exec_globals.get('results_df')
        best_model_name = exec_globals.get('best_model_name')
        # Gemini prompt now requests 'best_model' directly
        best_model_object = exec_globals.get('best_model') 
        feature_importance = exec_globals.get('feature_importance')
        
        if results_df is None or best_model_name is None:
            raise Exception("Failed to extract 'results_df' or 'best_model_name' from code execution.")
        
        best_model_row = results_df[results_df['model'] == best_model_name].iloc[0]
        
        # Prepare comprehensive results
        feature_cols = [col for col in df.columns if col != target_column]
        X = df[feature_cols]
        y = df[target_column]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42,
            stratify=y if problem_type == 'classification' else None
        )
        
        best_predictions = best_model_object.predict(X_test) if best_model_object else None
        
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
                'X_test': X_test # For SHAP analysis
            }
        }
        
        if problem_type == "classification" and best_predictions is not None:
            cm = confusion_matrix(y_test, best_predictions)
            results['best_model']['confusion_matrix'] = cm.tolist()
        elif problem_type == "regression" and best_predictions is not None:
            residuals = y_test - best_predictions
            results['best_model']['residuals'] = residuals.tolist()
        
        if progress_callback:
            progress_callback(100, "Training completed successfully!")
        
        return results
        
    except Exception as e:
        if progress_callback:
            progress_callback(0, f"Error: {str(e)}")
        if logs_callback:
            logs_callback(f"EXECUTION ERROR: {str(e)}\n\n--- CAPTURED LOGS ---\n{captured_output.getvalue()}")
        raise e
