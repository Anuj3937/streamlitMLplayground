import pandas as pd
import numpy as np
from sklearn.ensemble import (RandomForestClassifier, RandomForestRegressor, 
                             GradientBoostingClassifier, GradientBoostingRegressor,
                             VotingClassifier, VotingRegressor)
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.svm import SVC, SVR
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
from sklearn.metrics import accuracy_score, f1_score, r2_score, mean_squared_error
import xgboost as xgb
import lightgbm as lgb
import optuna
import time
from typing import Dict, List, Tuple, Any
import warnings
from .error_handler import handle_errors
from .config import CONFIG

warnings.filterwarnings('ignore')

class AdvancedMLPipeline:
    """Advanced ML pipeline with ensemble methods and auto-ML features"""
    
    def __init__(self):
        self.study_results = {}
        self.trained_models = {}
        
    def get_model_pool(self, problem_type: str) -> Dict[str, Any]:
        """Get comprehensive model pool based on problem type"""
        
        if problem_type == "classification":
            return {
                'RandomForest': RandomForestClassifier(random_state=CONFIG.random_state),
                'XGBoost': xgb.XGBClassifier(random_state=CONFIG.random_state, eval_metric='logloss'),
                'LightGBM': lgb.LGBMClassifier(random_state=CONFIG.random_state, verbose=-1),
                'LogisticRegression': LogisticRegression(random_state=CONFIG.random_state, max_iter=1000),
                'GradientBoosting': GradientBoostingClassifier(random_state=CONFIG.random_state),
                'SVM': SVC(random_state=CONFIG.random_state, probability=True),
                'NeuralNetwork': MLPClassifier(random_state=CONFIG.random_state, max_iter=500)
            }
        else:  # regression
            return {
                'RandomForest': RandomForestRegressor(random_state=CONFIG.random_state),
                'XGBoost': xgb.XGBRegressor(random_state=CONFIG.random_state),
                'LightGBM': lgb.LGBMRegressor(random_state=CONFIG.random_state, verbose=-1),
                'LinearRegression': LinearRegression(),
                'Ridge': Ridge(random_state=CONFIG.random_state),
                'Lasso': Lasso(random_state=CONFIG.random_state),
                'GradientBoosting': GradientBoostingRegressor(random_state=CONFIG.random_state),
                'SVR': SVR(),
                'NeuralNetwork': MLPRegressor(random_state=CONFIG.random_state, max_iter=500)
            }
    
    def get_search_space(self, model_name: str, problem_type: str) -> Dict[str, Any]:
        """Get hyperparameter search space for each model"""
        
        spaces = {
            'RandomForest': {
                'n_estimators': (50, 200),
                'max_depth': (5, 20),
                'min_samples_split': (2, 20),
                'min_samples_leaf': (1, 10),
                'max_features': ['sqrt', 'log2', None]
            },
            'XGBoost': {
                'n_estimators': (50, 200),
                'max_depth': (3, 10),
                'learning_rate': (0.01, 0.3),
                'subsample': (0.8, 1.0),
                'colsample_bytree': (0.8, 1.0),
            },
            'LightGBM': {
                'n_estimators': (50, 200),
                'max_depth': (3, 10),
                'learning_rate': (0.01, 0.3),
                'num_leaves': (20, 100),
                'subsample': (0.8, 1.0),
            },
            'LogisticRegression': {
                'C': (0.001, 100),
                'penalty': ['l1', 'l2', 'elasticnet'],
                'solver': ['liblinear', 'saga']
            },
            'Ridge': {
                'alpha': (0.001, 100)
            },
            'Lasso': {
                'alpha': (0.001, 100)
            },
            'GradientBoosting': {
                'n_estimators': (50, 200),
                'max_depth': (3, 10),
                'learning_rate': (0.01, 0.3),
                'subsample': (0.8, 1.0)
            },
            'SVM': {
                'C': (0.001, 100),
                'gamma': ['scale', 'auto'],
                'kernel': ['rbf', 'linear', 'poly']
            },
            'SVR': {
                'C': (0.001, 100),
                'gamma': ['scale', 'auto'],
                'kernel': ['rbf', 'linear', 'poly']
            },
            'NeuralNetwork': {
                'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
                'activation': ['relu', 'tanh'],
                'alpha': (0.0001, 0.01),
                'learning_rate': ['constant', 'adaptive']
            }
        }
        
        return spaces.get(model_name, {})
    
    @handle_errors()
    def optimize_model(self, model_name: str, model, X_train: pd.DataFrame, 
                      y_train: pd.Series, problem_type: str, n_trials: int = 30) -> Tuple[Any, Dict, float]:
        """Optimize single model using Optuna"""
        
        def objective(trial):
            params = {}
            search_space = self.get_search_space(model_name, problem_type)
            
            for param_name, param_range in search_space.items():
                if isinstance(param_range, tuple) and len(param_range) == 2:
                    if isinstance(param_range[0], float):
                        params[param_name] = trial.suggest_float(param_name, param_range[0], param_range[1], log=True)
                    else:
                        params[param_name] = trial.suggest_int(param_name, param_range[0], param_range[1])
                elif isinstance(param_range, list):
                    params[param_name] = trial.suggest_categorical(param_name, param_range)
            
            # Set parameters
            model.set_params(**params)
            
            # Cross-validation
            if problem_type == "classification":
                cv = StratifiedKFold(n_splits=CONFIG.cv_folds, shuffle=True, random_state=CONFIG.random_state)
                scoring = 'accuracy'
            else:
                cv = KFold(n_splits=CONFIG.cv_folds, shuffle=True, random_state=CONFIG.random_state)
                scoring = 'r2'
            
            scores = cross_val_score(model, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
            return scores.mean()
        
        study = optuna.create_study(direction='maximize', study_name=f"optimize_{model_name}")
        study.optimize(objective, n_trials=n_trials, timeout=CONFIG.max_training_time)
        
        # Get best parameters and score
        best_params = study.best_params
        best_score = study.best_value
        
        # Train final model with best parameters
        model.set_params(**best_params)
        
        return model, best_params, best_score
    
    @handle_errors()
    def train_and_evaluate_models(self, X_train: pd.DataFrame, X_test: pd.DataFrame,
                                 y_train: pd.Series, y_test: pd.Series, 
                                 problem_type: str) -> Tuple[List[Dict], Dict]:
        """Train and evaluate all models"""
        
        models = self.get_model_pool(problem_type)
        results = []
        best_model_info = {'score': -np.inf, 'model': None, 'name': None}
        
        for model_name, model in models.items():
            print(f"Training {model_name}...")
            start_time = time.time()
            
            try:
                # Optimize hyperparameters
                optimized_model, best_params, cv_score = self.optimize_model(
                    model_name, model, X_train, y_train, problem_type, n_trials=CONFIG.optuna_trials
                )
                
                # Train final model
                optimized_model.fit(X_train, y_train)
                
                # Predictions
                y_pred = optimized_model.predict(X_test)
                
                # Metrics
                if problem_type == "classification":
                    accuracy = accuracy_score(y_test, y_pred)
                    f1 = f1_score(y_test, y_pred, average='weighted')
                    
                    result = {
                        'model': model_name,
                        'accuracy': accuracy,
                        'f1_score': f1,
                        'cv_score': cv_score,
                        'training_time': time.time() - start_time,
                        'best_params': best_params
                    }
                    
                    score = accuracy
                    
                else:  # regression
                    r2 = r2_score(y_test, y_pred)
                    mse = mean_squared_error(y_test, y_pred)
                    rmse = np.sqrt(mse)
                    mae = np.mean(np.abs(y_test - y_pred))
                    
                    result = {
                        'model': model_name,
                        'r2_score': r2,
                        'mse': mse,
                        'rmse': rmse,
                        'mae': mae,
                        'cv_score': cv_score,
                        'training_time': time.time() - start_time,
                        'best_params': best_params
                    }
                    
                    score = r2
                
                results.append(result)
                self.trained_models[model_name] = optimized_model
                
                # Track best model
                if score > best_model_info['score']:
                    best_model_info.update({
                        'score': score,
                        'model': optimized_model,
                        'name': model_name,
                        'params': best_params,
                        'predictions': y_pred,
                        **result
                    })
                
            except Exception as e:
                print(f"Failed to train {model_name}: {str(e)}")
                continue
        
        return results, best_model_info
    
    @handle_errors()
    def create_ensemble_models(self, X_train: pd.DataFrame, X_test: pd.DataFrame,
                             y_train: pd.Series, y_test: pd.Series,
                             problem_type: str, base_models: Dict) -> Dict:
        """Create ensemble models from base models"""
        
        ensemble_results = {}
        
        try:
            # Select top 3 models for ensemble
            top_models = list(base_models.items())[:3]
            
            if problem_type == "classification":
                # Voting classifier
                voting_clf = VotingClassifier(
                    estimators=[(name, model) for name, model in top_models],
                    voting='soft'
                )
                
                voting_clf.fit(X_train, y_train)
                y_pred_voting = voting_clf.predict(X_test)
                
                ensemble_results['Voting'] = {
                    'model': voting_clf,
                    'accuracy': accuracy_score(y_test, y_pred_voting),
                    'f1_score': f1_score(y_test, y_pred_voting, average='weighted'),
                    'predictions': y_pred_voting
                }
                
            else:  # regression
                # Voting regressor
                voting_reg = VotingRegressor(
                    estimators=[(name, model) for name, model in top_models]
                )
                
                voting_reg.fit(X_train, y_train)
                y_pred_voting = voting_reg.predict(X_test)
                
                ensemble_results['Voting'] = {
                    'model': voting_reg,
                    'r2_score': r2_score(y_test, y_pred_voting),
                    'mse': mean_squared_error(y_test, y_pred_voting),
                    'rmse': np.sqrt(mean_squared_error(y_test, y_pred_voting)),
                    'mae': np.mean(np.abs(y_test - y_pred_voting)),
                    'predictions': y_pred_voting
                }
            
        except Exception as e:
            print(f"Ensemble creation failed: {str(e)}")
        
        return ensemble_results
    
    def get_feature_importance(self, model: Any, feature_names: List[str]) -> Dict[str, Any]:
        """Extract feature importance from model"""
        
        importance_dict = {'features': feature_names, 'importance': None}
        
        try:
            if hasattr(model, 'feature_importances_'):
                importance_dict['importance'] = model.feature_importances_.tolist()
            elif hasattr(model, 'coef_'):
                # For linear models, use absolute coefficients
                if len(model.coef_.shape) > 1:
                    importance_dict['importance'] = np.abs(model.coef_[0]).tolist()
                else:
                    importance_dict['importance'] = np.abs(model.coef_).tolist()
            else:
                # For models without built-in importance
                importance_dict['importance'] = [1.0 / len(feature_names)] * len(feature_names)
                
        except Exception as e:
            print(f"Feature importance extraction failed: {str(e)}")
            importance_dict['importance'] = [1.0 / len(feature_names)] * len(feature_names)
        
        return importance_dict

# Global ML pipeline instance
ml_pipeline = AdvancedMLPipeline()
