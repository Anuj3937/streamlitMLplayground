import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from .error_handler import handle_errors

@st.cache_data
def generate_shap_explanation(_model, X_test):
    """
    Generate SHAP explanation plot with enhanced error handling
    """
    try:
        import shap
        
        # Limit sample size for performance
        sample_size = min(100, len(X_test))
        X_sample = X_test.iloc[:sample_size] if hasattr(X_test, 'iloc') else X_test[:sample_size]
        
        # Create explainer based on model type
        explainer = None
        shap_values = None
        
        # Try TreeExplainer first (for tree-based models)
        try:
            explainer = shap.TreeExplainer(_model)
            shap_values = explainer.shap_values(X_sample)
            explainer_type = "Tree"
        except:
            # Fall back to KernelExplainer
            try:
                # Use a background sample for KernelExplainer
                background_size = min(50, len(X_test))
                background = X_test.iloc[:background_size] if hasattr(X_test, 'iloc') else X_test[:background_size]
                
                explainer = shap.KernelExplainer(_model.predict, background)
                shap_values = explainer.shap_values(X_sample)
                explainer_type = "Kernel"
            except:
                # Try LinearExplainer for linear models
                try:
                    explainer = shap.LinearExplainer(_model, X_sample)
                    shap_values = explainer.shap_values(X_sample)
                    explainer_type = "Linear"
                except:
                    return None
        
        # Handle different shap_values formats
        if isinstance(shap_values, list):
            # Multi-class classification - use the positive class
            if len(shap_values) == 2:
                shap_values_to_plot = shap_values[1]
            else:
                shap_values_to_plot = shap_values[0]
        else:
            shap_values_to_plot = shap_values
        
        # Create the plot
        plt.figure(figsize=(10, 8))
        
        try:
            # Try summary plot first
            shap.summary_plot(
                shap_values_to_plot, 
                X_sample, 
                show=False,
                max_display=15,
                plot_size=(10, 8)
            )
        except:
            # Fall back to bar plot
            shap.summary_plot(
                shap_values_to_plot, 
                X_sample, 
                plot_type="bar",
                show=False,
                max_display=15
            )
        
        plt.title(f"SHAP Feature Importance ({explainer_type} Explainer)")
        plt.tight_layout()
        
        return plt.gcf()
        
    except ImportError:
        st.error("❌ SHAP library not installed. Run: pip install shap")
        return None
    except Exception as e:
        st.warning(f"⚠️ Could not generate SHAP plot: {str(e)}")
        return None

@handle_errors()
def generate_feature_importance_plot(model, feature_names, top_n=15):
    """Generate feature importance plot as fallback to SHAP"""
    try:
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importance = np.abs(model.coef_[0] if len(model.coef_.shape) > 1 else model.coef_)
        else:
            return None
        
        # Create importance DataFrame
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=True).tail(top_n)
        
        # Create plot
        plt.figure(figsize=(10, 6))
        plt.barh(importance_df['feature'], importance_df['importance'])
        plt.xlabel('Importance')
        plt.title('Feature Importance')
        plt.tight_layout()
        
        return plt.gcf()
        
    except Exception as e:
        st.warning(f"Could not generate feature importance plot: {str(e)}")
        return None

@handle_errors()
def explain_prediction(model, X_sample, feature_names, prediction):
    """Provide text-based explanation of a prediction"""
    try:
        explanations = []
        
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
            
            # Get top contributing features
            feature_contributions = []
            for i, (feature, value) in enumerate(zip(feature_names, X_sample.iloc[0] if hasattr(X_sample, 'iloc') else X_sample[0])):
                contribution = importance[i] * abs(value)
                feature_contributions.append((feature, contribution, value))
            
            # Sort by contribution
            feature_contributions.sort(key=lambda x: x[1], reverse=True)
            
            explanations.append(f"**Prediction: {prediction}**")
            explanations.append("\n**Top Contributing Features:**")
            
            for feature, contribution, value in feature_contributions[:5]:
                explanations.append(f"• **{feature}**: {value:.3f} (contribution: {contribution:.3f})")
        
        return "\n".join(explanations)
        
    except Exception as e:
        return f"Could not generate explanation: {str(e)}"
