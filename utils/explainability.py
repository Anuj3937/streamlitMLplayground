# utils/explainability.py

import streamlit as st
import shap
import pandas as pd
import matplotlib.pyplot as plt

@st.cache_data
def generate_shap_explanation(_model, X_test):
    """
    Generates a SHAP summary plot figure.
    Using _model to align with st.cache_data hashing requirements.
    """
    try:
        # SHAP expects models that have a predict function.
        # For tree models, TreeExplainer is more efficient.
        if hasattr(_model, 'predict_proba'):
            explainer = shap.KernelExplainer(_model.predict_proba, X_test)
            shap_values = explainer.shap_values(X_test)
        else:
            explainer = shap.KernelExplainer(_model.predict, X_test)
            shap_values = explainer.shap_values(X_test)

        # For classification, shap_values can be a list of arrays.
        # We'll plot for the positive class if it is.
        if isinstance(shap_values, list):
            shap_values_to_plot = shap_values[1] # Typically the positive class
        else:
            shap_values_to_plot = shap_values

        fig, ax = plt.subplots()
        shap.summary_plot(shap_values_to_plot, X_test, show=False)
        plt.tight_layout()
        return fig
    except Exception as e:
        st.error(f"Could not generate SHAP plot: {e}")
        return None