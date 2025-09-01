# pages/5_Results_Dashboard.py

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pickle
from utils.explainability import generate_shap_explanation # New import

st.set_page_config(page_title="Results Dashboard", page_icon="🎯", layout="wide")

st.title("🎯 Results Dashboard")
st.markdown("Comprehensive analysis and download of your machine learning model.")

if st.session_state.get('model_results') is None:
    st.warning("No model results found. Please complete the previous checkpoints.")
    st.stop()

results = st.session_state.model_results
best_model = results['best_model']
problem_type = st.session_state.problem_type
target_column = st.session_state.target_column
X_test = results['data_info']['X_test']

# Executive Summary
st.header("📋 Executive Summary")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Final Model", best_model['name'])
with col2:
    if problem_type == "classification":
        main_metric = best_model.get('accuracy', best_model.get('Test Score', 0))
        st.metric("Accuracy", f"{main_metric:.1%}")
    else:
        main_metric = best_model.get('r2_score', best_model.get('Test Score', 0))
        st.metric("R² Score", f"{main_metric:.3f}")
with col3:
    st.metric("Training Samples", f"{results['data_info']['training_samples']:,}")
with col4:
    st.metric("Features Used", results['data_info']['features_count'])


# Detailed Results
st.header("📊 Detailed Analysis")
tab1, tab2, tab3, tab4 = st.tabs(["Model Comparison", "Performance Metrics", "Model Explanation (XAI)", "Download Model"])

with tab1:
    st.subheader("All Models Comparison")
    comparison_df = pd.DataFrame(results['model_comparison'])
    st.dataframe(comparison_df.style.format(precision=4), use_container_width=True)
    
    ranking_metric = 'accuracy' if problem_type == "classification" else 'r2_score'
    if 'Test Score' in comparison_df.columns:
        ranking_metric = 'Test Score'

    if ranking_metric in comparison_df.columns:
        ranked_df = comparison_df.sort_values(ranking_metric, ascending=False).reset_index(drop=True)
        fig = px.bar(ranked_df, x='model', y=ranking_metric,
                    title=f"Model Ranking by {ranking_metric.replace('_', ' ').title()}",
                    color=ranking_metric, color_continuous_scale='viridis')
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("Comprehensive Performance Metrics")
    if problem_type == "classification":
        st.subheader("Confusion Matrix")
        if 'confusion_matrix' in best_model:
            cm = np.array(best_model['confusion_matrix'])
            fig = go.Figure(data=go.Heatmap(z=cm, text=cm, texttemplate="%{text}", colorscale='Blues'))
            fig.update_layout(title="Confusion Matrix", xaxis_title="Predicted", yaxis_title="Actual")
            st.plotly_chart(fig, use_container_width=True)
    else: # Regression
        st.subheader("Residuals Analysis")
        if 'residuals' in best_model and 'predictions' in best_model:
            residuals_df = pd.DataFrame({
                'Predictions': best_model['predictions'],
                'Residuals': best_model['residuals']
            })
            fig = px.scatter(residuals_df, x='Predictions', y='Residuals', title='Residuals vs. Predictions',
                             marginal_y='histogram')
            fig.add_hline(y=0, line_dash="dash", line_color="red")
            st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.subheader("Model Explanation with SHAP")
    st.markdown("""
    SHAP (SHapley Additive exPlanations) values explain the output of any machine learning model.
    The plot below shows the impact of each feature on the model's predictions. Features at the top have the highest impact.
    """)
    with st.spinner("Generating SHAP explanation plot... This may take a moment."):
        model_object = best_model.get('model_object')
        if model_object:
            shap_fig = generate_shap_explanation(model_object, X_test)
            if shap_fig:
                st.pyplot(shap_fig)
        else:
            st.warning("Could not generate SHAP plot as the model object is not available.")

with tab4:
    st.subheader("Download Your Model")
    st.markdown("""
    The downloadable `.pkl` file contains a dictionary with the trained model object,
    performance metrics, and other metadata needed for deployment.
    """)
    model_package = {
        'model': best_model.get('model_object'),
        'model_name': best_model['name'],
        'problem_type': problem_type,
        'target_column': target_column,
        'performance_metrics': {k: v for k, v in best_model.items() if isinstance(v, (int, float))},
        'feature_names': best_model.get('feature_names', []),
        'preprocessing_steps': st.session_state.get('preprocessing_steps', [])
    }
    model_bytes = pickle.dumps(model_package)
    st.download_button(
        label="📥 Download Model (.pkl)",
        data=model_bytes,
        file_name=f"{best_model['name'].lower().replace(' ', '_')}_model.pkl",
        mime="application/octet-stream",
        use_container_width=True,
        type="primary"
    )

st.header("🚀 Next Steps")
col1, col2 = st.columns(2)
with col1:
    if st.button("🎮 Create Demo App", use_container_width=True, type="primary"):
        st.switch_page("pages/6_Demo_App_Generator.py")
with col2:
    if st.button("🔄 Start New Project", use_container_width=True):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.switch_page("app.py")