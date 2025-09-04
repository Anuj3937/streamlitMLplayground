# pages/4_Checkpoint_3_Model_Comparison.py

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

from utils.visualizations import create_confusion_matrix

st.set_page_config(page_title="Checkpoint 3: Model Comparison", page_icon="📈", layout="wide")

st.title("📈 Checkpoint 3: Model Comparison")
st.markdown("Compare model performance and select the best algorithm for your problem.")

# Update workflow stage
st.session_state.workflow_stage = 'checkpoint_3'

# Guard: results must exist
if st.session_state.get('model_results') is None:
    st.warning("No model results found. Please complete Checkpoint 2 first.")
    if st.button("← Go to Code Editor"):
        st.switch_page("pages/3_Checkpoint_2_Code_Editor.py")
    st.stop()

results = st.session_state.model_results
problem_type = st.session_state.problem_type
target_column = st.session_state.target_column

# Convert comparison to DataFrame
models_df = pd.DataFrame(results.get('model_comparison', []))
best_model = results.get('best_model', {})

# Utility to derive metric from table if missing in best_model payload
def _derive_metric(df: pd.DataFrame, name: str, higher_is_better: bool = True):
    try:
        if name in df.columns and len(df) > 0 and pd.api.types.is_numeric_dtype(df[name]):
            row = df.sort_values(name, ascending=not higher_is_better).iloc
            return float(row[name])
    except Exception:
        pass
    return None

# Results Overview
st.header("🏆 Model Performance Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Best Model", best_model.get('name', 'N/A'))

with col2:
    if problem_type == "classification":
        acc = best_model.get('accuracy')
        if acc is None:
            acc = _derive_metric(models_df, 'accuracy', higher_is_better=True)
        st.metric("Best Accuracy", f"{acc:.3f}" if acc is not None else "N/A")
    else:
        r2 = best_model.get('r2_score')
        if r2 is None:
            r2 = _derive_metric(models_df, 'r2_score', higher_is_better=True)
        st.metric("Best R² Score", f"{r2:.3f}" if r2 is not None else "N/A")

with col3:
    st.metric("Models Tested", len(models_df))

with col4:
    st.metric("Improvements", "Applied" if results.get('improvement_applied') else "None")

# Model Comparison Table
st.header("📊 Detailed Model Comparison")

# Ensure numeric columns are rounded for display
display_df = models_df.copy()
for col in display_df.columns:
    if pd.api.types.is_numeric_dtype(display_df[col]):
        display_df[col] = display_df[col].astype(float).round(4)
st.dataframe(display_df, use_container_width=True, hide_index=True)

# Performance Visualizations
st.header("📈 Performance Visualizations")
tab1, tab2, tab3, tab4 = st.tabs(["Model Comparison", "Best Model Details", "Feature Importance", "Confusion/Residuals"])

# Tab 1: Model Performance Comparison
with tab1:
    st.subheader("Model Performance Comparison")

    if problem_type == "classification":
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        present = [m for m in metrics if m in models_df.columns]
        if len(present) == 0:
            st.info("No classification metrics found in results. Please re-run training.")
        else:
            # Bar chart for primary metric (accuracy if present, else first available)
            primary = 'accuracy' if 'accuracy' in present else present
            fig = px.bar(
                models_df.sort_values(primary, ascending=False),
                x='model', y=primary, color=primary, color_continuous_scale='viridis',
                title=f"Model Ranking by {primary.replace('_',' ').title()}"
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        metrics = ['r2_score', 'mse', 'mae', 'rmse']
        present = [m for m in metrics if m in models_df.columns]
        if len(present) == 0:
            st.info("No regression metrics found in results. Please re-run training.")
        else:
            primary = 'r2_score' if 'r2_score' in present else present
            fig = px.bar(
                models_df.sort_values(primary, ascending=False),
                x='model', y=primary, color=primary, color_continuous_scale='viridis',
                title=f"Model Ranking by {primary.replace('_',' ').title()}"
            )
            st.plotly_chart(fig, use_container_width=True)

# Tab 2: Best Model Details
with tab2:
    st.subheader(f"Best Model: {best_model.get('name','N/A')}")

    colA, colB = st.columns(2)

    with colA:
        st.markdown("**Performance Metrics:**")
        if problem_type == "classification":
            metrics_to_show = [
                ('Accuracy', 'accuracy'),
                ('Precision', 'precision'),
                ('Recall', 'recall'),
                ('F1-Score', 'f1_score')
            ]
        else:
            metrics_to_show = [
                ('R² Score', 'r2_score'),
                ('Mean Squared Error', 'mse'),
                ('Mean Absolute Error', 'mae'),
                ('Root MSE', 'rmse')
            ]

        for label, key in metrics_to_show:
            val = best_model.get(key)
            # Derive from table if missing, only for primary keys
            if val is None and key in models_df.columns:
                val = _derive_metric(models_df, key, higher_is_better=(key not in ['mse','mae','rmse']))
            st.metric(label, f"{val:.4f}" if isinstance(val, (int, float)) else "N/A")

    with colB:
        st.markdown("**Model Parameters:**")
        params = best_model.get('best_params')
        if isinstance(params, dict) and len(params) > 0:
            params_df = pd.DataFrame(list(params.items()), columns=['Parameter', 'Value'])
            st.dataframe(params_df, use_container_width=True, hide_index=True)
        else:
            st.info("No hyperparameters to display.")

    # Optional training history if provided
    if 'training_history' in best_model and best_model['training_history']:
        st.subheader("Training Progress")
        history = best_model['training_history']
        fig = px.line(x=range(len(history)), y=history, title="Training Score Over Iterations")
        fig.update_xaxis(title="Iteration")
        fig.update_yaxis(title="Score")
        st.plotly_chart(fig, use_container_width=True)

# Tab 3: Feature Importance
with tab3:
    st.subheader("Feature Importance Analysis")
    fi = best_model.get('feature_importance')
    if isinstance(fi, dict) and len(fi) > 0:
        importance_df = pd.DataFrame({'Feature': list(fi.keys()), 'Importance': list(fi.values())})
        importance_df = importance_df.sort_values('Importance', ascending=True)
        top = importance_df.tail(15)
        fig = px.bar(
            top, x='Importance', y='Feature', orientation='h',
            title="Top 15 Most Important Features", color='Importance', color_continuous_scale='viridis'
        )
        fig.update_layout(height=max(400, len(top) * 30))
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Feature Insights")
        top_features = top['Feature'].tolist()[::-1][:5]
        st.success(f"🎯 Top 5 Features: {', '.join(top_features)}")
    else:
        st.info("Feature importance is not available for this model type.")

# Tab 4: Confusion (classification) or Residuals (regression)
with tab4:
    if problem_type == "classification":
        st.subheader("Confusion Matrix")
        cm = best_model.get('confusion_matrix')
        if cm is not None:
            cm = np.array(cm)
            fig = go.Figure(data=go.Heatmap(
                z=cm, text=cm, texttemplate="%{text}", textfont={"size": 16}, colorscale='Blues'
            ))
            fig.update_layout(title="Confusion Matrix", xaxis_title="Predicted", yaxis_title="Actual", height=400)
            st.plotly_chart(fig, use_container_width=True)

            if cm.size == 4:  # binary
                tn, fp, fn, tp = cm.ravel()
                c1, c2, c3, c4 = st.columns(4)
                with c1: st.metric("True Positives", tp)
                with c2: st.metric("True Negatives", tn)
                with c3: st.metric("False Positives", fp)
                with c4: st.metric("False Negatives", fn)
        else:
            st.info("Confusion matrix not available.")
    else:
        st.subheader("Residual Analysis")
        residuals = best_model.get('residuals')
        preds = best_model.get('predictions')
        if residuals is not None and preds is not None:
            residuals = np.array(residuals)
            preds = np.array(preds)
            c1, c2 = st.columns(2)
            with c1:
                fig = px.scatter(x=preds, y=residuals, title="Residuals vs Predictions",
                                 labels={'x': 'Predictions', 'y': 'Residuals'})
                fig.add_hline(y=0, line_dash="dash", line_color="red")
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                fig = px.histogram(residuals, title="Residuals Distribution",
                                   labels={'value': 'Residuals', 'count': 'Frequency'})
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Residual analysis not available.")

# Save Best Model
st.header("💾 Save Best Model")

colA, colB = st.columns([3, 1])
with colA:
    perf_val = None
    if problem_type == 'classification':
        perf_val = best_model.get('accuracy')
        if perf_val is None:
            perf_val = _derive_metric(models_df, 'accuracy', higher_is_better=True)
    else:
        perf_val = best_model.get('r2_score')
        if perf_val is None:
            perf_val = _derive_metric(models_df, 'r2_score', higher_is_better=True)

    st.markdown(f"""
    **Your best performing model is ready for deployment:**
    - **Model:** {best_model.get('name', 'N/A')}
    - **Performance:** {f"{perf_val:.3f}" if isinstance(perf_val, (int,float)) else "N/A"}
    - **Status:** Trained and optimized
    """)
with colB:
    if st.button("Approve Model →", type="primary", use_container_width=True):
        st.session_state.best_model = best_model
        st.session_state.workflow_stage = 'results'
        st.success("Model approved!")
        st.switch_page("pages/5_Results_Dashboard.py")

# Navigation
st.markdown("---")
c1, c2 = st.columns([1, 1])
with c1:
    if st.button("← Back to Code Editor", use_container_width=True):
        st.switch_page("pages/3_Checkpoint_2_Code_Editor.py")
with c2:
    if st.session_state.get('best_model') is not None:
        if st.button("Next: Results Dashboard →", use_container_width=True, type="primary"):
            st.switch_page("pages/5_Results_Dashboard.py")
