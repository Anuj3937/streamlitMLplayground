import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from utils.visualizations import create_confusion_matrix, create_feature_importance_plot

st.set_page_config(page_title="Checkpoint 3: Model Comparison", page_icon="📈", layout="wide")

st.title("📈 Checkpoint 3: Model Comparison")
st.markdown("Compare model performance and select the best algorithm for your problem.")

# Update workflow stage
st.session_state.workflow_stage = 'checkpoint_3'

# Check if model results exist
if st.session_state.get('model_results') is None:
    st.warning("No model results found. Please complete Checkpoint 2 first.")
    if st.button("← Go to Code Editor"):
        st.switch_page("pages/3_Checkpoint_2_Code_Editor.py")
    st.stop()

results = st.session_state.model_results
problem_type = st.session_state.problem_type
target_column = st.session_state.target_column

# Results Overview
st.header("🏆 Model Performance Overview")

# Extract performance metrics
models_df = pd.DataFrame(results['model_comparison'])
best_model = results['best_model']

# Display key metrics
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Best Model", best_model['name'])

with col2:
    if problem_type == "classification":
        st.metric("Best Accuracy", f"{best_model['accuracy']:.3f}")
    else:
        st.metric("Best R² Score", f"{best_model['r2_score']:.3f}")

with col3:
    st.metric("Models Tested", len(models_df))

with col4:
    if 'improvement_applied' in results:
        st.metric("Improvements", "Applied" if results['improvement_applied'] else "None")
    else:
        st.metric("Improvements", "None")

# Model Comparison Table
st.header("📊 Detailed Model Comparison")

# Format the comparison table
display_df = models_df.copy()
for col in display_df.columns:
    if display_df[col].dtype == 'float64':
        display_df[col] = display_df[col].round(4)

st.dataframe(display_df, use_container_width=True, hide_index=True)

# Performance Visualizations
st.header("📈 Performance Visualizations")

tab1, tab2, tab3, tab4 = st.tabs(["Model Comparison", "Best Model Details", "Feature Importance", "Confusion Matrix"])

with tab1:
    st.subheader("Model Performance Comparison")
    
    # Create comparison charts
    if problem_type == "classification":
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    else:
        metrics = ['r2_score', 'mse', 'mae']
        metric_names = ['R² Score', 'MSE', 'MAE']
    
    # Performance comparison bar chart
    fig = go.Figure()
    
    for i, (metric, name) in enumerate(zip(metrics[:2], metric_names[:2])):  # Show top 2 metrics
        if metric in models_df.columns:
            fig.add_trace(go.Bar(
                name=name,
                x=models_df['model'],
                y=models_df[metric],
                yaxis=f'y{i+1}' if i > 0 else 'y'
            ))
    
    fig.update_layout(
        title="Model Performance Comparison",
        xaxis_title="Models",
        barmode='group',
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Radar chart for comprehensive comparison
    if len(metrics) >= 3:
        st.subheader("Comprehensive Performance Radar")
        
        fig = go.Figure()
        
        for _, row in models_df.iterrows():
            values = []
            for metric in metrics[:4]:  # Use up to 4 metrics
                if metric in row and pd.notna(row[metric]):
                    # Normalize MSE and MAE (lower is better) for radar chart
                    if metric in ['mse', 'mae']:
                        max_val = models_df[metric].max()
                        normalized = 1 - (row[metric] / max_val) if max_val > 0 else 0
                    else:
                        normalized = row[metric]
                    values.append(normalized)
                else:
                    values.append(0)
            
            fig.add_trace(go.Scatterpolar(
                r=values + [values[0]],  # Close the polygon
                theta=metric_names[:len(values)] + [metric_names[0]],
                fill='toself',
                name=row['model']
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )),
            showlegend=True,
            title="Multi-Metric Performance Comparison"
        )
        
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader(f"Best Model: {best_model['name']}")
    
    col1, col2 = st.columns(2)
    
    with col1:
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
        
        for name, key in metrics_to_show:
            if key in best_model:
                st.metric(name, f"{best_model[key]:.4f}")
    
    with col2:
        st.markdown("**Model Parameters:**")
        if 'best_params' in best_model:
            params_df = pd.DataFrame(list(best_model['best_params'].items()), 
                                   columns=['Parameter', 'Value'])
            st.dataframe(params_df, use_container_width=True, hide_index=True)
        else:
            st.info("No hyperparameters to display")
    
    # Training progress if available
    if 'training_history' in best_model:
        st.subheader("Training Progress")
        history = best_model['training_history']
        if history:
            fig = px.line(x=range(len(history)), y=history, title="Training Score Over Iterations")
            fig.update_xaxis(title="Iteration")
            fig.update_yaxis(title="Score")
            st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.subheader("Feature Importance Analysis")
    
    if 'feature_importance' in best_model and best_model['feature_importance'] is not None:
        feature_data = best_model['feature_importance']
        if 'features' in feature_data and 'importance' in feature_data:
            features = feature_data['features']
            importance = feature_data['importance']
            
            # Ensure both lists have the same length
            if len(features) == len(importance):
                importance_df = pd.DataFrame({
                    'Feature': features,
                    'Importance': importance
                }).sort_values('Importance', ascending=True)
            else:
                st.error("Feature importance data is malformed.")
                importance_df = None
        else:
            st.error("Feature importance data is incomplete.")
            importance_df = None
        
        # Feature importance bar chart
        if importance_df is not None:
            fig = px.bar(importance_df.tail(15), 
                        x='Importance', 
                        y='Feature',
                        orientation='h',
                        title="Top 15 Most Important Features")
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
            
            # Feature importance insights
            st.subheader("Feature Insights")
            top_features = importance_df.tail(5)['Feature'].tolist()
            st.success(f"🎯 **Top 5 Most Important Features:** {', '.join(top_features)}")
            
            # Low importance features
            low_importance = importance_df[importance_df['Importance'] < 0.01]
            if len(low_importance) > 0:
                st.warning(f"⚠️ **{len(low_importance)} features** have very low importance (< 1%) and could potentially be removed.")
    
    else:
        st.info("Feature importance is not available for this model type.")

with tab4:
    if problem_type == "classification":
        st.subheader("Confusion Matrix")
        
        if 'confusion_matrix' in best_model:
            cm = best_model['confusion_matrix']
            
            # Create confusion matrix heatmap
            fig = go.Figure(data=go.Heatmap(
                z=cm,
                text=cm,
                texttemplate="%{text}",
                textfont={"size": 16},
                colorscale='Blues'
            ))
            
            fig.update_layout(
                title="Confusion Matrix",
                xaxis_title="Predicted",
                yaxis_title="Actual",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Classification insights
            if len(cm) == 2:  # Binary classification
                tn, fp, fn, tp = cm.ravel()
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("True Positives", tp)
                with col2:
                    st.metric("True Negatives", tn)
                with col3:
                    st.metric("False Positives", fp)
                with col4:
                    st.metric("False Negatives", fn)
        else:
            st.info("Confusion matrix not available.")
    
    else:
        st.subheader("Residual Analysis")
        
        if 'residuals' in best_model:
            residuals = best_model['residuals']
            predictions = best_model['predictions']
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Residuals vs Predictions
                fig = px.scatter(x=predictions, y=residuals, 
                               title="Residuals vs Predictions",
                               labels={'x': 'Predictions', 'y': 'Residuals'})
                fig.add_hline(y=0, line_dash="dash", line_color="red")
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Residuals distribution
                fig = px.histogram(residuals, title="Residuals Distribution",
                                 labels={'value': 'Residuals', 'count': 'Frequency'})
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Residual analysis not available.")

# Model Improvement Analysis
if 'improvement_analysis' in results:
    st.header("🔧 Automated Improvement Analysis")
    
    improvement = results['improvement_analysis']
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("**Improvement Techniques Applied:**")
        for technique in improvement.get('techniques_applied', []):
            st.markdown(f"✅ {technique}")
        
        if improvement.get('performance_gain', 0) > 0:
            st.success(f"🚀 **Performance Improvement:** +{improvement['performance_gain']:.3f}")
        else:
            st.info("🔍 **No significant improvement found** - Your model is already well-optimized!")
    
    with col2:
        if 'before_after' in improvement:
            before_after = improvement['before_after']
            st.metric("Before Optimization", f"{before_after['before']:.3f}")
            st.metric("After Optimization", f"{before_after['after']:.3f}")

# Save Best Model
st.header("💾 Save Best Model")

col1, col2 = st.columns([3, 1])

with col1:
    st.markdown(f"""
    **Your best performing model is ready for deployment:**
    
    - **Model:** {best_model['name']}
    - **Performance:** {best_model.get('accuracy' if problem_type == 'classification' else 'r2_score', 0):.3f}
    - **Status:** Trained and optimized
    
    You can now proceed to view detailed results or generate a demo application.
    """)

with col2:
    if st.button("Approve Model →", type="primary", use_container_width=True):
        st.session_state.best_model = best_model
        st.session_state.workflow_stage = 'results'
        st.success("Model approved!")
        st.switch_page("pages/5_Results_Dashboard.py")

# Navigation
st.markdown("---")
col1, col2 = st.columns([1, 1])

with col1:
    if st.button("← Back to Code Editor", use_container_width=True):
        st.switch_page("pages/3_Checkpoint_2_Code_Editor.py")

with col2:
    if st.session_state.get('best_model') is not None:
        if st.button("Next: Results Dashboard →", use_container_width=True, type="primary"):
            st.switch_page("pages/5_Results_Dashboard.py")
