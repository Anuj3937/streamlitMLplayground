import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from utils.ui_components import (inject_custom_css, render_progress_tracker, sidebar_navigation,
                                create_info_card, create_alert, create_comparison_table, display_feature_importance)
from utils.visualizations import (create_model_comparison_chart, create_confusion_matrix, 
                                create_residual_plot, create_roc_curve)
from utils.error_handler import handle_errors

st.set_page_config(page_title="Model Comparison", page_icon="📈", layout="wide")
inject_custom_css()

def initialize_comparison_state():
    """Initialize model comparison session state"""
    st.session_state.workflow_stage = 'checkpoint_3'

@handle_errors()
def create_performance_overview(results, problem_type):
    """Create comprehensive performance overview"""
    st.markdown("### 🏆 Model Performance Overview")
    
    models_df = pd.DataFrame(results['model_comparison'])
    best_model = results['best_model']
    data_info = results.get('data_info', {})
    
    # Key metrics cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Best Model", best_model.get('name', 'Unknown'))
    
    with col2:
        if problem_type == "classification":
            main_metric = best_model.get('accuracy', 0)
            st.metric("Best Accuracy", f"{main_metric:.1%}")
        else:
            main_metric = best_model.get('r2_score', 0)
            st.metric("Best R² Score", f"{main_metric:.3f}")
    
    with col3:
        st.metric("Models Tested", len(models_df))
    
    with col4:
        training_samples = data_info.get('training_samples', 0)
        st.metric("Training Samples", f"{training_samples:,}")
    
    return models_df, best_model

@handle_errors()
def create_detailed_comparison_table(models_df, problem_type):
    """Create detailed model comparison table"""
    st.markdown("### 📊 Detailed Model Comparison")
    
    # Format the comparison table
    display_df = models_df.copy()
    
    # Round numeric columns
    for col in display_df.columns:
        if display_df[col].dtype in ['float64', 'int64'] and col != 'training_time':
            display_df[col] = display_df[col].round(4)
        elif col == 'training_time':
            display_df[col] = display_df[col].round(2)
    
    # Create styled table
    if problem_type == "classification":
        highlight_cols = ['accuracy', 'f1_score', 'cv_score']
    else:
        highlight_cols = ['r2_score', 'cv_score']
    
    # Style the dataframe
    styled_df = display_df.style.format(precision=4)
    
    for col in highlight_cols:
        if col in display_df.columns:
            styled_df = styled_df.highlight_max(subset=[col], color='lightgreen')
    
    st.dataframe(styled_df, use_container_width=True, hide_index=True)

@handle_errors()
def create_performance_visualizations(models_df, problem_type):
    """Create comprehensive performance visualizations"""
    st.markdown("### 📈 Performance Visualizations")
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Model Comparison", "🎯 Performance Metrics", "⏱️ Training Analysis", "📈 Cross-Validation"
    ])
    
    with tab1:
        st.markdown("#### Model Performance Ranking")
        
        # Determine ranking metric
        if problem_type == "classification":
            ranking_metric = 'accuracy'
            secondary_metric = 'f1_score'
        else:
            ranking_metric = 'r2_score'
            secondary_metric = 'mae'
        
        # Create comparison chart
        if ranking_metric in models_df.columns:
            fig = create_model_comparison_chart(models_df, ranking_metric)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        
        # Performance distribution
        if ranking_metric in models_df.columns:
            fig = px.box(
                models_df, y=ranking_metric,
                title=f"{ranking_metric.replace('_', ' ').title()} Distribution",
                points="all"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.markdown("#### Multi-Metric Performance Analysis")
        
        if problem_type == "classification":
            metrics = ['accuracy', 'precision', 'recall', 'f1_score']
            available_metrics = [m for m in metrics if m in models_df.columns]
        else:
            metrics = ['r2_score', 'mse', 'mae', 'rmse']
            available_metrics = [m for m in metrics if m in models_df.columns]
        
        if len(available_metrics) >= 3:
            # Radar chart for comprehensive comparison
            fig = go.Figure()
            
            for _, row in models_df.iterrows():
                values = []
                for metric in available_metrics[:4]:  # Use top 4 metrics
                    if metric in row and pd.notna(row[metric]):
                        # Normalize MSE and MAE (lower is better)
                        if metric in ['mse', 'mae', 'rmse']:
                            max_val = models_df[metric].max()
                            normalized = 1 - (row[metric] / max_val) if max_val > 0 else 0
                        else:
                            normalized = row[metric]
                        values.append(max(0, min(1, normalized)))
                    else:
                        values.append(0)
                
                fig.add_trace(go.Scatterpolar(
                    r=values + [values[0]],
                    theta=[m.replace('_', ' ').title() for m in available_metrics[:4]] + [available_metrics[0].replace('_', ' ').title()],
                    fill='toself',
                    name=row['model']
                ))
            
            fig.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                showlegend=True,
                title="Multi-Metric Performance Comparison",
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.markdown("#### Training Efficiency Analysis")
        
        if 'training_time' in models_df.columns:
            # Training time vs performance scatter
            fig = px.scatter(
                models_df,
                x='training_time',
                y=ranking_metric,
                size=[1] * len(models_df),
                hover_name='model',
                title="Performance vs Training Time",
                labels={
                    'training_time': 'Training Time (seconds)',
                    ranking_metric: ranking_metric.replace('_', ' ').title()
                }
            )
            
            # Add model names as annotations
            for _, row in models_df.iterrows():
                fig.add_annotation(
                    x=row['training_time'],
                    y=row[ranking_metric],
                    text=row['model'],
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1,
                    arrowwidth=1
                )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Training time comparison
            fig_time = px.bar(
                models_df.sort_values('training_time'),
                x='training_time',
                y='model',
                orientation='h',
                title="Training Time Comparison",
                color='training_time',
                color_continuous_scale='viridis'
            )
            fig_time.update_layout(height=max(400, len(models_df) * 40))
            st.plotly_chart(fig_time, use_container_width=True)
    
    with tab4:
        st.markdown("#### Cross-Validation Analysis")
        
        if 'cv_score' in models_df.columns:
            # CV scores comparison
            fig = px.bar(
                models_df.sort_values('cv_score', ascending=True),
                x='cv_score',
                y='model',
                orientation='h',
                title="Cross-Validation Scores",
                color='cv_score',
                color_continuous_scale='blues'
            )
            fig.update_layout(height=max(400, len(models_df) * 40))
            st.plotly_chart(fig, use_container_width=True)
            
            # CV vs Test performance
            test_metric = ranking_metric
            if test_metric in models_df.columns:
                fig_cv = px.scatter(
                    models_df,
                    x='cv_score',
                    y=test_metric,
                    hover_name='model',
                    title="Cross-Validation vs Test Performance",
                    labels={
                        'cv_score': 'Cross-Validation Score',
                        test_metric: f'Test {test_metric.replace("_", " ").title()}'
                    }
                )
                
                # Add diagonal line for perfect correlation
                min_val = min(models_df['cv_score'].min(), models_df[test_metric].min())
                max_val = max(models_df['cv_score'].max(), models_df[test_metric].max())
                fig_cv.add_shape(
                    type="line",
                    x0=min_val, y0=min_val,
                    x1=max_val, y1=max_val,
                    line=dict(color="red", dash="dash"),
                )
                
                st.plotly_chart(fig_cv, use_container_width=True)

@handle_errors()
def create_best_model_analysis(best_model, problem_type, results):
    """Create detailed best model analysis"""
    st.markdown("### 🥇 Best Model Deep Dive")
    
    model_name = best_model.get('name', 'Unknown')
    create_info_card(
        f"Champion Model: {model_name}",
        f"Selected based on {'accuracy' if problem_type == 'classification' else 'R² score'} performance",
        "🏆"
    )
    
    tab1, tab2, tab3 = st.tabs(["📊 Performance Metrics", "🎛️ Model Parameters", "🔍 Detailed Analysis"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Key Performance Metrics:**")
            if problem_type == "classification":
                metrics_to_show = [
                    ('Accuracy', 'accuracy', '{:.1%}'),
                    ('Precision', 'precision', '{:.3f}'),
                    ('Recall', 'recall', '{:.3f}'),
                    ('F1-Score', 'f1_score', '{:.3f}')
                ]
            else:
                metrics_to_show = [
                    ('R² Score', 'r2_score', '{:.4f}'),
                    ('Mean Squared Error', 'mse', '{:.4f}'),
                    ('Root MSE', 'rmse', '{:.4f}'),
                    ('Mean Absolute Error', 'mae', '{:.4f}')
                ]
            
            for name, key, fmt in metrics_to_show:
                if key in best_model:
                    value = best_model[key]
                    formatted_value = fmt.format(value) if isinstance(value, (int, float)) else str(value)
                    st.metric(name, formatted_value)
        
        with col2:
            st.markdown("**Training Information:**")
            cv_score = best_model.get('cv_score', 0)
            training_time = best_model.get('training_time', 0)
            
            st.metric("Cross-Validation Score", f"{cv_score:.4f}")
            st.metric("Training Time", f"{training_time:.2f}s")
            
            # Training samples info
            data_info = results.get('data_info', {})
            if data_info:
                st.metric("Training Samples", f"{data_info.get('training_samples', 0):,}")
                st.metric("Test Samples", f"{data_info.get('test_samples', 0):,}")
    
    with tab2:
        st.markdown("**Optimized Hyperparameters:**")
        
        best_params = best_model.get('best_params', {})
        if best_params:
            # Create a nice display for parameters
            params_df = pd.DataFrame([
                {'Parameter': key, 'Value': str(value)[:50]}  # Limit string length
                for key, value in best_params.items()
            ])
            st.dataframe(params_df, use_container_width=True, hide_index=True)
        else:
            st.info("No hyperparameters available for this model.")
        
        # Model complexity indicator
        if best_params:
            st.markdown("**Model Complexity Analysis:**")
            complexity_score = len(best_params)
            st.progress(min(complexity_score / 10, 1.0))
            st.caption(f"Complexity Score: {complexity_score}/10 (based on number of parameters)")
    
    with tab3:
        st.markdown("#### Advanced Model Analysis")
        
        # Feature importance
        if 'feature_importance' in best_model and best_model['feature_importance']:
            importance_data = best_model['feature_importance']
            
            if isinstance(importance_data, dict) and 'features' in importance_data:
                display_feature_importance(
                    importance_data['features'],
                    importance_data['importance'],
                    top_n=15
                )
                
                # Feature insights
                features = importance_data['features']
                importance = importance_data['importance']
                
                if features and importance:
                    importance_df = pd.DataFrame({
                        'Feature': features,
                        'Importance': importance
                    }).sort_values('Importance', ascending=False)
                    
                    top_features = importance_df.head(5)['Feature'].tolist()
                    low_importance = importance_df[importance_df['Importance'] < 0.01]
                    
                    create_alert(
                        f"🎯 <strong>Top 5 Most Important Features:</strong><br>{'<br>'.join([f'• {feat}' for feat in top_features])}",
                        "info"
                    )
                    
                    if len(low_importance) > 0:
                        create_alert(
                            f"⚠️ <strong>{len(low_importance)} features</strong> have very low importance (<1%) and could be removed for model simplification.",
                            "warning"
                        )
        else:
            st.info("Feature importance not available for this model type.")
        
        # Confusion Matrix or Residuals
        if problem_type == "classification" and 'confusion_matrix' in best_model:
            st.markdown("#### Confusion Matrix Analysis")
            cm = np.array(best_model['confusion_matrix'])
            
            fig = go.Figure(data=go.Heatmap(
                z=cm,
                text=cm,
                texttemplate="%{text}",
                textfont={"size": 16},
                colorscale='Blues',
                showscale=True
            ))
            
            fig.update_layout(
                title="Confusion Matrix",
                xaxis_title="Predicted",
                yaxis_title="Actual",
                height=500,
                width=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Classification metrics from confusion matrix
            if cm.shape == (2, 2):  # Binary classification
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
                
        elif problem_type == "regression" and 'residuals' in best_model:
            st.markdown("#### Residual Analysis")
            
            residuals = best_model['residuals']
            predictions = best_model.get('predictions', [])
            
            if residuals and predictions:
                col1, col2 = st.columns(2)
                
                with col1:
                    # Residuals vs Predictions
                    fig = px.scatter(
                        x=predictions, y=residuals,
                        title="Residuals vs Predictions",
                        labels={'x': 'Predictions', 'y': 'Residuals'}
                    )
                    fig.add_hline(y=0, line_dash="dash", line_color="red")
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Residuals distribution
                    fig = px.histogram(
                        residuals, 
                        title="Residuals Distribution",
                        labels={'value': 'Residuals', 'count': 'Frequency'}
                    )
                    st.plotly_chart(fig, use_container_width=True)

def create_model_approval_section(best_model):
    """Create model approval section"""
    st.markdown("---")
    st.markdown("### ✅ Model Approval & Next Steps")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        model_name = best_model.get('name', 'Unknown')
        
        if st.session_state.problem_type == 'classification':
            performance = best_model.get('accuracy', 0)
            performance_text = f"{performance:.1%} accuracy"
        else:
            performance = best_model.get('r2_score', 0)
            performance_text = f"{performance:.3f} R² score"
        
        create_alert(
            f"🏆 <strong>{model_name}</strong> achieved <strong>{performance_text}</strong> and is ready for deployment!",
            "success"
        )
    
    with col2:
        if st.button("Approve & Continue →", type="primary", use_container_width=True):
            st.session_state.best_model = best_model
            st.session_state.workflow_stage = 'results'
            
            st.balloons()
            st.success("🎉 Model approved for deployment!")
            st.switch_page("pages/5_Results_Dashboard.py")

def main():
    """Main function for model comparison page"""
    initialize_comparison_state()
    
    # Check if model results exist
    if st.session_state.get('model_results') is None:
        st.warning("No model results found. Please complete the previous steps.")
        if st.button("← Go to Code Editor"):
            st.switch_page("pages/3_Code_Editor.py")
        st.stop()
    
    results = st.session_state.model_results
    problem_type = st.session_state.problem_type
    
    # Header
    st.title("📈 Model Comparison & Analysis")
    st.markdown("Compare model performance and select the best algorithm for deployment.")
    
    # Progress tracker
    render_progress_tracker('checkpoint_3')
    
    # Performance overview
    models_df, best_model = create_performance_overview(results, problem_type)
    
    st.markdown("---")
    
    # Detailed comparison table
    create_detailed_comparison_table(models_df, problem_type)
    
    st.markdown("---")
    
    # Performance visualizations
    create_performance_visualizations(models_df, problem_type)
    
    st.markdown("---")
    
    # Best model analysis
    create_best_model_analysis(best_model, problem_type, results)
    
    # Model approval
    create_model_approval_section(best_model)
    
    # Navigation
    st.markdown("---")
    col1, col2 = st.columns([1, 1])
    
    with col1:
        if st.button("← Back to Code Editor", use_container_width=True):
            st.switch_page("pages/3_Code_Editor.py")
    
    with col2:
        if st.session_state.get('best_model'):
            if st.button("Results Dashboard →", use_container_width=True, type="secondary"):
                st.switch_page("pages/5_Results_Dashboard.py")
    
    # Sidebar
    sidebar_navigation()

if __name__ == "__main__":
    main()
