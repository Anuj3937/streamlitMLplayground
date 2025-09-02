import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pickle
import io
from datetime import datetime
from utils.ui_components import (inject_custom_css, render_progress_tracker, sidebar_navigation,
                                create_info_card, create_alert, create_metric_card)
from utils.explainability import generate_shap_explanation
from utils.visualizations import create_prediction_vs_actual_plot
from utils.error_handler import handle_errors
from utils.cache_manager import cache_manager

st.set_page_config(page_title="Results Dashboard", page_icon="🎯", layout="wide")
inject_custom_css()

def initialize_results_state():
    """Initialize results dashboard session state"""
    st.session_state.workflow_stage = 'results'

@handle_errors()
def create_executive_summary(results, best_model, problem_type):
    """Create comprehensive executive summary"""
    st.markdown("### 📋 Executive Summary")
    
    data_info = results.get('data_info', {})
    
    # Key metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        create_metric_card("Final Model", best_model.get('name', 'Unknown'), "", "normal")
    
    with col2:
        if problem_type == "classification":
            main_metric = best_model.get('accuracy', 0)
            create_metric_card("Accuracy", f"{main_metric:.1%}", 
                             f"+{(main_metric-0.5)*100:.1f}% vs random", "positive" if main_metric > 0.7 else "normal")
        else:
            main_metric = best_model.get('r2_score', 0)
            create_metric_card("R² Score", f"{main_metric:.3f}",
                             "Higher is better", "positive" if main_metric > 0.7 else "normal")
    
    with col3:
        training_samples = data_info.get('training_samples', 0)
        create_metric_card("Training Data", f"{training_samples:,}", 
                         f"{data_info.get('features_count', 0)} features", "normal")
    
    with col4:
        model_comparison = results.get('model_comparison', [])
        create_metric_card("Models Tested", f"{len(model_comparison)}", 
                         "Algorithms compared", "normal")

@handle_errors()
def create_model_performance_deep_dive(results, best_model, problem_type):
    """Create detailed performance analysis"""
    st.markdown("### 🔍 Performance Deep Dive")
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 All Models", "🎯 Best Model", "📈 Performance Trends", "🔬 Model Analysis"
    ])
    
    with tab1:
        st.markdown("#### Complete Model Comparison")
        
        comparison_df = pd.DataFrame(results['model_comparison'])
        
        # Enhanced comparison table with styling
        if problem_type == "classification":
            key_columns = ['model', 'accuracy', 'f1_score', 'cv_score', 'training_time']
        else:
            key_columns = ['model', 'r2_score', 'mse', 'cv_score', 'training_time']
        
        display_df = comparison_df[key_columns] if all(col in comparison_df.columns for col in key_columns) else comparison_df
        
        # Style the dataframe
        styled_df = display_df.style.format(precision=4)
        
        # Highlight best performers
        numeric_cols = display_df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col != 'training_time':  # Higher is better for most metrics
                styled_df = styled_df.highlight_max(subset=[col], color='lightgreen')
            else:  # Lower is better for training time
                styled_df = styled_df.highlight_min(subset=[col], color='lightblue')
        
        st.dataframe(styled_df, use_container_width=True, hide_index=True)
        
        # Performance ranking visualization
        ranking_metric = 'accuracy' if problem_type == 'classification' else 'r2_score'
        if ranking_metric in comparison_df.columns:
            fig = px.bar(
                comparison_df.sort_values(ranking_metric, ascending=True),
                x=ranking_metric,
                y='model',
                orientation='h',
                title=f"Model Ranking by {ranking_metric.replace('_', ' ').title()}",
                color=ranking_metric,
                color_continuous_scale='viridis'
            )
            fig.update_layout(height=max(400, len(comparison_df) * 40))
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.markdown(f"#### Best Model: {best_model.get('name', 'Unknown')}")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**🎯 Performance Metrics:**")
            if problem_type == "classification":
                metrics = [
                    ('Accuracy', 'accuracy', '{:.1%}'),
                    ('Precision', 'precision', '{:.3f}'),
                    ('Recall', 'recall', '{:.3f}'),
                    ('F1-Score', 'f1_score', '{:.3f}')
                ]
            else:
                metrics = [
                    ('R² Score', 'r2_score', '{:.4f}'),
                    ('MSE', 'mse', '{:.6f}'),
                    ('RMSE', 'rmse', '{:.4f}'),
                    ('MAE', 'mae', '{:.4f}')
                ]
            
            for name, key, fmt in metrics:
                if key in best_model:
                    value = best_model[key]
                    if isinstance(value, (int, float)):
                        st.metric(name, fmt.format(value))
        
        with col2:
            st.markdown("**⚙️ Model Configuration:**")
            
            # Cross-validation score
            cv_score = best_model.get('cv_score', 0)
            st.metric("CV Score", f"{cv_score:.4f}")
            
            # Training time
            training_time = best_model.get('training_time', 0)
            st.metric("Training Time", f"{training_time:.2f}s")
            
            # Hyperparameters count
            best_params = best_model.get('best_params', {})
            st.metric("Optimized Parameters", len(best_params))
        
        # Hyperparameters details
        if best_params:
            st.markdown("**🎛️ Optimized Hyperparameters:**")
            with st.expander("View Parameters", expanded=False):
                params_df = pd.DataFrame([
                    {'Parameter': key, 'Value': str(value)}
                    for key, value in best_params.items()
                ])
                st.dataframe(params_df, use_container_width=True, hide_index=True)
    
    with tab3:
        st.markdown("#### Performance Trends & Insights")
        
        comparison_df = pd.DataFrame(results['model_comparison'])
        
        if len(comparison_df) > 1:
            # Performance vs Complexity scatter
            if 'training_time' in comparison_df.columns:
                ranking_metric = 'accuracy' if problem_type == 'classification' else 'r2_score'
                
                if ranking_metric in comparison_df.columns:
                    fig = px.scatter(
                        comparison_df,
                        x='training_time',
                        y=ranking_metric,
                        size=[20] * len(comparison_df),  # Uniform size
                        hover_name='model',
                        title="Performance vs Training Time Trade-off",
                        labels={
                            'training_time': 'Training Time (seconds)',
                            ranking_metric: ranking_metric.replace('_', ' ').title()
                        }
                    )
                    
                    # Highlight best model
                    best_model_row = comparison_df[comparison_df['model'] == best_model.get('name')]
                    if not best_model_row.empty:
                        fig.add_trace(go.Scatter(
                            x=best_model_row['training_time'],
                            y=best_model_row[ranking_metric],
                            mode='markers',
                            marker=dict(color='red', size=15, symbol='star'),
                            name='Best Model',
                            hovertemplate='<b>Best Model</b><br>' +
                                        f'Training Time: %{{x:.2f}}s<br>' +
                                        f'{ranking_metric.title()}: %{{y:.3f}}'
                        ))
                    
                    st.plotly_chart(fig, use_container_width=True)
            
            # Model performance distribution
            numeric_cols = comparison_df.select_dtypes(include=[np.number]).columns
            performance_cols = [col for col in numeric_cols if col not in ['training_time']]
            
            if performance_cols:
                selected_metric = st.selectbox("Select metric for distribution analysis:", performance_cols)
                
                fig = px.histogram(
                    comparison_df,
                    x=selected_metric,
                    title=f"Distribution of {selected_metric.replace('_', ' ').title()}",
                    nbins=min(10, len(comparison_df))
                )
                
                # Add vertical line for best model
                if selected_metric in best_model:
                    fig.add_vline(
                        x=best_model[selected_metric],
                        line_dash="dash",
                        line_color="red",
                        annotation_text="Best Model"
                    )
                
                st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.markdown("#### Advanced Model Analysis")
        
        # Feature importance analysis
        if 'feature_importance' in best_model and best_model['feature_importance']:
            st.markdown("**🔍 Feature Importance Analysis**")
            
            importance_data = best_model['feature_importance']
            if isinstance(importance_data, dict) and 'features' in importance_data:
                features = importance_data['features']
                importance = importance_data['importance']
                
                if features and importance and len(features) == len(importance):
                    importance_df = pd.DataFrame({
                        'Feature': features,
                        'Importance': importance
                    }).sort_values('Importance', ascending=True)
                    
                    # Top features visualization
                    fig = px.bar(
                        importance_df.tail(15),
                        x='Importance',
                        y='Feature',
                        orientation='h',
                        title="Top 15 Most Important Features",
                        color='Importance',
                        color_continuous_scale='viridis'
                    )
                    fig.update_layout(height=500)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Feature insights
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        top_features = importance_df.tail(5)['Feature'].tolist()
                        create_alert(
                            f"🎯 **Top 5 Features:**<br>" + "<br>".join([f"• {feat}" for feat in top_features]),
                            "info"
                        )
                    
                    with col2:
                        low_importance = importance_df[importance_df['Importance'] < 0.01]
                        if len(low_importance) > 0:
                            create_alert(
                                f"⚠️ **{len(low_importance)} features** have very low importance and could be removed.",
                                "warning"
                            )
                        else:
                            create_alert("✅ All features contribute meaningfully to the model.", "success")
            else:
                st.warning("Feature importance data is not properly formatted.")
        else:
            st.info("Feature importance not available for this model type.")
        
        # Model predictions analysis
        if problem_type == "regression" and 'predictions' in best_model and 'residuals' in best_model:
            st.markdown("**📈 Prediction Analysis**")
            
            predictions = best_model['predictions']
            residuals = best_model['residuals']
            
            # Get actual values from test data
            data_info = results.get('data_info', {})
            X_test = data_info.get('X_test')
            
            if X_test is not None and hasattr(X_test, 'index'):
                # Residual analysis
                col1, col2 = st.columns(2)
                
                with col1:
                    fig = px.scatter(
                        x=predictions, y=residuals,
                        title="Residuals vs Predictions",
                        labels={'x': 'Predictions', 'y': 'Residuals'}
                    )
                    fig.add_hline(y=0, line_dash="dash", line_color="red")
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    fig = px.histogram(
                        residuals,
                        title="Residuals Distribution",
                        nbins=30
                    )
                    st.plotly_chart(fig, use_container_width=True)

@handle_errors()
def create_model_interpretability_section(best_model, results):
    """Create model interpretability with SHAP"""
    st.markdown("### 🔬 Model Interpretability (SHAP Analysis)")
    
    create_info_card(
        "SHAP Explanation",
        """
        SHAP (SHapley Additive exPlanations) shows how each feature contributes to individual predictions.<br>
        This helps understand <strong>why</strong> the model makes specific predictions.
        """,
        "🔬"
    )
    
    model_object = best_model.get('model_object')
    data_info = results.get('data_info', {})
    X_test = data_info.get('X_test')
    
    if model_object and X_test is not None:
        with st.spinner("🔬 Generating SHAP explanations... This may take a moment."):
            try:
                # Generate SHAP plot
                shap_fig = generate_shap_explanation(model_object, X_test.iloc[:100])  # Use first 100 samples
                
                if shap_fig:
                    st.pyplot(shap_fig, use_container_width=True)
                    
                    create_alert(
                        "💡 **How to read this plot:**<br>" +
                        "• Each dot represents one prediction<br>" +
                        "• Red dots push prediction higher, blue dots push lower<br>" +
                        "• Features at the top have the highest average impact",
                        "info"
                    )
                else:
                    st.warning("Could not generate SHAP plot for this model type.")
                    
            except Exception as e:
                st.warning(f"SHAP analysis failed: {str(e)}")
                st.info("SHAP analysis is not available for all model types.")
    else:
        st.warning("Model object or test data not available for SHAP analysis.")

@handle_errors()
def create_model_download_section(best_model, results):
    """Create model download and export section"""
    st.markdown("### 💾 Model Export & Download")
    
    model_name = best_model.get('name', 'Unknown').lower().replace(' ', '_')
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    col1, col2 = st.columns(2)
    
    with col1:
        create_info_card(
            "Model Package Contents",
            """
            • <strong>Trained model object</strong> ready for predictions<br>
            • <strong>Performance metrics</strong> and validation scores<br>
            • <strong>Feature names</strong> and preprocessing info<br>
            • <strong>Hyperparameters</strong> and model configuration
            """,
            "📦"
        )
        
        # Model metadata
        st.markdown("**📊 Model Metadata:**")
        metadata = {
            "Model Type": best_model.get('name', 'Unknown'),
            "Problem Type": st.session_state.get('problem_type', 'Unknown'),
            "Target Column": st.session_state.get('target_column', 'Unknown'),
            "Training Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Features Count": len(best_model.get('feature_names', [])),
        }
        
        for key, value in metadata.items():
            st.text(f"{key}: {value}")
    
    with col2:
        st.markdown("**📥 Download Options:**")
        
        # Prepare model package
        model_package = {
            'model': best_model.get('model_object'),
            'metadata': {
                'model_name': best_model.get('name', 'Unknown'),
                'problem_type': st.session_state.get('problem_type'),
                'target_column': st.session_state.get('target_column'),
                'training_timestamp': datetime.now().isoformat(),
                'performance_metrics': {k: v for k, v in best_model.items() 
                                     if isinstance(v, (int, float, str)) and k != 'model_object'},
                'feature_names': best_model.get('feature_names', []),
                'preprocessing_steps': st.session_state.get('preprocessing_steps', []),
                'hyperparameters': best_model.get('best_params', {})
            }
        }
        
        # Create download data
        try:
            model_bytes = pickle.dumps(model_package)
            
            st.download_button(
                label="📥 Download Complete Model Package (.pkl)",
                data=model_bytes,
                file_name=f"{model_name}_model_{timestamp}.pkl",
                mime="application/octet-stream",
                use_container_width=True,
                type="primary",
                help="Download the complete trained model with all metadata"
            )
        except Exception as e:
            st.error(f"Failed to prepare model for download: {str(e)}")
        
        # Export model summary
        model_summary = {
            'Model Performance Summary': {
                'Algorithm': best_model.get('name', 'Unknown'),
                'Problem Type': st.session_state.get('problem_type', 'Unknown'),
                'Primary Metric': best_model.get('accuracy' if st.session_state.get('problem_type') == 'classification' else 'r2_score', 'N/A'),
                'Cross-Validation Score': best_model.get('cv_score', 'N/A'),
                'Training Time': f"{best_model.get('training_time', 0):.2f}s",
                'Features Used': len(best_model.get('feature_names', [])),
                'Training Samples': results.get('data_info', {}).get('training_samples', 'N/A'),
            }
        }
        
        summary_text = ""
        for section, data in model_summary.items():
            summary_text += f"=== {section} ===\n"
            for key, value in data.items():
                summary_text += f"{key}: {value}\n"
            summary_text += "\n"
        
        st.download_button(
            label="📄 Download Model Summary (.txt)",
            data=summary_text,
            file_name=f"{model_name}_summary_{timestamp}.txt",
            mime="text/plain",
            use_container_width=True,
            help="Download a human-readable summary of model performance"
        )

def create_next_steps_section():
    """Create next steps and deployment options"""
    st.markdown("### 🚀 Next Steps & Deployment")
    
    col1, col2 = st.columns(2)
    
    with col1:
        create_info_card(
            "Deployment Options",
            """
            • <strong>Demo App:</strong> Create interactive testing interface<br>
            • <strong>API Integration:</strong> Use downloaded model in production<br>
            • <strong>Batch Predictions:</strong> Apply to new datasets<br>
            • <strong>Model Monitoring:</strong> Track performance over time
            """,
            "🚀"
        )
        
        if st.button("🎮 Create Interactive Demo", use_container_width=True, type="primary"):
            st.switch_page("pages/6_Demo_Generator.py")
    
    with col2:
        create_info_card(
            "Model Usage Guide",
            """
            <strong>To use your trained model:</strong><br>
            1. Load the .pkl file using pickle.load()<br>
            2. Access model via package['model']<br>
            3. Use model.predict(X) for new predictions<br>
            4. Refer to feature_names for input format
            """,
            "📘"
        )
        
        if st.button("🔄 Train New Model", use_container_width=True):
            # Clear session state and start over
            keys_to_keep = ['workflow_stage']  # Keep only essential keys
            for key in list(st.session_state.keys()):
                if key not in keys_to_keep:
                    del st.session_state[key]
            st.switch_page("app.py")

def main():
    """Main function for results dashboard"""
    initialize_results_state()
    
    # Check prerequisites
    if st.session_state.get('model_results') is None:
        st.warning("No model results found. Please complete the previous steps.")
        if st.button("← Go to Model Comparison"):
            st.switch_page("pages/4_Model_Comparison.py")
        st.stop()
    
    results = st.session_state.model_results
    best_model = results.get('best_model', {})
    problem_type = st.session_state.problem_type
    
    # Header
    st.title("🎯 Results Dashboard")
    st.markdown("Comprehensive analysis and deployment-ready model package.")
    
    # Progress tracker
    render_progress_tracker('results')
    
    # Executive summary
    create_executive_summary(results, best_model, problem_type)
    
    st.markdown("---")
    
    # Detailed performance analysis
    create_model_performance_deep_dive(results, best_model, problem_type)
    
    st.markdown("---")
    
    # Model interpretability
    create_model_interpretability_section(best_model, results)
    
    st.markdown("---")
    
    # Model download section
    create_model_download_section(best_model, results)
    
    st.markdown("---")
    
    # Next steps
    create_next_steps_section()
    
    # Navigation
    st.markdown("---")
    col1, col2 = st.columns([1, 1])
    
    with col1:
        if st.button("← Back to Model Comparison", use_container_width=True):
            st.switch_page("pages/4_Model_Comparison.py")
    
    with col2:
        if st.button("Create Demo App →", use_container_width=True, type="secondary"):
            st.switch_page("pages/6_Demo_Generator.py")
    
    # Sidebar
    sidebar_navigation()

if __name__ == "__main__":
    main()
