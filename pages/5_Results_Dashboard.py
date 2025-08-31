import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pickle
import io
import base64

st.set_page_config(page_title="Results Dashboard", page_icon="🎯", layout="wide")

st.title("🎯 Results Dashboard")
st.markdown("Comprehensive analysis and download of your machine learning model.")

# Update workflow stage
st.session_state.workflow_stage = 'results'

# Check if model results exist
if st.session_state.get('model_results') is None or st.session_state.get('best_model') is None:
    st.warning("No model results found. Please complete the previous checkpoints.")
    if st.button("← Go to Model Comparison"):
        st.switch_page("pages/4_Checkpoint_3_Model_Comparison.py")
    st.stop()

results = st.session_state.model_results
best_model = st.session_state.best_model
problem_type = st.session_state.problem_type
target_column = st.session_state.target_column

# Executive Summary
st.header("📋 Executive Summary")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Final Model", best_model['name'])

with col2:
    if problem_type == "classification":
        main_metric = best_model.get('accuracy', 0)
        st.metric("Accuracy", f"{main_metric:.1%}")
    else:
        main_metric = best_model.get('r2_score', 0)
        st.metric("R² Score", f"{main_metric:.3f}")

with col3:
    dataset_size = len(st.session_state.get('cleaned_dataset', []))
    st.metric("Training Samples", f"{dataset_size:,}")

with col4:
    feature_count = len(st.session_state.get('cleaned_dataset', pd.DataFrame()).columns) - 1
    st.metric("Features Used", feature_count)

# Performance Summary
st.subheader("🎯 Performance Summary")

performance_summary = f"""
**Model Performance Analysis:**

Your **{best_model['name']}** model has been successfully trained and optimized for {problem_type} on the **{target_column}** target variable.

"""

if problem_type == "classification":
    accuracy = best_model.get('accuracy', 0)
    precision = best_model.get('precision', 0)
    recall = best_model.get('recall', 0)
    f1 = best_model.get('f1_score', 0)
    
    performance_summary += f"""
**Key Metrics:**
- **Accuracy:** {accuracy:.1%} - Overall correctness of predictions
- **Precision:** {precision:.1%} - Reliability of positive predictions  
- **Recall:** {recall:.1%} - Ability to find all positive cases
- **F1-Score:** {f1:.3f} - Balanced measure of precision and recall
"""
    
    if accuracy >= 0.9:
        performance_summary += "\n🟢 **Excellent Performance** - Model shows outstanding accuracy"
    elif accuracy >= 0.8:
        performance_summary += "\n🟡 **Good Performance** - Model performs well for most cases"
    elif accuracy >= 0.7:
        performance_summary += "\n🟠 **Fair Performance** - Model may need additional optimization"
    else:
        performance_summary += "\n🔴 **Needs Improvement** - Consider feature engineering or more data"

else:  # Regression
    r2 = best_model.get('r2_score', 0)
    mse = best_model.get('mse', 0)
    mae = best_model.get('mae', 0)
    
    performance_summary += f"""
**Key Metrics:**
- **R² Score:** {r2:.3f} - Percentage of variance explained by the model
- **Mean Squared Error:** {mse:.3f} - Average squared prediction errors
- **Mean Absolute Error:** {mae:.3f} - Average absolute prediction errors
"""
    
    if r2 >= 0.8:
        performance_summary += "\n🟢 **Excellent Performance** - Model explains most data variance"
    elif r2 >= 0.6:
        performance_summary += "\n🟡 **Good Performance** - Model captures key patterns well"
    elif r2 >= 0.4:
        performance_summary += "\n🟠 **Fair Performance** - Model shows moderate predictive power"
    else:
        performance_summary += "\n🔴 **Needs Improvement** - Consider feature engineering or different approaches"

st.markdown(performance_summary)

# Detailed Results
st.header("📊 Detailed Analysis")

tab1, tab2, tab3, tab4 = st.tabs(["Model Comparison", "Feature Analysis", "Performance Metrics", "Model Details"])

with tab1:
    st.subheader("All Models Comparison")
    
    if 'model_comparison' in results:
        comparison_df = pd.DataFrame(results['model_comparison'])
        
        # Format numerical columns
        for col in comparison_df.columns:
            if comparison_df[col].dtype == 'float64':
                comparison_df[col] = comparison_df[col].round(4)
        
        st.dataframe(comparison_df, use_container_width=True, hide_index=True)
        
        # Performance ranking
        st.subheader("Performance Ranking")
        
        if problem_type == "classification":
            ranking_metric = 'accuracy'
        else:
            ranking_metric = 'r2_score'
        
        if ranking_metric in comparison_df.columns:
            ranked_df = comparison_df.sort_values(ranking_metric, ascending=False).reset_index(drop=True)
            ranked_df.index += 1
            
            fig = px.bar(ranked_df, 
                        x='model', 
                        y=ranking_metric,
                        title=f"Model Ranking by {ranking_metric.replace('_', ' ').title()}",
                        color=ranking_metric,
                        color_continuous_scale='viridis')
            
            st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("Feature Importance & Analysis")
    
    if 'feature_importance' in best_model and best_model['feature_importance'] is not None:
        importance_data = best_model['feature_importance']
        importance_df = pd.DataFrame({
            'Feature': importance_data['features'],
            'Importance': importance_data['importance']
        }).sort_values('Importance', ascending=False)
        
        # Top features chart
        top_features = importance_df.head(10)
        fig = px.bar(top_features, 
                    x='Importance', 
                    y='Feature',
                    orientation='h',
                    title="Top 10 Most Important Features",
                    color='Importance',
                    color_continuous_scale='blues')
        fig.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig, use_container_width=True)
        
        # Feature importance table
        st.subheader("Complete Feature Importance")
        st.dataframe(importance_df, use_container_width=True, hide_index=True)
        
        # Feature insights
        st.subheader("🔍 Feature Insights")
        
        total_importance = importance_df['Importance'].sum()
        top_5_importance = importance_df.head(5)['Importance'].sum()
        contribution_pct = (top_5_importance / total_importance * 100)
        
        st.info(f"💡 The top 5 features contribute **{contribution_pct:.1f}%** of the total predictive power")
        
        low_importance_features = importance_df[importance_df['Importance'] < 0.01]
        if len(low_importance_features) > 0:
            st.warning(f"⚠️ {len(low_importance_features)} features have very low importance and could be removed for model simplification")
    else:
        st.info("Feature importance analysis is not available for this model type.")

with tab3:
    st.subheader("Comprehensive Performance Metrics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Primary Metrics**")
        
        if problem_type == "classification":
            metrics = [
                ("Accuracy", best_model.get('accuracy', 0)),
                ("Precision", best_model.get('precision', 0)),
                ("Recall", best_model.get('recall', 0)),
                ("F1-Score", best_model.get('f1_score', 0))
            ]
        else:
            metrics = [
                ("R² Score", best_model.get('r2_score', 0)),
                ("MSE", best_model.get('mse', 0)),
                ("MAE", best_model.get('mae', 0)),
                ("RMSE", best_model.get('rmse', 0))
            ]
        
        for name, value in metrics:
            if value is not None:
                st.metric(name, f"{value:.4f}")
    
    with col2:
        st.markdown("**Model Configuration**")
        if 'best_params' in best_model and best_model['best_params']:
            params_df = pd.DataFrame(list(best_model['best_params'].items()), 
                                   columns=['Parameter', 'Value'])
            st.dataframe(params_df, use_container_width=True, hide_index=True)
        else:
            st.info("No hyperparameters to display")
    
    # Performance visualization
    if problem_type == "classification" and 'confusion_matrix' in best_model:
        st.subheader("Confusion Matrix")
        cm = best_model['confusion_matrix']
        
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
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.subheader("Model Technical Details")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Training Information**")
        st.markdown(f"- **Algorithm:** {best_model['name']}")
        st.markdown(f"- **Problem Type:** {problem_type.title()}")
        st.markdown(f"- **Target Variable:** {target_column}")
        
        if 'training_time' in best_model:
            st.markdown(f"- **Training Time:** {best_model['training_time']:.2f} seconds")
        
        if 'cv_scores' in best_model:
            cv_mean = np.mean(best_model['cv_scores'])
            cv_std = np.std(best_model['cv_scores'])
            st.markdown(f"- **CV Score:** {cv_mean:.3f} ± {cv_std:.3f}")
    
    with col2:
        st.markdown("**Data Information**")
        if 'data_info' in results:
            data_info = results['data_info']
            for key, value in data_info.items():
                st.markdown(f"- **{key.replace('_', ' ').title()}:** {value}")

# Model Download
st.header("💾 Download Your Model")

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("""
    **Download your trained model for deployment:**
    
    The model file includes:
    - Trained algorithm with optimized parameters
    - Preprocessing pipeline
    - Feature encoders and scalers
    - Model metadata and performance metrics
    
    Use this file to make predictions on new data in your own applications.
    """)

with col2:
    # Create downloadable model file
    if st.button("📥 Download Model (.pkl)", type="primary", use_container_width=True):
        try:
            # Prepare model package
            model_package = {
                'model': best_model.get('model_object'),
                'model_name': best_model['name'],
                'problem_type': problem_type,
                'target_column': target_column,
                'performance_metrics': {
                    key: value for key, value in best_model.items() 
                    if key in ['accuracy', 'precision', 'recall', 'f1_score', 'r2_score', 'mse', 'mae', 'rmse']
                },
                'feature_names': best_model.get('feature_names', []),
                'preprocessing_steps': st.session_state.get('preprocessing_steps', [])
            }
            
            # Serialize model
            model_bytes = pickle.dumps(model_package)
            
            # Create download
            st.download_button(
                label="📥 Download Model File",
                data=model_bytes,
                file_name=f"{best_model['name'].lower().replace(' ', '_')}_model.pkl",
                mime="application/octet-stream",
                use_container_width=True
            )
            
            st.success("✅ Model ready for download!")
            
        except Exception as e:
            st.error(f"Error preparing model download: {str(e)}")

# Next Steps
st.header("🚀 Next Steps")

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("🎮 Create Demo App", use_container_width=True, type="primary"):
        st.switch_page("pages/6_Demo_App_Generator.py")

with col2:
    if st.button("🔄 Start New Project", use_container_width=True):
        # Clear session state
        for key in list(st.session_state.keys()):
            if key != 'workflow_stage':
                del st.session_state[key]
        st.session_state.workflow_stage = 'welcome'
        st.switch_page("app.py")

with col3:
    if st.button("📊 Model Comparison", use_container_width=True):
        st.switch_page("pages/4_Checkpoint_3_Model_Comparison.py")

# Project Summary
st.markdown("---")
st.markdown("### 🎉 Project Completed Successfully!")
st.success(f"""
**Your {problem_type} model is ready for deployment!**

✅ Data processed and cleaned  
✅ {len(results.get('model_comparison', []))} algorithms tested and compared  
✅ Best model selected and optimized  
✅ Performance metrics calculated  
✅ Model package prepared for download  

**Main Achievement:** {main_metric:.1%} accuracy on {target_column} prediction
""")
