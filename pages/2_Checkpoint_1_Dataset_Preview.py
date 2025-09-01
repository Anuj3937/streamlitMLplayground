# pages/2_Checkpoint_1_Dataset_Preview.py

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from utils.data_processing import clean_and_process_data, perform_feature_engineering
from utils.gemini_integration import get_insights_with_gemini

st.set_page_config(page_title="Checkpoint 1: Dataset Preview", page_icon="📊", layout="wide")

st.title("📊 Checkpoint 1: Dataset Preview & Cleaning")

st.session_state.workflow_stage = 'checkpoint_1'

if st.session_state.get('dataset') is None:
    st.warning("No dataset found. Please go back to Data Input.")
    if st.button("← Go to Data Input"):
        st.switch_page("pages/1_Data_Input.py")
    st.stop()

df = st.session_state.dataset
target_column = st.session_state.target_column
problem_type = st.session_state.problem_type

# Dataset Overview
st.header("📋 Raw Dataset Overview")
st.dataframe(df.head(), use_container_width=True)

# Automated Feature Engineering (Optional)
with st.expander("🛠️ Automated Feature Engineering (Optional)"):
    st.markdown("""
    This step can automatically create new features from your existing data, which may improve model performance.
    - **Datetime Features**: Extracts month, year, day of week, etc. from date columns.
    - **Mathematical Combinations**: Creates new features by combining the most variant numerical columns (e.g., sum, product).
    """)
    if st.button("Run Automated Feature Engineering"):
        with st.spinner("Engineering new features..."):
            engineered_df, steps = perform_feature_engineering(df)
            st.session_state.dataset = engineered_df # Update the main dataframe
            df = engineered_df # Update local df variable
            st.success("Feature engineering complete!")
            for step in steps:
                st.info(f"✅ {step}")
            st.dataframe(df.head())
            st.rerun()


# Data Cleaning and Processing
st.header("🧹 Data Cleaning & Preprocessing")
with st.spinner("Analyzing and cleaning data..."):
    cleaned_df, preprocessing_steps = clean_and_process_data(df, target_column, problem_type)

st.subheader("Applied Preprocessing Steps")
for i, step in enumerate(preprocessing_steps, 1):
    st.markdown(f"{i}. {step}")
st.dataframe(cleaned_df.head(), use_container_width=True)

# Feature Analysis and Visualizations (keep this section as it was)
st.header("📈 Feature Analysis")
# ... (All the tabs for Target Distribution, Correlations, etc. remain the same) ...
tab1, tab2, tab3 = st.tabs(["Target Distribution", "Feature Correlations", "Feature Distributions"])
with tab1:
    st.subheader(f"Target Variable: {target_column}")
    if problem_type == "classification":
        fig = px.pie(cleaned_df, names=target_column, title="Target Class Distribution")
        st.plotly_chart(fig, use_container_width=True)
    else:
        fig = px.histogram(cleaned_df, x=target_column, title="Target Distribution", nbins=50)
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("Feature Correlation Matrix")
    numeric_cols = cleaned_df.select_dtypes(include=['float64', 'int64']).columns
    if len(numeric_cols) > 1:
        corr_matrix = cleaned_df[numeric_cols].corr()
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values, x=corr_matrix.columns, y=corr_matrix.columns,
            colorscale='RdBu', zmid=0
        ))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Not enough numeric features for correlation analysis.")
        
with tab3:
    st.subheader("Feature Distributions")
    feature_cols = [col for col in cleaned_df.columns if col != target_column]
    selected_feature = st.selectbox("Select a feature to visualize", feature_cols)
    if selected_feature:
        fig = px.histogram(cleaned_df, x=selected_feature, color=target_column, 
                           title=f"Distribution of {selected_feature} by {target_column}",
                           marginal="box")
        st.plotly_chart(fig, use_container_width=True)


# AI-Powered Data Insights
st.header("💡 AI-Powered Key Insights")
with st.spinner("💡 Generating AI-powered insights..."):
    insights = get_insights_with_gemini(cleaned_df, target_column)
for insight in insights:
    st.info(f"💡 {insight}")


# Approval Section
st.header("✅ Approve Dataset for Model Training")
if st.button("Approve & Continue →", type="primary", use_container_width=True):
    st.session_state.cleaned_dataset = cleaned_df
    st.switch_page("pages/3_Checkpoint_2_Code_Editor.py")