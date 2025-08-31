import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from utils.data_processing import clean_and_process_data, get_data_insights
from utils.visualizations import create_correlation_heatmap, create_feature_distribution

st.set_page_config(page_title="Checkpoint 1: Dataset Preview", page_icon="📊", layout="wide")

st.title("📊 Checkpoint 1: Dataset Preview")
st.markdown("Review your dataset, examine features, and approve data preprocessing.")

# Update workflow stage
st.session_state.workflow_stage = 'checkpoint_1'

# Check if dataset exists
if st.session_state.dataset is None:
    st.warning("No dataset found. Please go back to Data Input.")
    if st.button("← Go to Data Input"):
        st.switch_page("pages/1_Data_Input.py")
    st.stop()

df = st.session_state.dataset
target_column = st.session_state.target_column
problem_type = st.session_state.problem_type

# Dataset Overview
st.header("📋 Dataset Overview")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Rows", len(df))
with col2:
    st.metric("Total Columns", len(df.columns))
with col3:
    st.metric("Target Column", target_column)
with col4:
    st.metric("Problem Type", problem_type.title())

# Raw Dataset Preview
st.subheader("Raw Dataset Preview")
st.dataframe(df.head(10), use_container_width=True)

# Data Quality Analysis
st.header("🔍 Data Quality Analysis")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Missing Values")
    missing_data = df.isnull().sum()
    missing_df = pd.DataFrame({
        'Column': missing_data.index,
        'Missing Count': missing_data.values,
        'Missing %': (missing_data.values / len(df) * 100).round(2)
    })
    missing_df = missing_df[missing_df['Missing Count'] > 0]
    
    if len(missing_df) > 0:
        st.dataframe(missing_df, use_container_width=True)
        st.warning(f"Found missing values in {len(missing_df)} columns")
    else:
        st.success("✅ No missing values found!")

with col2:
    st.subheader("Data Types")
    dtype_df = pd.DataFrame({
        'Column': df.dtypes.index,
        'Data Type': df.dtypes.values.astype(str),
        'Unique Values': [df[col].nunique() for col in df.columns]
    })
    st.dataframe(dtype_df, use_container_width=True)

# Data Processing and Cleaning
st.header("🧹 Data Cleaning & Preprocessing")

with st.spinner("Analyzing and cleaning data..."):
    # Clean and process the data
    cleaned_df, preprocessing_steps = clean_and_process_data(df, target_column, problem_type)
    
    # Get data insights
    insights = get_data_insights(cleaned_df, target_column, problem_type)

# Display preprocessing steps
st.subheader("Applied Preprocessing Steps")
for i, step in enumerate(preprocessing_steps, 1):
    st.markdown(f"{i}. {step}")

# Cleaned Dataset Preview
st.subheader("Cleaned Dataset Preview")
st.dataframe(cleaned_df.head(10), use_container_width=True)

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Rows After Cleaning", len(cleaned_df))
with col2:
    st.metric("Features for Training", len(cleaned_df.columns) - 1)
with col3:
    improvement = len(cleaned_df) / len(df) * 100
    st.metric("Data Retention", f"{improvement:.1f}%")

# Feature Analysis
st.header("📈 Feature Analysis")

tab1, tab2, tab3 = st.tabs(["Target Distribution", "Feature Correlations", "Feature Distributions"])

with tab1:
    st.subheader(f"Target Variable: {target_column}")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if problem_type == "classification":
            target_counts = cleaned_df[target_column].value_counts()
            fig = px.bar(x=target_counts.index, y=target_counts.values,
                        title="Target Class Distribution",
                        labels={'x': target_column, 'y': 'Count'})
            st.plotly_chart(fig, use_container_width=True)
        else:
            fig = px.histogram(cleaned_df, x=target_column,
                             title="Target Distribution",
                             nbins=30)
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Target Statistics")
        target_stats = cleaned_df[target_column].describe()
        st.dataframe(target_stats, use_container_width=True)
        
        if problem_type == "classification":
            st.subheader("Class Balance")
            class_balance = cleaned_df[target_column].value_counts(normalize=True) * 100
            for class_name, percentage in class_balance.items():
                st.metric(f"Class {class_name}", f"{percentage:.1f}%")

with tab2:
    st.subheader("Feature Correlation Matrix")
    
    # Select only numeric columns for correlation
    numeric_cols = cleaned_df.select_dtypes(include=['float64', 'int64']).columns
    if len(numeric_cols) > 1:
        corr_matrix = cleaned_df[numeric_cols].corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0
        ))
        fig.update_layout(title="Feature Correlation Heatmap")
        st.plotly_chart(fig, use_container_width=True)
        
        # High correlation warnings
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.8:
                    high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_val))
        
        if high_corr_pairs:
            st.warning("⚠️ High correlations detected:")
            for col1, col2, corr_val in high_corr_pairs:
                st.markdown(f"- **{col1}** ↔ **{col2}**: {corr_val:.3f}")
    else:
        st.info("Not enough numeric features for correlation analysis.")

with tab3:
    st.subheader("Feature Distributions")
    
    # Select features to display
    feature_cols = [col for col in cleaned_df.columns if col != target_column]
    selected_features = st.multiselect("Select features to visualize", feature_cols, default=feature_cols[:4])
    
    if selected_features:
        for feature in selected_features[:6]:  # Limit to 6 features
            col1, col2 = st.columns(2)
            
            with col1:
                if cleaned_df[feature].dtype in ['float64', 'int64']:
                    fig = px.histogram(cleaned_df, x=feature, title=f"Distribution: {feature}")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    value_counts = cleaned_df[feature].value_counts().head(10)
                    fig = px.bar(x=value_counts.index, y=value_counts.values, title=f"Top Values: {feature}")
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader(f"{feature} Statistics")
                if cleaned_df[feature].dtype in ['float64', 'int64']:
                    stats = cleaned_df[feature].describe()
                    st.dataframe(stats, use_container_width=True)
                else:
                    stats = pd.DataFrame({
                        'Unique Values': [cleaned_df[feature].nunique()],
                        'Most Frequent': [cleaned_df[feature].mode().iloc[0] if len(cleaned_df[feature].mode()) > 0 else 'N/A'],
                        'Frequency': [cleaned_df[feature].value_counts().iloc[0] if len(cleaned_df[feature].value_counts()) > 0 else 0]
                    })
                    st.dataframe(stats, use_container_width=True)

# Data Insights
st.header("💡 Key Insights")
for insight in insights:
    st.info(f"💡 {insight}")

# Approval Section
st.header("✅ Approve Dataset for Model Training")

col1, col2 = st.columns([3, 1])

with col1:
    st.markdown("""
    **Review Summary:**
    - Dataset has been cleaned and preprocessed
    - Missing values have been handled appropriately
    - Features are properly encoded for machine learning
    - Data quality checks have been completed
    
    Proceed to the next checkpoint to review and edit the generated ML code.
    """)

with col2:
    if st.button("Approve & Continue →", type="primary", use_container_width=True):
        st.session_state.cleaned_dataset = cleaned_df
        st.session_state.preprocessing_steps = preprocessing_steps
        st.session_state.data_insights = insights
        st.session_state.workflow_stage = 'checkpoint_2'
        st.success("Dataset approved!")
        st.switch_page("pages/3_Checkpoint_2_Code_Editor.py")

# Navigation
st.markdown("---")
col1, col2 = st.columns([1, 1])

with col1:
    if st.button("← Back to Data Input", use_container_width=True):
        st.switch_page("pages/1_Data_Input.py")

with col2:
    if st.session_state.get('cleaned_dataset') is not None:
        if st.button("Next: Code Editor →", use_container_width=True, type="primary"):
            st.switch_page("pages/3_Checkpoint_2_Code_Editor.py")
