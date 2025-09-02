import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from utils.ui_components import (inject_custom_css, render_progress_tracker, sidebar_navigation, 
                                create_info_card, create_alert, create_comparison_table, display_feature_importance)
from utils.data_processing import processor
from utils.gemini_integration import get_insights_with_gemini
from utils.visualizations import create_correlation_heatmap, create_feature_distribution
from utils.data_validator import validator
from utils.error_handler import handle_errors
from utils.cache_manager import cache_manager

st.set_page_config(page_title="Dataset Preview", page_icon="📊", layout="wide")
inject_custom_css()

def initialize_preview_state():
    """Initialize dataset preview session state"""
    st.session_state.workflow_stage = 'checkpoint_1'
    
    if 'processing_applied' not in st.session_state:
        st.session_state.processing_applied = False
    
    if 'feature_engineering_applied' not in st.session_state:
        st.session_state.feature_engineering_applied = False

@handle_errors()
def display_dataset_overview(df, target_column):
    """Display comprehensive dataset overview"""
    st.markdown("### 📋 Dataset Overview")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Rows", f"{len(df):,}")
    with col2:
        st.metric("Features", f"{len(df.columns) - 1:,}")
    with col3:
        missing_pct = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
        st.metric("Missing Data", f"{missing_pct:.1f}%")
    with col4:
        memory_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
        st.metric("Memory Usage", f"{memory_mb:.1f} MB")
    
    # Data preview
    st.markdown("#### 👀 Data Sample")
    st.dataframe(df.head(10), use_container_width=True)
    
    # Data types summary
    with st.expander("📊 Column Analysis"):
        col1, col2 = st.columns(2)
        
        with col1:
            # Data types
            dtype_counts = df.dtypes.value_counts()
            st.markdown("**Data Types Distribution:**")
            st.dataframe(dtype_counts.to_frame('Count'), use_container_width=True)
        
        with col2:
            # Missing values by column
            missing_values = df.isnull().sum()
            missing_values = missing_values[missing_values > 0].sort_values(ascending=False)
            
            if len(missing_values) > 0:
                st.markdown("**Missing Values by Column:**")
                st.dataframe(missing_values.to_frame('Missing Count'), use_container_width=True)
            else:
                st.success("✅ No missing values found!")

@handle_errors()
def create_data_processing_options():
    """Create advanced data processing options"""
    st.markdown("### ⚙️ Data Processing Configuration")
    
    with st.expander("🛠️ Advanced Processing Options", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Missing Data Handling:**")
            missing_strategy = st.selectbox(
                "Strategy",
                ["auto", "median", "knn"],
                help="Auto: median for numeric, mode for categorical"
            )
            
            st.markdown("**Outlier Detection:**")
            outlier_method = st.selectbox(
                "Method",
                ["iqr", "zscore", "winsorize"],
                help="IQR: Interquartile Range, Z-score: Standard deviation"
            )
        
        with col2:
            st.markdown("**Feature Scaling:**")
            scaling_method = st.selectbox(
                "Method",
                ["standard", "robust"],
                help="Standard: mean=0, std=1; Robust: less sensitive to outliers"
            )
            
            st.markdown("**Categorical Encoding:**")
            encoding_method = st.selectbox(
                "Method",
                ["auto", "label", "onehot"],
                help="Auto: label encoding for high cardinality, one-hot for low"
            )
    
    processing_options = {
        'missing_strategy': missing_strategy,
        'outlier_method': outlier_method,
        'scaling_method': scaling_method,
        'encoding_method': encoding_method
    }
    
    return processing_options

@handle_errors()
def apply_data_cleaning(df, target_column, problem_type, processing_options):
    """Apply data cleaning with progress tracking"""
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        status_text.text("🧹 Starting data cleaning...")
        progress_bar.progress(20)
        
        # Apply cleaning
        cleaned_df, preprocessing_steps = processor.clean_and_process_data(
            df, target_column, problem_type, processing_options
        )
        
        progress_bar.progress(80)
        status_text.text("✅ Data cleaning completed!")
        
        progress_bar.progress(100)
        
        # Display results
        st.success(f"✅ Data cleaning completed! Applied {len(preprocessing_steps)} preprocessing steps.")
        
        # Show preprocessing steps
        with st.expander("📋 Applied Preprocessing Steps", expanded=True):
            for i, step in enumerate(preprocessing_steps, 1):
                st.markdown(f"**{i}.** {step}")
        
        return cleaned_df, preprocessing_steps
        
    except Exception as e:
        progress_bar.empty()
        status_text.empty()
        st.error(f"❌ Error during data cleaning: {str(e)}")
        return None, []

@handle_errors()
def create_feature_engineering_section(df, target_column):
    """Create feature engineering section"""
    st.markdown("### 🔧 Automated Feature Engineering")
    
    create_info_card(
        "Feature Engineering Options",
        """
        • <strong>Datetime Features:</strong> Extract month, year, day of week from date columns<br>
        • <strong>Mathematical Combinations:</strong> Create sum, mean, std of top variant features<br>
        • <strong>Polynomial Features:</strong> Create interaction terms between features<br>
        • <strong>Smart Selection:</strong> Only applies relevant techniques to your data
        """,
        "🔧"
    )
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        **Benefits:**
        - Can improve model performance by 5-15%
        - Automatically selects relevant transformations
        - Creates interpretable new features
        """)
    
    with col2:
        if st.button("🚀 Apply Feature Engineering", use_container_width=True, type="secondary"):
            with st.spinner("🔧 Engineering new features..."):
                engineered_df, steps = processor.perform_feature_engineering(df, target_column)
                
                if steps:
                    st.session_state.dataset = engineered_df
                    st.session_state.feature_engineering_applied = True
                    
                    st.success("✅ Feature engineering completed!")
                    for step in steps:
                        st.info(f"🔧 {step}")
                    
                    st.rerun()
                else:
                    st.warning("ℹ️ No applicable feature engineering transformations found for this dataset.")

@handle_errors()
def create_visualization_tabs(df, target_column, problem_type):
    """Create comprehensive data visualization tabs"""
    st.markdown("### 📈 Data Exploration & Insights")
    
    tab1, tab2, tab3, tab4 = st.tabs(["🎯 Target Analysis", "📊 Feature Correlations", "📈 Distributions", "🔍 Data Quality"])
    
    with tab1:
        st.markdown("#### Target Variable Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Target statistics
            st.markdown("**Target Statistics:**")
            if problem_type == "classification":
                value_counts = df[target_column].value_counts()
                st.dataframe(value_counts.to_frame('Count'), use_container_width=True)
            else:
                target_stats = df[target_column].describe()
                st.dataframe(target_stats.to_frame('Value'), use_container_width=True)
        
        with col2:
            # Target visualization
            if problem_type == "classification":
                fig = px.pie(
                    values=df[target_column].value_counts().values,
                    names=df[target_column].value_counts().index,
                    title="Target Class Distribution"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                fig = px.histogram(
                    df, x=target_column, 
                    title="Target Distribution",
                    nbins=50
                )
                st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.markdown("#### Feature Correlation Analysis")
        
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        if len(numeric_cols) > 1:
            fig = create_correlation_heatmap(df[numeric_cols], "Feature Correlation Matrix")
            if fig:
                st.plotly_chart(fig, use_container_width=True)
                
                # Correlation insights
                corr_matrix = df[numeric_cols].corr()
                high_corr_pairs = []
                
                for i in range(len(corr_matrix.columns)):
                    for j in range(i+1, len(corr_matrix.columns)):
                        corr_val = abs(corr_matrix.iloc[i, j])
                        if corr_val > 0.7:
                            high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_val))
                
                if high_corr_pairs:
                    st.warning("⚠️ **High Correlations Detected:**")
                    for col1_name, col2_name, corr_val in high_corr_pairs[:5]:
                        st.write(f"• {col1_name} ↔ {col2_name}: {corr_val:.3f}")
            else:
                st.info("Need at least 2 numeric columns for correlation analysis.")
        else:
            st.info("Not enough numeric features for correlation analysis.")
    
    with tab3:
        st.markdown("#### Feature Distributions")
        
        feature_cols = [col for col in df.columns if col != target_column]
        
        if feature_cols:
            selected_feature = st.selectbox("Select feature to analyze:", feature_cols)
            
            if selected_feature:
                fig = create_feature_distribution(df, selected_feature, target_column, problem_type)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                
                # Feature statistics
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"**{selected_feature} Statistics:**")
                    if pd.api.types.is_numeric_dtype(df[selected_feature]):
                        stats = df[selected_feature].describe()
                        st.dataframe(stats.to_frame('Value'))
                    else:
                        value_counts = df[selected_feature].value_counts().head(10)
                        st.dataframe(value_counts.to_frame('Count'))
                
                with col2:
                    st.markdown("**Quality Metrics:**")
                    missing = df[selected_feature].isnull().sum()
                    unique = df[selected_feature].nunique()
                    
                    st.metric("Missing Values", missing)
                    st.metric("Unique Values", unique)
                    st.metric("Data Type", str(df[selected_feature].dtype))
    
    with tab4:
        st.markdown("#### Data Quality Assessment")
        
        # Missing data heatmap
        if df.isnull().sum().sum() > 0:
            missing_data = df.isnull().sum().sort_values(ascending=False)
            missing_data = missing_data[missing_data > 0]
            
            fig = px.bar(
                x=missing_data.values,
                y=missing_data.index,
                orientation='h',
                title="Missing Values by Column"
            )
            fig.update_layout(height=max(400, len(missing_data) * 30))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.success("✅ No missing values detected!")
        
        # Data quality summary
        quality_metrics = {
            'Total Missing Values': df.isnull().sum().sum(),
            'Duplicate Rows': df.duplicated().sum(),
            'Constant Columns': sum(df.nunique() <= 1),
            'High Cardinality Categorical': sum(
                (df.select_dtypes(include=['object']).nunique() > 50).values
            )
        }
        
        col1, col2 = st.columns(2)
        for i, (metric, value) in enumerate(quality_metrics.items()):
            with col1 if i % 2 == 0 else col2:
                color = "normal" if value == 0 else "warning"
                st.metric(metric, value)

@handle_errors()
def create_ai_insights_section(df, target_column):
    """Create AI-powered insights section"""
    st.markdown("### 💡 AI-Powered Dataset Insights")
    
    with st.spinner("🧠 AI is analyzing your dataset..."):
        try:
            insights = get_insights_with_gemini(df, target_column)
            
            if insights:
                for i, insight in enumerate(insights):
                    if insight.strip():  # Skip empty insights
                        create_alert(f"💡 {insight}", "info")
            else:
                st.warning("Could not generate AI insights at this time.")
                
        except Exception as e:
            st.warning(f"AI insights unavailable: {str(e)}")

def main():
    """Main function for dataset preview page"""
    initialize_preview_state()
    
    # Check if dataset exists
    if st.session_state.get('dataset') is None:
        st.warning("No dataset found. Please go back to Data Input.")
        if st.button("← Go to Data Input"):
            st.switch_page("pages/1_Data_Input.py")
        st.stop()
    
    df = st.session_state.dataset
    target_column = st.session_state.target_column
    problem_type = st.session_state.problem_type
    
    # Header
    st.title("📊 Dataset Preview & Processing")
    st.markdown("Review your data, apply preprocessing, and prepare for model training.")
    
    # Progress tracker
    render_progress_tracker('checkpoint_1')
    
    # Dataset overview
    display_dataset_overview(df, target_column)
    
    st.markdown("---")
    
    # Feature engineering section
    if not st.session_state.feature_engineering_applied:
        create_feature_engineering_section(df, target_column)
        st.markdown("---")
    else:
        create_alert("✅ Feature engineering has been applied to this dataset", "success")
    
    # Data processing options
    processing_options = create_data_processing_options()
    
    # Apply data cleaning
    if st.button("🧹 Clean & Process Data", type="primary", use_container_width=True):
        cleaned_df, preprocessing_steps = apply_data_cleaning(df, target_column, problem_type, processing_options)
        
        if cleaned_df is not None:
            st.session_state.cleaned_dataset = cleaned_df
            st.session_state.preprocessing_steps = preprocessing_steps
            st.session_state.processing_applied = True
            st.rerun()
    
    # Show results if processing applied
    if st.session_state.get('processing_applied', False) and st.session_state.get('cleaned_dataset') is not None:
        st.markdown("---")
        st.markdown("### ✅ Processed Dataset")
        
        cleaned_df = st.session_state.cleaned_dataset
        
        # Show before/after comparison
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Original Rows", len(df))
            st.metric("Original Columns", len(df.columns))
        
        with col2:
            st.metric("Processed Rows", len(cleaned_df))
            st.metric("Processed Columns", len(cleaned_df.columns))
        
        # Visualization tabs
        create_visualization_tabs(cleaned_df, target_column, problem_type)
        
        # AI insights
        create_ai_insights_section(cleaned_df, target_column)
        
        # Final approval
        st.markdown("---")
        st.markdown("### ✅ Ready for Model Training")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            create_alert(
                f"Dataset is ready for training with <strong>{len(cleaned_df):,} samples</strong> and <strong>{len(cleaned_df.columns)-1} features</strong>",
                "success"
            )
        
        with col2:
            if st.button("Continue to Code Editor →", type="primary", use_container_width=True):
                st.success("🎉 Moving to code generation phase!")
                st.switch_page("pages/3_Code_Editor.py")
    
    # Navigation
    st.markdown("---")
    if st.button("← Back to Data Input"):
        st.switch_page("pages/1_Data_Input.py")
    
    # Sidebar
    sidebar_navigation()

if __name__ == "__main__":
    main()
