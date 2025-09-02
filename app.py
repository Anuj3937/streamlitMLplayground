import streamlit as st
import pandas as pd
from utils.ui_components import inject_custom_css, render_progress_tracker, sidebar_navigation, create_info_card
from utils.synthetic_data import generate_sample_datasets
from utils.error_handler import setup_logging
from utils.config import CONFIG

# Configure page
st.set_page_config(
    page_title="AI ML Teammate Platform",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Setup logging and inject CSS
setup_logging()
inject_custom_css()

def initialize_session_state():
    """Initialize all session state variables"""
    defaults = {
        'workflow_stage': 'welcome',
        'dataset': None,
        'target_column': None,
        'problem_type': None,
        'generated_code': "",
        'model_results': None,
        'best_model': None,
        'processing_options': {},
        'feature_engineering_applied': False,
        'data_validation_passed': False
    }
    
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

def create_welcome_hero():
    """Create the main welcome hero section"""
    st.markdown("""
    <div style="text-align: center; padding: 2rem 0;">
        <h1 style="font-size: 3.5rem; margin-bottom: 1rem; background: linear-gradient(135deg, var(--primary-color), var(--secondary-color)); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
            🤖 AI ML Teammate
        </h1>
        <h2 style="color: var(--text-secondary); font-weight: 400; margin-bottom: 2rem;">
            Your Intelligent Machine Learning Partner
        </h2>
    </div>
    """, unsafe_allow_html=True)

def create_feature_showcase():
    """Create feature showcase section"""
    st.markdown("### ✨ Platform Capabilities")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        create_info_card(
            "🔧 Intelligent Processing",
            """
            • Automatic data cleaning & validation<br>
            • Smart feature engineering<br>
            • Advanced outlier detection<br>
            • Missing data imputation
            """,
            "🔧"
        )
    
    with col2:
        create_info_card(
            "🚀 Advanced ML",
            """
            • 9+ algorithms with auto-tuning<br>
            • Ensemble methods<br>
            • Hyperparameter optimization<br>
            • Cross-validation & metrics
            """,
            "🚀"
        )
    
    with col3:
        create_info_card(
            "📊 Smart Analytics",
            """
            • Interactive visualizations<br>
            • Model interpretability (SHAP)<br>
            • Performance comparison<br>
            • Automated insights
            """,
            "📊"
        )

def create_workflow_overview():
    """Create workflow overview"""
    st.markdown("### 🔄 ML Workflow Process")
    
    render_progress_tracker('welcome')
    
    st.markdown("""
    <div style="margin: 2rem 0;">
        <h4>📋 Complete ML Pipeline:</h4>
        <ol style="font-size: 1.1rem; line-height: 1.8;">
            <li><strong>Data Input:</strong> Upload CSV, generate synthetic data, or search Kaggle datasets</li>
            <li><strong>Data Preview:</strong> Validate, clean, and engineer features automatically</li>
            <li><strong>Code Generation:</strong> AI generates optimized ML code with latest techniques</li>
            <li><strong>Model Training:</strong> Train multiple algorithms with hyperparameter tuning</li>
            <li><strong>Results Analysis:</strong> Compare models, analyze performance, and interpret results</li>
            <li><strong>Deployment:</strong> Generate interactive demo applications for testing</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)

def create_quick_start_section():
    """Create quick start with sample prompts"""
    st.markdown("### 🚀 Quick Start Templates")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🎯 Customer Churn Prediction", use_container_width=True, type="secondary"):
            st.session_state.sample_prompt = "customer_churn"
            st.switch_page("pages/1_Data_Input.py")
        
        st.markdown("""
        <small style="color: var(--text-secondary);">
        Predict customer behavior using demographic and usage data
        </small>
        """, unsafe_allow_html=True)
    
    with col2:
        if st.button("📧 Email Spam Detection", use_container_width=True, type="secondary"):
            st.session_state.sample_prompt = "spam_detection"
            st.switch_page("pages/1_Data_Input.py")
        
        st.markdown("""
        <small style="color: var(--text-secondary);">
        Classify emails as spam or legitimate using content features
        </small>
        """, unsafe_allow_html=True)
    
    with col3:
        if st.button("💰 House Price Prediction", use_container_width=True, type="secondary"):
            st.session_state.sample_prompt = "house_prices"
            st.switch_page("pages/1_Data_Input.py")
        
        st.markdown("""
        <small style="color: var(--text-secondary);">
        Predict real estate prices based on property characteristics
        </small>
        """, unsafe_allow_html=True)

def create_custom_project_section():
    """Create custom project section"""
    st.markdown("### 🛠️ Custom ML Project")
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown("""
        <div style="padding: 1.5rem; background: var(--background-color); border-radius: var(--border-radius); margin: 1rem 0;">
            <h4 style="margin-top: 0;">Ready for Your Own Data?</h4>
            <p style="color: var(--text-secondary); margin-bottom: 1.5rem;">
                Start with your own dataset and let our AI teammate guide you through 
                the complete machine learning pipeline with full customization options.
            </p>
            <ul style="color: var(--text-secondary);">
                <li>Support for CSV files up to 200MB</li>
                <li>Automatic data type detection</li>
                <li>Comprehensive data validation</li>
                <li>Advanced preprocessing options</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🚀 Start Custom Project", use_container_width=True, type="primary"):
            if 'sample_prompt' in st.session_state:
                del st.session_state.sample_prompt
            st.switch_page("pages/1_Data_Input.py")
        
        # Configuration options
        with st.expander("⚙️ Advanced Options"):
            st.selectbox("Default Problem Type", ["Auto-detect", "Classification", "Regression"])
            st.selectbox("Processing Strategy", ["Balanced", "Speed-optimized", "Accuracy-focused"])
            st.slider("Max Training Time (minutes)", 1, 30, 5)

def display_system_status():
    """Display system status and statistics"""
    with st.expander("📊 System Status"):
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Available Algorithms", "9+")
        with col2:
            st.metric("Max Dataset Size", "200MB")
        with col3:
            st.metric("Default Optuna Trials", CONFIG.optuna_trials)
        with col4:
            st.metric("Cross-Validation Folds", CONFIG.cv_folds)

def main():
    """Main application function"""
    
    # Initialize session state
    initialize_session_state()
    
    # Create welcome hero
    create_welcome_hero()
    
    # Feature showcase
    create_feature_showcase()
    
    # Workflow overview
    create_workflow_overview()
    
    st.markdown("---")
    
    # Quick start section
    create_quick_start_section()
    
    st.markdown("---")
    
    # Custom project section
    create_custom_project_section()
    
    # System status
    display_system_status()
    
    # Sidebar navigation
    sidebar_navigation()

if __name__ == "__main__":
    main()
