import streamlit as st
import pandas as pd
from utils.synthetic_data import generate_sample_datasets

# Configure page
st.set_page_config(
    page_title="AI ML Teammate",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'workflow_stage' not in st.session_state:
    st.session_state.workflow_stage = 'welcome'
if 'dataset' not in st.session_state:
    st.session_state.dataset = None
if 'target_column' not in st.session_state:
    st.session_state.target_column = None
if 'problem_type' not in st.session_state:
    st.session_state.problem_type = None
if 'generated_code' not in st.session_state:
    st.session_state.generated_code = ""
if 'model_results' not in st.session_state:
    st.session_state.model_results = None
if 'best_model' not in st.session_state:
    st.session_state.best_model = None

def main():
    st.title("🤖 AI ML Teammate")
    st.subheader("Your Intelligent Machine Learning Partner")
    
    # Welcome Section
    st.markdown("""
    ### Welcome to Your AI-Powered ML Platform
    
    Transform your machine learning workflow from complex manual processes into a simple, guided conversation. 
    Our intelligent platform acts as your senior AI teammate, automating the entire process of building 
    high-quality machine learning models.
    
    #### What This Platform Does:
    - **Intelligent Data Processing**: Automatically clean and prepare your data
    - **Smart Algorithm Selection**: Choose the best ML algorithms for your specific problem
    - **Automated Optimization**: Improve model performance through hyperparameter tuning
    - **Transparent Pipeline**: See exactly what's happening at each step
    - **Interactive Results**: Compare models and visualize performance metrics
    - **Instant Deployment**: Generate demo apps to test your models
    
    #### How It Works:
    1. **Provide Your Data** - Upload CSV, generate synthetic data, or find datasets
    2. **Review & Clean** - Preview and validate your processed dataset
    3. **Edit & Approve Code** - Review and modify the generated ML pipeline
    4. **Compare Models** - Analyze performance across multiple algorithms
    5. **Deploy & Test** - Create interactive demo applications
    """)
    
    st.markdown("---")
    
    # Sample Prompts Section
    st.markdown("### 🚀 Quick Start with Sample Prompts")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🎯 Predict Customer Churn", use_container_width=True):
            st.session_state.sample_prompt = "customer_churn"
            st.switch_page("pages/1_Data_Input.py")
    
    with col2:
        if st.button("📧 Email Spam Detection", use_container_width=True):
            st.session_state.sample_prompt = "spam_detection"
            st.switch_page("pages/1_Data_Input.py")
    
    with col3:
        if st.button("💰 House Price Prediction", use_container_width=True):
            st.session_state.sample_prompt = "house_prices"
            st.switch_page("pages/1_Data_Input.py")
    
    st.markdown("---")
    
    # Custom Project Section
    st.markdown("### 🛠️ Start Your Custom Project")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown("""
        Ready to work with your own data or custom requirements? 
        Click below to begin the guided ML workflow with full control over your project parameters.
        """)
    
    with col2:
        if st.button("Start Custom Project →", use_container_width=True, type="primary"):
            if 'sample_prompt' in st.session_state:
                del st.session_state.sample_prompt
            st.switch_page("pages/1_Data_Input.py")
    
    # Workflow Progress Sidebar
    with st.sidebar:
        st.markdown("### 📋 Workflow Progress")
        
        progress_stages = [
            ("Welcome", "welcome"),
            ("Data Input", "data_input"),
            ("Checkpoint 1: Dataset Preview", "checkpoint_1"),
            ("Checkpoint 2: Code Editor", "checkpoint_2"),
            ("Checkpoint 3: Model Comparison", "checkpoint_3"),
            ("Results Dashboard", "results"),
            ("Demo App Generator", "demo_app")
        ]
        
        current_stage = st.session_state.get('workflow_stage', 'welcome')
        
        for i, (stage_name, stage_key) in enumerate(progress_stages):
            if stage_key == current_stage:
                st.markdown(f"**→ {stage_name}** ✅")
            elif i < [s[1] for s in progress_stages].index(current_stage):
                st.markdown(f"✅ {stage_name}")
            else:
                st.markdown(f"⏳ {stage_name}")
        
        st.markdown("---")
        
        # Quick Navigation
        st.markdown("### 🧭 Quick Navigation")
        if st.session_state.dataset is not None:
            if st.button("📊 Dataset Preview", use_container_width=True):
                st.switch_page("pages/2_Checkpoint_1_Dataset_Preview.py")
        
        if st.session_state.generated_code:
            if st.button("💻 Code Editor", use_container_width=True):
                st.switch_page("pages/3_Checkpoint_2_Code_Editor.py")
        
        if st.session_state.model_results is not None:
            if st.button("📈 Model Results", use_container_width=True):
                st.switch_page("pages/4_Checkpoint_3_Model_Comparison.py")

if __name__ == "__main__":
    main()
