import streamlit as st
import pandas as pd
import io
from utils.synthetic_data import generate_sample_datasets, create_synthetic_dataset
from utils.data_processing import clean_and_process_data

st.set_page_config(page_title="Data Input", page_icon="📁", layout="wide")

st.title("📁 Data Input")
st.markdown("Choose how you want to provide data for your machine learning project.")

# Update workflow stage
st.session_state.workflow_stage = 'data_input'

# Handle sample prompts
sample_prompt = st.session_state.get('sample_prompt', None)
if sample_prompt:
    st.info(f"🎯 Quick Start: {sample_prompt.replace('_', ' ').title()}")

# Three modes of data input
tab1, tab2, tab3 = st.tabs(["📤 Upload Dataset", "🎲 Generate Synthetic Data", "🔍 Find Dataset"])

with tab1:
    st.header("Upload Your Dataset")
    st.markdown("Upload a CSV file containing your dataset.")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            # Read the uploaded file
            df = pd.read_csv(uploaded_file)
            st.success(f"Successfully loaded dataset with {len(df)} rows and {len(df.columns)} columns")
            
            # Display basic info
            st.subheader("Dataset Preview")
            st.dataframe(df.head())
            
            # Ask for target column
            st.subheader("Select Target Column")
            target_column = st.selectbox("Which column do you want to predict?", df.columns.tolist())
            
            if target_column:
                # Determine problem type
                unique_values = df[target_column].nunique()
                is_numeric = pd.api.types.is_numeric_dtype(df[target_column])
                
                if is_numeric and unique_values > 10:
                    problem_type = "regression"
                    st.info("🎯 Detected: **Regression** problem (predicting continuous values)")
                else:
                    problem_type = "classification"
                    st.info("🎯 Detected: **Classification** problem (predicting categories)")
                
                if st.button("Proceed with this dataset", type="primary"):
                    st.session_state.dataset = df
                    st.session_state.target_column = target_column
                    st.session_state.problem_type = problem_type
                    st.session_state.workflow_stage = 'checkpoint_1'
                    st.success("Dataset loaded successfully!")
                    st.switch_page("pages/2_Checkpoint_1_Dataset_Preview.py")
        
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")

with tab2:
    st.header("Generate Synthetic Data")
    st.markdown("Create a realistic synthetic dataset for your machine learning project.")
    
    # Sample datasets based on prompt
    if sample_prompt:
        st.subheader("Suggested Dataset")
        sample_datasets = generate_sample_datasets()
        
        if sample_prompt in sample_datasets:
            dataset_info = sample_datasets[sample_prompt]
            st.markdown(f"**{dataset_info['name']}**")
            st.markdown(f"*{dataset_info['description']}*")
            
            col1, col2 = st.columns([1, 1])
            with col1:
                st.markdown(f"**Problem Type:** {dataset_info['problem_type'].title()}")
            with col2:
                st.markdown(f"**Features:** {len(dataset_info['features'])} columns")
            
            num_samples = st.slider("Number of samples to generate", 100, 5000, 1000)
            
            if st.button(f"Generate {dataset_info['name']}", type="primary"):
                with st.spinner("Generating synthetic dataset..."):
                    df = create_synthetic_dataset(sample_prompt, num_samples)
                    
                    st.session_state.dataset = df
                    st.session_state.target_column = dataset_info['target']
                    st.session_state.problem_type = dataset_info['problem_type']
                    st.session_state.workflow_stage = 'checkpoint_1'
                    
                    st.success(f"Generated {len(df)} samples successfully!")
                    st.switch_page("pages/2_Checkpoint_1_Dataset_Preview.py")
    
    st.markdown("---")
    
    # Custom synthetic data generation
    st.subheader("Custom Synthetic Dataset")
    
    col1, col2 = st.columns(2)
    
    with col1:
        dataset_name = st.text_input("Dataset Name", "My Custom Dataset")
        num_samples = st.slider("Number of samples", 100, 5000, 1000)
        problem_type = st.selectbox("Problem Type", ["classification", "regression"])
    
    with col2:
        num_features = st.slider("Number of features", 3, 20, 5)
        noise_level = st.slider("Noise Level", 0.0, 0.5, 0.1)
    
    dataset_description = st.text_area("Dataset Description", "Custom synthetic dataset for machine learning")
    
    if st.button("Generate Custom Dataset", type="primary"):
        with st.spinner("Generating custom synthetic dataset..."):
            df = create_synthetic_dataset("custom", num_samples, {
                'num_features': num_features,
                'problem_type': problem_type,
                'noise_level': noise_level
            })
            
            st.session_state.dataset = df
            st.session_state.target_column = 'target'
            st.session_state.problem_type = problem_type
            st.session_state.workflow_stage = 'checkpoint_1'
            
            st.success(f"Generated custom dataset with {len(df)} samples!")
            st.switch_page("pages/2_Checkpoint_1_Dataset_Preview.py")

with tab3:
    st.header("Find Dataset")
    st.markdown("Search for suitable datasets from popular sources.")
    
    # Mock dataset discovery interface
    search_query = st.text_input("Search for datasets", placeholder="e.g., house prices, customer data, sentiment analysis")
    
    if search_query:
        st.info("🔍 Dataset discovery feature would integrate with platforms like Kaggle, UCI ML Repository, and other data sources.")
        st.markdown("""
        **Available Integration Options:**
        - 🏆 Kaggle Datasets API
        - 🎓 UCI Machine Learning Repository
        - 📊 OpenML Datasets
        - 🌐 Google Dataset Search
        - 📈 Financial Data APIs
        
        *Note: This feature requires API keys and external service integration.*
        """)
        
        st.warning("⚠️ Dataset discovery is not implemented in this demo. Please use Upload or Generate options.")

# Navigation
st.markdown("---")
col1, col2 = st.columns([1, 1])

with col1:
    if st.button("← Back to Welcome", use_container_width=True):
        st.switch_page("app.py")

with col2:
    if st.session_state.dataset is not None:
        if st.button("Next: Dataset Preview →", use_container_width=True, type="primary"):
            st.switch_page("pages/2_Checkpoint_1_Dataset_Preview.py")
