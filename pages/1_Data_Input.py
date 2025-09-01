# pages/1_Data_Input.py

import streamlit as st
import pandas as pd
import io
from utils.synthetic_data import generate_sample_datasets, create_synthetic_dataset
from utils.dataset_finder import search_kaggle_datasets, download_and_load_kaggle_dataset

st.set_page_config(page_title="Data Input", page_icon="📁", layout="wide")

st.title("📁 Data Input")
st.markdown("Choose how you want to provide data for your machine learning project.")

st.session_state.workflow_stage = 'data_input'

sample_prompt = st.session_state.get('sample_prompt', None)
if sample_prompt:
    st.info(f"🎯 Quick Start: {sample_prompt.replace('_', ' ').title()}")

tab1, tab2, tab3 = st.tabs(["📤 Upload Dataset", "🎲 Generate Synthetic Data", "🔍 Find Dataset (Kaggle)"])

with tab1:
    st.header("Upload Your Dataset")
    uploaded_file = st.file_uploader("Upload a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"Successfully loaded dataset with {len(df)} rows and {len(df.columns)} columns")
            st.dataframe(df.head())
            
            target_column = st.selectbox("Which column do you want to predict?", df.columns.tolist(), key="upload_target")
            
            if target_column:
                unique_values = df[target_column].nunique()
                is_numeric = pd.api.types.is_numeric_dtype(df[target_column])
                
                problem_type = "regression" if is_numeric and unique_values > 10 else "classification"
                st.info(f"🎯 Detected: **{problem_type.title()}** problem")
                
                if st.button("Proceed with this dataset", type="primary", key="upload_proceed"):
                    st.session_state.dataset = df
                    st.session_state.target_column = target_column
                    st.session_state.problem_type = problem_type
                    st.switch_page("pages/2_Checkpoint_1_Dataset_Preview.py")
        
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")

with tab2:
    st.header("Generate Synthetic Data")
    # ... (Keep the synthetic data generation code as it was)
    if sample_prompt:
        st.subheader("Suggested Dataset")
        sample_datasets = generate_sample_datasets()
        
        if sample_prompt in sample_datasets:
            dataset_info = sample_datasets[sample_prompt]
            st.markdown(f"**{dataset_info['name']}**")
            st.markdown(f"*{dataset_info['description']}*")
            
            num_samples = st.slider("Number of samples to generate", 100, 5000, 1000, key="gen_samples")
            
            if st.button(f"Generate {dataset_info['name']}", type="primary", key="gen_button"):
                with st.spinner("Generating synthetic dataset..."):
                    df = create_synthetic_dataset(sample_prompt, num_samples)
                    st.session_state.dataset = df
                    st.session_state.target_column = dataset_info['target']
                    st.session_state.problem_type = dataset_info['problem_type']
                    st.switch_page("pages/2_Checkpoint_1_Dataset_Preview.py")
    
    st.markdown("---")
    st.subheader("Custom Synthetic Dataset")
    # ... (Keep the custom synthetic data generation code as it was)
    col1, col2 = st.columns(2)
    with col1:
        num_samples_custom = st.slider("Number of samples", 100, 5000, 1000, key="custom_samples")
        problem_type_custom = st.selectbox("Problem Type", ["classification", "regression"], key="custom_problem")
    with col2:
        num_features_custom = st.slider("Number of features", 3, 20, 5, key="custom_features")
        noise_level_custom = st.slider("Noise Level", 0.0, 0.5, 0.1, key="custom_noise")
    if st.button("Generate Custom Dataset", type="primary", key="custom_button"):
        with st.spinner("Generating custom synthetic dataset..."):
            df = create_synthetic_dataset("custom", num_samples_custom, {
                'num_features': num_features_custom,
                'problem_type': problem_type_custom,
                'noise_level': noise_level_custom
            })
            st.session_state.dataset = df
            st.session_state.target_column = 'target'
            st.session_state.problem_type = problem_type_custom
            st.switch_page("pages/2_Checkpoint_1_Dataset_Preview.py")

with tab3:
    st.header("Find Dataset on Kaggle")
    search_query = st.text_input("Search for datasets", placeholder="e.g., house prices, customer churn")
    
    if search_query:
        with st.spinner("Searching Kaggle..."):
            search_results = search_kaggle_datasets(search_query)
        
        if search_results:
            st.success(f"Found {len(search_results)} datasets.")
            
            for d in search_results:
                with st.expander(f"**{d.title}** (Votes: {d.usability_rating})"):
                    # st.markdown(f"Owner: `{d.ownerSlug}` | Size: `{d.size}`")
                    st.markdown(f"Reference: `{d.ref}`")
                    
                    if st.button("Download and Preview this dataset", key=f"download_{d.ref}"):
                        with st.spinner(f"Downloading {d.ref}..."):
                            df = download_and_load_kaggle_dataset(d.ref)
                            if df is not None:
                                st.session_state.temp_df = df
                                st.session_state.temp_df_name = d.title

        else:
            st.warning("No datasets found matching your query.")

    if 'temp_df' in st.session_state:
        st.markdown("---")
        st.subheader(f"Preview: {st.session_state.temp_df_name}")
        st.dataframe(st.session_state.temp_df.head())

        target_column_kaggle = st.selectbox(
            "Select the target column to predict", 
            st.session_state.temp_df.columns.tolist(),
            key="kaggle_target"
        )
        if target_column_kaggle:
            df = st.session_state.temp_df
            unique_values = df[target_column_kaggle].nunique()
            is_numeric = pd.api.types.is_numeric_dtype(df[target_column_kaggle])
            
            problem_type_kaggle = "regression" if is_numeric and unique_values > 10 else "classification"
            st.info(f"🎯 Detected: **{problem_type_kaggle.title()}** problem")

            if st.button("Proceed with this Kaggle dataset", type="primary", key="kaggle_proceed"):
                st.session_state.dataset = st.session_state.temp_df
                st.session_state.target_column = target_column_kaggle
                st.session_state.problem_type = problem_type_kaggle
                del st.session_state['temp_df']
                del st.session_state['temp_df_name']
                st.switch_page("pages/2_Checkpoint_1_Dataset_Preview.py")

st.markdown("---")
if st.button("← Back to Welcome"):
    st.switch_page("app.py")