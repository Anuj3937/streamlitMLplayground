# utils/dataset_finder.py

import streamlit as st
import kaggle
import pandas as pd
import os
import json

def setup_kaggle_api():
    """
    Sets up the Kaggle API credentials by creating the kaggle.json file
    from Streamlit secrets.
    """
    # Define the path for the kaggle.json file
    kaggle_dir = os.path.join(os.path.expanduser("~"), ".kaggle")
    kaggle_json_path = os.path.join(kaggle_dir, "kaggle.json")

    # Check if secrets are available
    if 'KAGGLE_USERNAME' not in st.secrets or 'KAGGLE_KEY' not in st.secrets:
        st.error("Kaggle credentials not found in Streamlit secrets.")
        st.info("Please add `KAGGLE_USERNAME` and `KAGGLE_KEY` to your `.streamlit/secrets.toml` file.")
        return False

    # Create the directory if it doesn't exist
    os.makedirs(kaggle_dir, exist_ok=True)

    # Create the kaggle.json file
    kaggle_credentials = {
        "username": st.secrets["KAGGLE_USERNAME"],
        "key": st.secrets["KAGGLE_KEY"]
    }
    with open(kaggle_json_path, "w") as f:
        json.dump(kaggle_credentials, f)

    # Set permissions for the file to avoid warnings
    os.chmod(kaggle_json_path, 0o600)

    try:
        kaggle.api.authenticate()
        return True
    except Exception as e:
        st.error(f"Kaggle authentication failed: {e}")
        return False

@st.cache_data
def search_kaggle_datasets(query):
    """Searches for datasets on Kaggle."""
    if not setup_kaggle_api():
        return []
    try:
        # Use a generator to fetch a limited number of datasets
        datasets = kaggle.api.dataset_list(search=query, sort_by='votes', file_type='csv')
        return datasets[:20] # Return top 20 results
    except Exception as e:
        st.error(f"Error searching Kaggle: {e}")
        return []

def download_and_load_kaggle_dataset(dataset_ref):
    """Downloads a dataset file from Kaggle and loads it into a DataFrame."""
    if not setup_kaggle_api():
        return None
    
    download_path = "temp_data"
    os.makedirs(download_path, exist_ok=True)
    
    try:
        files = kaggle.api.dataset_list_files(dataset_ref).files
        csv_file = next((f for f in files if f.name.endswith('.csv')), None)

        if not csv_file:
            st.error("No CSV files found in this dataset.")
            return None
        
        kaggle.api.dataset_download_file(dataset_ref, file_name=csv_file.name, path=download_path, quiet=False)
        
        file_path = os.path.join(download_path, csv_file.name)
        zip_path = file_path + ".zip"
        if os.path.exists(zip_path):
            import zipfile
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(download_path)
            os.remove(zip_path)

        df = pd.read_csv(file_path, encoding='latin1', on_bad_lines='skip')
        os.remove(file_path)
        
        return df
    except Exception as e:
        st.error(f"Failed to download or load dataset: {e}")
        return None