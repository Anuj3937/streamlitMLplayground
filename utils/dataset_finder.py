import streamlit as st
import kaggle
import pandas as pd
import os
import json
import zipfile
from .error_handler import handle_errors

@handle_errors()
def setup_kaggle_api():
    """Sets up the Kaggle API credentials"""
    kaggle_dir = os.path.join(os.path.expanduser("~"), ".kaggle")
    kaggle_json_path = os.path.join(kaggle_dir, "kaggle.json")
    
    # Check if secrets are available
    if 'KAGGLE_USERNAME' not in st.secrets or 'KAGGLE_KEY' not in st.secrets:
        st.error("🔑 Kaggle credentials not found in Streamlit secrets.")
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
    
    # Set permissions for the file
    os.chmod(kaggle_json_path, 0o600)
    
    try:
        kaggle.api.authenticate()
        return True
    except Exception as e:
        st.error(f"❌ Kaggle authentication failed: {e}")
        return False

@st.cache_data(ttl=3600)  # Cache for 1 hour
def search_kaggle_datasets(query, max_results=20):
    """Searches for datasets on Kaggle with caching"""
    if not setup_kaggle_api():
        return []
    
    try:
        with st.spinner(f"🔍 Searching Kaggle for '{query}'..."):
            datasets = kaggle.api.dataset_list(
                search=query, 
                sort_by='votes', 
                file_type='csv',
                max_size=200*1024*1024  # 200MB max
            )
            
            # Filter and sort results
            filtered_datasets = []
            for dataset in datasets[:max_results]:
                if hasattr(dataset, 'usabilityRating') and dataset.usabilityRating > 5.0:
                    filtered_datasets.append(dataset)
            
            return filtered_datasets[:max_results]
            
    except Exception as e:
        st.error(f"❌ Error searching Kaggle: {e}")
        return []

@handle_errors()
def download_and_load_kaggle_dataset(dataset_ref):
    """Downloads and loads a Kaggle dataset"""
    if not setup_kaggle_api():
        return None
    
    download_path = "temp_kaggle_data"
    os.makedirs(download_path, exist_ok=True)
    
    try:
        # Get dataset files info
        files = kaggle.api.dataset_list_files(dataset_ref).files
        csv_files = [f for f in files if f.name.endswith('.csv')]
        
        if not csv_files:
            st.error("❌ No CSV files found in this dataset.")
            return None
        
        # Download the first (usually largest) CSV file
        csv_file = csv_files[0]
        
        # Check file size
        if hasattr(csv_file, 'size') and csv_file.size > 200*1024*1024:  # 200MB
            st.warning(f"⚠️ File is large ({csv_file.size/1024/1024:.1f}MB). This may take time to download.")
        
        kaggle.api.dataset_download_file(
            dataset_ref, 
            file_name=csv_file.name, 
            path=download_path,
            quiet=False
        )
        
        file_path = os.path.join(download_path, csv_file.name)
        zip_path = file_path + ".zip"
        
        # Extract if zipped
        if os.path.exists(zip_path):
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(download_path)
            os.remove(zip_path)
        
        # Load CSV with multiple encoding attempts
        encodings = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']
        df = None
        
        for encoding in encodings:
            try:
                df = pd.read_csv(
                    file_path, 
                    encoding=encoding, 
                    on_bad_lines='skip',
                    low_memory=False,
                    nrows=50000  # Limit rows for performance
                )
                break
            except UnicodeDecodeError:
                continue
            except Exception as e:
                st.warning(f"⚠️ Error with encoding {encoding}: {str(e)}")
                continue
        
        # Cleanup
        if os.path.exists(file_path):
            os.remove(file_path)
        
        if df is not None:
            st.success(f"✅ Loaded {len(df):,} rows and {len(df.columns)} columns from Kaggle!")
            return df
        else:
            st.error("❌ Failed to load dataset with any encoding.")
            return None
            
    except Exception as e:
        st.error(f"❌ Failed to download dataset: {e}")
        return None
    finally:
        # Cleanup temp directory
        try:
            import shutil
            if os.path.exists(download_path):
                shutil.rmtree(download_path)
        except:
            pass
