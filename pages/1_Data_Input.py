import streamlit as st
import pandas as pd
import io
from utils.ui_components import inject_custom_css, render_progress_tracker, sidebar_navigation, create_info_card, create_alert
from utils.synthetic_data import generate_sample_datasets, create_synthetic_dataset
from utils.dataset_finder import search_kaggle_datasets, download_and_load_kaggle_dataset
from utils.data_validator import validator
from utils.error_handler import handle_errors
from utils.cache_manager import cache_manager
from utils.interactive_synthetic_generator import InteractiveSyntheticDataGenerator

st.set_page_config(page_title="Data Input", page_icon="📁", layout="wide")
inject_custom_css()

def initialize_data_input_state():
    """Initialize data input specific session state"""
    st.session_state.workflow_stage = 'data_input'
    
    if 'upload_key' not in st.session_state:
        st.session_state.upload_key = 0
    
    if 'kaggle_search_results' not in st.session_state:
        st.session_state.kaggle_search_results = []
    
    if 'synthetic_generator' not in st.session_state:
        st.session_state.synthetic_generator = InteractiveSyntheticDataGenerator()

@handle_errors()
def process_uploaded_file(uploaded_file):
    """Process uploaded CSV file with validation"""
    try:
        # Read CSV with multiple encodings
        encodings = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']
        df = None
        
        for encoding in encodings:
            try:
                uploaded_file.seek(0)  # Reset file pointer
                df = pd.read_csv(uploaded_file, encoding=encoding)
                break
            except UnicodeDecodeError:
                continue
        
        if df is None:
            st.error("❌ Could not read the file. Please ensure it's a valid CSV file.")
            return None
        
        # Validate dataset
        is_valid, issues, metadata = validator.validate_dataset(df)
        validator.display_validation_results(is_valid, issues, metadata)
        
        if not is_valid and any("❌" in issue for issue in issues):
            st.error("Dataset validation failed. Please fix the issues above.")
            return None
        
        return df
        
    except Exception as e:
        st.error(f"❌ Error processing file: {str(e)}")
        return None

@handle_errors()
def display_dataset_preview(df, title="Dataset Preview"):
    """Display dataset preview with enhanced information"""
    st.markdown(f"### {title}")
    
    # Dataset statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Rows", f"{len(df):,}")
    with col2:
        st.metric("Columns", f"{len(df.columns):,}")
    with col3:
        st.metric("Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB")
    with col4:
        st.metric("Missing Values", f"{df.isnull().sum().sum():,}")
    
    # Data preview
    st.dataframe(df.head(10), use_container_width=True)
    
    # Column information
    with st.expander("📋 Column Information"):
        col_info = pd.DataFrame({
            'Column': df.columns,
            'Type': df.dtypes.astype(str),
            'Non-Null Count': df.count().values,
            'Null Count': df.isnull().sum().values,
            'Unique Values': df.nunique().values
        })
        st.dataframe(col_info, use_container_width=True)

@handle_errors()
def target_column_selection(df):
    """Enhanced target column selection with suggestions"""
    st.markdown("### 🎯 Target Variable Selection")
    
    # Get suggestions from validator
    suggestions = validator.suggest_target_column(df)
    
    if suggestions:
        st.markdown("**💡 Suggested target columns:**")
        suggestion_cols = st.columns(len(suggestions[:3]))
        
        for i, col in enumerate(suggestions[:3]):
            with suggestion_cols[i]:
                if st.button(f"✨ {col}", use_container_width=True, type="secondary"):
                    st.session_state.suggested_target = col
    
    # Target column selection
    selected_col = st.session_state.get('suggested_target', df.columns[0])
    target_column = st.selectbox(
        "Select the target column (what you want to predict):",
        df.columns.tolist(),
        index=df.columns.tolist().index(selected_col) if selected_col in df.columns else 0,
        key="target_selection",
        help="This is the variable you want to predict. The AI will automatically detect if it's a classification or regression problem."
    )
    
    if target_column:
        # Auto-detect problem type
        problem_type = validator.detect_problem_type(df, target_column)
        
        # Display target information
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Unique Values", df[target_column].nunique())
            st.metric("Data Type", str(df[target_column].dtype))
        
        with col2:
            st.metric("Missing Values", df[target_column].isnull().sum())
            st.metric("Problem Type", problem_type.title())
        
        # Target distribution preview
        if problem_type == "classification":
            st.markdown("**Target Distribution:**")
            value_counts = df[target_column].value_counts().head(10)
            st.bar_chart(value_counts)
        else:
            st.markdown("**Target Distribution:**")
            st.line_chart(df[target_column].value_counts().sort_index())
        
        return target_column, problem_type
    
    return None, None

def create_upload_tab():
    """Create upload dataset tab"""
    st.markdown("### 📤 Upload Your Dataset")
    
    create_info_card(
        "Supported Formats",
        """
        • <strong>CSV files</strong> up to 200MB<br>
        • Multiple encodings supported (UTF-8, Latin1, etc.)<br>
        • Automatic data type detection<br>
        • Comprehensive data validation
        """,
        "📋"
    )
    
    # File uploader with custom key for refresh
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type="csv",
        key=f"file_uploader_{st.session_state.upload_key}",
        help="Upload your CSV file. The system will automatically validate and process it."
    )
    
    if uploaded_file is not None:
        with st.spinner("🔄 Processing uploaded file..."):
            df = process_uploaded_file(uploaded_file)
        
        if df is not None:
            display_dataset_preview(df, "📊 Uploaded Dataset Preview")
            
            target_column, problem_type = target_column_selection(df)
            
            if target_column and problem_type:
                st.markdown("---")
                
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    create_alert(
                        f"✅ Ready to proceed with <strong>{problem_type}</strong> problem using target column <strong>{target_column}</strong>",
                        "success"
                    )
                
                with col2:
                    if st.button("Continue →", type="primary", use_container_width=True):
                        # Cache the dataset
                        dataset_hash = cache_manager.get_hash(df)
                        cache_manager.cache_data(f"dataset_{dataset_hash}", df, "data")
                        
                        # Store in session state
                        st.session_state.dataset = df
                        st.session_state.target_column = target_column
                        st.session_state.problem_type = problem_type
                        st.session_state.data_source = "upload"
                        
                        st.success("✅ Dataset uploaded successfully!")
                        st.switch_page("pages/2_Dataset_Preview.py")

def create_synthetic_tab():
    """Create synthetic data generation tab"""
    st.markdown("### 🎲 Generate Synthetic Data")
    
    sample_prompt = st.session_state.get('sample_prompt', None)
    
    if sample_prompt:
        st.markdown("#### 🌟 Recommended Dataset")
        sample_datasets = generate_sample_datasets()
        
        if sample_prompt in sample_datasets:
            dataset_info = sample_datasets[sample_prompt]
            
            create_info_card(
                dataset_info['name'],
                f"""
                <strong>Description:</strong> {dataset_info['description']}<br>
                <strong>Problem Type:</strong> {dataset_info['problem_type'].title()}<br>
                <strong>Features:</strong> {len(dataset_info['features'])} columns<br>
                <strong>Target:</strong> {dataset_info['target']}
                """,
                "🎯"
            )
            
            col1, col2 = st.columns(2)
            
            with col1:
                num_samples = st.slider(
                    "Number of samples to generate",
                    100, 10000, 2000,
                    key="recommended_samples",
                    help="More samples generally lead to better model performance"
                )
            
            with col2:
                st.markdown("<br>", unsafe_allow_html=True)
                if st.button(f"Generate {dataset_info['name']}", type="primary", use_container_width=True):
                    with st.spinner("🎲 Generating synthetic dataset..."):
                        df = create_synthetic_dataset(sample_prompt, num_samples)
                        
                        st.session_state.dataset = df
                        st.session_state.target_column = dataset_info['target']
                        st.session_state.problem_type = dataset_info['problem_type']
                        st.session_state.data_source = "synthetic"
                        
                        st.success("✅ Synthetic dataset generated successfully!")
                        st.switch_page("pages/2_Dataset_Preview.py")
        
        st.markdown("---")
    
    # Custom synthetic dataset
    st.markdown("#### 🛠️ Custom Synthetic Dataset")
    
    col1, col2 = st.columns(2)
    
    with col1:
        num_samples_custom = st.slider("Number of samples", 100, 10000, 1000, key="custom_samples")
        problem_type_custom = st.selectbox("Problem Type", ["classification", "regression"], key="custom_problem")
    
    with col2:
        num_features_custom = st.slider("Number of features", 3, 25, 8, key="custom_features")
        noise_level_custom = st.slider("Noise Level", 0.0, 0.5, 0.1, key="custom_noise")
    
    if st.button("Generate Custom Dataset", type="primary", use_container_width=True):
        with st.spinner("🎲 Generating custom synthetic dataset..."):
            df = create_synthetic_dataset("custom", num_samples_custom, {
                'num_features': num_features_custom,
                'problem_type': problem_type_custom,
                'noise_level': noise_level_custom
            })
            
            st.session_state.dataset = df
            st.session_state.target_column = 'target'
            st.session_state.problem_type = problem_type_custom
            st.session_state.data_source = "synthetic"
            
            st.success("✅ Custom dataset generated successfully!")
            st.switch_page("pages/2_Dataset_Preview.py")

def create_interactive_ai_tab():
    """Create AI-powered interactive synthetic data generator tab"""
    st.markdown("### 🤖 AI-Powered Data Designer")
    
    create_info_card(
        "Intelligent Dataset Creation",
        """
        • <strong>Natural Language Interface:</strong> Describe your dataset in plain English<br>
        • <strong>AI Configuration:</strong> Let AI understand and configure your requirements<br>
        • <strong>Custom Statistics:</strong> Define exact statistical properties for each feature<br>
        • <strong>Domain Templates:</strong> Quick start with business, healthcare, or finance templates<br>
        • <strong>Interactive Design:</strong> Chat-like interface guides you through the process
        """,
        "🤖"
    )
    
    # Run the interactive generator
    st.session_state.synthetic_generator.run_interactive_generator()

def create_kaggle_tab():
    """Create Kaggle dataset finder tab"""
    st.markdown("### 🔍 Find Dataset on Kaggle")
    
    create_info_card(
        "Kaggle Integration",
        """
        Search and download datasets directly from Kaggle's vast collection.<br>
        <strong>Requirements:</strong> Kaggle API credentials in secrets.toml
        """,
        "🏆"
    )
    
    # Search interface
    col1, col2 = st.columns([3, 1])
    
    with col1:
        search_query = st.text_input(
            "Search for datasets",
            placeholder="e.g., house prices, customer churn, iris classification",
            help="Enter keywords related to your problem domain"
        )
    
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        search_clicked = st.button("🔍 Search", use_container_width=True)
    
    if search_query and search_clicked:
        with st.spinner("🔍 Searching Kaggle datasets..."):
            search_results = search_kaggle_datasets(search_query)
            st.session_state.kaggle_search_results = search_results
    
    # Display search results
    if st.session_state.kaggle_search_results:
        st.markdown(f"### 📊 Found {len(st.session_state.kaggle_search_results)} Datasets")
        
        for i, dataset in enumerate(st.session_state.kaggle_search_results[:10]):  # Show top 10
            with st.expander(f"🏆 **{dataset.title}** (Usability: {dataset.usabilityRating:.1f}/10)"):
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.markdown(f"**Owner:** `{dataset.ownerName}`")
                    st.markdown(f"**Reference:** `{dataset.ref}`")
                    st.markdown(f"**Size:** {dataset.totalBytes / 1024 / 1024:.1f} MB")
                    st.markdown(f"**Downloads:** {dataset.downloadCount:,}")
                
                with col2:
                    if st.button("📥 Download", key=f"download_{i}", use_container_width=True):
                        with st.spinner(f"📥 Downloading {dataset.ref}..."):
                            df = download_and_load_kaggle_dataset(dataset.ref)
                            
                            if df is not None:
                                st.session_state.temp_kaggle_df = df
                                st.session_state.temp_kaggle_title = dataset.title
                                st.rerun()
    
    # Preview downloaded dataset
    if 'temp_kaggle_df' in st.session_state:
        st.markdown("---")
        df = st.session_state.temp_kaggle_df
        
        display_dataset_preview(df, f"📊 Preview: {st.session_state.temp_kaggle_title}")
        
        target_column, problem_type = target_column_selection(df)
        
        if target_column and problem_type:
            st.markdown("---")
            
            col1, col2 = st.columns([3, 1])
            
            with col1:
                create_alert(
                    f"✅ Ready to proceed with <strong>{problem_type}</strong> problem using target column <strong>{target_column}</strong>",
                    "success"
                )
            
            with col2:
                if st.button("Continue with Kaggle Dataset →", type="primary", use_container_width=True):
                    st.session_state.dataset = st.session_state.temp_kaggle_df
                    st.session_state.target_column = target_column
                    st.session_state.problem_type = problem_type
                    st.session_state.data_source = "kaggle"
                    
                    # Clean up temporary variables
                    del st.session_state['temp_kaggle_df']
                    del st.session_state['temp_kaggle_title']
                    
                    st.success("✅ Kaggle dataset loaded successfully!")
                    st.switch_page("pages/2_Dataset_Preview.py")

def create_data_source_comparison():
    """Create comparison of different data sources"""
    st.markdown("### 📊 Data Source Comparison")
    
    comparison_data = {
        'Data Source': ['Upload CSV', 'Synthetic Data', 'AI Designer', 'Kaggle Search'],
        'Speed': ['Fast', 'Very Fast', 'Fast', 'Medium'],
        'Customization': ['High', 'Medium', 'Very High', 'Low'],
        'Data Quality': ['Variable', 'Controlled', 'Controlled', 'High'],
        'Best For': ['Real Data', 'Testing/Learning', 'Custom Requirements', 'Research/Learning']
    }
    
    comparison_df = pd.DataFrame(comparison_data)
    st.dataframe(comparison_df, use_container_width=True, hide_index=True)

def main():
    """Main function for data input page"""
    initialize_data_input_state()
    
    # Header
    st.title("📁 Data Input")
    st.markdown("Choose how you want to provide data for your machine learning project.")
    
    # Progress tracker
    render_progress_tracker('data_input')
    
    # Quick start info
    sample_prompt = st.session_state.get('sample_prompt', None)
    if sample_prompt:
        create_alert(
            f"🎯 <strong>Quick Start:</strong> {sample_prompt.replace('_', ' ').title()} Project",
            "info"
        )
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📤 Upload Dataset", 
        "🎲 Generate Synthetic Data", 
        "🤖 AI Data Designer",
        "🔍 Find Dataset (Kaggle)"
    ])
    
    with tab1:
        create_upload_tab()
    
    with tab2:
        create_synthetic_tab()
    
    with tab3:
        create_interactive_ai_tab()
    
    with tab4:
        create_kaggle_tab()
    
    # Additional information section
    st.markdown("---")
    
    # Data source comparison
    with st.expander("📊 Compare Data Sources"):
        create_data_source_comparison()
    
    # Tips section
    with st.expander("💡 Tips for Better Results"):
        st.markdown("""
        **📤 For Upload:**
        - Ensure your CSV has clear column names
        - Remove or handle missing values beforehand
        - Keep file size under 200MB for best performance
        
        **🎲 For Synthetic Data:**
        - Start with recommended templates for common use cases
        - Use larger sample sizes (>1000) for better model training
        - Consider the noise level based on your use case
        
        **🤖 For AI Designer:**
        - Be specific about your domain and requirements
        - Describe statistical properties you need
        - Use templates as starting points for customization
        
        **🔍 For Kaggle:**
        - Search with specific keywords related to your domain
        - Check dataset usability ratings (>7.0 recommended)
        - Preview data before downloading
        """)
    
    # Navigation
    st.markdown("---")
    navigation_col1, navigation_col2 = st.columns([1, 1])
    
    with navigation_col1:
        if st.button("← Back to Welcome", use_container_width=True):
            st.switch_page("app.py")
    
    with navigation_col2:
        # Show continue button if dataset is loaded
        if st.session_state.get('dataset') is not None:
            if st.button("Continue to Dataset Preview →", use_container_width=True, type="primary"):
                st.switch_page("pages/2_Dataset_Preview.py")
    
    # Sidebar
    sidebar_navigation()

if __name__ == "__main__":
    main()
