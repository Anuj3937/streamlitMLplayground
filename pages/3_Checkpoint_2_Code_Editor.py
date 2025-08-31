import streamlit as st
from streamlit_ace import st_ace
from utils.ml_pipeline import generate_ml_code, execute_ml_code

st.set_page_config(page_title="Checkpoint 2: Code Editor", page_icon="💻", layout="wide")

st.title("💻 Checkpoint 2: Code Editor")
st.markdown("Review, edit, and approve the generated machine learning code before execution.")

# Update workflow stage
st.session_state.workflow_stage = 'checkpoint_2'

# Check if cleaned dataset exists
if st.session_state.get('cleaned_dataset') is None:
    st.warning("No cleaned dataset found. Please complete Checkpoint 1 first.")
    if st.button("← Go to Dataset Preview"):
        st.switch_page("pages/2_Checkpoint_1_Dataset_Preview.py")
    st.stop()

cleaned_df = st.session_state.cleaned_dataset
target_column = st.session_state.target_column
problem_type = st.session_state.problem_type

# Generate ML code if not already generated
if not st.session_state.generated_code:
    with st.spinner("Generating machine learning code..."):
        generated_code = generate_ml_code(cleaned_df, target_column, problem_type)
        st.session_state.generated_code = generated_code

# Code Overview
st.header("📋 Generated ML Pipeline")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Problem Type", problem_type.title())
with col2:
    st.metric("Algorithms", "5 Models")
with col3:
    st.metric("Features", len(cleaned_df.columns) - 1)

st.markdown("""
**The generated code includes:**
- Data splitting and preprocessing
- 5 machine learning algorithms optimized for your problem type
- Hyperparameter tuning with GridSearchCV
- Model evaluation and comparison
- Performance metrics calculation
""")

# Code Editor
st.header("🛠️ Code Editor")

tab1, tab2 = st.tabs(["Edit Code", "Code Explanation"])

with tab1:
    st.subheader("Machine Learning Pipeline Code")
    st.markdown("Review and modify the generated code below. Click 'Execute Code' when ready.")
    
    # Code editor
    edited_code = st_ace(
        value=st.session_state.generated_code,
        language='python',
        theme='monokai',
        key="ml_code_editor",
        height=600,
        auto_update=True,
        font_size=14,
        tab_size=4,
        show_gutter=True,
        show_print_margin=True,
        wrap=False,
        annotations=None
    )
    
    # Update session state with edited code
    if edited_code != st.session_state.generated_code:
        st.session_state.generated_code = edited_code
    
    st.markdown("---")
    
    # Code execution controls
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.markdown("**Ready to execute?** The code will run in a secure environment and display live progress.")
    
    with col2:
        if st.button("Reset to Original", use_container_width=True):
            original_code = generate_ml_code(cleaned_df, target_column, problem_type)
            st.session_state.generated_code = original_code
            st.rerun()
    
    with col3:
        execute_button = st.button("Execute Code", type="primary", use_container_width=True)

with tab2:
    st.subheader("Code Explanation")
    
    st.markdown("""
    ### 📚 What This Code Does:
    
    #### 1. Data Preparation
    - Splits your dataset into training and testing sets (80/20 split)
    - Applies any necessary preprocessing steps
    - Prepares features and target variables
    
    #### 2. Algorithm Selection
    The code tests these algorithms optimized for your problem type:
    """)
    
    if problem_type == "classification":
        algorithms = [
            "**Logistic Regression** - Linear classifier, good baseline",
            "**Random Forest** - Ensemble method, handles mixed data types",
            "**XGBoost** - Gradient boosting, often high-performing",
            "**Support Vector Machine** - Effective for complex boundaries",
            "**Decision Tree** - Interpretable, good for feature importance"
        ]
    else:
        algorithms = [
            "**Linear Regression** - Simple linear relationship modeling",
            "**Random Forest** - Ensemble method for non-linear patterns",
            "**XGBoost** - Gradient boosting for complex relationships",
            "**Support Vector Regression** - Effective for high-dimensional data",
            "**Decision Tree** - Interpretable non-linear regression"
        ]
    
    for alg in algorithms:
        st.markdown(f"- {alg}")
    
    st.markdown("""
    #### 3. Hyperparameter Tuning
    - Uses GridSearchCV to find optimal parameters for each algorithm
    - Tests multiple parameter combinations automatically
    - Selects best parameters based on cross-validation performance
    
    #### 4. Model Evaluation
    - Evaluates each model on the test set
    - Calculates comprehensive performance metrics
    - Identifies the best-performing model
    
    #### 5. Results Generation
    - Creates detailed performance comparisons
    - Generates visualizations and metrics
    - Prepares the final model for deployment
    """)

# Code Execution
if execute_button:
    st.header("🚀 Code Execution")
    
    # Create execution container
    execution_container = st.container()
    
    with execution_container:
        st.subheader("Live Execution Progress")
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        logs_container = st.empty()
        
        try:
            # Execute the ML code
            results = execute_ml_code(
                st.session_state.generated_code,
                cleaned_df,
                target_column,
                problem_type,
                progress_callback=lambda progress, message: (
                    progress_bar.progress(progress),
                    status_text.text(message)
                ),
                logs_callback=lambda logs: logs_container.text_area("Execution Logs", logs, height=200)
            )
            
            if results:
                st.session_state.model_results = results
                st.session_state.workflow_stage = 'checkpoint_3'
                
                progress_bar.progress(100)
                status_text.text("✅ Execution completed successfully!")
                
                st.success("🎉 Machine learning pipeline executed successfully!")
                st.markdown("**Next:** Review model comparison results and performance metrics.")
                
                # Auto-navigate after short delay
                import time
                time.sleep(2)
                st.switch_page("pages/4_Checkpoint_3_Model_Comparison.py")
            else:
                st.error("❌ Code execution failed. Please check the logs and try again.")
                
        except Exception as e:
            st.error(f"❌ Execution error: {str(e)}")
            status_text.text("❌ Execution failed")
            progress_bar.progress(0)

# Navigation
st.markdown("---")
col1, col2 = st.columns([1, 1])

with col1:
    if st.button("← Back to Dataset Preview", use_container_width=True):
        st.switch_page("pages/2_Checkpoint_1_Dataset_Preview.py")

with col2:
    if st.session_state.get('model_results') is not None:
        if st.button("Next: Model Comparison →", use_container_width=True, type="primary"):
            st.switch_page("pages/4_Checkpoint_3_Model_Comparison.py")
