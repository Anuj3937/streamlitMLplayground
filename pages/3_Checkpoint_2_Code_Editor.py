# pages/3_Checkpoint_2_Code_Editor.py

import streamlit as st
from streamlit_ace import st_ace
from utils.ml_pipeline import execute_ml_code
from utils.gemini_integration import generate_code_with_gemini, explain_code_with_gemini

st.set_page_config(page_title="Checkpoint 2: Code Editor", page_icon="💻", layout="wide")

st.title("💻 Checkpoint 2: AI Code Generation")
st.markdown("Your AI Teammate has generated the ML code. Review, edit, and execute it.")

st.session_state.workflow_stage = 'checkpoint_2'

if st.session_state.get('cleaned_dataset') is None:
    st.warning("No cleaned dataset found. Please complete Checkpoint 1 first.")
    if st.button("← Go to Dataset Preview"):
        st.switch_page("pages/2_Checkpoint_1_Dataset_Preview.py")
    st.stop()

cleaned_df = st.session_state.cleaned_dataset
target_column = st.session_state.target_column
problem_type = st.session_state.problem_type

# Generate ML code with Gemini if not already generated
if not st.session_state.get('generated_code'):
    with st.spinner("🧠 Calling AI Teammate to write ML code with Optuna..."):
        generated_code = generate_code_with_gemini(
            df=cleaned_df,
            target_column=target_column,
            problem_type=problem_type
        )
        st.session_state.generated_code = generated_code
        st.session_state.edited_code = generated_code

tab1, tab2 = st.tabs(["Edit & Execute Code", "🤖 AI Code Explanation"])

with tab1:
    st.subheader("Machine Learning Pipeline Code")
    edited_code = st_ace(
        value=st.session_state.edited_code,
        language='python',
        theme='monokai',
        key="ml_code_editor",
        height=600,
        auto_update=True
    )
    st.session_state.edited_code = edited_code # Persist edits
    
    col1, col2 = st.columns([2,1])
    with col1:
        execute_button = st.button("🚀 Execute Code", type="primary", use_container_width=True)
    with col2:
        if st.button("🔄 Reset to AI Original", use_container_width=True):
            st.session_state.edited_code = st.session_state.generated_code
            st.rerun()

with tab2:
    st.subheader("Code Explanation from AI Teammate")
    if 'code_explanation' not in st.session_state or st.button("Get Fresh Explanation"):
        with st.spinner("AI Teammate is reading the code..."):
            explanation = explain_code_with_gemini(st.session_state.edited_code)
            st.session_state.code_explanation = explanation
    
    st.markdown(st.session_state.get('code_explanation', "Click the button to generate an explanation."))


if execute_button:
    st.header("🚀 Code Execution")
    progress_bar = st.progress(0, text="Starting execution...")
    logs_container = st.expander("Live Execution Logs", expanded=True)
    log_area = logs_container.empty()
    
    try:
        def progress_callback(progress, message):
            progress_bar.progress(progress, text=message)
        def logs_callback(logs):
            log_area.code(logs)

        results = execute_ml_code(
            st.session_state.edited_code,
            cleaned_df,
            target_column,
            problem_type,
            progress_callback=progress_callback,
            logs_callback=logs_callback
        )
        
        if results:
            st.session_state.model_results = results
            st.success("🎉 Pipeline executed successfully!")
            st.switch_page("pages/4_Checkpoint_3_Model_Comparison.py")
        else:
            st.error("❌ Code execution finished but returned no results.")
            
    except Exception as e:
        st.error(f"❌ Execution failed with an error.")
        # The error is already printed in the logs via the callback

st.markdown("---")
if st.button("← Back to Dataset Preview"):
    st.switch_page("pages/2_Checkpoint_1_Dataset_Preview.py")