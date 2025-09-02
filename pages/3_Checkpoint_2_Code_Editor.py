import streamlit as st
from streamlit_ace import st_ace
from utils.ui_components import (inject_custom_css, render_progress_tracker, sidebar_navigation,
                                create_info_card, create_alert, create_expandable_code)
from utils.ml_pipeline import execute_ml_code
from utils.gemini_integration import generate_code_with_gemini, explain_code_with_gemini
from utils.error_handler import handle_errors, streamlit_handler
from utils.cache_manager import cache_manager
import time

st.set_page_config(page_title="Code Editor", page_icon="💻", layout="wide")
inject_custom_css()

def initialize_code_editor_state():
    """Initialize code editor session state"""
    st.session_state.workflow_stage = 'checkpoint_2'
    
    if 'code_generation_complete' not in st.session_state:
        st.session_state.code_generation_complete = False
    
    if 'code_execution_complete' not in st.session_state:
        st.session_state.code_execution_complete = False
    
    if 'execution_logs' not in st.session_state:
        st.session_state.execution_logs = ""

@handle_errors()
def generate_ml_code(df, target_column, problem_type):
    """Generate ML code using Gemini AI"""
    
    # Check cache first
    code_hash = cache_manager.get_hash(f"{df.shape}_{target_column}_{problem_type}")
    cached_code = cache_manager.load_cached_data(f"code_{code_hash}", "features")
    
    if cached_code:
        st.info("✅ Using cached code generation")
        return cached_code
    
    # Generate new code
    with st.spinner("🧠 AI is writing optimized ML code..."):
        progress_bar = st.progress(0)
        
        progress_bar.progress(25)
        time.sleep(1)  # Simulate processing time
        
        generated_code = generate_code_with_gemini(df, target_column, problem_type)
        
        progress_bar.progress(75)
        
        if generated_code and "import" in generated_code:
            # Cache the generated code
            cache_manager.cache_data(f"code_{code_hash}", generated_code, "features")
            progress_bar.progress(100)
            
            return generated_code
        else:
            st.error("❌ Failed to generate valid ML code. Please try again.")
            return None

@handle_errors()
def create_code_editor_interface(code):
    """Create the enhanced code editor interface"""
    st.markdown("### 💻 Machine Learning Pipeline Code")
    
    create_info_card(
        "AI-Generated Features",
        """
        • <strong>Multi-Algorithm Training:</strong> Tests 7+ different algorithms<br>
        • <strong>Hyperparameter Optimization:</strong> Uses Optuna for automatic tuning<br>
        • <strong>Cross-Validation:</strong> Ensures robust model evaluation<br>
        • <strong>Advanced Metrics:</strong> Comprehensive performance analysis
        """,
        "🤖"
    )
    
    # Code editor
    edited_code = st_ace(
        value=code,
        language='python',
        theme='monokai',
        key="ml_code_editor",
        height=600,
        auto_update=True,
        wrap=False,
        font_size=14,
        tab_size=4,
        show_gutter=True,
        show_print_margin=True
    )
    
    return edited_code

@handle_errors()
def create_code_actions(original_code, edited_code):
    """Create code action buttons"""
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        execute_button = st.button(
            "🚀 Execute ML Pipeline",
            type="primary",
            use_container_width=True,
            help="Train all models with hyperparameter optimization"
        )
    
    with col2:
        if st.button("🔄 Reset Code", use_container_width=True):
            st.session_state.edited_code = original_code
            st.rerun()
    
    with col3:
        if st.button("💾 Save Code", use_container_width=True):
            code_hash = cache_manager.get_hash(edited_code)
            cache_manager.cache_data(f"user_code_{code_hash}", edited_code, "features")
            st.success("✅ Code saved!")
    
    return execute_button

@handle_errors()
def execute_code_with_monitoring(code, df, target_column, problem_type):
    """Execute code with enhanced monitoring and logging"""
    
    st.markdown("### 🚀 Training Progress")
    
    # Create monitoring interface
    progress_container = st.container()
    logs_container = st.container()
    
    with progress_container:
        progress_bar = st.progress(0, text="Initializing ML pipeline...")
        status_text = st.empty()
    
    with logs_container:
        st.markdown("#### 📋 Live Execution Logs")
        log_expander = st.expander("View Detailed Logs", expanded=True)
        log_area = log_expander.empty()
    
    # Clear previous logs
    streamlit_handler.clear_logs()
    
    try:
        def progress_callback(progress, message):
            progress_bar.progress(progress, text=message)
            status_text.info(f"🔄 {message}")
        
        def logs_callback(logs):
            current_logs = streamlit_handler.get_logs()
            combined_logs = f"{current_logs}\n{logs}" if current_logs else logs
            log_area.code(combined_logs, language="text")
            st.session_state.execution_logs = combined_logs
        
        # Execute the ML pipeline
        results = execute_ml_code(
            code,
            df,
            target_column,
            problem_type,
            progress_callback=progress_callback,
            logs_callback=logs_callback
        )
        
        if results:
            progress_bar.progress(100, text="✅ Training completed successfully!")
            status_text.success("🎉 All models trained and evaluated!")
            
            st.session_state.model_results = results
            st.session_state.code_execution_complete = True
            
            return True
        else:
            status_text.error("❌ Training failed - check logs for details")
            return False
            
    except Exception as e:
        progress_bar.progress(0, text="❌ Execution failed")
        status_text.error(f"❌ Error: {str(e)}")
        return False

@handle_errors()
def create_code_explanation_tab(code):
    """Create code explanation tab"""
    st.markdown("### 🤖 AI Code Explanation")
    
    if 'code_explanation' not in st.session_state:
        if st.button("📖 Generate Code Explanation", use_container_width=True):
            with st.spinner("🤖 AI is analyzing the code..."):
                explanation = explain_code_with_gemini(code)
                st.session_state.code_explanation = explanation
                st.rerun()
    else:
        st.markdown(st.session_state.code_explanation)
        
        if st.button("🔄 Refresh Explanation", use_container_width=True):
            with st.spinner("🤖 Generating fresh explanation..."):
                explanation = explain_code_with_gemini(code)
                st.session_state.code_explanation = explanation
                st.rerun()

def create_execution_summary(results):
    """Create execution results summary"""
    if not results:
        return
    
    st.markdown("---")
    st.markdown("### 🎉 Execution Summary")
    
    model_comparison = results.get('model_comparison', [])
    best_model = results.get('best_model', {})
    
    if model_comparison:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Models Trained", len(model_comparison))
        
        with col2:
            st.metric("Best Model", best_model.get('name', 'Unknown'))
        
        with col3:
            problem_type = st.session_state.problem_type
            if problem_type == 'classification':
                score = best_model.get('accuracy', 0)
                st.metric("Best Accuracy", f"{score:.1%}")
            else:
                score = best_model.get('r2_score', 0)
                st.metric("Best R² Score", f"{score:.3f}")
        
        with col4:
            training_time = sum(model.get('training_time', 0) for model in model_comparison)
            st.metric("Total Training Time", f"{training_time:.1f}s")
    
    create_alert("✅ Ready to proceed to model comparison and analysis!", "success")

def main():
    """Main function for code editor page"""
    initialize_code_editor_state()
    
    # Check prerequisites
    if st.session_state.get('cleaned_dataset') is None:
        st.warning("No processed dataset found. Please complete the previous steps.")
        if st.button("← Go to Dataset Preview"):
            st.switch_page("pages/2_Dataset_Preview.py")
        st.stop()
    
    cleaned_df = st.session_state.cleaned_dataset
    target_column = st.session_state.target_column
    problem_type = st.session_state.problem_type
    
    # Header
    st.title("💻 AI Code Generation & Execution")
    st.markdown("Review, customize, and execute the AI-generated machine learning pipeline.")
    
    # Progress tracker
    render_progress_tracker('checkpoint_2')
    
    # Generate or load code
    if not st.session_state.get('generated_code') or not st.session_state.code_generation_complete:
        generated_code = generate_ml_code(cleaned_df, target_column, problem_type)
        
        if generated_code:
            st.session_state.generated_code = generated_code
            st.session_state.edited_code = generated_code
            st.session_state.code_generation_complete = True
            st.success("✅ ML code generated successfully!")
        else:
            st.error("❌ Failed to generate code. Please try refreshing the page.")
            st.stop()
    
    # Main interface tabs
    tab1, tab2, tab3 = st.tabs(["💻 Code Editor", "📖 Code Explanation", "📊 Execution Results"])
    
    with tab1:
        if st.session_state.code_generation_complete:
            edited_code = create_code_editor_interface(st.session_state.edited_code)
            st.session_state.edited_code = edited_code
            
            # Code actions
            execute_button = create_code_actions(st.session_state.generated_code, edited_code)
            
            # Execute code
            if execute_button and not st.session_state.code_execution_complete:
                success = execute_code_with_monitoring(
                    edited_code, cleaned_df, target_column, problem_type
                )
                
                if success:
                    st.balloons()
                    time.sleep(2)
                    st.switch_page("pages/4_Model_Comparison.py")
        else:
            st.info("Code generation in progress...")
    
    with tab2:
        if st.session_state.code_generation_complete:
            create_code_explanation_tab(st.session_state.edited_code)
        else:
            st.info("Generate code first to see explanation.")
    
    with tab3:
        if st.session_state.code_execution_complete and st.session_state.get('model_results'):
            create_execution_summary(st.session_state.model_results)
            
            col1, col2 = st.columns([1, 1])
            with col1:
                if st.button("📊 View Model Comparison →", type="primary", use_container_width=True):
                    st.switch_page("pages/4_Model_Comparison.py")
            
            with col2:
                if st.button("🔄 Re-run Training", use_container_width=True):
                    st.session_state.code_execution_complete = False
                    st.rerun()
        else:
            if st.session_state.execution_logs:
                st.markdown("#### 📋 Previous Execution Logs")
                st.code(st.session_state.execution_logs, language="text")
            else:
                st.info("Execute the code to see results here.")
    
    # Navigation
    st.markdown("---")
    if st.button("← Back to Dataset Preview"):
        st.switch_page("pages/2_Dataset_Preview.py")
    
    # Sidebar
    sidebar_navigation()

if __name__ == "__main__":
    main()
