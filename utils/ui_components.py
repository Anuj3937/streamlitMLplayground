import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from typing import Dict, Any, List, Optional

def inject_custom_css():
    """Inject modern CSS styling"""
    st.markdown("""
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Hide default Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {visibility: hidden;}
    
    /* Root variables */
    :root {
        --primary-color: #2E86AB;
        --secondary-color: #A23B72;
        --accent-color: #F18F01;
        --success-color: #28a745;
        --warning-color: #ffc107;
        --danger-color: #dc3545;
        --background-color: #F8F9FA;
        --surface-color: #FFFFFF;
        --text-primary: #2C3E50;
        --text-secondary: #6C757D;
        --border-color: #E9ECEF;
        --shadow: 0 2px 4px rgba(0,0,0,0.1);
        --border-radius: 8px;
    }
    
    /* Global styles */
    .stApp {
        font-family: 'Inter', sans-serif;
    }
    
    /* Custom containers */
    .metric-container {
        background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
        color: white;
        padding: 1.5rem;
        border-radius: var(--border-radius);
        margin: 0.5rem 0;
        box-shadow: var(--shadow);
        transition: transform 0.3s ease;
    }
    
    .metric-container:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
    }
    
    .info-card {
        background: var(--surface-color);
        border: 1px solid var(--border-color);
        border-radius: var(--border-radius);
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: var(--shadow);
    }
    
    .info-header {
        display: flex;
        align-items: center;
        margin-bottom: 1rem;
        border-bottom: 1px solid var(--border-color);
        padding-bottom: 0.5rem;
    }
    
    .info-icon {
        font-size: 1.5rem;
        margin-right: 0.5rem;
    }
    
    .insight-box {
        background: var(--background-color);
        border-left: 4px solid var(--accent-color);
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0 var(--border-radius) var(--border-radius) 0;
    }
    
    /* Enhanced sidebar styling */
    .css-1d391kg {
        background-color: var(--background-color);
    }
    
    .sidebar-nav {
        background: white;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        box-shadow: var(--shadow);
    }
    
    /* Progress indicators */
    .progress-step {
        display: flex;
        align-items: center;
        padding: 0.5rem;
        margin: 0.25rem 0;
        border-radius: var(--border-radius);
        transition: all 0.3s ease;
    }
    
    .progress-step.completed {
        background: var(--success-color);
        color: white;
    }
    
    .progress-step.active {
        background: var(--primary-color);
        color: white;
    }
    
    .progress-step.pending {
        background: var(--background-color);
        color: var(--text-secondary);
    }
    
    /* Button enhancements */
    .stButton > button {
        border-radius: var(--border-radius);
        border: none;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
    }
    
    /* Alert boxes */
    .alert-success {
        background: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 1rem;
        border-radius: var(--border-radius);
        margin: 1rem 0;
    }
    
    .alert-warning {
        background: #fff3cd;
        border: 1px solid #ffeaa7;
        color: #856404;
        padding: 1rem;
        border-radius: var(--border-radius);
        margin: 1rem 0;
    }
    
    .alert-error {
        background: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        padding: 1rem;
        border-radius: var(--border-radius);
        margin: 1rem 0;
    }
    
    /* Animation effects */
    .fade-in {
        animation: fadeIn 0.6s ease-in;
    }
    
    @keyframes fadeIn {
        from { 
            opacity: 0; 
            transform: translateY(20px); 
        }
        to { 
            opacity: 1; 
            transform: translateY(0); 
        }
    }
    
    /* Radio button styling for navigation */
    .stRadio > div {
        background: white;
        border-radius: 10px;
        padding: 1rem;
        box-shadow: var(--shadow);
    }
    
    .stRadio > div > div {
        gap: 0.5rem;
    }
    </style>
    """, unsafe_allow_html=True)

def create_metric_card(title: str, value: str, delta: str = None, delta_color: str = "normal") -> None:
    """Create an enhanced metric card"""
    delta_colors = {
        "normal": "#666",
        "positive": "#28a745", 
        "negative": "#dc3545"
    }
    
    delta_html = ""
    if delta:
        color = delta_colors.get(delta_color, delta_colors["normal"])
        delta_html = f'<div style="color: {color}; font-size: 0.9rem; margin-top: 0.25rem;">({delta})</div>'
    
    st.markdown(f"""
    <div class="metric-container fade-in">
        <div style="font-size: 0.9rem; opacity: 0.9; margin-bottom: 0.5rem;">{title}</div>
        <div style="font-size: 2rem; font-weight: 600; margin-bottom: 0.25rem;">{value}</div>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)

def create_info_card(title: str, content: str, icon: str = "ℹ️") -> None:
    """Create an informational card"""
    st.markdown(f"""
    <div class="info-card fade-in">
        <div class="info-header">
            <span class="info-icon">{icon}</span>
            <h3 style="margin: 0; color: var(--text-primary);">{title}</h3>
        </div>
        <div class="info-content" style="color: var(--text-secondary);">
            {content}
        </div>
    </div>
    """, unsafe_allow_html=True)

def create_alert(message: str, alert_type: str = "info") -> None:
    """Create styled alert boxes"""
    alert_classes = {
        "success": "alert-success",
        "warning": "alert-warning", 
        "error": "alert-error",
        "info": "info-card"
    }
    
    icons = {
        "success": "✅",
        "warning": "⚠️",
        "error": "❌", 
        "info": "ℹ️"
    }
    
    class_name = alert_classes.get(alert_type, "info-card")
    icon = icons.get(alert_type, "ℹ️")
    
    st.markdown(f"""
    <div class="{class_name} fade-in">
        <span style="margin-right: 0.5rem;">{icon}</span>
        {message}
    </div>
    """, unsafe_allow_html=True)

def render_progress_tracker(current_stage: str) -> None:
    """Simple progress tracker using Streamlit widgets"""
    st.markdown("### 🔄 ML Pipeline Progress")
    
    stages = [
        ("🏠", "Welcome", "welcome"),
        ("📁", "Data Input", "data_input"),
        ("🔍", "Preview", "checkpoint_1"), 
        ("💻", "Code Gen", "checkpoint_2"),
        ("📊", "Comparison", "checkpoint_3"),
        ("🎯", "Results", "results"),
        ("🎮", "Deploy", "demo_app")
    ]
    
    try:
        current_idx = [s[2] for s in stages].index(current_stage)
    except ValueError:
        current_idx = 0
    
    # Progress bar
    progress = (current_idx + 1) / len(stages)
    st.progress(progress, text=f"Step {current_idx + 1}/{len(stages)}: {stages[current_idx][1]}")
    
    # Step indicators in columns
    cols = st.columns(len(stages))
    for i, (icon, name, _) in enumerate(stages):
        with cols[i]:
            if i < current_idx:
                st.markdown(f"✅ {icon}")
                st.caption(f"~~{name}~~")
            elif i == current_idx:
                st.markdown(f"🔄 {icon}")
                st.caption(f"**{name}**")
            else:
                st.markdown(f"⏳ {icon}")
                st.caption(name)

def sidebar_navigation() -> None:
    """Enhanced sidebar navigation with proper Streamlit widgets"""
    with st.sidebar:
        st.markdown("## 🔘 Navigation")
        
        # Get current stage
        current_stage = st.session_state.get('workflow_stage', 'welcome')
        
        # Navigation options
        nav_options = [
            ("🏠", "Welcome", "welcome", "app.py"),
            ("📁", "Data Input", "data_input", "pages/1_Data_Input.py"),
            ("🔍", "Dataset Preview", "checkpoint_1", "pages/2_Dataset_Preview.py"),
            ("💻", "Code Editor", "checkpoint_2", "pages/3_Code_Editor.py"),
            ("📊", "Model Comparison", "checkpoint_3", "pages/4_Model_Comparison.py"),
            ("🎯", "Results Dashboard", "results", "pages/5_Results_Dashboard.py"),
            ("🎮", "Demo Generator", "demo_app", "pages/6_Demo_Generator.py")
        ]
        
        # Create navigation menu
        nav_labels = [f"{icon} {name}" for icon, name, _, _ in nav_options]
        nav_keys = [key for _, _, key, _ in nav_options]
        nav_pages = [page for _, _, _, page in nav_options]
        
        try:
            current_index = nav_keys.index(current_stage)
        except ValueError:
            current_index = 0
        
        # Custom styling for radio buttons
        st.markdown('<div class="sidebar-nav">', unsafe_allow_html=True)
        
        selected = st.radio(
            "Navigate to:",
            nav_labels,
            index=current_index,
            key="sidebar_nav",
            label_visibility="collapsed"
        )
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Handle navigation
        selected_index = nav_labels.index(selected)
        selected_key = nav_keys[selected_index]
        selected_page = nav_pages[selected_index]
        
        if selected_key != current_stage:
            st.switch_page(selected_page)
        
        st.markdown("---")
        
        # Workflow status
        st.markdown("### 📊 Workflow Status")
        
        # Progress indicators
        progress_stages = [
            ("welcome", "Welcome", "✅"),
            ("data_input", "Data Input", "✅" if st.session_state.get('dataset') is not None else "⏳"),
            ("checkpoint_1", "Data Processing", "✅" if st.session_state.get('data_validation_passed') else "⏳"),
            ("checkpoint_2", "Code Generation", "✅" if st.session_state.get('generated_code') else "⏳"),
            ("checkpoint_3", "Model Training", "✅" if st.session_state.get('model_results') else "⏳"),
            ("results", "Results Analysis", "✅" if st.session_state.get('best_model') else "⏳"),
            ("demo_app", "Demo Ready", "✅" if st.session_state.get('demo_generated') else "⏳")
        ]
        
        for stage_key, stage_name, status in progress_stages:
            if stage_key == current_stage:
                st.markdown(f"🔄 **{stage_name}**")
            else:
                st.markdown(f"{status} {stage_name}")
        
        st.markdown("---")
        
        # Dataset info
        if st.session_state.get('dataset') is not None:
            st.markdown("### 📈 Dataset Info")
            df = st.session_state.dataset
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Rows", f"{len(df):,}")
                st.metric("Columns", f"{len(df.columns):,}")
            
            with col2:
                if st.session_state.get('target_column'):
                    st.metric("Target", st.session_state.target_column)
                if st.session_state.get('problem_type'):
                    st.metric("Type", st.session_state.problem_type.title())
        
        # Model info
        if st.session_state.get('best_model'):
            st.markdown("### 🏆 Best Model")
            best_model = st.session_state.best_model
            
            st.metric("Algorithm", best_model.get('name', 'Unknown'))
            
            if st.session_state.get('problem_type') == 'classification':
                acc = best_model.get('accuracy', 0)
                st.metric("Accuracy", f"{acc:.1%}")
            else:
                r2 = best_model.get('r2_score', 0)
                st.metric("R² Score", f"{r2:.3f}")
        
        st.markdown("---")
        
        # Quick actions
        st.markdown("### ⚡ Quick Actions")
        
        if st.button("🔄 Reset Workflow", use_container_width=True, help="Clear all data and start over"):
            # Clear session state
            keys_to_clear = [
                'dataset', 'target_column', 'problem_type', 'generated_code',
                'model_results', 'best_model', 'data_validation_passed'
            ]
            for key in keys_to_clear:
                if key in st.session_state:
                    del st.session_state[key]
            st.session_state.workflow_stage = 'welcome'
            st.switch_page("app.py")
        
        if st.session_state.get('dataset') is not None:
            if st.button("💾 Export Data", use_container_width=True, help="Download current dataset"):
                df = st.session_state.dataset
                csv = df.to_csv(index=False)
                st.download_button(
                    label="📥 Download CSV",
                    data=csv,
                    file_name="exported_dataset.csv",
                    mime="text/csv",
                    use_container_width=True
                )

def create_comparison_table(df: pd.DataFrame, highlight_cols: List[str] = None) -> None:
    """Create an enhanced comparison table"""
    if highlight_cols is None:
        highlight_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    
    # Style the dataframe
    styled_df = df.style.format(precision=4)
    
    # Highlight best values
    for col in highlight_cols:
        if col in df.columns:
            if df[col].dtype in ['float64', 'int64']:
                styled_df = styled_df.highlight_max(subset=[col], color='lightgreen')
    
    st.dataframe(styled_df, use_container_width=True)

def create_expandable_code(code: str, title: str = "Code", language: str = "python") -> None:
    """Create expandable code section"""
    with st.expander(f"📝 {title}"):
        st.code(code, language=language)

def create_loading_animation(text: str = "Processing...") -> Any:
    """Create a loading animation context manager"""
    return st.spinner(text)

def display_feature_importance(feature_names: List[str], importance_values: List[float], top_n: int = 10) -> None:
    """Display feature importance with enhanced styling"""
    if not feature_names or not importance_values:
        st.warning("No feature importance data available")
        return
    
    # Create DataFrame
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance_values
    }).sort_values('Importance', ascending=False).head(top_n)
    
    # Create horizontal bar chart
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=importance_df['Importance'],
        y=importance_df['Feature'],
        orientation='h',
        marker=dict(
            color=importance_df['Importance'],
            colorscale='Viridis',
            colorbar=dict(title="Importance")
        )
    ))
    
    fig.update_layout(
        title=f"Top {top_n} Feature Importance",
        xaxis_title="Importance Score",
        yaxis_title="Features",
        height=max(400, len(importance_df) * 30),
        yaxis={'categoryorder': 'total ascending'}
    )
    
    st.plotly_chart(fig, use_container_width=True)
