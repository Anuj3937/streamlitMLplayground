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
    
    /* Progress indicators */
    .progress-tracker {
        display: flex;
        justify-content: space-between;
        margin: 2rem 0;
        padding: 1rem;
        background: var(--surface-color);
        border-radius: var(--border-radius);
        box-shadow: var(--shadow);
    }
    
    .progress-step {
        display: flex;
        flex-direction: column;
        align-items: center;
        padding: 1rem;
        border-radius: var(--border-radius);
        transition: all 0.3s ease;
        flex: 1;
        margin: 0 0.25rem;
        position: relative;
    }
    
    .progress-step:not(:last-child)::after {
        content: '';
        position: absolute;
        top: 50%;
        right: -50%;
        width: 100%;
        height: 2px;
        background: var(--border-color);
        z-index: -1;
    }
    
    .progress-step.completed::after {
        background: var(--success-color);
    }
    
    .progress-step.active {
        background: var(--primary-color);
        color: white;
        transform: scale(1.05);
    }
    
    .progress-step.completed {
        background: var(--success-color);
        color: white;
    }
    
    .progress-step.pending {
        background: var(--background-color);
        color: var(--text-secondary);
    }
    
    .step-icon {
        font-size: 1.5rem;
        margin-bottom: 0.5rem;
    }
    
    .step-name {
        font-size: 0.9rem;
        font-weight: 500;
        text-align: center;
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
    
    .slide-up {
        animation: slideUp 0.5s ease-out;
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
    
    @keyframes slideUp {
        from { 
            transform: translateY(30px); 
            opacity: 0; 
        }
        to { 
            transform: translateY(0); 
            opacity: 1; 
        }
    }
    
    /* Improved dataframe styling */
    .dataframe {
        border: 1px solid var(--border-color);
        border-radius: var(--border-radius);
        overflow: hidden;
    }
    
    /* Sidebar enhancements */
    .css-1d391kg {
        background-color: var(--background-color);
    }
    
    /* Hide Streamlit menu and footer */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
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
    """Render an enhanced progress tracker"""
    stages = [
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
        current_idx = -1
    
    progress_html = '<div class="progress-tracker">'
    
    for i, (icon, name, stage_key) in enumerate(stages):
        if i == current_idx:
            status = "active"
        elif i < current_idx:
            status = "completed"
        else:
            status = "pending"
        
        progress_html += f"""
        <div class="progress-step {status}">
            <div class="step-icon">{icon}</div>
            <div class="step-name">{name}</div>
        </div>
        """
    
    progress_html += "</div>"
    st.markdown(progress_html, unsafe_allow_html=True)

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

def sidebar_navigation() -> None:
    """Enhanced sidebar navigation"""
    with st.sidebar:
        st.markdown("### 📋 Workflow Progress")
        
        # Get current stage
        current_stage = st.session_state.get('workflow_stage', 'welcome')
        
        # Progress stages
        progress_stages = [
            ("🏠", "Welcome", "welcome"),
            ("📁", "Data Input", "data_input"),
            ("🔍", "Dataset Preview", "checkpoint_1"),
            ("💻", "Code Editor", "checkpoint_2"),
            ("📊", "Model Comparison", "checkpoint_3"),
            ("🎯", "Results Dashboard", "results"),
            ("🎮", "Demo Generator", "demo_app")
        ]
        
        for icon, stage_name, stage_key in progress_stages:
            if stage_key == current_stage:
                st.markdown(f"**{icon} {stage_name}** ✅")
            elif stage_key in ['welcome', 'data_input']:
                st.markdown(f"✅ {icon} {stage_name}")
            elif st.session_state.get(f'{stage_key}_completed', False):
                st.markdown(f"✅ {icon} {stage_name}")
            else:
                st.markdown(f"⏳ {icon} {stage_name}")
        
        st.markdown("---")
        
        # Quick stats
        if st.session_state.get('dataset') is not None:
            df = st.session_state.dataset
            st.markdown("### 📊 Dataset Info")
            st.markdown(f"**Rows:** {len(df):,}")
            st.markdown(f"**Columns:** {len(df.columns):,}")
            
            if st.session_state.get('target_column'):
                st.markdown(f"**Target:** {st.session_state.target_column}")
            
            if st.session_state.get('problem_type'):
                st.markdown(f"**Type:** {st.session_state.problem_type.title()}")
        
        # Model info
        if st.session_state.get('best_model'):
            st.markdown("### 🏆 Best Model")
            best_model = st.session_state.best_model
            st.markdown(f"**Algorithm:** {best_model.get('name', 'Unknown')}")
            
            if st.session_state.problem_type == 'classification':
                acc = best_model.get('accuracy', 0)
                st.markdown(f"**Accuracy:** {acc:.1%}")
            else:
                r2 = best_model.get('r2_score', 0)
                st.markdown(f"**R² Score:** {r2:.3f}")

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
