# utils/interactive_synthetic_generator.py

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, List, Any
import json
from .gemini_integration import get_gemini_model, configure_gemini
from .error_handler import handle_errors

class InteractiveSyntheticDataGenerator:
    """Interactive chat-like synthetic data generator using AI"""
    
    def __init__(self):
        self.conversation_history = []
        self.dataset_config = {}
        
    @handle_errors()
    def run_interactive_generator(self):
        """Main interactive interface"""
        st.title("🤖 AI-Powered Synthetic Data Generator")
        st.markdown("Tell me about the dataset you want to create, and I'll generate it for you!")
        
        # Initialize session state
        if 'generator_stage' not in st.session_state:
            st.session_state.generator_stage = 'description'
        if 'dataset_description' not in st.session_state:
            st.session_state.dataset_description = ""
        if 'feature_configs' not in st.session_state:
            st.session_state.feature_configs = []
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        
        # Chat interface tabs
        tab1, tab2, tab3 = st.tabs(["💬 Dataset Designer", "🔧 Feature Configuration", "📊 Generate Data"])
        
        with tab1:
            self.create_chat_interface()
        
        with tab2:
            self.create_feature_configuration()
        
        with tab3:
            self.generate_synthetic_data()
    
    def create_chat_interface(self):
        """Create conversational interface for dataset design"""
        st.markdown("### 💬 Describe Your Dataset")
        
        # Display chat history
        if st.session_state.chat_history:
            st.markdown("**Conversation History:**")
            for i, msg in enumerate(st.session_state.chat_history):
                if msg['role'] == 'user':
                    st.markdown(f"**You:** {msg['content']}")
                else:
                    st.markdown(f"**AI:** {msg['content']}")
        
        # Chat input
        user_input = st.chat_input("Describe the dataset you want to create...")
        
        if user_input:
            # Add user message to history
            st.session_state.chat_history.append({'role': 'user', 'content': user_input})
            
            # Process with AI
            ai_response = self.process_user_input(user_input)
            st.session_state.chat_history.append({'role': 'assistant', 'content': ai_response})
            st.rerun()
        
        # Quick start templates
        st.markdown("### 🚀 Quick Start Templates")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📊 Business Analytics Dataset"):
                self.load_template("business_analytics")
        
        with col2:
            if st.button("🏥 Healthcare Dataset"):
                self.load_template("healthcare")
        
        with col3:
            if st.button("💰 Financial Dataset"):
                self.load_template("financial")
    
    @handle_errors()
    def process_user_input(self, user_input: str) -> str:
        """Process user input with AI to extract dataset requirements"""
        configure_gemini()
        model = get_gemini_model()
        
        prompt = f"""
You are an expert data scientist helping to design synthetic datasets. 
Analyze this user request: "{user_input}"

Extract and suggest:
1. Problem type (classification/regression)
2. Target variable name and characteristics
3. Feature suggestions with data types and statistical properties
4. Number of samples recommended
5. Any domain-specific considerations

Respond in a helpful, conversational tone. Ask clarifying questions if needed.
If the user provides specific statistical requirements, acknowledge and incorporate them.
"""
        
        try:
            response = model.generate_content(prompt)
            
            # Also try to extract structured information
            self.extract_dataset_config(user_input, response.text)
            
            return response.text
            
        except Exception as e:
            return f"I'm having trouble understanding your request. Could you please rephrase it? (Error: {str(e)})"
    
    def extract_dataset_config(self, user_input: str, ai_response: str):
        """Extract structured dataset configuration from conversation"""
        configure_gemini()
        model = get_gemini_model()
        
        extraction_prompt = f"""
Based on this conversation about dataset requirements:
User: "{user_input}"
AI Response: "{ai_response}"

Extract structured information in JSON format:
{{
    "problem_type": "classification" or "regression",
    "target_variable": {{
        "name": "target_name",
        "type": "categorical" or "numeric",
        "properties": {{"classes": [...] or "mean": X, "std": Y}}
    }},
    "features": [
        {{
            "name": "feature_name",
            "type": "numeric" or "categorical",
            "properties": {{"mean": X, "std": Y}} or {{"categories": [...], "probabilities": [...]}}
        }}
    ],
    "sample_size": 1000,
    "domain": "business/healthcare/finance/other"
}}

Only return valid JSON. If information is missing, use reasonable defaults.
"""
        
        try:
            response = model.generate_content(extraction_prompt)
            config_text = response.text.strip()
            
            # Clean up JSON response
            if config_text.startswith('```'):
                config_text = config_text[7:]
            if config_text.endswith('```'):
                config_text = config_text[:-3]
            
            config = json.loads(config_text)
            st.session_state.dataset_config = config
            
        except Exception as e:
            st.warning(f"Could not extract structured config: {str(e)}")
    
    def create_feature_configuration(self):
        """Create detailed feature configuration interface"""
        st.markdown("### 🔧 Feature Configuration")
        
        if not st.session_state.dataset_config:
            st.info("💡 Please describe your dataset in the chat first, or use a quick start template.")
            return
        
        config = st.session_state.dataset_config
        
        # Problem type selection
        problem_type = st.selectbox(
            "Problem Type",
            ["classification", "regression"],
            index=0 if config.get('problem_type') == 'classification' else 1
        )
        
        # Sample size
        sample_size = st.number_input(
            "Number of samples",
            min_value=100,
            max_value=100000,
            value=config.get('sample_size', 1000)
        )
        
        # Target variable configuration
        st.markdown("#### 🎯 Target Variable")
        target_config = config.get('target_variable', {})
        
        col1, col2 = st.columns(2)
        with col1:
            target_name = st.text_input("Target name", value=target_config.get('name', 'target'))
        
        with col2:
            if problem_type == 'classification':
                num_classes = st.number_input("Number of classes", min_value=2, max_value=20, value=2)
            else:
                target_mean = st.number_input("Target mean", value=target_config.get('properties', {}).get('mean', 0.0))
                target_std = st.number_input("Target std", value=target_config.get('properties', {}).get('std', 1.0))
        
        # Features configuration
        st.markdown("#### 📊 Feature Configuration")
        
        features = config.get('features', [])
        if not features:
            features = [{'name': 'feature_1', 'type': 'numeric', 'properties': {'mean': 0, 'std': 1}}]
        
        # Dynamic feature creation
        num_features = st.number_input("Number of features", min_value=1, max_value=50, value=len(features))
        
        configured_features = []
        for i in range(num_features):
            st.markdown(f"**Feature {i+1}:**")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                feat_name = st.text_input(f"Name", value=features[i]['name'] if i < len(features) else f"feature_{i+1}", key=f"feat_name_{i}")
            
            with col2:
                feat_type = st.selectbox(f"Type", ["numeric", "categorical"], 
                                       index=0 if features[i].get('type') == 'numeric' else 1 if i < len(features) else 0, key=f"feat_type_{i}")
            
            if feat_type == "numeric":
                with col3:
                    feat_mean = st.number_input(f"Mean", value=features[i].get('properties', {}).get('mean', 0.0) if i < len(features) else 0.0, key=f"feat_mean_{i}")
                with col4:
                    feat_std = st.number_input(f"Std Dev", value=features[i].get('properties', {}).get('std', 1.0) if i < len(features) else 1.0, key=f"feat_std_{i}")
                
                configured_features.append({
                    'name': feat_name,
                    'type': 'numeric',
                    'properties': {'mean': feat_mean, 'std': feat_std}
                })
            else:
                with col3:
                    categories_text = st.text_input(f"Categories (comma-separated)", 
                                                  value=",".join(features[i].get('properties', {}).get('categories', ['A', 'B', 'C'])) if i < len(features) else "A,B,C", 
                                                  key=f"feat_cats_{i}")
                with col4:
                    probs_text = st.text_input(f"Probabilities", value="equal", key=f"feat_probs_{i}")
                
                categories = [c.strip() for c in categories_text.split(',')]
                if probs_text == "equal":
                    probabilities = [1.0/len(categories)] * len(categories)
                else:
                    try:
                        probabilities = [float(p.strip()) for p in probs_text.split(',')]
                        if abs(sum(probabilities) - 1.0) > 0.01:
                            probabilities = [1.0/len(categories)] * len(categories)
                    except:
                        probabilities = [1.0/len(categories)] * len(categories)
                
                configured_features.append({
                    'name': feat_name,
                    'type': 'categorical',
                    'properties': {'categories': categories, 'probabilities': probabilities}
                })
        
        # Save configuration
        if st.button("💾 Save Configuration"):
            updated_config = {
                'problem_type': problem_type,
                'sample_size': sample_size,
                'target_variable': {
                    'name': target_name,
                    'type': 'categorical' if problem_type == 'classification' else 'numeric',
                    'properties': {'num_classes': num_classes} if problem_type == 'classification' else {'mean': target_mean, 'std': target_std}
                },
                'features': configured_features
            }
            
            st.session_state.dataset_config = updated_config
            st.success("✅ Configuration saved!")
    
    def generate_synthetic_data(self):
        """Generate synthetic dataset based on configuration"""
        st.markdown("### 📊 Generate Synthetic Data")
        
        if not st.session_state.dataset_config:
            st.info("💡 Please configure your dataset first.")
            return
        
        config = st.session_state.dataset_config
        
        # Display configuration summary
        st.markdown("#### 📋 Configuration Summary")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Problem Type", config.get('problem_type', 'Unknown').title())
            st.metric("Sample Size", config.get('sample_size', 0))
        
        with col2:
            st.metric("Target Variable", config.get('target_variable', {}).get('name', 'Unknown'))
            st.metric("Number of Features", len(config.get('features', [])))
        
        # Generation options
        st.markdown("#### ⚙️ Generation Options")
        
        col1, col2 = st.columns(2)
        with col1:
            add_noise = st.checkbox("Add realistic noise", value=True)
            correlations = st.checkbox("Add feature correlations", value=False)
        
        with col2:
            random_seed = st.number_input("Random seed (for reproducibility)", value=42, min_value=0)
        
        # Generate button
        if st.button("🚀 Generate Synthetic Dataset", type="primary"):
            with st.spinner("🔄 Generating synthetic data..."):
                df = self.create_dataset_from_config(config, add_noise, correlations, random_seed)
                
                if df is not None:
                    st.success(f"✅ Generated synthetic dataset with {len(df)} samples and {len(df.columns)} columns!")
                    
                    # Store in session state
                    st.session_state.dataset = df
                    st.session_state.target_column = config['target_variable']['name']
                    st.session_state.problem_type = config['problem_type']
                    st.session_state.data_source = "interactive_synthetic"
                    
                    # Display preview
                    st.markdown("#### 👀 Dataset Preview")
                    st.dataframe(df.head(10), use_container_width=True)
                    
                    # Dataset statistics
                    st.markdown("#### 📊 Dataset Statistics")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB")
                    with col2:
                        missing_pct = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
                        st.metric("Missing Data", f"{missing_pct:.1f}%")
                    with col3:
                        st.metric("Data Types", len(df.dtypes.unique()))
                    
                    # Download options
                    st.markdown("#### 💾 Export Options")
                    
                    csv_data = df.to_csv(index=False)
                    st.download_button(
                        "📥 Download as CSV",
                        csv_data,
                        f"synthetic_dataset_{config['problem_type']}.csv",
                        "text/csv",
                        use_container_width=True
                    )
                    
                    # Continue to next step
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("🔄 Generate New Dataset", use_container_width=True):
                            st.session_state.dataset_config = {}
                            st.rerun()
                    
                    with col2:
                        if st.button("➡️ Continue to ML Pipeline", use_container_width=True, type="primary"):
                            st.switch_page("pages/2_Dataset_Preview.py")
    
    @handle_errors()
    def create_dataset_from_config(self, config: Dict, add_noise: bool, correlations: bool, random_seed: int) -> pd.DataFrame:
        """Create synthetic dataset based on configuration"""
        np.random.seed(random_seed)
        
        sample_size = config['sample_size']
        features = config['features']
        target_config = config['target_variable']
        problem_type = config['problem_type']
        
        data = {}
        
        # Generate features
        for feature in features:
            name = feature['name']
            feat_type = feature['type']
            properties = feature['properties']
            
            if feat_type == 'numeric':
                mean = properties['mean']
                std = properties['std']
                
                # Add noise if requested
                if add_noise:
                    noise_factor = np.random.uniform(0.9, 1.1)
                    std *= noise_factor
                
                data[name] = np.random.normal(mean, std, sample_size)
                
            else:  # categorical
                categories = properties['categories']
                probabilities = properties.get('probabilities', [1.0/len(categories)] * len(categories))
                
                data[name] = np.random.choice(categories, size=sample_size, p=probabilities)
        
        # Generate target variable
        target_name = target_config['name']
        
        if problem_type == 'classification':
            num_classes = target_config['properties'].get('num_classes', 2)
            classes = [f"class_{i}" for i in range(num_classes)]
            data[target_name] = np.random.choice(classes, size=sample_size)
            
        else:  # regression
            mean = target_config['properties']['mean']
            std = target_config['properties']['std']
            
            if add_noise:
                noise_factor = np.random.uniform(0.9, 1.1)
                std *= noise_factor
            
            data[target_name] = np.random.normal(mean, std, sample_size)
        
        # Add correlations if requested
        if correlations and len(features) > 1:
            self.add_feature_correlations(data, features, target_name)
        
        return pd.DataFrame(data)
    
    def add_feature_correlations(self, data: Dict, features: List, target_name: str):
        """Add realistic correlations between features"""
        numeric_features = [f['name'] for f in features if f['type'] == 'numeric']
        
        if len(numeric_features) >= 2:
            # Add some correlation between first two numeric features
            feat1, feat2 = numeric_features[0], numeric_features[1]
            correlation_strength = np.random.uniform(0.3, 0.7)
            
            # Adjust second feature to correlate with first
            data[feat2] = (data[feat2] + correlation_strength * data[feat1]) / (1 + correlation_strength)
    
    def load_template(self, template_name: str):
        """Load predefined templates"""
        templates = {
            'business_analytics': {
                'problem_type': 'classification',
                'sample_size': 5000,
                'target_variable': {
                    'name': 'customer_segment',
                    'type': 'categorical',
                    'properties': {'num_classes': 3}
                },
                'features': [
                    {'name': 'age', 'type': 'numeric', 'properties': {'mean': 35, 'std': 12}},
                    {'name': 'income', 'type': 'numeric', 'properties': {'mean': 50000, 'std': 20000}},
                    {'name': 'spending_score', 'type': 'numeric', 'properties': {'mean': 50, 'std': 25}},
                    {'name': 'region', 'type': 'categorical', 'properties': {'categories': ['North', 'South', 'East', 'West'], 'probabilities': [0.25, 0.25, 0.25, 0.25]}}
                ]
            },
            'healthcare': {
                'problem_type': 'classification',
                'sample_size': 3000,
                'target_variable': {
                    'name': 'diagnosis',
                    'type': 'categorical',
                    'properties': {'num_classes': 2}
                },
                'features': [
                    {'name': 'age', 'type': 'numeric', 'properties': {'mean': 45, 'std': 15}},
                    {'name': 'bmi', 'type': 'numeric', 'properties': {'mean': 25, 'std': 4}},
                    {'name': 'blood_pressure', 'type': 'numeric', 'properties': {'mean': 120, 'std': 20}},
                    {'name': 'gender', 'type': 'categorical', 'properties': {'categories': ['Male', 'Female'], 'probabilities': [0.5, 0.5]}}
                ]
            },
            'financial': {
                'problem_type': 'regression',
                'sample_size': 4000,
                'target_variable': {
                    'name': 'credit_score',
                    'type': 'numeric',
                    'properties': {'mean': 650, 'std': 100}
                },
                'features': [
                    {'name': 'income', 'type': 'numeric', 'properties': {'mean': 60000, 'std': 25000}},
                    {'name': 'debt_to_income', 'type': 'numeric', 'properties': {'mean': 0.3, 'std': 0.15}},
                    {'name': 'employment_length', 'type': 'numeric', 'properties': {'mean': 8, 'std': 5}},
                    {'name': 'loan_purpose', 'type': 'categorical', 'properties': {'categories': ['home', 'auto', 'personal', 'business'], 'probabilities': [0.4, 0.3, 0.2, 0.1]}}
                ]
            }
        }
        
        if template_name in templates:
            st.session_state.dataset_config = templates[template_name]
            st.session_state.chat_history.append({
                'role': 'assistant', 
                'content': f"I've loaded the {template_name.replace('_', ' ').title()} template for you! You can customize it in the Feature Configuration tab."
            })
            st.success(f"✅ Loaded {template_name.replace('_', ' ').title()} template!")
            st.rerun()

# Global generator instance
synthetic_generator = InteractiveSyntheticDataGenerator()
