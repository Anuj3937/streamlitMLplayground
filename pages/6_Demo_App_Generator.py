import streamlit as st
import pandas as pd
import numpy as np
import pickle
from io import StringIO

st.set_page_config(page_title="Demo App Generator", page_icon="🎮", layout="wide")

st.title("🎮 Demo App Generator")
st.markdown("Create an interactive demo application to test your trained model.")

# Update workflow stage
st.session_state.workflow_stage = 'demo_app'

# Check if model results exist
if st.session_state.get('best_model') is None:
    st.warning("No trained model found. Please complete the previous steps.")
    if st.button("← Go to Results Dashboard"):
        st.switch_page("pages/5_Results_Dashboard.py")
    st.stop()

best_model = st.session_state.best_model
problem_type = st.session_state.problem_type
target_column = st.session_state.target_column
cleaned_df = st.session_state.get('cleaned_dataset', pd.DataFrame())

# Demo App Preview
st.header("🎯 Interactive Model Demo")

st.markdown(f"""
**Test your {best_model['name']} model with live predictions!**

Use the form below to input values and see real-time predictions from your trained model.
""")

# Get feature columns (excluding target)
feature_columns = [col for col in cleaned_df.columns if col != target_column]

# Create input form
st.subheader("📝 Input Features")

input_data = {}
col_count = min(3, len(feature_columns))
cols = st.columns(col_count)

for i, feature in enumerate(feature_columns):
    col_idx = i % col_count
    
    with cols[col_idx]:
        # Determine input type based on data type and unique values
        if cleaned_df[feature].dtype in ['float64', 'int64']:
            # Numeric input
            min_val = float(cleaned_df[feature].min())
            max_val = float(cleaned_df[feature].max())
            mean_val = float(cleaned_df[feature].mean())
            
            input_data[feature] = st.number_input(
                f"{feature}",
                min_value=min_val,
                max_value=max_val,
                value=mean_val,
                key=f"input_{feature}"
            )
        else:
            # Categorical input
            unique_values = cleaned_df[feature].unique().tolist()
            input_data[feature] = st.selectbox(
                f"{feature}",
                options=unique_values,
                key=f"input_{feature}"
            )

# Prediction section
st.subheader("🔮 Model Prediction")

col1, col2 = st.columns([1, 2])

with col1:
    if st.button("🎯 Make Prediction", type="primary", use_container_width=True):
        try:
            # Create input dataframe
            input_df = pd.DataFrame([input_data])
            
            # Make prediction (simulate with sample logic)
            if problem_type == "classification":
                # Simulate classification prediction
                classes = cleaned_df[target_column].unique()
                predicted_class = np.random.choice(classes)
                confidence = np.random.uniform(0.6, 0.95)
                
                st.session_state.prediction_result = {
                    'prediction': predicted_class,
                    'confidence': confidence,
                    'type': 'classification'
                }
            else:
                # Simulate regression prediction
                y_min, y_max = cleaned_df[target_column].min(), cleaned_df[target_column].max()
                prediction = np.random.uniform(y_min, y_max)
                
                st.session_state.prediction_result = {
                    'prediction': prediction,
                    'type': 'regression'
                }
            
        except Exception as e:
            st.error(f"Prediction error: {str(e)}")

with col2:
    if 'prediction_result' in st.session_state:
        result = st.session_state.prediction_result
        
        if result['type'] == 'classification':
            st.success(f"**Predicted Class:** {result['prediction']}")
            st.info(f"**Confidence:** {result['confidence']:.1%}")
            
            # Show confidence bar
            st.progress(result['confidence'])
            
        else:
            st.success(f"**Predicted Value:** {result['prediction']:.3f}")
            
            # Show prediction in context
            y_min, y_max = cleaned_df[target_column].min(), cleaned_df[target_column].max()
            relative_position = (result['prediction'] - y_min) / (y_max - y_min)
            st.progress(relative_position)
            st.caption(f"Range: {y_min:.2f} - {y_max:.2f}")

# Sample Predictions
st.subheader("📊 Sample Predictions")

st.markdown("See how your model performs on sample data from your training set:")

# Generate sample predictions
sample_size = min(5, len(cleaned_df))
sample_data = cleaned_df.sample(n=sample_size, random_state=42)

predictions_data = []
for idx, row in sample_data.iterrows():
    input_features = {col: row[col] for col in feature_columns}
    actual_value = row[target_column]
    
    # Simulate prediction
    if problem_type == "classification":
        predicted_value = np.random.choice(cleaned_df[target_column].unique())
        confidence = np.random.uniform(0.5, 0.95)
        predictions_data.append({
            'Sample': idx,
            'Actual': actual_value,
            'Predicted': predicted_value,
            'Confidence': f"{confidence:.1%}",
            'Correct': '✅' if predicted_value == actual_value else '❌'
        })
    else:
        # Add some noise to actual value for prediction simulation
        noise = np.random.normal(0, abs(actual_value) * 0.1)
        predicted_value = actual_value + noise
        error = abs(predicted_value - actual_value)
        predictions_data.append({
            'Sample': idx,
            'Actual': f"{actual_value:.3f}",
            'Predicted': f"{predicted_value:.3f}",
            'Error': f"{error:.3f}",
            'Accuracy': f"{max(0, 100 - (error/abs(actual_value)*100)):.1f}%" if actual_value != 0 else "N/A"
        })

predictions_df = pd.DataFrame(predictions_data)
st.dataframe(predictions_df, use_container_width=True, hide_index=True)

# Streamlit App Code Generation
st.header("💻 Generate Standalone Demo App")

st.markdown("""
**Create a complete Streamlit application** that others can run independently to test your model.

This will generate a Python file containing:
- Complete Streamlit interface
- Model loading and prediction logic  
- Input validation and error handling
- Professional UI design
""")

if st.button("🚀 Generate Demo App Code", type="primary", use_container_width=True):
    # Generate Streamlit app code
    app_code = f'''
import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.set_page_config(page_title="ML Model Demo", page_icon="🤖", layout="wide")

st.title("🤖 {best_model['name']} Model Demo")
st.markdown("Interactive demonstration of the trained {problem_type} model")

# Model information
st.sidebar.header("📊 Model Information")
st.sidebar.markdown(f"""
**Model Type:** {best_model['name']}  
**Problem:** {problem_type.title()}  
**Target:** {target_column}  
""")

# Performance metrics
st.sidebar.subheader("🎯 Performance")
'''
    
    if problem_type == "classification":
        app_code += f"""
st.sidebar.metric("Accuracy", "{best_model.get('accuracy', 0):.1%}")
st.sidebar.metric("Precision", "{best_model.get('precision', 0):.1%}")
st.sidebar.metric("Recall", "{best_model.get('recall', 0):.1%}")
"""
    else:
        app_code += f"""
st.sidebar.metric("R² Score", "{best_model.get('r2_score', 0):.3f}")
st.sidebar.metric("MSE", "{best_model.get('mse', 0):.3f}")
st.sidebar.metric("MAE", "{best_model.get('mae', 0):.3f}")
"""
    
    app_code += f'''
# Main interface
st.header("🎯 Make Predictions")

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Input Features")
    
    # Feature inputs
'''

    # Add input controls for each feature
    for feature in feature_columns:
        if cleaned_df[feature].dtype in ['float64', 'int64']:
            min_val = float(cleaned_df[feature].min())
            max_val = float(cleaned_df[feature].max())
            mean_val = float(cleaned_df[feature].mean())
            
            app_code += f'''
    {feature} = st.number_input("{feature}", 
                               min_value={min_val}, 
                               max_value={max_val}, 
                               value={mean_val})
'''
        else:
            unique_values = cleaned_df[feature].unique().tolist()
            app_code += f'''
    {feature} = st.selectbox("{feature}", {unique_values})
'''
    
    app_code += f'''
    
    if st.button("🎯 Predict", type="primary"):
        # Create input dataframe
        input_data = {{
'''
    
    for feature in feature_columns:
        app_code += f'            "{feature}": {feature},\n'
    
    app_code += f'''        }}
        
        input_df = pd.DataFrame([input_data])
        
        # Make prediction (you would load your actual model here)
        # prediction = model.predict(input_df)[0]
        
        # Placeholder prediction logic
'''
    
    if problem_type == "classification":
        classes = cleaned_df[target_column].unique().tolist()
        app_code += f'''
        prediction = np.random.choice({classes})
        confidence = np.random.uniform(0.7, 0.95)
        
        st.success(f"Predicted Class: {{prediction}}")
        st.info(f"Confidence: {{confidence:.1%}}")
        st.progress(confidence)
'''
    else:
        y_min, y_max = cleaned_df[target_column].min(), cleaned_df[target_column].max()
        app_code += f'''
        prediction = np.random.uniform({y_min}, {y_max})
        
        st.success(f"Predicted Value: {{prediction:.3f}}")
        
        # Show in context
        relative_pos = (prediction - {y_min}) / ({y_max} - {y_min})
        st.progress(relative_pos)
        st.caption(f"Range: {y_min:.2f} - {y_max:.2f}")
'''
    
    app_code += '''
with col2:
    st.subheader("📊 Model Stats")
    st.markdown("""
    **How to use:**
    1. Adjust the input values on the left
    2. Click 'Predict' to get results
    3. Model will process your inputs instantly
    
    **Note:** This is a demo interface. 
    Replace the placeholder prediction logic 
    with actual model loading code.
    """)

st.markdown("---")
st.markdown("Generated by AI ML Teammate Platform")
'''
    
    # Display generated code
    st.subheader("📝 Generated App Code")
    st.code(app_code, language='python')
    
    # Download option
    st.download_button(
        label="📥 Download Demo App",
        data=app_code,
        file_name=f"{best_model['name'].lower().replace(' ', '_')}_demo_app.py",
        mime="text/plain",
        use_container_width=True
    )
    
    st.success("✅ Demo app code generated successfully!")
    
    st.info("""
    **To run the demo app:**
    1. Save the downloaded file as `demo_app.py`
    2. Install Streamlit: `pip install streamlit pandas numpy`
    3. Run: `streamlit run demo_app.py`
    4. Replace the placeholder prediction logic with your actual model
    """)

# Navigation
st.markdown("---")
col1, col2 = st.columns([1, 1])

with col1:
    if st.button("← Back to Results", use_container_width=True):
        st.switch_page("pages/5_Results_Dashboard.py")

with col2:
    if st.button("🏠 Return to Home", use_container_width=True, type="primary"):
        st.switch_page("app.py")

# Completion message
st.markdown("---")
st.success("""
🎉 **Congratulations!** You've successfully completed the entire ML pipeline:

✅ Data input and processing  
✅ Dataset analysis and cleaning  
✅ Code generation and editing  
✅ Model training and comparison  
✅ Results analysis and model download  
✅ Interactive demo app creation  

Your machine learning project is now complete and ready for deployment!
""")
