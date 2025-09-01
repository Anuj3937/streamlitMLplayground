# utils/gemini_integration.py

import streamlit as st
import google.generativeai as genai
import pandas as pd

def configure_gemini():
    """Configures the Gemini API key."""
    try:
        genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    except Exception as e:
        st.error(f"Failed to configure Gemini API: {e}. Please ensure your GEMINI_API_KEY is set in .streamlit/secrets.toml")
        st.stop()

def get_gemini_model():
    """Returns an instance of the Gemini Pro model."""
    return genai.GenerativeModel('gemini-1.5-flash-latest')

def get_insights_with_gemini(df: pd.DataFrame, target_column: str):
    """
    Generates deep insights about the dataset using the Gemini API.
    
    Returns:
        A list of insight strings.
    """
    configure_gemini()
    model = get_gemini_model()
    
    prompt = f"""
    As an expert data scientist, analyze the following dataset summary and provide key insights in a bulleted list.

    Dataset Summary (df.describe()):
    {df.describe().to_string()}

    Dataset Head (first 5 rows):
    {df.head().to_string()}

    Target Column: '{target_column}'

    Provide a bulleted list of insights covering:
    - Initial observations and data quality assessment.
    - Important relationships between features and the target variable '{target_column}'.
    - Recommendations for feature engineering that could improve model performance.
    - Any potential sources of bias you can infer from this summary.
    """
    try:
        response = model.generate_content(prompt)
        # Clean up the response and split into a list
        insights = [line.strip() for line in response.text.split('* ') if line.strip()]
        return insights
    except Exception as e:
        st.error(f"Error generating insights with Gemini: {e}")
        return ["Failed to generate AI-powered insights."]

def generate_code_with_gemini(df: pd.DataFrame, target_column: str, problem_type: str):
    """
    Generates an ML pipeline script using Gemini, incorporating advanced techniques.
    """
    configure_gemini()
    model = get_gemini_model()
    
    prompt = f"""
    You are an expert machine learning engineer. Write a complete Python script to find the best ML model for the given dataset.

    Dataset Information:
    - Problem Type: {problem_type}
    - Target Column: '{target_column}'
    - All Columns: {df.columns.tolist()}
    - Data Types:\n{df.dtypes.to_string()}
    - First 3 rows:\n{df.head(3).to_string()}

    The Python script MUST:
    1.  Use pandas and numpy. Import all necessary libraries from scikit-learn and xgboost.
    2.  Define `X` (features) and `y` (target).
    3.  Split the data into training and testing sets (80/20 split, random_state=42).
    4.  Define a list of at least 3 appropriate models for a '{problem_type}' problem. Include RandomForest, XGBoost, and one other model.
    5.  Use Optuna for hyperparameter tuning for each model. The optimization function should perform 3-fold cross-validation. Run the study for a maximum of 15 trials per model.
    6.  Train each model with its best-found hyperparameters.
    7.  Evaluate each tuned model on the test set using appropriate metrics.
    8.  Store all results (model name, metrics, best_params, training_time) in a list of dictionaries called `results`.
    9.  Convert the `results` list into a pandas DataFrame called `results_df`.
    10. Identify the best performing model based on accuracy (for classification) or R-squared score (for regression) and store its name in a variable called `best_model_name`.
    11. Extract the trained model object of the best model and store it in a variable called `best_model_object`.
    12. Create a feature importance dictionary for the best model called `feature_importance`.
    13. Print statements to show progress (e.g., "Training Logistic Regression...").
    14. BE SURE to wrap the entire code in a single Python code block. Do not use markdown backticks like ```python.
    """
    try:
        response = model.generate_content(prompt)
        # Basic cleaning of the generated code
        return response.text.replace("`", "").strip()
    except Exception as e:
        st.error(f"Error generating code with Gemini: {e}")
        return "# Failed to generate code."

def explain_code_with_gemini(code: str):
    """
    Explains a given Python script using Gemini.
    """
    configure_gemini()
    model = get_gemini_model()
    
    prompt = f"""
    As a senior Python developer, explain the following machine learning script step-by-step.
    Focus on what each major block of code does and why it's important.
    Use markdown for clear formatting.

    Script to explain:
    ---
    {code}
    ---
    """
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        st.error(f"Error explaining code with Gemini: {e}")
        return "Failed to generate explanation."