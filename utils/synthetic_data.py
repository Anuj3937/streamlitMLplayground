import pandas as pd
import numpy as np
from sklearn.datasets import make_classification, make_regression

def generate_sample_datasets():
    """
    Generate sample dataset definitions for quick start prompts.
    
    Returns:
        sample_datasets: Dictionary of sample dataset configurations
    """
    
    sample_datasets = {
        'customer_churn': {
            'name': 'Customer Churn Prediction',
            'description': 'Predict whether customers will churn based on usage patterns and demographics',
            'problem_type': 'classification',
            'target': 'churn',
            'features': [
                'monthly_charges', 'total_charges', 'tenure', 'contract_length',
                'payment_method', 'internet_service', 'phone_service', 'multiple_lines',
                'online_security', 'online_backup', 'device_protection', 'tech_support',
                'streaming_tv', 'streaming_movies', 'senior_citizen', 'partner', 'dependents'
            ]
        },
        'spam_detection': {
            'name': 'Email Spam Detection',
            'description': 'Classify emails as spam or legitimate based on content features',
            'problem_type': 'classification',
            'target': 'is_spam',
            'features': [
                'word_freq_make', 'word_freq_address', 'word_freq_all', 'word_freq_3d',
                'word_freq_our', 'word_freq_over', 'word_freq_remove', 'word_freq_internet',
                'word_freq_order', 'word_freq_mail', 'word_freq_receive', 'word_freq_will',
                'word_freq_people', 'word_freq_report', 'word_freq_addresses', 'word_freq_free',
                'char_freq_semicolon', 'char_freq_parentheses', 'char_freq_bracket',
                'char_freq_exclamation', 'char_freq_dollar', 'char_freq_hash',
                'capital_run_length_average', 'capital_run_length_longest', 'capital_run_length_total'
            ]
        },
        'house_prices': {
            'name': 'House Price Prediction',
            'description': 'Predict house prices based on property features and location',
            'problem_type': 'regression',
            'target': 'price',
            'features': [
                'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors',
                'waterfront', 'view', 'condition', 'grade', 'sqft_above',
                'sqft_basement', 'yr_built', 'yr_renovated', 'zipcode',
                'lat', 'long', 'sqft_living15', 'sqft_lot15'
            ]
        }
    }
    
    return sample_datasets

def create_synthetic_dataset(dataset_type, num_samples=1000, custom_params=None):
    """
    Create synthetic datasets for machine learning.
    
    Args:
        dataset_type: Type of dataset to generate
        num_samples: Number of samples to generate
        custom_params: Custom parameters for dataset generation
    
    Returns:
        df: Generated pandas DataFrame
    """
    
    np.random.seed(42)  # For reproducibility
    
    if dataset_type == "customer_churn":
        return _create_customer_churn_data(num_samples)
    elif dataset_type == "spam_detection":
        return _create_spam_detection_data(num_samples)
    elif dataset_type == "house_prices":
        return _create_house_price_data(num_samples)
    elif dataset_type == "custom":
        return _create_custom_dataset(num_samples, custom_params)
    else:
        # Default classification dataset
        return _create_default_classification_data(num_samples)

def _create_customer_churn_data(num_samples):
    """Create customer churn synthetic dataset."""
    
    data = {}
    
    # Numeric features
    data['monthly_charges'] = np.random.normal(70, 20, num_samples).clip(20, 120)
    data['total_charges'] = data['monthly_charges'] * np.random.uniform(1, 72, num_samples)
    data['tenure'] = np.random.randint(1, 73, num_samples)
    data['contract_length'] = np.random.choice([1, 12, 24], num_samples, p=[0.4, 0.3, 0.3])
    
    # Categorical features
    data['payment_method'] = np.random.choice(
        ['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'],
        num_samples, p=[0.3, 0.2, 0.25, 0.25]
    )
    data['internet_service'] = np.random.choice(
        ['DSL', 'Fiber optic', 'No'], num_samples, p=[0.4, 0.4, 0.2]
    )
    
    # Binary features
    binary_features = ['phone_service', 'multiple_lines', 'online_security', 'online_backup',
                      'device_protection', 'tech_support', 'streaming_tv', 'streaming_movies',
                      'senior_citizen', 'partner', 'dependents']
    
    for feature in binary_features:
        data[feature] = np.random.choice([0, 1], num_samples, p=[0.6, 0.4])
    
    # Create target with realistic logic
    churn_prob = (
        0.1 +  # Base probability
        0.3 * (data['monthly_charges'] > 80) +  # High charges increase churn
        0.2 * (data['tenure'] < 12) +  # New customers more likely to churn
        0.15 * (data['contract_length'] == 1) +  # Month-to-month contracts
        0.1 * (data['payment_method'] == 'Electronic check') +  # Payment method effect
        0.05 * data['senior_citizen'] -  # Senior citizens
        0.1 * data['partner'] -  # Having partner reduces churn
        0.05 * data['dependents']  # Having dependents reduces churn
    )
    
    data['churn'] = np.random.binomial(1, churn_prob.clip(0, 1), num_samples)
    
    return pd.DataFrame(data)

def _create_spam_detection_data(num_samples):
    """Create spam detection synthetic dataset."""
    
    # Use scikit-learn's make_classification for realistic spam detection
    X, y = make_classification(
        n_samples=num_samples,
        n_features=25,
        n_informative=20,
        n_redundant=5,
        n_clusters_per_class=1,
        class_sep=0.8,
        random_state=42
    )
    
    # Create feature names that resemble spam detection features
    feature_names = [
        'word_freq_make', 'word_freq_address', 'word_freq_all', 'word_freq_3d',
        'word_freq_our', 'word_freq_over', 'word_freq_remove', 'word_freq_internet',
        'word_freq_order', 'word_freq_mail', 'word_freq_receive', 'word_freq_will',
        'word_freq_people', 'word_freq_report', 'word_freq_addresses', 'word_freq_free',
        'char_freq_semicolon', 'char_freq_parentheses', 'char_freq_bracket',
        'char_freq_exclamation', 'char_freq_dollar', 'char_freq_hash',
        'capital_run_length_average', 'capital_run_length_longest', 'capital_run_length_total'
    ]
    
    # Create DataFrame
    df = pd.DataFrame(X, columns=feature_names)
    df['is_spam'] = y
    
    # Ensure all features are non-negative (frequencies can't be negative)
    for col in feature_names:
        df[col] = df[col].clip(lower=0)
    
    return df

def _create_house_price_data(num_samples):
    """Create house price synthetic dataset."""
    
    data = {}
    
    # Basic house features
    data['bedrooms'] = np.random.choice([1, 2, 3, 4, 5, 6], num_samples, p=[0.1, 0.2, 0.3, 0.25, 0.1, 0.05])
    data['bathrooms'] = data['bedrooms'] * np.random.uniform(0.5, 1.5, num_samples)
    data['sqft_living'] = np.random.normal(2000, 800, num_samples).clip(500, 6000)
    data['sqft_lot'] = np.random.normal(8000, 4000, num_samples).clip(1000, 30000)
    data['floors'] = np.random.choice([1, 1.5, 2, 2.5, 3], num_samples, p=[0.3, 0.1, 0.4, 0.1, 0.1])
    
    # Quality features
    data['condition'] = np.random.choice([1, 2, 3, 4, 5], num_samples, p=[0.05, 0.1, 0.6, 0.2, 0.05])
    data['grade'] = np.random.choice(range(3, 14), num_samples)
    data['view'] = np.random.choice([0, 1, 2, 3, 4], num_samples, p=[0.6, 0.2, 0.1, 0.07, 0.03])
    
    # Binary features
    data['waterfront'] = np.random.choice([0, 1], num_samples, p=[0.95, 0.05])
    
    # Calculated features
    data['sqft_above'] = data['sqft_living'] * np.random.uniform(0.7, 1.0, num_samples)
    data['sqft_basement'] = data['sqft_living'] - data['sqft_above']
    data['sqft_living15'] = data['sqft_living'] * np.random.uniform(0.8, 1.2, num_samples)
    data['sqft_lot15'] = data['sqft_lot'] * np.random.uniform(0.8, 1.2, num_samples)
    
    # Time features
    data['yr_built'] = np.random.randint(1900, 2020, num_samples)
    data['yr_renovated'] = np.where(
        np.random.random(num_samples) < 0.3,
        np.random.randint(1990, 2020, num_samples),
        0
    )
    
    # Location features (simplified)
    data['zipcode'] = np.random.choice(range(98001, 98200), num_samples)
    data['lat'] = np.random.normal(47.6, 0.2, num_samples)
    data['long'] = np.random.normal(-122.3, 0.3, num_samples)
    
    # Create realistic price based on features
    price_base = (
        data['sqft_living'] * 150 +  # Base price per sqft
        data['bedrooms'] * 20000 +   # Bedroom premium
        data['bathrooms'] * 15000 +  # Bathroom premium
        data['grade'] * 30000 +      # Grade premium
        data['view'] * 50000 +       # View premium
        data['waterfront'] * 500000 + # Waterfront premium
        (2020 - data['yr_built']) * -1000  # Age discount
    )
    
    # Add some noise and ensure positive prices
    data['price'] = (price_base + np.random.normal(0, 50000, num_samples)).clip(50000, 5000000)
    
    return pd.DataFrame(data)

def _create_custom_dataset(num_samples, params):
    """Create custom synthetic dataset."""
    
    if params is None:
        params = {}
    
    num_features = params.get('num_features', 5)
    problem_type = params.get('problem_type', 'classification')
    noise_level = params.get('noise_level', 0.1)
    
    if problem_type == 'classification':
        X, y = make_classification(
            n_samples=num_samples,
            n_features=num_features,
            n_informative=max(2, num_features - 2),
            n_redundant=min(2, num_features // 3),
            n_clusters_per_class=1,
            class_sep=1.0 - noise_level,
            random_state=42
        )
    else:
        X, y = make_regression(
            n_samples=num_samples,
            n_features=num_features,
            n_informative=max(2, num_features - 2),
            noise=noise_level * 10,
            random_state=42
        )
    
    # Create feature names
    feature_names = [f'feature_{i+1}' for i in range(num_features)]
    
    # Create DataFrame
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y
    
    return df

def _create_default_classification_data(num_samples):
    """Create default classification dataset."""
    
    X, y = make_classification(
        n_samples=num_samples,
        n_features=10,
        n_informative=8,
        n_redundant=2,
        n_clusters_per_class=1,
        class_sep=0.8,
        random_state=42
    )
    
    feature_names = [f'feature_{i+1}' for i in range(10)]
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y
    
    return df
