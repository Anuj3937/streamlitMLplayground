import pandas as pd
import numpy as np
from sklearn.datasets import make_classification, make_regression
from datetime import datetime, timedelta
import random
from .error_handler import handle_errors

def generate_sample_datasets():
    """Generate comprehensive sample dataset definitions"""
    sample_datasets = {
        'customer_churn': {
            'name': 'Customer Churn Prediction',
            'description': 'Predict customer churn based on usage patterns, demographics, and service history',
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
            'description': 'Classify emails as spam or legitimate based on content analysis and metadata',
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
            'description': 'Predict real estate prices using property features, location, and market conditions',
            'problem_type': 'regression',
            'target': 'price',
            'features': [
                'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors',
                'waterfront', 'view', 'condition', 'grade', 'sqft_above',
                'sqft_basement', 'yr_built', 'yr_renovated', 'zipcode',
                'lat', 'long', 'sqft_living15', 'sqft_lot15'
            ]
        },
        'credit_approval': {
            'name': 'Credit Card Approval',
            'description': 'Predict credit card approval based on applicant financial profile',
            'problem_type': 'classification',
            'target': 'approved',
            'features': [
                'age', 'income', 'employment_length', 'debt_to_income', 'credit_score',
                'num_credit_cards', 'num_bank_accounts', 'education_level', 'marital_status',
                'home_ownership', 'loan_purpose', 'loan_amount'
            ]
        },
        'stock_prediction': {
            'name': 'Stock Price Movement',
            'description': 'Predict next-day stock price movement using technical indicators',
            'problem_type': 'regression',
            'target': 'next_day_return',
            'features': [
                'open_price', 'high_price', 'low_price', 'close_price', 'volume',
                'sma_10', 'sma_30', 'ema_12', 'ema_26', 'rsi', 'macd',
                'bollinger_upper', 'bollinger_lower', 'volatility'
            ]
        }
    }
    
    return sample_datasets

@handle_errors()
def create_synthetic_dataset(dataset_type, num_samples=1000, custom_params=None):
    """Create comprehensive synthetic datasets"""
    np.random.seed(42)  # For reproducibility
    
    if dataset_type == "customer_churn":
        return _create_customer_churn_data(num_samples)
    elif dataset_type == "spam_detection":
        return _create_spam_detection_data(num_samples)
    elif dataset_type == "house_prices":
        return _create_house_price_data(num_samples)
    elif dataset_type == "credit_approval":
        return _create_credit_approval_data(num_samples)
    elif dataset_type == "stock_prediction":
        return _create_stock_prediction_data(num_samples)
    elif dataset_type == "custom":
        return _create_custom_dataset(num_samples, custom_params)
    else:
        return _create_default_classification_data(num_samples)

def _create_customer_churn_data(num_samples):
    """Create realistic customer churn dataset"""
    data = {}
    
    # Numeric features with realistic distributions
    data['monthly_charges'] = np.random.gamma(3, 20, num_samples).clip(20, 150)
    data['total_charges'] = data['monthly_charges'] * np.random.exponential(10, num_samples).clip(1, 72)
    data['tenure'] = np.random.exponential(12, num_samples).clip(1, 72).astype(int)
    data['contract_length'] = np.random.choice([1, 12, 24], num_samples, p=[0.4, 0.35, 0.25])
    
    # Categorical features
    data['payment_method'] = np.random.choice(
        ['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'],
        num_samples, p=[0.33, 0.19, 0.22, 0.26]
    )
    
    data['internet_service'] = np.random.choice(
        ['DSL', 'Fiber optic', 'No'], num_samples, p=[0.35, 0.45, 0.20]
    )
    
    # Binary service features
    service_features = ['phone_service', 'multiple_lines', 'online_security', 'online_backup',
                       'device_protection', 'tech_support', 'streaming_tv', 'streaming_movies']
    
    for feature in service_features:
        # Services are correlated with higher charges
        prob = 0.3 + 0.4 * (data['monthly_charges'] > 70)
        data[feature] = np.random.binomial(1, prob, num_samples)
    
    # Demographics
    data['senior_citizen'] = np.random.binomial(1, 0.16, num_samples)
    data['partner'] = np.random.binomial(1, 0.48, num_samples)
    data['dependents'] = np.random.binomial(1, 0.30, num_samples)
    
    # Create realistic churn with business logic
    churn_prob = (
        0.05 +  # Base probability
        0.25 * (data['monthly_charges'] > 80) +  # High charges
        0.20 * (data['tenure'] < 6) +  # New customers
        0.15 * (data['contract_length'] == 1) +  # Month-to-month
        0.10 * (data['total_charges'] > 5000) +  # High total spend
        0.05 * data['senior_citizen'] -  # Senior citizens slightly more loyal
        0.10 * data['partner'] -  # Partners more stable
        0.08 * data['dependents']  # Dependents increase stability
    )
    
    data['churn'] = np.random.binomial(1, churn_prob.clip(0, 0.8), num_samples)
    
    return pd.DataFrame(data)

def _create_spam_detection_data(num_samples):
    """Create spam detection dataset with realistic features"""
    # Use scikit-learn's make_classification with realistic parameters
    X, y = make_classification(
        n_samples=num_samples,
        n_features=25,
        n_informative=20,
        n_redundant=3,
        n_clusters_per_class=2,
        class_sep=1.2,
        random_state=42,
        weights=[0.75, 0.25]  # 25% spam, 75% ham
    )
    
    # Create realistic feature names
    feature_names = [
        'word_freq_make', 'word_freq_address', 'word_freq_all', 'word_freq_3d',
        'word_freq_our', 'word_freq_over', 'word_freq_remove', 'word_freq_internet',
        'word_freq_order', 'word_freq_mail', 'word_freq_receive', 'word_freq_will',
        'word_freq_people', 'word_freq_report', 'word_freq_addresses', 'word_freq_free',
        'word_freq_business', 'word_freq_email', 'word_freq_you', 'word_freq_credit',
        'char_freq_semicolon', 'char_freq_parentheses', 'char_freq_bracket',
        'char_freq_exclamation', 'char_freq_dollar'
    ]
    
    # Ensure all features are non-negative (frequencies can't be negative)
    X = np.abs(X)
    
    # Scale to realistic frequency ranges
    X = X * np.random.uniform(0.1, 5.0, X.shape[1])
    
    df = pd.DataFrame(X, columns=feature_names)
    df['is_spam'] = y
    
    return df

def _create_house_price_data(num_samples):
    """Create realistic house price dataset"""
    data = {}
    
    # Basic house features with correlations
    data['bedrooms'] = np.random.choice([1, 2, 3, 4, 5, 6], num_samples, 
                                       p=[0.05, 0.15, 0.35, 0.30, 0.12, 0.03])
    
    # Bathrooms correlated with bedrooms
    bathroom_base = data['bedrooms'] * 0.75
    data['bathrooms'] = (bathroom_base + np.random.normal(0, 0.5, num_samples)).clip(1, 6)
    
    # Square footage correlated with bedrooms
    base_sqft = 500 + data['bedrooms'] * 400
    data['sqft_living'] = (base_sqft + np.random.normal(0, 300, num_samples)).clip(400, 8000)
    
    # Lot size
    data['sqft_lot'] = np.random.lognormal(9, 0.5, num_samples).clip(1000, 50000)
    
    # Other features
    data['floors'] = np.random.choice([1, 1.5, 2, 2.5, 3], num_samples, 
                                     p=[0.3, 0.15, 0.4, 0.1, 0.05])
    
    data['waterfront'] = np.random.binomial(1, 0.08, num_samples)  # 8% waterfront
    data['view'] = np.random.choice([0, 1, 2, 3, 4], num_samples, 
                                   p=[0.60, 0.20, 0.12, 0.06, 0.02])
    data['condition'] = np.random.choice([1, 2, 3, 4, 5], num_samples, 
                                        p=[0.03, 0.07, 0.65, 0.20, 0.05])
    data['grade'] = np.random.choice(range(3, 14), num_samples)
    
    # Derived features
    data['sqft_above'] = data['sqft_living'] * np.random.uniform(0.6, 1.0, num_samples)
    data['sqft_basement'] = data['sqft_living'] - data['sqft_above']
    
    # Time features
    current_year = datetime.now().year
    data['yr_built'] = np.random.randint(1900, current_year, num_samples)
    
    # Some houses renovated
    renovation_mask = np.random.random(num_samples) < 0.25
    data['yr_renovated'] = np.where(
        renovation_mask,
        np.random.randint(1990, current_year, num_samples),
        0
    )
    
    # Location features (simplified)
    data['zipcode'] = np.random.choice(range(98001, 98200), num_samples)
    data['lat'] = np.random.normal(47.6, 0.3, num_samples)
    data['long'] = np.random.normal(-122.3, 0.4, num_samples)
    
    # Neighborhood features
    data['sqft_living15'] = data['sqft_living'] * np.random.uniform(0.8, 1.3, num_samples)
    data['sqft_lot15'] = data['sqft_lot'] * np.random.uniform(0.7, 1.4, num_samples)
    
    # Create realistic price with complex interactions
    price_base = (
        data['sqft_living'] * 180 +  # Base price per sqft
        data['bedrooms'] * 15000 +   # Bedroom premium
        data['bathrooms'] * 12000 +  # Bathroom premium
        data['grade'] * 25000 +      # Grade premium
        data['view'] * 30000 +       # View premium
        data['waterfront'] * 400000 + # Waterfront premium
        (current_year - data['yr_built']) * -800 +  # Age discount
        (data['yr_renovated'] > 0) * 50000  # Renovation bonus
    )
    
    # Add location premium based on zipcode
    high_value_zips = np.random.choice(range(98001, 98200), 20)
    location_premium = np.where(
        np.isin(data['zipcode'], high_value_zips),
        100000,  # Premium locations
        0
    )
    
    price_base += location_premium
    
    # Add realistic noise and ensure positive prices
    data['price'] = (price_base + np.random.normal(0, 50000, num_samples)).clip(50000, 8000000)
    
    return pd.DataFrame(data)

def _create_credit_approval_data(num_samples):
    """Create credit approval dataset"""
    data = {}
    
    # Demographics
    data['age'] = np.random.normal(40, 12, num_samples).clip(18, 80).astype(int)
    
    # Income with age correlation
    income_base = 30000 + (data['age'] - 25) * 1200
    data['income'] = (income_base + np.random.normal(0, 20000, num_samples)).clip(15000, 250000)
    
    # Employment length
    data['employment_length'] = np.random.exponential(5, num_samples).clip(0, 40)
    
    # Financial metrics
    data['debt_to_income'] = np.random.beta(2, 5, num_samples) * 0.6  # 0 to 60%
    data['credit_score'] = np.random.normal(650, 100, num_samples).clip(300, 850).astype(int)
    
    # Credit history
    data['num_credit_cards'] = np.random.poisson(3, num_samples).clip(0, 15)
    data['num_bank_accounts'] = np.random.poisson(2, num_samples).clip(1, 8)
    
    # Categorical features
    data['education_level'] = np.random.choice(
        ['High School', 'Some College', 'Bachelor', 'Master', 'PhD'],
        num_samples, p=[0.25, 0.20, 0.35, 0.15, 0.05]
    )
    
    data['marital_status'] = np.random.choice(
        ['Single', 'Married', 'Divorced'], num_samples, p=[0.4, 0.45, 0.15]
    )
    
    data['home_ownership'] = np.random.choice(
        ['Rent', 'Own', 'Mortgage'], num_samples, p=[0.35, 0.25, 0.40]
    )
    
    data['loan_purpose'] = np.random.choice(
        ['debt_consolidation', 'home_improvement', 'major_purchase', 'other'],
        num_samples, p=[0.40, 0.25, 0.20, 0.15]
    )
    
    # Loan amount based on income
    loan_ratio = np.random.uniform(0.1, 0.4, num_samples)
    data['loan_amount'] = (data['income'] * loan_ratio).clip(1000, 50000)
    
    # Create approval decision with realistic business logic
    approval_score = (
        (data['credit_score'] - 500) / 100 +  # Credit score impact
        (data['income'] / 50000) +             # Income impact
        (data['employment_length'] / 10) +     # Employment stability
        -(data['debt_to_income'] * 5) +        # Debt burden
        (data['num_bank_accounts'] / 10) +     # Banking relationship
        -(data['loan_amount'] / data['income'] * 2)  # Loan size relative to income
    )
    
    # Add categorical impacts
    education_bonus = {
        'High School': 0, 'Some College': 0.2, 'Bachelor': 0.4, 
        'Master': 0.6, 'PhD': 0.8
    }
    approval_score += np.array([education_bonus[edu] for edu in data['education_level']])
    
    # Approval probability
    approval_prob = 1 / (1 + np.exp(-approval_score))  # Sigmoid
    data['approved'] = np.random.binomial(1, approval_prob, num_samples)
    
    # Encode categorical variables
    from sklearn.preprocessing import LabelEncoder
    for col in ['education_level', 'marital_status', 'home_ownership', 'loan_purpose']:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
    
    return pd.DataFrame(data)

def _create_stock_prediction_data(num_samples):
    """Create stock prediction dataset with technical indicators"""
    data = {}
    
    # Base price series with trend and volatility
    np.random.seed(42)
    
    # Generate price series
    initial_price = 100
    returns = np.random.normal(0.0005, 0.02, num_samples)  # Daily returns
    prices = [initial_price]
    
    for ret in returns:
        prices.append(prices[-1] * (1 + ret))
    
    prices = np.array(prices[1:])  # Remove initial price
    
    # OHLC data
    data['close_price'] = prices
    data['open_price'] = prices * np.random.uniform(0.98, 1.02, num_samples)
    
    # High and low based on volatility
    daily_vol = np.abs(np.random.normal(0, 0.015, num_samples))
    data['high_price'] = prices * (1 + daily_vol)
    data['low_price'] = prices * (1 - daily_vol)
    
    # Volume with mean reversion
    base_volume = 1000000
    data['volume'] = base_volume * np.random.lognormal(0, 0.5, num_samples)
    
    # Technical indicators
    # Simple moving averages
    data['sma_10'] = pd.Series(prices).rolling(10, min_periods=1).mean().values
    data['sma_30'] = pd.Series(prices).rolling(30, min_periods=1).mean().values
    
    # Exponential moving averages
    data['ema_12'] = pd.Series(prices).ewm(span=12).mean().values
    data['ema_26'] = pd.Series(prices).ewm(span=26).mean().values
    
    # RSI (simplified)
    price_changes = np.diff(prices, prepend=prices[0])
    gains = np.maximum(price_changes, 0)
    losses = np.maximum(-price_changes, 0)
    
    avg_gains = pd.Series(gains).rolling(14, min_periods=1).mean().values
    avg_losses = pd.Series(losses).rolling(14, min_periods=1).mean().values
    
    rs = avg_gains / (avg_losses + 1e-10)
    data['rsi'] = 100 - (100 / (1 + rs))
    
    # MACD
    data['macd'] = data['ema_12'] - data['ema_26']
    
    # Bollinger Bands
    rolling_mean = pd.Series(prices).rolling(20, min_periods=1).mean().values
    rolling_std = pd.Series(prices).rolling(20, min_periods=1).std().values
    data['bollinger_upper'] = rolling_mean + (rolling_std * 2)
    data['bollinger_lower'] = rolling_mean - (rolling_std * 2)
    
    # Volatility
    data['volatility'] = pd.Series(returns).rolling(10, min_periods=1).std().values
    
    # Target: next day return
    next_day_returns = np.roll(returns, -1)
    next_day_returns[-1] = 0  # Last day has no next day
    data['next_day_return'] = next_day_returns
    
    return pd.DataFrame(data)

def _create_custom_dataset(num_samples, params):
    """Create custom synthetic dataset based on parameters"""
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
            class_sep=max(0.5, 1.5 - noise_level),
            random_state=42
        )
    else:  # regression
        X, y = make_regression(
            n_samples=num_samples,
            n_features=num_features,
            n_informative=max(2, num_features - 2),
            noise=noise_level * 50,
            random_state=42
        )
    
    # Create feature names
    feature_names = [f'feature_{i+1}' for i in range(num_features)]
    
    # Create DataFrame
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y
    
    return df

def _create_default_classification_data(num_samples):
    """Create default classification dataset"""
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
