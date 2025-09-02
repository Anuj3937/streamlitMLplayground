import streamlit as st
from dataclasses import dataclass
from typing import Dict, Any, Optional

@dataclass
class MLConfig:
    """Configuration class for ML Platform"""
    
    # Data Processing
    max_samples: int = 50000
    test_size: float = 0.2
    cv_folds: int = 5
    random_state: int = 42
    
    # Model Training
    optuna_trials: int = 30
    max_training_time: int = 300  # seconds
    early_stopping_rounds: int = 10
    
    # UI Settings
    max_features_display: int = 20
    chart_height: int = 500
    chart_width: int = 700
    
    # Cache Settings
    cache_ttl: int = 3600  # 1 hour
    max_cache_size: int = 100  # MB
    
    @classmethod
    def load_from_secrets(cls) -> 'MLConfig':
        """Load configuration from Streamlit secrets"""
        config = cls()
        
        # Override with secrets if available
        secrets = st.secrets.get("ml_config", {})
        for key, value in secrets.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        return config
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return {
            field.name: getattr(self, field.name)
            for field in self.__dataclass_fields__.values()
        }

# Global config instance
CONFIG = MLConfig.load_from_secrets()
