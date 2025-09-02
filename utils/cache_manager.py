import streamlit as st
import pickle
import hashlib
import os
from pathlib import Path
from typing import Any, Optional
import pandas as pd
from .config import CONFIG
from .error_handler import handle_errors

class CacheManager:
    """Enhanced caching system for ML operations"""
    
    def __init__(self, cache_dir: str = ".ml_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        (self.cache_dir / "models").mkdir(exist_ok=True)
        (self.cache_dir / "data").mkdir(exist_ok=True)
        (self.cache_dir / "features").mkdir(exist_ok=True)
    
    def get_hash(self, data: Any) -> str:
        """Generate hash for data"""
        if isinstance(data, pd.DataFrame):
            # Hash based on shape and column names
            content = f"{data.shape}_{list(data.columns)}_{data.iloc[0].to_string() if len(data) > 0 else ''}"
        else:
            content = str(data)
        
        return hashlib.md5(content.encode()).hexdigest()[:16]
    
    @handle_errors()
    def cache_data(self, key: str, data: Any, category: str = "data") -> bool:
        """Cache data with given key"""
        try:
            cache_path = self.cache_dir / category / f"{key}.pkl"
            with open(cache_path, 'wb') as f:
                pickle.dump(data, f)
            return True
        except Exception as e:
            st.warning(f"Failed to cache data: {str(e)}")
            return False
    
    @handle_errors()
    def load_cached_data(self, key: str, category: str = "data") -> Optional[Any]:
        """Load cached data by key"""
        try:
            cache_path = self.cache_dir / category / f"{key}.pkl"
            if cache_path.exists():
                with open(cache_path, 'rb') as f:
                    return pickle.load(f)
        except Exception as e:
            st.warning(f"Failed to load cached data: {str(e)}")
        return None
    
    def clear_cache(self, category: Optional[str] = None) -> int:
        """Clear cache files"""
        files_removed = 0
        
        if category:
            cache_path = self.cache_dir / category
            if cache_path.exists():
                for file in cache_path.glob("*.pkl"):
                    file.unlink()
                    files_removed += 1
        else:
            for file in self.cache_dir.rglob("*.pkl"):
                file.unlink()
                files_removed += 1
        
        return files_removed
    
    def get_cache_size(self) -> float:
        """Get total cache size in MB"""
        total_size = 0
        for file in self.cache_dir.rglob("*"):
            if file.is_file():
                total_size += file.stat().st_size
        
        return total_size / (1024 * 1024)  # Convert to MB
    
    def cleanup_old_cache(self, max_age_hours: int = 24) -> int:
        """Remove cache files older than specified hours"""
        import time
        
        current_time = time.time()
        files_removed = 0
        
        for file in self.cache_dir.rglob("*.pkl"):
            if file.is_file():
                file_age_hours = (current_time - file.stat().st_mtime) / 3600
                if file_age_hours > max_age_hours:
                    file.unlink()
                    files_removed += 1
        
        return files_removed

# Enhanced Streamlit caching decorators
@st.cache_data(ttl=CONFIG.cache_ttl)
def cached_data_processing(df_hash: str, processing_func, *args, **kwargs):
    """Cache data processing results"""
    return processing_func(*args, **kwargs)

@st.cache_resource
def cached_model_training(model_hash: str, training_func, *args, **kwargs):
    """Cache model training results"""
    return training_func(*args, **kwargs)

# Global cache manager
cache_manager = CacheManager()
