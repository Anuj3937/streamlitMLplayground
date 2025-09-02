import streamlit as st
import logging
import traceback
from functools import wraps
from typing import Any, Callable
import sys
from io import StringIO

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ml_platform.log', mode='a'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

def handle_errors(show_error: bool = True, return_none: bool = True):
    """Decorator for handling errors gracefully"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                error_msg = f"Error in {func.__name__}: {str(e)}"
                logger.error(f"{error_msg}\n{traceback.format_exc()}")
                
                if show_error:
                    st.error(f"❌ {error_msg}")
                    
                    with st.expander("🐛 Debug Information"):
                        st.code(traceback.format_exc())
                
                return None if return_none else False
        return wrapper
    return decorator

def setup_logging(log_level: str = "INFO"):
    """Setup logging configuration"""
    log_levels = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR
    }
    
    logging.getLogger().setLevel(log_levels.get(log_level.upper(), logging.INFO))
    logger.info("Logging setup completed")

class StreamlitLogHandler(logging.Handler):
    """Custom log handler for Streamlit"""
    
    def __init__(self):
        super().__init__()
        self.log_capture_string = StringIO()
        
    def emit(self, record):
        log_entry = self.format(record)
        self.log_capture_string.write(f"{log_entry}\n")
        
    def get_logs(self) -> str:
        return self.log_capture_string.getvalue()
        
    def clear_logs(self):
        self.log_capture_string.truncate(0)
        self.log_capture_string.seek(0)

# Global log handler
streamlit_handler = StreamlitLogHandler()
logging.getLogger().addHandler(streamlit_handler)
