# utils/model_monitor.py
import streamlit as st
import time
from datetime import datetime

class ModelMonitor:
    def __init__(self):
        self.metrics_history = []
    
    def log_prediction(self, input_data, prediction, actual=None):
        log_entry = {
            'timestamp': datetime.now(),
            'input_hash': hash(str(input_data)),
            'prediction': prediction,
            'actual': actual
        }
        self.metrics_history.append(log_entry)
    
    def detect_drift(self, recent_predictions, baseline_mean):
        if len(recent_predictions) < 10:
            return False
        
        current_mean = np.mean(recent_predictions)
        drift_threshold = 0.1  # 10% change
        
        return abs(current_mean - baseline_mean) / baseline_mean > drift_threshold
    
    def render_monitoring_dashboard(self):
        st.subheader("🔍 Model Monitoring Dashboard")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Predictions", len(self.metrics_history))
        with col2:
            recent_accuracy = self._calculate_recent_accuracy()
            st.metric("Recent Accuracy", f"{recent_accuracy:.2%}")
        with col3:
            drift_detected = self._check_recent_drift()
            st.metric("Drift Status", "⚠️ Detected" if drift_detected else "✅ Normal")
