import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Any
import great_expectations as ge
from .error_handler import handle_errors
import streamlit as st

class DataValidator:
    """Enhanced data validation with comprehensive checks"""
    
    def __init__(self):
        self.validation_rules = {
            'min_rows': 10,
            'max_rows': 100000,
            'max_missing_ratio': 0.8,
            'min_unique_ratio': 0.01,
            'max_memory_mb': 500
        }
    
    @handle_errors()
    def validate_dataset(self, df: pd.DataFrame) -> Tuple[bool, List[str], Dict[str, Any]]:
        """
        Comprehensive dataset validation
        
        Returns:
            Tuple of (is_valid, issues, metadata)
        """
        issues = []
        metadata = {}
        
        # Basic size validation
        if len(df) < self.validation_rules['min_rows']:
            issues.append(f"❌ Dataset too small: {len(df)} rows (minimum: {self.validation_rules['min_rows']})")
        
        if len(df) > self.validation_rules['max_rows']:
            issues.append(f"⚠️ Large dataset: {len(df):,} rows may slow processing")
        
        # Memory usage check
        memory_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
        metadata['memory_usage_mb'] = memory_mb
        
        if memory_mb > self.validation_rules['max_memory_mb']:
            issues.append(f"⚠️ High memory usage: {memory_mb:.1f}MB")
        
        # Missing data validation
        missing_ratio = df.isnull().sum().sum() / (len(df) * len(df.columns))
        metadata['missing_data_ratio'] = missing_ratio
        
        if missing_ratio > self.validation_rules['max_missing_ratio']:
            issues.append(f"❌ Too much missing data: {missing_ratio:.1%}")
        
        # Data quality checks
        quality_issues = self._check_data_quality(df)
        issues.extend(quality_issues)
        
        # Column analysis
        column_analysis = self._analyze_columns(df)
        metadata.update(column_analysis)
        
        return len(issues) == 0, issues, metadata
    
    def _check_data_quality(self, df: pd.DataFrame) -> List[str]:
        """Check data quality issues"""
        issues = []
        
        # Duplicate rows
        duplicates = df.duplicated().sum()
        if duplicates > 0:
            issues.append(f"⚠️ Found {duplicates} duplicate rows")
        
        # Constant columns
        constant_cols = [col for col in df.columns if df[col].nunique() <= 1]
        if constant_cols:
            issues.append(f"⚠️ Constant columns detected: {', '.join(constant_cols[:5])}")
        
        # High cardinality categorical columns
        high_card_cols = []
        for col in df.select_dtypes(include=['object']).columns:
            if df[col].nunique() > 0.5 * len(df):
                high_card_cols.append(col)
        
        if high_card_cols:
            issues.append(f"⚠️ High cardinality categorical columns: {', '.join(high_card_cols[:3])}")
        
        return issues
    
    def _analyze_columns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze column characteristics"""
        analysis = {
            'total_columns': len(df.columns),
            'numeric_columns': len(df.select_dtypes(include=[np.number]).columns),
            'categorical_columns': len(df.select_dtypes(include=['object', 'category']).columns),
            'datetime_columns': len(df.select_dtypes(include=['datetime64']).columns),
        }
        
        # Missing data per column
        missing_per_col = df.isnull().sum()
        analysis['columns_with_missing'] = (missing_per_col > 0).sum()
        analysis['max_missing_per_column'] = missing_per_col.max()
        
        return analysis
    
    @handle_errors()
    def suggest_target_column(self, df: pd.DataFrame) -> List[str]:
        """Suggest potential target columns"""
        suggestions = []
        
        # Common target column names
        target_keywords = [
            'target', 'label', 'class', 'y', 'output', 'result',
            'price', 'value', 'amount', 'cost', 'revenue', 'profit',
            'churn', 'fraud', 'default', 'approval', 'success',
            'rating', 'score', 'performance'
        ]
        
        for col in df.columns:
            col_lower = col.lower()
            for keyword in target_keywords:
                if keyword in col_lower:
                    suggestions.append(col)
                    break
        
        # If no keyword matches, suggest columns with reasonable cardinality
        if not suggestions:
            for col in df.columns:
                nunique = df[col].nunique()
                # Good targets: 2-20 unique values for classification, continuous for regression
                if 2 <= nunique <= min(20, len(df) * 0.1):
                    suggestions.append(col)
        
        return suggestions[:5]  # Return top 5 suggestions
    
    @handle_errors()
    def detect_problem_type(self, df: pd.DataFrame, target_column: str) -> str:
        """Automatically detect problem type"""
        if target_column not in df.columns:
            return "unknown"
        
        target_series = df[target_column]
        nunique = target_series.nunique()
        
        # Classification indicators
        if nunique <= 10:
            return "classification"
        
        # Check if numeric and continuous
        if pd.api.types.is_numeric_dtype(target_series):
            # If many unique values, likely regression
            if nunique > 20 and nunique / len(df) > 0.05:
                return "regression"
        
        # Default to classification for categorical
        if target_series.dtype == 'object':
            return "classification"
        
        return "regression"
    
    def display_validation_results(self, is_valid: bool, issues: List[str], metadata: Dict[str, Any]) -> None:
        """Display validation results in Streamlit"""
        if is_valid:
            st.success("✅ Dataset validation passed!")
        else:
            st.warning("⚠️ Dataset validation found some issues")
        
        if issues:
            with st.expander("🔍 Validation Issues", expanded=not is_valid):
                for issue in issues:
                    st.markdown(f"• {issue}")
        
        # Display metadata
        with st.expander("📊 Dataset Analysis"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Memory Usage", f"{metadata.get('memory_usage_mb', 0):.1f} MB")
                st.metric("Missing Data", f"{metadata.get('missing_data_ratio', 0):.1%}")
            
            with col2:
                st.metric("Total Columns", metadata.get('total_columns', 0))
                st.metric("Numeric Columns", metadata.get('numeric_columns', 0))
            
            with col3:
                st.metric("Categorical Columns", metadata.get('categorical_columns', 0))
                st.metric("Columns w/ Missing", metadata.get('columns_with_missing', 0))

# Global validator instance
validator = DataValidator()
