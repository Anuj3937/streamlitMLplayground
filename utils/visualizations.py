import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import warnings
warnings.filterwarnings('ignore')

def create_correlation_heatmap(df, title="Feature Correlation Matrix"):
    """
    Create a correlation heatmap for numeric features.
    
    Args:
        df: DataFrame with numeric features
        title: Title for the heatmap
    
    Returns:
        fig: Plotly figure object
    """
    
    # Select only numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    if len(numeric_cols) < 2:
        return None
    
    # Calculate correlation matrix
    corr_matrix = df[numeric_cols].corr()
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu',
        zmid=0,
        text=np.round(corr_matrix.values, 2),
        texttemplate="%{text}",
        textfont={"size": 10},
        hoverongaps=False
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Features",
        yaxis_title="Features",
        height=600,
        width=600
    )
    
    return fig

def create_feature_distribution(df, feature_name, target_column=None, problem_type=None):
    """
    Create distribution plot for a feature, optionally colored by target.
    
    Args:
        df: DataFrame containing the feature
        feature_name: Name of the feature to plot
        target_column: Name of target column for coloring
        problem_type: 'classification' or 'regression'
    
    Returns:
        fig: Plotly figure object
    """
    
    if feature_name not in df.columns:
        return None
    
    feature_data = df[feature_name]
    
    # Check if feature is numeric or categorical
    if pd.api.types.is_numeric_dtype(feature_data):
        # Numeric feature - histogram
        if target_column and target_column in df.columns:
            if problem_type == "classification":
                # Color by target classes
                fig = px.histogram(
                    df, 
                    x=feature_name, 
                    color=target_column,
                    title=f"Distribution of {feature_name} by {target_column}",
                    nbins=30,
                    marginal="box"
                )
            else:
                # Scatter plot for regression
                fig = px.scatter(
                    df, 
                    x=feature_name, 
                    y=target_column,
                    title=f"{feature_name} vs {target_column}",
                    trendline="ols"
                )
        else:
            # Simple histogram
            fig = px.histogram(
                df, 
                x=feature_name,
                title=f"Distribution of {feature_name}",
                nbins=30
            )
    else:
        # Categorical feature - bar chart
        value_counts = feature_data.value_counts().head(20)  # Top 20 categories
        
        if target_column and target_column in df.columns and problem_type == "classification":
            # Stacked bar chart by target
            crosstab = pd.crosstab(df[feature_name], df[target_column])
            crosstab_pct = crosstab.div(crosstab.sum(axis=1), axis=0) * 100
            
            fig = go.Figure()
            for target_class in crosstab_pct.columns:
                fig.add_trace(go.Bar(
                    name=f"{target_column}={target_class}",
                    x=crosstab_pct.index,
                    y=crosstab_pct[target_class]
                ))
            
            fig.update_layout(
                title=f"{feature_name} Distribution by {target_column}",
                xaxis_title=feature_name,
                yaxis_title="Percentage",
                barmode='stack'
            )
        else:
            # Simple bar chart
            fig = px.bar(
                x=value_counts.index,
                y=value_counts.values,
                title=f"Distribution of {feature_name}",
                labels={'x': feature_name, 'y': 'Count'}
            )
    
    return fig

def create_confusion_matrix(y_true, y_pred, class_names=None, title="Confusion Matrix"):
    """
    Create a confusion matrix heatmap.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: Names of classes (optional)
        title: Title for the plot
    
    Returns:
        fig: Plotly figure object
    """
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Create class names if not provided
    if class_names is None:
        unique_labels = sorted(list(set(y_true) | set(y_pred)))
        class_names = [f"Class {label}" for label in unique_labels]
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=class_names,
        y=class_names,
        text=cm,
        texttemplate="%{text}",
        textfont={"size": 16},
        colorscale='Blues',
        showscale=True
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Predicted",
        yaxis_title="Actual",
        height=500,
        width=500
    )
    
    # Add accuracy annotations
    accuracy = np.trace(cm) / np.sum(cm)
    fig.add_annotation(
        text=f"Accuracy: {accuracy:.3f}",
        xref="paper", yref="paper",
        x=0.02, y=0.98,
        showarrow=False,
        font=dict(size=14, color="black"),
        bgcolor="white",
        bordercolor="black",
        borderwidth=1
    )
    
    return fig

def create_feature_importance_plot(feature_names, importance_values, title="Feature Importance", top_n=15):
    """
    Create a horizontal bar plot for feature importance.
    
    Args:
        feature_names: List of feature names
        importance_values: List of importance values
        title: Title for the plot
        top_n: Number of top features to show
    
    Returns:
        fig: Plotly figure object
    """
    
    if len(feature_names) != len(importance_values):
        return None
    
    # Create DataFrame and sort by importance
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance_values
    }).sort_values('Importance', ascending=True)
    
    # Take top N features
    importance_df = importance_df.tail(top_n)
    
    # Create horizontal bar chart
    fig = px.bar(
        importance_df,
        x='Importance',
        y='Feature',
        orientation='h',
        title=title,
        color='Importance',
        color_continuous_scale='viridis'
    )
    
    fig.update_layout(
        height=max(400, len(importance_df) * 30),
        yaxis={'categoryorder': 'total ascending'},
        showlegend=False
    )
    
    return fig

def create_model_comparison_chart(results_df, metric_name, title=None):
    """
    Create a bar chart comparing model performance.
    
    Args:
        results_df: DataFrame with model results
        metric_name: Name of the metric to compare
        title: Title for the chart
    
    Returns:
        fig: Plotly figure object
    """
    
    if metric_name not in results_df.columns:
        return None
    
    if title is None:
        title = f"Model Comparison: {metric_name.replace('_', ' ').title()}"
    
    # Sort by metric value
    sorted_df = results_df.sort_values(metric_name, ascending=True)
    
    # Create bar chart
    fig = px.bar(
        sorted_df,
        x=metric_name,
        y='model',
        orientation='h',
        title=title,
        color=metric_name,
        color_continuous_scale='viridis',
        text=metric_name
    )
    
    fig.update_traces(texttemplate='%{text:.3f}', textposition='outside')
    fig.update_layout(
        height=max(400, len(sorted_df) * 50),
        yaxis={'categoryorder': 'total ascending'},
        showlegend=False
    )
    
    return fig

def create_residual_plot(y_true, y_pred, title="Residual Plot"):
    """
    Create a residual plot for regression models.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        title: Title for the plot
    
    Returns:
        fig: Plotly figure object
    """
    
    residuals = np.array(y_true) - np.array(y_pred)
    
    # Create scatter plot
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=y_pred,
        y=residuals,
        mode='markers',
        name='Residuals',
        marker=dict(
            color='blue',
            opacity=0.6,
            size=6
        )
    ))
    
    # Add horizontal line at y=0
    fig.add_hline(y=0, line_dash="dash", line_color="red", line_width=2)
    
    fig.update_layout(
        title=title,
        xaxis_title="Predicted Values",
        yaxis_title="Residuals",
        height=500
    )
    
    return fig

def create_learning_curve(train_scores, val_scores, train_sizes, title="Learning Curve"):
    """
    Create a learning curve plot.
    
    Args:
        train_scores: Training scores for different training sizes
        val_scores: Validation scores for different training sizes
        train_sizes: Different training set sizes
        title: Title for the plot
    
    Returns:
        fig: Plotly figure object
    """
    
    fig = go.Figure()
    
    # Training scores
    fig.add_trace(go.Scatter(
        x=train_sizes,
        y=np.mean(train_scores, axis=1),
        mode='lines+markers',
        name='Training Score',
        line=dict(color='blue'),
        error_y=dict(
            type='data',
            array=np.std(train_scores, axis=1),
            visible=True
        )
    ))
    
    # Validation scores
    fig.add_trace(go.Scatter(
        x=train_sizes,
        y=np.mean(val_scores, axis=1),
        mode='lines+markers',
        name='Validation Score',
        line=dict(color='red'),
        error_y=dict(
            type='data',
            array=np.std(val_scores, axis=1),
            visible=True
        )
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Training Set Size",
        yaxis_title="Score",
        height=500
    )
    
    return fig

def create_roc_curve(y_true, y_pred_proba, title="ROC Curve"):
    """
    Create ROC curve for binary classification.
    
    Args:
        y_true: True binary labels
        y_pred_proba: Predicted probabilities for positive class
        title: Title for the plot
    
    Returns:
        fig: Plotly figure object
    """
    
    try:
        from sklearn.metrics import roc_curve, auc
        
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        fig = go.Figure()
        
        # ROC curve
        fig.add_trace(go.Scatter(
            x=fpr,
            y=tpr,
            mode='lines',
            name=f'ROC Curve (AUC = {roc_auc:.3f})',
            line=dict(color='blue', width=2)
        ))
        
        # Diagonal line (random classifier)
        fig.add_trace(go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode='lines',
            name='Random Classifier',
            line=dict(color='red', dash='dash')
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            height=500,
            width=500
        )
        
        return fig
        
    except ImportError:
        return None

def create_precision_recall_curve(y_true, y_pred_proba, title="Precision-Recall Curve"):
    """
    Create precision-recall curve for binary classification.
    
    Args:
        y_true: True binary labels
        y_pred_proba: Predicted probabilities for positive class
        title: Title for the plot
    
    Returns:
        fig: Plotly figure object
    """
    
    try:
        from sklearn.metrics import precision_recall_curve, average_precision_score
        
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
        avg_precision = average_precision_score(y_true, y_pred_proba)
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=recall,
            y=precision,
            mode='lines',
            name=f'PR Curve (AP = {avg_precision:.3f})',
            line=dict(color='blue', width=2)
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Recall",
            yaxis_title="Precision",
            height=500,
            width=500
        )
        
        return fig
        
    except ImportError:
        return None

def create_class_distribution_pie(y, title="Class Distribution"):
    """
    Create a pie chart showing class distribution.
    
    Args:
        y: Target variable
        title: Title for the plot
    
    Returns:
        fig: Plotly figure object
    """
    
    class_counts = pd.Series(y).value_counts()
    
    fig = px.pie(
        values=class_counts.values,
        names=class_counts.index,
        title=title
    )
    
    fig.update_traces(textposition='inside', textinfo='percent+label')
    
    return fig

def create_prediction_vs_actual_plot(y_true, y_pred, title="Predictions vs Actual"):
    """
    Create a scatter plot of predictions vs actual values for regression.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        title: Title for the plot
    
    Returns:
        fig: Plotly figure object
    """
    
    fig = go.Figure()
    
    # Scatter plot
    fig.add_trace(go.Scatter(
        x=y_true,
        y=y_pred,
        mode='markers',
        name='Predictions',
        marker=dict(
            color='blue',
            opacity=0.6,
            size=6
        )
    ))
    
    # Perfect prediction line (y=x)
    min_val = min(min(y_true), min(y_pred))
    max_val = max(max(y_true), max(y_pred))
    
    fig.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode='lines',
        name='Perfect Prediction',
        line=dict(color='red', dash='dash')
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Actual Values",
        yaxis_title="Predicted Values",
        height=500,
        width=500
    )
    
    return fig
