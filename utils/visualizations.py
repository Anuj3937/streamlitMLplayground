import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
import warnings
from .error_handler import handle_errors

warnings.filterwarnings('ignore')

@handle_errors()
def create_correlation_heatmap(df, title="Feature Correlation Matrix"):
    """Create an enhanced correlation heatmap"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) < 2:
        return None
    
    # Calculate correlation matrix
    corr_matrix = df[numeric_cols].corr()
    
    # Create mask for better visualization
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
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
        hoverongaps=False,
        colorbar=dict(title="Correlation")
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Features",
        yaxis_title="Features",
        height=600,
        width=600
    )
    
    return fig

@handle_errors()
def create_feature_distribution(df, feature_name, target_column=None, problem_type=None):
    """Create enhanced feature distribution plot"""
    if feature_name not in df.columns:
        return None
    
    feature_data = df[feature_name]
    
    if pd.api.types.is_numeric_dtype(feature_data):
        # Numeric feature
        if target_column and target_column in df.columns:
            if problem_type == "classification":
                fig = px.histogram(
                    df,
                    x=feature_name,
                    color=target_column,
                    title=f"Distribution of {feature_name} by {target_column}",
                    nbins=30,
                    marginal="box",
                    barmode='overlay',
                    opacity=0.7
                )
            else:
                fig = px.scatter(
                    df,
                    x=feature_name,
                    y=target_column,
                    title=f"{feature_name} vs {target_column}",
                    trendline="ols",
                    opacity=0.6
                )
        else:
            fig = px.histogram(
                df,
                x=feature_name,
                title=f"Distribution of {feature_name}",
                nbins=30,
                marginal="box"
            )
    else:
        # Categorical feature
        value_counts = feature_data.value_counts().head(20)
        
        if target_column and target_column in df.columns and problem_type == "classification":
            # Stacked bar chart
            crosstab = pd.crosstab(df[feature_name], df[target_column])
            
            fig = go.Figure()
            for target_class in crosstab.columns:
                fig.add_trace(go.Bar(
                    name=f"{target_column}={target_class}",
                    x=crosstab.index,
                    y=crosstab[target_class]
                ))
            
            fig.update_layout(
                title=f"{feature_name} Distribution by {target_column}",
                xaxis_title=feature_name,
                yaxis_title="Count",
                barmode='stack'
            )
        else:
            fig = px.bar(
                x=value_counts.index,
                y=value_counts.values,
                title=f"Distribution of {feature_name}",
                labels={'x': feature_name, 'y': 'Count'}
            )
    
    return fig

@handle_errors()
def create_model_comparison_chart(results_df, metric_name, title=None):
    """Create enhanced model comparison chart"""
    if metric_name not in results_df.columns:
        return None
    
    if title is None:
        title = f"Model Comparison: {metric_name.replace('_', ' ').title()}"
    
    # Sort by metric value
    sorted_df = results_df.sort_values(metric_name, ascending=True)
    
    # Create horizontal bar chart
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

@handle_errors()
def create_confusion_matrix_plot(y_true, y_pred, class_names=None, title="Confusion Matrix"):
    """Create enhanced confusion matrix visualization"""
    cm = confusion_matrix(y_true, y_pred)
    
    if class_names is None:
        unique_labels = sorted(list(set(y_true) | set(y_pred)))
        class_names = [f"Class {label}" for label in unique_labels]
    
    # Calculate percentages
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    # Create annotations
    annotations = []
    for i in range(len(cm)):
        for j in range(len(cm[0])):
            annotations.append(f"{cm[i][j]}<br>({cm_percent[i][j]:.1f}%)")
    
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=class_names,
        y=class_names,
        text=np.array(annotations).reshape(cm.shape),
        texttemplate="%{text}",
        textfont={"size": 14},
        colorscale='Blues',
        showscale=True,
        colorbar=dict(title="Count")
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Predicted",
        yaxis_title="Actual",
        height=500,
        width=500
    )
    
    # Add accuracy annotation
    accuracy = np.trace(cm) / np.sum(cm)
    fig.add_annotation(
        text=f"Accuracy: {accuracy:.1%}",
        xref="paper", yref="paper",
        x=0.02, y=0.98,
        showarrow=False,
        font=dict(size=16, color="black"),
        bgcolor="white",
        bordercolor="black",
        borderwidth=1
    )
    
    return fig

@handle_errors()
def create_roc_curve_plot(y_true, y_pred_proba, title="ROC Curve"):
    """Create ROC curve visualization"""
    try:
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        fig = go.Figure()
        
        # ROC curve
        fig.add_trace(go.Scatter(
            x=fpr,
            y=tpr,
            mode='lines',
            name=f'ROC Curve (AUC = {roc_auc:.3f})',
            line=dict(color='blue', width=3)
        ))
        
        # Diagonal line
        fig.add_trace(go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode='lines',
            name='Random Classifier',
            line=dict(color='red', dash='dash', width=2)
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            height=500,
            width=500,
            showlegend=True
        )
        
        return fig
        
    except Exception as e:
        return None

@handle_errors()
def create_precision_recall_curve_plot(y_true, y_pred_proba, title="Precision-Recall Curve"):
    """Create precision-recall curve visualization"""
    try:
        from sklearn.metrics import average_precision_score
        
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
        avg_precision = average_precision_score(y_true, y_pred_proba)
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=recall,
            y=precision,
            mode='lines',
            name=f'PR Curve (AP = {avg_precision:.3f})',
            line=dict(color='blue', width=3),
            fill='tonexty'
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Recall",
            yaxis_title="Precision",
            height=500,
            width=500
        )
        
        return fig
        
    except Exception as e:
        return None

@handle_errors()
def create_residual_plot(y_true, y_pred, title="Residual Analysis"):
    """Create residual plot for regression"""
    residuals = np.array(y_true) - np.array(y_pred)
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Residuals vs Predictions", "Residuals Distribution")
    )
    
    # Residuals vs Predictions
    fig.add_trace(
        go.Scatter(
            x=y_pred,
            y=residuals,
            mode='markers',
            name='Residuals',
            marker=dict(color='blue', opacity=0.6)
        ),
        row=1, col=1
    )
    
    # Add horizontal line at y=0
    fig.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=1)
    
    # Residuals distribution
    fig.add_trace(
        go.Histogram(
            x=residuals,
            name='Distribution',
            marker=dict(color='lightblue'),
            showlegend=False
        ),
        row=1, col=2
    )
    
    fig.update_layout(
        title=title,
        height=400,
        showlegend=False
    )
    
    fig.update_xaxes(title_text="Predicted Values", row=1, col=1)
    fig.update_yaxes(title_text="Residuals", row=1, col=1)
    fig.update_xaxes(title_text="Residuals", row=1, col=2)
    fig.update_yaxes(title_text="Frequency", row=1, col=2)
    
    return fig

@handle_errors()
def create_learning_curve_plot(train_scores, val_scores, train_sizes, title="Learning Curve"):
    """Create learning curve visualization"""
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

@handle_errors()
def create_prediction_vs_actual_plot(y_true, y_pred, title="Predictions vs Actual"):
    """Create predictions vs actual scatter plot"""
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
        line=dict(color='red', dash='dash', width=2)
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Actual Values",
        yaxis_title="Predicted Values",
        height=500,
        width=500
    )
    
    return fig

@handle_errors()
def create_feature_importance_plot(feature_names, importance_values, title="Feature Importance", top_n=15):
    """Create feature importance horizontal bar plot"""
    if len(feature_names) != len(importance_values):
        return None
    
    # Create DataFrame and sort
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance_values
    }).sort_values('Importance', ascending=True).tail(top_n)
    
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
