import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix

def evaluate_model_by_group(model, test_data, features, target, sensitive_attributes):
    """
    Evaluate model performance broken down by sensitive attribute categories
    
    Parameters:
    model: Trained XGBoost model
    test_data: Test dataset
    features: List of feature names
    target: Target variable name
    sensitive_attributes: List of sensitive attribute column names
    
    Returns:
    Dictionary containing metrics for each group within each sensitive attribute
    """
    X_test = test_data[features]
    y_test = test_data[target]
    
    # Get predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Overall metrics (for reference)
    overall_metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'auc': roc_auc_score(y_test, y_pred_proba),
        'precision_recall_f1': precision_recall_fscore_support(y_test, y_pred, average=None),
        'confusion_matrix': confusion_matrix(y_test, y_pred)
    }
    
    # Group-specific metrics
    group_metrics = {'overall': overall_metrics}
    
    # For each sensitive attribute
    for attr in sensitive_attributes:
        group_metrics[attr] = {}
        
        # Get unique values for this attribute
        unique_groups = test_data[attr].unique()
        
        # Calculate metrics for each group
        for group in unique_groups:
            # Filter test data for this group
            group_indices = test_data[attr] == group
            group_y_test = y_test[group_indices]
            group_y_pred = y_pred[group_indices]
            group_y_pred_proba = y_pred_proba[group_indices]
            
            # Skip if too few samples
            if len(group_y_test) < 10:
                continue
                
            try:
                group_metrics[attr][group] = {
                    'count': len(group_y_test),
                    'positive_rate': group_y_test.mean(),
                    'accuracy': accuracy_score(group_y_test, group_y_pred),
                    'precision_recall_f1': precision_recall_fscore_support(
                        group_y_test, group_y_pred, average=None
                    ),
                    'auc': roc_auc_score(group_y_test, group_y_pred_proba) if len(np.unique(group_y_test)) > 1 else np.nan,
                    'confusion_matrix': confusion_matrix(group_y_test, group_y_pred)
                }
            except Exception as e:
                print(f"Error calculating metrics for {attr}={group}: {e}")
    
    return group_metrics

def compare_group_metrics(original_metrics, augmented_metrics):
    """
    Compare metrics between original and augmented models for each group
    """
    comparison = {}
    
    # Iterate through all attributes
    all_attrs = set(list(original_metrics.keys()) + list(augmented_metrics.keys()))
    
    for attr in all_attrs:
        if attr == 'overall':
            continue
            
        comparison[attr] = {}
        
        # Get all groups for this attribute
        all_groups = set(list(original_metrics.get(attr, {}).keys()) + 
                         list(augmented_metrics.get(attr, {}).keys()))
        
        for group in all_groups:
            orig_metrics = original_metrics.get(attr, {}).get(group, {})
            aug_metrics = augmented_metrics.get(attr, {}).get(group, {})
            
            if not orig_metrics or not aug_metrics:
                continue
                
            # Calculate differences in key metrics
            comparison[attr][group] = {
                'count': aug_metrics['count'],
                'accuracy_diff': aug_metrics['accuracy'] - orig_metrics['accuracy'],
                'auc_diff': aug_metrics['auc'] - orig_metrics['auc'] if not np.isnan(aug_metrics.get('auc', np.nan)) and not np.isnan(orig_metrics.get('auc', np.nan)) else np.nan,
                'recall_diff_class1': aug_metrics['precision_recall_f1'][1][1] - orig_metrics['precision_recall_f1'][1][1] if len(aug_metrics['precision_recall_f1'][1]) > 1 and len(orig_metrics['precision_recall_f1'][1]) > 1 else np.nan,
                'precision_diff_class1': aug_metrics['precision_recall_f1'][0][1] - orig_metrics['precision_recall_f1'][0][1] if len(aug_metrics['precision_recall_f1'][0]) > 1 and len(orig_metrics['precision_recall_f1'][0]) > 1 else np.nan,
                'f1_diff_class1': aug_metrics['precision_recall_f1'][2][1] - orig_metrics['precision_recall_f1'][2][1] if len(aug_metrics['precision_recall_f1'][2]) > 1 and len(orig_metrics['precision_recall_f1'][2]) > 1 else np.nan,
            }
            
    return comparison


def analyse_intersectional_groups(model, test_data, features, target, attr1, attr2):
    """Analyse model performance for intersectional groups"""
    results = {}
    
    # Create combination groups
    test_data['intersect_group'] = test_data[attr1].astype(str) + '_' + test_data[attr2].astype(str)
    
    # Get predictions
    X_test = test_data[features]
    y_test = test_data[target]
    y_pred = model.predict(X_test)
    
    # Analyse each intersectional group
    for group in test_data['intersect_group'].unique():
        group_mask = test_data['intersect_group'] == group
        if group_mask.sum() >= 20:  # Only analyse groups with sufficient samples
            group_y_test = y_test[group_mask]
            group_y_pred = y_pred[group_mask]
            
            results[group] = {
                'count': group_mask.sum(),
                'accuracy': accuracy_score(group_y_test, group_y_pred),
                'precision_recall_f1': precision_recall_fscore_support(
                    group_y_test, group_y_pred, average=None
                )
            }
    
    return results