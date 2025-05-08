"""
Bias Analysis and Ethical Review Toolkit for Income Classification Data

This toolkit provides functions for:
1. Distribution analysis of sensitive characteristics
2. ML model training and evaluation
3. Fairness metrics calculation (demographic parity, equalized odds, etc.)
4. Visualization of bias and fairness metrics
5. Synthetic data generation with fairness constraints
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
import scipy.stats as stats
from sklearn.impute import SimpleImputer
import pickle
import os
import json
from datetime import datetime

# Define constants
INCOME_THRESHOLD = "50K"  # Income threshold (">50K" is positive class)
DEFAULT_SENSITIVE_ATTRS = ["race", "gender", "native_country"]
DEFAULT_TARGET = "income"
DEFAULT_POSITIVE_CLASS = 1  # Binary classification: 1 = >50K, 0 = <=50K

#########################
# Data Exploration Functions
#########################

def load_and_preprocess_data(file_path, target_col=DEFAULT_TARGET, sensitive_attrs=DEFAULT_SENSITIVE_ATTRS):
    """
    Load and perform basic preprocessing on the dataset.
    
    Parameters:
    -----------
    file_path : str
        Path to the CSV file containing the dataset
    target_col : str
        Name of the target column (default: "income")
    sensitive_attrs : list
        List of sensitive attribute columns
        
    Returns:
    --------
    df : pandas.DataFrame
        Preprocessed dataframe
    """
    # Load data
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"Error loading data: {e}")
        return None
    
    # Basic info
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Convert target to binary if needed
    if target_col in df.columns:
        if df[target_col].dtype == 'object':
            df[target_col] = df[target_col].apply(lambda x: 1 if INCOME_THRESHOLD in str(x) else 0)
            print(f"Converted {target_col} to binary (1: >{INCOME_THRESHOLD}, 0: ≤{INCOME_THRESHOLD})")
    
    # Check for missing values
    missing_values = df.isnull().sum()
    print("\nMissing values:")
    print(missing_values[missing_values > 0])
    
    # Basic data type checks and conversions
    for col in df.columns:
        if col in sensitive_attrs and df[col].dtype == 'object':
            # Keep as categorical
            print(f"Treating {col} as categorical sensitive attribute")
        elif df[col].dtype == 'object' and col != target_col:
            # Convert other categorical features
            try:
                if df[col].str.contains('\d').any():
                    # Try converting to numeric if it contains digits
                    df[col] = pd.to_numeric(df[col], errors='ignore')
            except:
                pass
    
    return df

def analyze_distributions(df, sensitive_attrs=DEFAULT_SENSITIVE_ATTRS, target_col=DEFAULT_TARGET):
    """
    Analyze and visualize the distribution of sensitive attributes in the dataset.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The dataset to analyze
    sensitive_attrs : list
        List of sensitive attribute columns
    target_col : str
        Name of the target column
        
    Returns:
    --------
    distributions : dict
        Dictionary containing distribution statistics
    """
    distributions = {}
    
    for attr in sensitive_attrs:
        if attr not in df.columns:
            print(f"Warning: {attr} not found in the dataset")
            continue
        
        # Overall distribution
        overall_dist = df[attr].value_counts(normalize=True).to_dict()
        
        # Distribution by target class
        if target_col in df.columns:
            dist_by_class = {}
            for class_val in sorted(df[target_col].unique()):
                class_label = f"{'>'+INCOME_THRESHOLD if class_val == 1 else '≤'+INCOME_THRESHOLD}"
                dist_by_class[class_label] = df[df[target_col]==class_val][attr].value_counts(normalize=True).to_dict()
            
            # Calculate representation disparity
            positive_class = 1  # Assuming 1 represents >50K
            representation_ratios = {}
            
            for category in overall_dist.keys():
                # Calculate ratio of positive outcomes for this category vs overall
                if category in dist_by_class[f">{INCOME_THRESHOLD}"]:
                    category_positive_rate = df[(df[attr]==category) & (df[target_col]==positive_class)].shape[0] / df[df[attr]==category].shape[0]
                    overall_positive_rate = df[df[target_col]==positive_class].shape[0] / df.shape[0]
                    representation_ratios[category] = category_positive_rate / overall_positive_rate
        
            distributions[attr] = {
                'overall': overall_dist,
                'by_class': dist_by_class,
                'representation_ratios': representation_ratios
            }
        else:
            distributions[attr] = {'overall': overall_dist}
    
    return distributions

def visualize_distributions(distributions, output_dir="plots"):
    """
    Create visualizations for the distribution analysis.
    
    Parameters:
    -----------
    distributions : dict
        Dictionary containing distribution statistics
    output_dir : str
        Directory to save plots (default: "plots")
        
    Returns:
    --------
    None
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for attr, dist_data in distributions.items():
        # Overall distribution
        plt.figure(figsize=(10, 6))
        overall_df = pd.DataFrame.from_dict(dist_data['overall'], orient='index', columns=['proportion'])
        overall_df.sort_values('proportion', ascending=False).plot(kind='bar', color='skyblue')
        plt.title(f'Distribution of {attr}')
        plt.ylabel('Proportion')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{attr}_overall_distribution.png")
        plt.close()
        
        # Distribution by class
        if 'by_class' in dist_data:
            plt.figure(figsize=(12, 8))
            by_class_df = pd.DataFrame(dist_data['by_class'])
            by_class_df.sort_index().plot(kind='bar')
            plt.title(f'Distribution of {attr} by Income Class')
            plt.ylabel('Proportion')
            plt.xticks(rotation=45, ha='right')
            plt.legend(title='Income Class')
            plt.tight_layout()
            plt.savefig(f"{output_dir}/{attr}_class_distribution.png")
            plt.close()
        
        # Representation ratios
        if 'representation_ratios' in dist_data:
            plt.figure(figsize=(12, 6))
            ratios_df = pd.DataFrame.from_dict(
                dist_data['representation_ratios'], 
                orient='index', 
                columns=['ratio']
            )
            # Sort by ratio value
            ratios_df.sort_values('ratio', ascending=False).plot(kind='bar', color='lightgreen')
            plt.axhline(y=1.0, color='red', linestyle='--', label='Parity')
            plt.title(f'Representation Ratio for {attr}\n(Ratio of >50K rate to overall >50K rate)')
            plt.ylabel('Ratio')
            plt.xticks(rotation=45, ha='right')
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"{output_dir}/{attr}_representation_ratio.png")
            plt.close()

def generate_distribution_report(distributions, output_file="distribution_report.md"):
    """
    Generate a Markdown report of the distribution analysis.
    
    Parameters:
    -----------
    distributions : dict
        Dictionary containing distribution statistics
    output_file : str
        Filename for the output report (default: "distribution_report.md")
        
    Returns:
    --------
    None
    """
    with open(output_file, 'w') as f:
        f.write("# Distribution Analysis Report\n\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        for attr, dist_data in distributions.items():
            f.write(f"## {attr.title()} Distribution\n\n")
            
            # Overall distribution
            f.write("### Overall Distribution\n\n")
            f.write("| Category | Proportion |\n")
            f.write("|----------|------------|\n")
            overall_sorted = sorted(dist_data['overall'].items(), key=lambda x: x[1], reverse=True)
            for category, prop in overall_sorted:
                f.write(f"| {category} | {prop:.2%} |\n")
            f.write("\n")
            
            # Distribution by class
            if 'by_class' in dist_data:
                f.write("### Distribution by Income Class\n\n")
                classes = list(dist_data['by_class'].keys())
                f.write(f"| Category | {' | '.join(classes)} |\n")
                f.write("|" + "-"*10 + "|" + "".join(["-"*10 + "|" for _ in classes]) + "\n")
                
                # Get all categories across all classes
                all_categories = set()
                for class_data in dist_data['by_class'].values():
                    all_categories.update(class_data.keys())
                
                for category in sorted(all_categories):
                    row = f"| {category} "
                    for class_name in classes:
                        prop = dist_data['by_class'][class_name].get(category, 0)
                        row += f"| {prop:.2%} "
                    row += "|\n"
                    f.write(row)
                f.write("\n")
            
            # Representation ratios
            if 'representation_ratios' in dist_data:
                f.write("### Representation Ratio (High Income)\n\n")
                f.write("| Category | Ratio |\n")
                f.write("|----------|------|\n")
                ratio_sorted = sorted(dist_data['representation_ratios'].items(), key=lambda x: x[1], reverse=True)
                for category, ratio in ratio_sorted:
                    color = "🔴" if ratio < 0.8 else "🟡" if ratio < 0.95 else "🟢"
                    f.write(f"| {category} | {ratio:.2f} {color} |\n")
                f.write("\n")
                f.write("🔴 Less than 80% of parity\n")
                f.write("🟡 Between 80% and 95% of parity\n")
                f.write("🟢 At least 95% of parity\n\n")
            
            f.write("\n")

#########################
# Feature Engineering Functions
#########################

def prepare_features(df, target_col=DEFAULT_TARGET, sensitive_attrs=DEFAULT_SENSITIVE_ATTRS, 
                     categorical_cols=None, numerical_cols=None, drop_cols=None):
    """
    Prepare features for ML model training with preprocessing pipeline.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The dataset
    target_col : str
        Name of the target column
    sensitive_attrs : list
        List of sensitive attributes
    categorical_cols : list
        List of categorical columns (if None, will be inferred)
    numerical_cols : list
        List of numerical columns (if None, will be inferred)
    drop_cols : list
        List of columns to drop (if None, will be empty list)
        
    Returns:
    --------
    X : numpy.ndarray
        Feature matrix
    y : numpy.ndarray
        Target vector
    preprocessor : sklearn.compose.ColumnTransformer
        Preprocessing pipeline
    feature_names : list
        List of feature names after preprocessing
    sensitive_idx : dict
        Dictionary mapping sensitive attributes to their indices in the feature matrix
    """
    if drop_cols is None:
        drop_cols = []
    
    # Create copies to avoid modifying original lists
    df = df.copy()
    
    # Separate features and target
    if target_col in df.columns:
        y = df[target_col].values
        X = df.drop(columns=[target_col] + drop_cols)
    else:
        print(f"Warning: Target column '{target_col}' not found in the dataset")
        y = None
        X = df.drop(columns=drop_cols)
    
    # Identify column types if not provided
    if categorical_cols is None:
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    
    if numerical_cols is None:
        numerical_cols = X.select_dtypes(include=['int', 'float']).columns.tolist()
    
    # Ensure all columns are accounted for
    all_cols = categorical_cols + numerical_cols
    for col in X.columns:
        if col not in all_cols:
            print(f"Warning: Column '{col}' not classified as categorical or numerical")
            # Determine type and add to appropriate list
            if X[col].dtype in ['object', 'category']:
                categorical_cols.append(col)
            else:
                numerical_cols.append(col)
    
    # Create preprocessing pipelines
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Create ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ]
    )
    
    # Fit and transform
    X_processed = preprocessor.fit_transform(X)
    
    # Get feature names
    feature_names = []
    # Numerical features keep their names
    feature_names.extend(numerical_cols)
    
    # Get categorical feature names
    cat_feature_names = []
    for i, col in enumerate(categorical_cols):
        ohe = preprocessor.transformers_[1][1].named_steps['onehot']
        categories = ohe.categories_[i]
        for category in categories:
            cat_feature_names.append(f"{col}_{category}")
    
    feature_names.extend(cat_feature_names)
    
    # Create mapping of sensitive attributes to their indices
    sensitive_idx = {}
    for attr in sensitive_attrs:
        if attr in categorical_cols:
            # Find the indices of the one-hot encoded features for this attribute
            idx = categorical_cols.index(attr)
            ohe = preprocessor.transformers_[1][1].named_steps['onehot']
            categories = ohe.categories_[idx]
            
            # Calculate start and end indices
            start_idx = len(numerical_cols)
            for i in range(idx):
                start_idx += len(preprocessor.transformers_[1][1].named_steps['onehot'].categories_[i])
            
            end_idx = start_idx + len(categories)
            
            sensitive_idx[attr] = (start_idx, end_idx, list(categories))
        elif attr in numerical_cols:
            # For numerical sensitive attributes, just get the index
            sensitive_idx[attr] = numerical_cols.index(attr)
    
    return X_processed, y, preprocessor, feature_names, sensitive_idx

#########################
# ML Training Functions
#########################

def train_and_evaluate_model(X, y, model_type='random_forest', test_size=0.2, random_state=42):
    """
    Train and evaluate an ML model.
    
    Parameters:
    -----------
    X : numpy.ndarray
        Feature matrix
    y : numpy.ndarray
        Target vector
    model_type : str
        Type of model to train (default: 'random_forest')
    test_size : float
        Proportion of data to use for testing (default: 0.2)
    random_state : int
        Random seed (default: 42)
        
    Returns:
    --------
    results : dict
        Dictionary containing model, predictions, and evaluation metrics
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Initialize model
    if model_type == 'random_forest':
        model = RandomForestClassifier(n_estimators=100, random_state=random_state)
    elif model_type == 'gradient_boosting':
        model = GradientBoostingClassifier(random_state=random_state)
    elif model_type == 'logistic_regression':
        model = LogisticRegression(max_iter=1000, random_state=random_state)
    else:
        print(f"Warning: Unknown model type '{model_type}'. Using RandomForest.")
        model = RandomForestClassifier(n_estimators=100, random_state=random_state)
    
    # Train model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
    
    # Evaluate model
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
    }
    
    if y_pred_proba is not None:
        metrics['roc_auc'] = roc_auc_score(y_test, y_pred_proba)
    
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred, output_dict=True)
    
    # Store results
    results = {
        'model': model,
        'model_type': model_type,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba,
        'metrics': metrics,
        'confusion_matrix': conf_matrix,
        'classification_report': class_report
    }
    
    return results

def visualize_model_results(results, output_dir="plots"):
    """
    Visualize model evaluation results.
    
    Parameters:
    -----------
    results : dict
        Dictionary containing model evaluation results
    output_dir : str
        Directory to save plots (default: "plots")
        
    Returns:
    --------
    None
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Confusion matrix
    plt.figure(figsize=(8, 6))
    conf_matrix = results['confusion_matrix']
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Predicted ≤50K', 'Predicted >50K'],
                yticklabels=['Actual ≤50K', 'Actual >50K'])
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/confusion_matrix.png")
    plt.close()
    
    # Metrics bar chart
    plt.figure(figsize=(10, 6))
    metrics = results['metrics']
    plt.bar(metrics.keys(), metrics.values(), color='skyblue')
    plt.title('Model Performance Metrics')
    plt.ylabel('Score')
    plt.ylim([0, 1])
    for i, v in enumerate(metrics.values()):
        plt.text(i, v + 0.02, f'{v:.3f}', ha='center')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/model_metrics.png")
    plt.close()
    
    # ROC curve if available
    if 'y_pred_proba' in results and results['y_pred_proba'] is not None:
        try:
            from sklearn.metrics import roc_curve, auc
            fpr, tpr, _ = roc_curve(results['y_test'], results['y_pred_proba'])
            roc_auc = auc(fpr, tpr)
            
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic (ROC) Curve')
            plt.legend(loc="lower right")
            plt.tight_layout()
            plt.savefig(f"{output_dir}/roc_curve.png")
            plt.close()
        except Exception as e:
            print(f"Error plotting ROC curve: {e}")
    
    # Feature importance if available
    if hasattr(results['model'], 'feature_importances_'):
        try:
            importances = results['model'].feature_importances_
            indices = np.argsort(importances)[-20:]  # Top 20 features
            
            plt.figure(figsize=(12, 8))
            plt.barh(range(len(indices)), importances[indices], color='lightgreen')
            plt.yticks(range(len(indices)), [f"Feature {i}" for i in indices])
            plt.xlabel('Feature Importance')
            plt.title('Top 20 Feature Importances')
            plt.tight_layout()
            plt.savefig(f"{output_dir}/feature_importance.png")
            plt.close()
        except Exception as e:
            print(f"Error plotting feature importance: {e}")

#########################
# Fairness Metrics Functions
#########################

def get_groups_from_sensitive_attr(X, sensitive_idx, attr_name):
    """
    Get group indices for a sensitive attribute.
    
    Parameters:
    -----------
    X : numpy.ndarray
        Feature matrix
    sensitive_idx : dict or tuple
        Dictionary mapping sensitive attributes to their indices in X
    attr_name : str
        Name of the sensitive attribute
        
    Returns:
    --------
    groups : dict
        Dictionary mapping group names to row indices
    """
    groups = {}
    
    if isinstance(sensitive_idx, dict):
        # Handle dict format
        attr_info = sensitive_idx[attr_name]
        
        if isinstance(attr_info, tuple) and len(attr_info) == 3:
            # Categorical attribute with one-hot encoding
            start_idx, end_idx, categories = attr_info
            
            # For each category, find rows where the corresponding column is 1
            for i, category in enumerate(categories):
                col_idx = start_idx + i
                group_indices = np.where(X[:, col_idx] == 1)[0]
                groups[str(category)] = group_indices
        else:
            # Numerical attribute
            col_idx = attr_info
            # Discretize into groups (example: split at median)
            median_val = np.median(X[:, col_idx])
            groups['Below_Median'] = np.where(X[:, col_idx] < median_val)[0]
            groups['Above_Median'] = np.where(X[:, col_idx] >= median_val)[0]
    
    return groups

def compute_fairness_metrics(results, sensitive_idx, sensitive_attrs=DEFAULT_SENSITIVE_ATTRS):
    """
    Compute fairness metrics for each sensitive attribute.
    
    Parameters:
    -----------
    results : dict
        Dictionary containing model evaluation results
    sensitive_idx : dict
        Dictionary mapping sensitive attributes to their indices in feature matrix
    sensitive_attrs : list
        List of sensitive attributes to analyze
        
    Returns:
    --------
    fairness_metrics : dict
        Dictionary containing fairness metrics for each sensitive attribute
    """
    fairness_metrics = {}
    
    # Get predictions and ground truth
    y_test = results['y_test']
    y_pred = results['y_pred']
    X_test = results['X_test']
    
    for attr in sensitive_attrs:
        if attr not in sensitive_idx:
            print(f"Warning: Sensitive attribute '{attr}' not found in sensitive_idx")
            continue
        
        # Get groups for this attribute
        groups = get_groups_from_sensitive_attr(X_test, sensitive_idx, attr)
        
        if not groups:
            print(f"Warning: No groups found for attribute '{attr}'")
            continue
        
        # Initialize metrics for this attribute
        attr_metrics = {
            'demographic_parity': {},
            'equal_opportunity': {},
            'equalized_odds': {},
            'accuracy_parity': {},
            'group_metrics': {}
        }
        
        # Calculate positive prediction rates for each group
        positive_rates = {}
        for group_name, indices in groups.items():
            if len(indices) == 0:
                continue
            positive_rates[group_name] = np.mean(y_pred[indices] == 1)
            
            # Calculate group-specific metrics
            group_y_true = y_test[indices]
            group_y_pred = y_pred[indices]
            
            if len(np.unique(group_y_true)) > 1:  # Ensure we have both classes
                attr_metrics['group_metrics'][group_name] = {
                    'accuracy': accuracy_score(group_y_true, group_y_pred),
                    'precision': precision_score(group_y_true, group_y_pred, zero_division=0),
                    'recall': recall_score(group_y_true, group_y_pred, zero_division=0),
                    'f1': f1_score(group_y_true, group_y_pred, zero_division=0),
                    'positive_rate': positive_rates[group_name],
                    'count': len(indices)
                }
        
        # Demographic Parity: difference in positive prediction rates
        group_names = list(positive_rates.keys())
        for i, group1 in enumerate(group_names):
            for group2 in group_names[i+1:]:
                diff = abs(positive_rates[group1] - positive_rates[group2])
                ratio = positive_rates[group1] / positive_rates[group2] if positive_rates[group2] > 0 else float('inf')
                attr_metrics['demographic_parity'][f'{group1}_vs_{group2}'] = {
                    'difference': diff,
                    'ratio': ratio,
                    'group1_rate': positive_rates[group1],
                    'group2_rate': positive_rates[group2]
                }
        
        # Equal Opportunity: difference in true positive rates
        true_positive_rates = {}
        for group_name, indices in groups.items():
            if len(indices) == 0:
                continue
            
            # Filter to only positive ground truth
            positive_indices = indices[y_test[indices] == 1]
            
            if len(positive_indices) > 0:
                true_positive_rates[group_name] = np.mean(y_pred[positive_indices] == 1)
        
        group_names = list(true_positive_rates.keys())
        for i, group1 in enumerate(group_names):
            for group2 in group_names[i+1:]:
                diff = abs(true_positive_rates[group1] - true_positive_rates[group2])
                ratio = true_positive_rates[group1] / true_positive_rates[group2] if true_positive_rates[group2] > 0 else float('inf')
                attr_metrics['equal_opportunity'][f'{group1}_vs_{group2}'] = {
                    'difference': diff,
                    'ratio': ratio,
                    'group1_tpr': true_positive_rates[group1],
                    'group2_tpr': true_positive_rates[group2]
                }
        
        # Equalized Odds: difference in TPR and FPR
        false_positive_rates = {}
        for group_name, indices in groups.items():
            if len(indices) == 0:
                continue
            
            # Filter to only negative ground truth
            negative_indices = indices[y_test[indices] == 0]
            
            if len(negative_indices) > 0:
                false_positive_rates[group_name] = np.mean(y_pred[negative_indices] == 1)
        
        group_names = list(set(true_positive_rates.keys()) & set(false_positive_rates.keys()))
        for i, group1 in enumerate(group_names):
            for group2 in group_names[i+1:]:
                tpr_diff = abs(true_positive_rates[group1] - true_positive_rates[group2])
                fpr_diff = abs(false_positive_rates[group1] - false_positive_rates[group2])
                attr_metrics['equalized_odds'][f'{group1}_vs_{group2}'] = {
                    'tpr_difference': tpr_diff,
                    'fpr_difference': fpr_diff,
                    'mean_difference': (tpr_diff + fpr_diff) / 2,
                    'group1_tpr': true_positive_rates[group1],
                    'group2_tpr': true_positive_rates[group2],
                    'group1_fpr': false_positive_rates[group1],
                    'group2_fpr': false_positive_rates[group2]
                }
        
        # Accuracy Parity: difference in accuracy rates
        accuracy_rates = {}
        for group_name, indices in groups.items():
            if len(indices) == 0:
                continue
            accuracy_rates[group_name] = accuracy_score(y_test[indices], y_pred[indices])
        
        group_names = list(accuracy_rates.keys())
        for i, group1 in enumerate(group_names):
            for group2 in group_names[i+1:]:
                diff = abs(accuracy_rates[group1] - accuracy_rates[group2])
                ratio = accuracy_rates[group1] / accuracy_rates[group2] if accuracy_rates[group2] > 0 else float('inf')
                attr_metrics['accuracy_parity'][f'{group1}_vs_{group2}'] = {
                    'difference': diff,
                    'ratio': ratio,
                    'group1_accuracy': accuracy_rates[group1],
                    'group2_accuracy': accuracy_rates[group2]
                }
        
        fairness_metrics[attr] = attr_metrics
    
    return fairness_metrics

def visualize_fairness_metrics(fairness_metrics, output_dir="plots"):
    """
    Visualize fairness metrics.
    
    Parameters:
    -----------
    fairness_metrics : dict
        Dictionary containing fairness metrics
    output_dir : str
        Directory to save plots (default: "plots")
        
    Returns:
    --------
    None
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for attr, metrics in fairness_metrics.items():
        # Demographic Parity
        if metrics['demographic_parity']:
            plt.figure(figsize=(12, 6))
            groups = []
            diffs = []
            ratios = []
            
            for comparison, values in metrics['demographic_parity'].items():
                groups.append(comparison)
                diffs.append(values['difference'])
                ratios.append(values['ratio'])
            
            x = np.arange(len(groups))
            width = 0.35
            
            fig, ax1 = plt.subplots(figsize=(14, 7))
            
            # Plot difference bars
            rects1 = ax1.bar(x - width/2, diffs, width, label='Difference', color='skyblue')
            ax1.set_ylabel('Difference in Positive Rate')
            ax1.set_title(f'Demographic Parity for {attr}')
            
            # Create second y-axis for ratios
            ax2 = ax1.twinx()
            
            # Plot ratio bars
            capped_ratios = [min(r, 5) for r in ratios]  # Cap ratios for better visualization
            rects2 = ax2.bar(x + width/2, capped_ratios, width, label='Ratio', color='lightgreen')
            ax2.set_ylabel('Ratio of Positive Rates')
            ax2.axhline(y=1.0, color='red', linestyle='--', alpha=0.7)
            
            # Add labels and legend
            ax1.set_xticks(x)
            ax1.set_xticklabels(groups, rotation=45, ha='right')
            
            # Add a legend
            ax1.legend(loc='upper left')
            ax2.legend(loc='upper right')
            
            plt.tight_layout()
            plt.savefig(f"{output_dir}/{attr}_demographic_parity.png")
            plt.close()
        
        # Group-specific metrics
        if metrics['group_metrics']:
            group_metrics_df = pd.DataFrame({
                group: values for group, values in metrics['group_metrics'].items()
            }).T
            
            # Metrics comparison
            plt.figure(figsize=(14, 8))
            
            # Create a grouped bar chart
            bar_width = 0.15
            x = np.arange(len(group_metrics_df))
            metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1', 'positive_rate']
            
            for i, metric in enumerate(metrics_to_plot):
                plt.bar(x + i*bar_width, group_metrics_df[metric], 
                       width=bar_width, label=metric.capitalize())
            
            plt.xlabel('Group')
            plt.ylabel('Score')
            plt.title(f'Performance Metrics by {attr} Group')
            plt.xticks(x + bar_width*2, group_metrics_df.index, rotation=45, ha='right')
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"{output_dir}/{attr}_group_metrics.png")
            plt.close()
            
            # Group counts
            plt.figure(figsize=(10, 6))
            plt.bar(group_metrics_df.index, group_metrics_df['count'], color='lightblue')
            plt.title(f'Sample Count by {attr} Group')
            plt.ylabel('Count')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(f"{output_dir}/{attr}_group_counts.png")
            plt.close()

def generate_fairness_report(fairness_metrics, output_file="fairness_report.md"):
    """
    Generate a Markdown report of fairness metrics.
    
    Parameters:
    -----------
    fairness_metrics : dict
        Dictionary containing fairness metrics
    output_file : str
        Filename for the output report (default: "fairness_report.md")
        
    Returns:
    --------
    None
    """
    with open(output_file, 'w') as f:
        f.write("# Fairness Metrics Report\n\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        for attr, metrics in fairness_metrics.items():
            f.write(f"## {attr.title()} Fairness Analysis\n\n")
            
            # Group metrics
            if metrics['group_metrics']:
                f.write("### Group-Specific Performance\n\n")
                f.write("| Group | Count | Accuracy | Precision | Recall | F1 | Positive Rate |\n")
                f.write("|-------|-------|----------|-----------|--------|----|--------------|\n")
                
                for group, vals in metrics['group_metrics'].items():
                    f.write(f"| {group} | {vals['count']} | {vals['accuracy']:.3f} | {vals['precision']:.3f} | {vals['recall']:.3f} | {vals['f1']:.3f} | {vals['positive_rate']:.3f} |\n")
                f.write("\n")
            
            # Demographic parity
            if metrics['demographic_parity']:
                f.write("### Demographic Parity (Equal Positive Rate)\n\n")
                f.write("| Comparison | Group 1 Rate | Group 2 Rate | Difference | Ratio | Assessment |\n")
                f.write("|------------|-------------|-------------|------------|-------|------------|\n")
                
                for comparison, vals in metrics['demographic_parity'].items():
                    # Assessment based on the 80% rule (ratio between 0.8 and 1.25)
                    ratio = vals['ratio']
                    if 0.8 <= ratio <= 1.25:
                        assessment = "✅ Fair"
                    elif 0.7 <= ratio <= 1.43:
                        assessment = "⚠️ Borderline"
                    else:
                        assessment = "❌ Potential bias"
                    
                    f.write(f"| {comparison} | {vals['group1_rate']:.3f} | {vals['group2_rate']:.3f} | {vals['difference']:.3f} | {vals['ratio']:.2f} | {assessment} |\n")
                f.write("\n")
            
            # Equal opportunity
            if metrics['equal_opportunity']:
                f.write("### Equal Opportunity (Equal True Positive Rate)\n\n")
                f.write("| Comparison | Group 1 TPR | Group 2 TPR | Difference | Ratio | Assessment |\n")
                f.write("|------------|------------|------------|------------|-------|------------|\n")
                
                for comparison, vals in metrics['equal_opportunity'].items():
                    # Assessment based on the 80% rule
                    ratio = vals['ratio']
                    if 0.8 <= ratio <= 1.25:
                        assessment = "✅ Fair"
                    elif 0.7 <= ratio <= 1.43:
                        assessment = "⚠️ Borderline"
                    else:
                        assessment = "❌ Potential bias"
                    
                    f.write(f"| {comparison} | {vals['group1_tpr']:.3f} | {vals['group2_tpr']:.3f} | {vals['difference']:.3f} | {vals['ratio']:.2f} | {assessment} |\n")
                f.write("\n")
            
            # Equalized odds
            if metrics['equalized_odds']:
                f.write("### Equalized Odds (Equal TPR and FPR)\n\n")
                f.write("| Comparison | TPR Diff | FPR Diff | Mean Diff | Assessment |\n")
                f.write("|------------|----------|----------|-----------|------------|\n")
                
                for comparison, vals in metrics['equalized_odds'].items():
                    # Assessment based on mean difference
                    mean_diff = vals['mean_difference']
                    if mean_diff <= 0.05:
                        assessment = "✅ Fair"
                    elif mean_diff <= 0.10:
                        assessment = "⚠️ Borderline"
                    else:
                        assessment = "❌ Potential bias"
                    
                    f.write(f"| {comparison} | {vals['tpr_difference']:.3f} | {vals['fpr_difference']:.3f} | {vals['mean_difference']:.3f} | {assessment} |\n")
                f.write("\n")
            
            # Accuracy parity
            if metrics['accuracy_parity']:
                f.write("### Accuracy Parity\n\n")
                f.write("| Comparison | Group 1 Accuracy | Group 2 Accuracy | Difference | Assessment |\n")
                f.write("|------------|-----------------|------------------|------------|------------|\n")
                
                for comparison, vals in metrics['accuracy_parity'].items():
                    # Assessment based on difference
                    diff = vals['difference']
                    if diff <= 0.03:
                        assessment = "✅ Fair"
                    elif diff <= 0.05:
                        assessment = "⚠️ Borderline"
                    else:
                        assessment = "❌ Potential bias"
                    
                    f.write(f"| {comparison} | {vals['group1_accuracy']:.3f} | {vals['group2_accuracy']:.3f} | {vals['difference']:.3f} | {assessment} |\n")
                f.write("\n")
            
            f.write("\n")

#########################
# Synthetic Data Generation Functions
#########################

def generate_synthetic_data(df, sensitive_attrs=DEFAULT_SENSITIVE_ATTRS, target_col=DEFAULT_TARGET, 
                          mode='preserve', balancing=None, noise_level=0.1, n_samples=None):
    """
    Generate synthetic data with fairness considerations.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Original dataframe
    sensitive_attrs : list
        List of sensitive attributes
    target_col : str
        Name of the target column
    mode : str
        Mode of generation ('preserve', 'balance', 'remove')
    balancing : dict
        Dictionary mapping sensitive attributes to target ratios
    noise_level : float
        Level of noise to add (0-1)
    n_samples : int
        Number of samples to generate (if None, same as original)
        
    Returns:
    --------
    df_synthetic : pandas.DataFrame
        Synthetic dataframe
    """
    if n_samples is None:
        n_samples = len(df)
    
    # Make a copy of the original dataframe
    df_orig = df.copy()
    
    if mode == 'remove':
        # Remove sensitive attributes completely
        for attr in sensitive_attrs:
            if attr in df_orig.columns:
                df_orig = df_orig.drop(columns=[attr])
    
    # Analyze original distributions
    orig_distributions = {}
    for attr in sensitive_attrs:
        if attr in df_orig.columns:
            # Overall distribution
            attr_dist = df_orig[attr].value_counts(normalize=True).to_dict()
            
            # Distribution by target
            if target_col in df_orig.columns:
                target_vals = df_orig[target_col].unique()
                attr_by_target = {}
                for val in target_vals:
                    attr_by_target[val] = df_orig[df_orig[target_col] == val][attr].value_counts(normalize=True).to_dict()
                
                orig_distributions[attr] = {
                    'overall': attr_dist,
                    'by_target': attr_by_target
                }
            else:
                orig_distributions[attr] = {'overall': attr_dist}
    
    # Create synthetic data
    df_synthetic = pd.DataFrame(columns=df_orig.columns)
    
    # Determine sampling strategy based on mode
    if mode == 'preserve':
        # Sample with replacement from original data
        sample_idx = np.random.choice(len(df_orig), size=n_samples, replace=True)
        df_synthetic = df_orig.iloc[sample_idx].reset_index(drop=True)
        
        # Add noise to numerical columns
        for col in df_synthetic.select_dtypes(include=['int', 'float']).columns:
            if col != target_col:  # Don't add noise to target
                col_std = df_synthetic[col].std()
                if col_std > 0:
                    noise = np.random.normal(0, noise_level * col_std, size=len(df_synthetic))
                    df_synthetic[col] = df_synthetic[col] + noise
                    # Round to nearest integer if original was integer
                    if df_orig[col].dtype == 'int64':
                        df_synthetic[col] = np.round(df_synthetic[col]).astype('int64')
    
    elif mode == 'balance':
        # Create balanced dataset with respect to sensitive attributes and target
        if balancing is None:
            balancing = {}
            for attr in sensitive_attrs:
                if attr in df_orig.columns:
                    # Default to equal distribution
                    categories = df_orig[attr].unique()
                    balancing[attr] = {cat: 1/len(categories) for cat in categories}
        
        # Calculate samples per group
        samples_per_group = {}
        
        for attr, ratios in balancing.items():
            # Normalize ratios to sum to 1
            total = sum(ratios.values())
            normalized_ratios = {k: v/total for k, v in ratios.items()}
            
            # Calculate samples per category
            samples_per_group[attr] = {k: int(v * n_samples) for k, v in normalized_ratios.items()}
            
            # Adjust for rounding errors
            total_samples = sum(samples_per_group[attr].values())
            if total_samples < n_samples:
                # Add remaining samples to the largest group
                largest_group = max(normalized_ratios, key=normalized_ratios.get)
                samples_per_group[attr][largest_group] += n_samples - total_samples
        
        # Generate data for each group
        for attr, samples in samples_per_group.items():
            for category, n in samples.items():
                if n > 0:
                    # Get original data for this category
                    category_data = df_orig[df_orig[attr] == category]
                    
                    if len(category_data) > 0:
                        # Sample with replacement
                        category_sample_idx = np.random.choice(len(category_data), size=n, replace=True)
                        category_synthetic = category_data.iloc[category_sample_idx].reset_index(drop=True)
                        
                        # Add noise to numerical columns
                        for col in category_synthetic.select_dtypes(include=['int', 'float']).columns:
                            if col != target_col:  # Don't add noise to target
                                col_std = category_synthetic[col].std()
                                if col_std > 0:
                                    noise = np.random.normal(0, noise_level * col_std, size=len(category_synthetic))
                                    category_synthetic[col] = category_synthetic[col] + noise
                                    # Round to nearest integer if original was integer
                                    if df_orig[col].dtype == 'int64':
                                        category_synthetic[col] = np.round(category_synthetic[col]).astype('int64')
                        
                        # Append to synthetic data
                        df_synthetic = pd.concat([df_synthetic, category_synthetic])
    
    # Reset index
    df_synthetic = df_synthetic.reset_index(drop=True)
    
    return df_synthetic

def compare_distributions(df_original, df_synthetic, sensitive_attrs=DEFAULT_SENSITIVE_ATTRS, 
                         target_col=DEFAULT_TARGET, output_dir="plots"):
    """
    Compare distributions between original and synthetic data.
    
    Parameters:
    -----------
    df_original : pandas.DataFrame
        Original dataframe
    df_synthetic : pandas.DataFrame
        Synthetic dataframe
    sensitive_attrs : list
        List of sensitive attributes
    target_col : str
        Name of the target column
    output_dir : str
        Directory to save plots (default: "plots")
        
    Returns:
    --------
    comparison : dict
        Dictionary containing distribution comparisons
    """
    os.makedirs(output_dir, exist_ok=True)
    comparison = {}
    
    # Overall target distribution
    if target_col in df_original.columns and target_col in df_synthetic.columns:
        orig_target_dist = df_original[target_col].value_counts(normalize=True)
        synth_target_dist = df_synthetic[target_col].value_counts(normalize=True)
        
        plt.figure(figsize=(10, 6))
        bar_width = 0.35
        x = np.arange(len(orig_target_dist))
        
        plt.bar(x - bar_width/2, orig_target_dist, bar_width, label='Original', color='skyblue')
        plt.bar(x + bar_width/2, synth_target_dist, bar_width, label='Synthetic', color='lightgreen')
        
        plt.xlabel('Target Class')
        plt.ylabel('Proportion')
        plt.title('Target Distribution Comparison')
        plt.xticks(x, orig_target_dist.index)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{output_dir}/target_distribution_comparison.png")
        plt.close()
        
        comparison['target'] = {
            'original': orig_target_dist.to_dict(),
            'synthetic': synth_target_dist.to_dict(),
            'js_divergence': stats.entropy(orig_target_dist, synth_target_dist) / 2 + stats.entropy(synth_target_dist, orig_target_dist) / 2
        }
    
    # Sensitive attribute distributions
    for attr in sensitive_attrs:
        if attr in df_original.columns and attr in df_synthetic.columns:
            orig_attr_dist = df_original[attr].value_counts(normalize=True)
            synth_attr_dist = df_synthetic[attr].value_counts(normalize=True)
            
            # Align distributions (handle categories present in one but not the other)
            all_categories = sorted(set(orig_attr_dist.index) | set(synth_attr_dist.index))
            orig_aligned = pd.Series({cat: orig_attr_dist.get(cat, 0) for cat in all_categories})
            synth_aligned = pd.Series({cat: synth_attr_dist.get(cat, 0) for cat in all_categories})
            
            # Plot comparison
            plt.figure(figsize=(12, 6))
            bar_width = 0.35
            x = np.arange(len(all_categories))
            
            plt.bar(x - bar_width/2, orig_aligned, bar_width, label='Original', color='skyblue')
            plt.bar(x + bar_width/2, synth_aligned, bar_width, label='Synthetic', color='lightgreen')
            
            plt.xlabel(attr)
            plt.ylabel('Proportion')
            plt.title(f'{attr} Distribution Comparison')
            plt.xticks(x, all_categories, rotation=45, ha='right')
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"{output_dir}/{attr}_distribution_comparison.png")
            plt.close()
            
            # Compute JS divergence
            js_div = stats.entropy(orig_aligned, synth_aligned) / 2 + stats.entropy(synth_aligned, orig_aligned) / 2
            
            comparison[attr] = {
                'original': orig_aligned.to_dict(),
                'synthetic': synth_aligned.to_dict(),
                'js_divergence': js_div
            }
            
            # Conditional distributions by target
            if target_col in df_original.columns and target_col in df_synthetic.columns:
                for target_val in sorted(df_original[target_col].unique()):
                    orig_cond_dist = df_original[df_original[target_col] == target_val][attr].value_counts(normalize=True)
                    synth_cond_dist = df_synthetic[df_synthetic[target_col] == target_val][attr].value_counts(normalize=True)
                    
                    # Align distributions
                    orig_cond_aligned = pd.Series({cat: orig_cond_dist.get(cat, 0) for cat in all_categories})
                    synth_cond_aligned = pd.Series({cat: synth_cond_dist.get(cat, 0) for cat in all_categories})
                    
                    # Plot comparison
                    plt.figure(figsize=(12, 6))
                    
                    plt.bar(x - bar_width/2, orig_cond_aligned, bar_width, label='Original', color='skyblue')
                    plt.bar(x + bar_width/2, synth_cond_aligned, bar_width, label='Synthetic', color='lightgreen')
                    
                    target_label = f">{INCOME_THRESHOLD}" if target_val == 1 else f"≤{INCOME_THRESHOLD}"
                    plt.xlabel(attr)
                    plt.ylabel('Proportion')
                    plt.title(f'{attr} Distribution Comparison (Target = {target_label})')
                    plt.xticks(x, all_categories, rotation=45, ha='right')
                    plt.legend()
                    plt.tight_layout()
                    plt.savefig(f"{output_dir}/{attr}_target{target_val}_distribution_comparison.png")
                    plt.close()
                    
                    # Compute JS divergence
                    js_div_cond = stats.entropy(orig_cond_aligned, synth_cond_aligned) / 2 + stats.entropy(synth_cond_aligned, orig_cond_aligned) / 2
                    
                    if 'conditional' not in comparison[attr]:
                        comparison[attr]['conditional'] = {}
                    
                    comparison[attr]['conditional'][target_val] = {
                        'original': orig_cond_aligned.to_dict(),
                        'synthetic': synth_cond_aligned.to_dict(),
                        'js_divergence': js_div_cond
                    }
    
    return comparison

#########################
# Main Analysis Functions
#########################

def run_full_analysis(df, sensitive_attrs=DEFAULT_SENSITIVE_ATTRS, target_col=DEFAULT_TARGET, 
                     model_type='random_forest', output_dir="bias_analysis_results"):
    """
    Run a complete bias analysis pipeline.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The dataset to analyze
    sensitive_attrs : list
        List of sensitive attributes
    target_col : str
        Name of the target column
    model_type : str
        Type of model to train
    output_dir : str
        Directory for output files
        
    Returns:
    --------
    results : dict
        Dictionary containing all analysis results
    """
    os.makedirs(output_dir, exist_ok=True)
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    print("Starting comprehensive bias analysis...")
    results = {}
    
    # Step 1: Distribution analysis
    print("\n1. Analyzing distributions of sensitive attributes...")
    distributions = analyze_distributions(df, sensitive_attrs, target_col)
    visualize_distributions(distributions, output_dir=plots_dir)
    generate_distribution_report(distributions, output_file=os.path.join(output_dir, "distribution_report.md"))
    results['distributions'] = distributions
    print("✅ Distribution analysis complete")
    
    # Step 2: Prepare features
    print("\n2. Preparing features for ML model...")
    X_processed, y, preprocessor, feature_names, sensitive_idx = prepare_features(df, target_col, sensitive_attrs)
    
    if y is None:
        print("❌ Target column not found. Cannot proceed with model training.")
        return results
    
    results['feature_info'] = {
        'preprocessor': preprocessor,
        'feature_names': feature_names,
        'sensitive_idx': sensitive_idx
    }
    print("✅ Feature preparation complete")
    
    # Step 3: Train and evaluate model
    print("\n3. Training and evaluating ML model...")
    model_results = train_and_evaluate_model(X_processed, y, model_type=model_type)
    visualize_model_results(model_results, output_dir=plots_dir)
    results['model_results'] = model_results
    print(f"✅ Model training complete. Accuracy: {model_results['metrics']['accuracy']:.4f}")
    
    # Step 4: Compute fairness metrics
    print("\n4. Computing fairness metrics...")
    fairness_metrics = compute_fairness_metrics(model_results, sensitive_idx, sensitive_attrs)
    visualize_fairness_metrics(fairness_metrics, output_dir=plots_dir)
    generate_fairness_report(fairness_metrics, output_file=os.path.join(output_dir, "fairness_report.md"))
    results['fairness_metrics'] = fairness_metrics
    print("✅ Fairness analysis complete")
    
    # Step 5: Generate synthetic data if requested
    print("\n5. Generating balanced synthetic data...")
    # Example: Generate balanced synthetic data with equal representation
    balancing = {}
    for attr in sensitive_attrs:
        if attr in df.columns:
            categories = df[attr].unique()
            balancing[attr] = {cat: 1/len(categories) for cat in categories}
    
    df_synthetic = generate_synthetic_data(
        df, 
        sensitive_attrs=sensitive_attrs, 
        target_col=target_col,
        mode='balance', 
        balancing=balancing
    )
    
    distribution_comparison = compare_distributions(
        df, df_synthetic, 
        sensitive_attrs=sensitive_attrs, 
        target_col=target_col,
        output_dir=plots_dir
    )
    
    # Save synthetic data
    df_synthetic.to_csv(os.path.join(output_dir, "balanced_synthetic_data.csv"), index=False)
    
    results['synthetic_data'] = {
        'synthetic_df_shape': df_synthetic.shape,
        'distribution_comparison': distribution_comparison
    }
    print("✅ Synthetic data generation complete")
    
    # Step 6: Train model on synthetic data and compare fairness
    print("\n6. Evaluating model on synthetic data...")
    X_synth, y_synth, synth_preprocessor, synth_feature_names, synth_sensitive_idx = prepare_features(
        df_synthetic, target_col, sensitive_attrs
    )
    
    synth_model_results = train_and_evaluate_model(X_synth, y_synth, model_type=model_type)
    
    synth_fairness_metrics = compute_fairness_metrics(
        synth_model_results, synth_sensitive_idx, sensitive_attrs
    )
    
    # Generate comparison report
    with open(os.path.join(output_dir, "synthetic_vs_original_report.md"), 'w') as f:
        f.write("# Synthetic Data vs Original Data Comparison\n\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Model Performance Comparison\n\n")
        f.write("| Metric | Original Data | Synthetic Data | Difference |\n")
        f.write("|--------|---------------|---------------|------------|\n")
        
        for metric in ['accuracy', 'precision', 'recall', 'f1']:
            orig_val = model_results['metrics'][metric]
            synth_val = synth_model_results['metrics'][metric]
            diff = synth_val - orig_val
            f.write(f"| {metric.capitalize()} | {orig_val:.4f} | {synth_val:.4f} | {diff:+.4f} |\n")
        
        f.write("\n## Fairness Metrics Comparison\n\n")
        
        for attr in sensitive_attrs:
            if attr in fairness_metrics and attr in synth_fairness_metrics:
                f.write(f"### {attr.title()} Fairness Comparison\n\n")
                
                # Compare demographic parity
                if fairness_metrics[attr]['demographic_parity'] and synth_fairness_metrics[attr]['demographic_parity']:
                    f.write("#### Demographic Parity Comparison\n\n")
                    f.write("| Comparison | Original Difference | Synthetic Difference | Change |\n")
                    f.write("|------------|---------------------|----------------------|-------|\n")
                    
                    for comparison in fairness_metrics[attr]['demographic_parity']:
                        if comparison in synth_fairness_metrics[attr]['demographic_parity']:
                            orig_diff = fairness_metrics[attr]['demographic_parity'][comparison]['difference']
                            synth_diff = synth_fairness_metrics[attr]['demographic_parity'][comparison]['difference']
                            change = synth_diff - orig_diff
                            change_str = f"{change:+.4f}"
                            
                            # Add emoji to indicate improvement or worsening
                            if change < -0.01:  # Improvement (less disparity)
                                change_str += " ✅"
                            elif change > 0.01:  # Worsening (more disparity)
                                change_str += " ❌"
                            else:  # No significant change
                                change_str += " ➖"
                            
                            f.write(f"| {comparison} | {orig_diff:.4f} | {synth_diff:.4f} | {change_str} |\n")
                    
                    f.write("\n")
    
    results['synthetic_evaluation'] = {
        'model_results': synth_model_results,
        'fairness_metrics': synth_fairness_metrics
    }
    print("✅ Synthetic data evaluation complete")
    
    # Step 7: Save all results
    print("\n7. Saving all results...")
    
    # Save summary report
    with open(os.path.join(output_dir, "summary_report.md"), 'w') as f:
        f.write("# Bias Analysis and Ethical Review Summary\n\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Dataset Overview\n\n")
        f.write(f"- Total samples: {len(df)}\n")
        if target_col in df.columns:
            pos_rate = df[df[target_col] == 1].shape[0] / df.shape[0]
            f.write(f"- Positive class (>{INCOME_THRESHOLD}) rate: {pos_rate:.2%}\n")
        
        f.write("\n## Sensitive Attributes Analysis\n\n")
        for attr in sensitive_attrs:
            if attr in distributions:
                f.write(f"### {attr.title()}\n\n")
                
                # Top 3 representation ratios
                if 'representation_ratios' in distributions[attr]:
                    ratios = distributions[attr]['representation_ratios']
                    sorted_ratios = sorted(ratios.items(), key=lambda x: x[1], reverse=True)
                    
                    f.write("#### Representation Analysis\n\n")
                    f.write("Top 3 overrepresented groups in high income class:\n")
                    for i, (group, ratio) in enumerate(sorted_ratios[:3], 1):
                        f.write(f"{i}. {group}: {ratio:.2f}x average rate\n")
                    
                    f.write("\nBottom 3 underrepresented groups in high income class:\n")
                    for i, (group, ratio) in enumerate(sorted_ratios[-3:], 1):
                        f.write(f"{i}. {group}: {ratio:.2f}x average rate\n")
                    
                    f.write("\n")
        
        f.write("## Model Performance\n\n")
        f.write(f"- Model type: {model_results['model_type']}\n")
        f.write(f"- Overall accuracy: {model_results['metrics']['accuracy']:.4f}\n")
        f.write(f"- Precision: {model_results['metrics']['precision']:.4f}\n")
        f.write(f"- Recall: {model_results['metrics']['recall']:.4f}\n")
        f.write(f"- F1 score: {model_results['metrics']['f1']:.4f}\n")
        
        f.write("\n## Fairness Concerns\n\n")
        
        # Identify top fairness concerns
        concerns = []
        
        for attr, metrics in fairness_metrics.items():
            # Check demographic parity
            for comparison, values in metrics['demographic_parity'].items():
                if values['ratio'] < 0.8 or values['ratio'] > 1.25:
                    concerns.append({
                        'attribute': attr,
                        'comparison': comparison,
                        'metric': 'Demographic Parity',
                        'value': values['ratio'],
                        'severity': abs(1 - values['ratio'])
                    })
            
            # Check accuracy parity
            for comparison, values in metrics['accuracy_parity'].items():
                if values['difference'] > 0.05:
                    concerns.append({
                        'attribute': attr,
                        'comparison': comparison,
                        'metric': 'Accuracy Parity',
                        'value': values['difference'],
                        'severity': values['difference']
                    })
        
        # Sort concerns by severity
        concerns.sort(key=lambda x: x['severity'], reverse=True)
        
        if concerns:
            f.write("Top fairness concerns identified:\n\n")
            for i, concern in enumerate(concerns[:5], 1):
                f.write(f"{i}. **{concern['attribute']} - {concern['comparison']}** ({concern['metric']}): ")
                
                if concern['metric'] == 'Demographic Parity':
                    f.write(f"Ratio of {concern['value']:.2f} indicates ")
                    if concern['value'] < 1:
                        f.write(f"underrepresentation by {(1-concern['value'])*100:.1f}%\n")
                    else:
                        f.write(f"overrepresentation by {(concern['value']-1)*100:.1f}%\n")
                else:
                    f.write(f"Difference of {concern['value']:.4f}\n")
            
            f.write("\n")
        else:
            f.write("No major fairness concerns identified.\n\n")
        
        f.write("## Mitigation Strategies\n\n")
        f.write("1. **Synthetic Data Generation**: Balanced synthetic data was generated to address representation disparities.\n")
        f.write("   - Impact: ")
        
        # Calculate overall impact
        orig_disparity = 0
        synth_disparity = 0
        count = 0
        
        for attr in sensitive_attrs:
            if attr in fairness_metrics and attr in synth_fairness_metrics:
                for comparison in fairness_metrics[attr]['demographic_parity']:
                    if comparison in synth_fairness_metrics[attr]['demographic_parity']:
                        orig_diff = fairness_metrics[attr]['demographic_parity'][comparison]['difference']
                        synth_diff = synth_fairness_metrics[attr]['demographic_parity'][comparison]['difference']
                        orig_disparity += orig_diff
                        synth_disparity += synth_diff
                        count += 1
        
        if count > 0:
            avg_orig = orig_disparity / count
            avg_synth = synth_disparity / count
            pct_change = (avg_synth - avg_orig) / avg_orig * 100
            
            if pct_change < -5:
                f.write(f"Reduced average disparity by {abs(pct_change):.1f}%\n")
            elif pct_change > 5:
                f.write(f"Increased average disparity by {pct_change:.1f}%\n")
            else:
                f.write("No significant change in overall disparity\n")
        
        f.write("\n2. **Additional Recommended Strategies**:\n")
        f.write("   - Collect more data from underrepresented groups\n")
        f.write("   - Apply fairness constraints during model training\n")
        f.write("   - Consider using explainable AI techniques to understand model decisions\n")
        f.write("   - Implement post-processing techniques to equalize error rates\n")
        
        f.write("\n## Next Steps\n\n")
        f.write("1. Conduct stakeholder consultations with representatives from affected groups\n")
        f.write("2. Develop a monitoring plan to track model fairness metrics over time\n")
        f.write("3. Create clear documentation for model users about limitations and potential biases\n")
        f.write("4. Establish a process for receiving and addressing fairness-related feedback\n")
    
    # Save model and preprocessor
    with open(os.path.join(output_dir, "model.pkl"), 'wb') as f:
        pickle.dump(model_results['model'], f)
    
    with open(os.path.join(output_dir, "preprocessor.pkl"), 'wb') as f:
        pickle.dump(preprocessor, f)
    
    print("✅ All results saved to", output_dir)
    
    return results

def command_line_interface():
    """
    Simple command line interface for running the bias analysis toolkit.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Bias Analysis and Ethical Review Toolkit')
    parser.add_argument('--file', type=str, required=True, help='Path to the CSV file')
    parser.add_argument('--target', type=str, default='income', help='Name of the target column')
    parser.add_argument('--sensitive', type=str, nargs='+', default=['race', 'gender', 'native_country'],
                       help='List of sensitive attributes')
    parser.add_argument('--model', type=str, choices=['random_forest', 'gradient_boosting', 'logistic_regression'],
                       default='random_forest', help='Model type to use')
    parser.add_argument('--output', type=str, default='bias_analysis_results',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Load data
    df = load_and_preprocess_data(args.file, args.target, args.sensitive)
    
    if df is not None:
        # Run analysis
        results = run_full_analysis(
            df, 
            sensitive_attrs=args.sensitive,
            target_col=args.target,
            model_type=args.model,
            output_dir=args.output
        )
        
        print("\nAnalysis complete!")
        print(f"Results saved to {args.output}/")
    else:
        print("Error loading data. Exiting.")

#########################
# Example
#########################

def example_analysis():
    """
    Example of how to use the bias analysis toolkit with a sample dataset.
    
    Note: This is typically used with income prediction datasets like Adult Census Income.
    """
    # Sample usage with Adult dataset
    from sklearn.datasets import fetch_openml
    
    # Load the Adult dataset
    print("Loading Adult dataset...")
    adult = fetch_openml(name='adult', version=2, as_frame=True)
    df = adult.frame
    
    # Rename columns to match expected names
    df = df.rename(columns={
        'class': 'income',
        'sex': 'gender',
        'race': 'race',
        'native-country': 'native_country'
    })
    
    # Run analysis
    results = run_full_analysis(
        df,
        sensitive_attrs=['race', 'gender', 'native_country'],
        target_col='income',
        model_type='random_forest',
        output_dir='adult_bias_analysis'
    )
    
    print("\nExample analysis complete!")
    print("Results saved to adult_bias_analysis/")
