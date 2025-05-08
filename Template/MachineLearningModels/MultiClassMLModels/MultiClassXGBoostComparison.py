import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def train_xgb_model(train_df, features, target, params=None):
    """
    Trains an XGBoost classifier on the provided training DataFrame for multi-class classification.
    
    Parameters:
      - train_df: pandas DataFrame with training data.
      - features: list of feature column names.
      - target: target column name.
      - params: dictionary of XGBoost parameters (optional).
      
    Returns:
      - model: a trained XGBClassifier.
    """
    X_train = train_df[features]
    y_train = train_df[target]
    num_class = len(np.unique(y_train))
    if params is None:
        params = {
            'objective': 'multi:softprob',
            'eval_metric': 'mlogloss',
            'use_label_encoder': False,
            'seed': 42,
            'num_class': num_class
        }
    else:
        params['num_class'] = num_class
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, test_df, features, target):
    """
    Evaluates a trained model on the test dataset for multi-class classification.
    
    Returns a dictionary containing:
      - accuracy, AUC, classification report, confusion matrix,
      - y_test, y_pred, and predicted probabilities.
    """
    X_test = test_df[features]
    y_test = test_df[target]
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    acc = accuracy_score(y_test, y_pred)
    # Compute multi-class AUC using one-vs-rest
    auc = roc_auc_score(y_test, y_proba, multi_class='ovr')
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)
    return {
        'accuracy': acc,
        'auc': auc,
        'classification_report': report,
        'confusion_matrix': cm,
        'y_test': y_test,
        'y_pred': y_pred,
        'y_proba': y_proba
    }

def compare_models(original_metrics, augmented_metrics, target_names):
    """
    Prints evaluation metrics for both models using actual target names in the classification report.
    """
    print("=== Original Training Model Metrics ===")
    print(f"Accuracy: {original_metrics['accuracy']:.3f}")
    print(f"AUC: {original_metrics['auc']:.3f}")
    print("Classification Report:")
    print(classification_report(original_metrics['y_test'], original_metrics['y_pred'], target_names=target_names))
    print("Confusion Matrix:")
    print(original_metrics['confusion_matrix'])
    
    print("\n=== Augmented Training Model Metrics ===")
    print(f"Accuracy: {augmented_metrics['accuracy']:.3f}")
    print(f"AUC: {augmented_metrics['auc']:.3f}")
    print("Classification Report:")
    print(classification_report(augmented_metrics['y_test'], augmented_metrics['y_pred'], target_names=target_names))
    print("Confusion Matrix:")
    print(augmented_metrics['confusion_matrix'])
    
