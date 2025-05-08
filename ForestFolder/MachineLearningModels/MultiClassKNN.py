#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, confusion_matrix, roc_curve
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

def train_knn_model(train_df, features, target, params=None):
    """
    Trains a K-Nearest Neighbors classifier on the provided training DataFrame for multi-class classification.
    
    Parameters:
      - train_df: pandas DataFrame with training data.
      - features: list of feature column names.
      - target: target column name.
      - params: dictionary of KNN parameters (optional).
      
    Returns:
      - model: a trained KNeighborsClassifier.
    """
    X_train = train_df[features]
    y_train = train_df[target]
    
    if params is None:
        params = {'n_neighbors': 5}
    model = KNeighborsClassifier(**params)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, test_df, features, target):
    X_test = test_df[features]
    y_test = test_df[target]
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba, multi_class='ovr') if y_proba is not None else None
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

def plot_roc_curve(y_test, y_proba, label):
    fpr, tpr, _ = roc_curve(y_test, y_proba.max(axis=1))
    plt.plot(fpr, tpr, label=label)

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