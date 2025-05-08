import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, confusion_matrix, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

def train_logistic_model(train_df, features, target, params=None):
    """
    Trains a Logistic Regression classifier on the provided training DataFrame.
    
    Parameters:
      - train_df: pandas DataFrame with training data.
      - features: list of feature column names.
      - target: target column name.
      - params: dictionary of Logistic Regression parameters (optional).
      
    Returns:
      - model: a trained LogisticRegression model.
    """
    X_train = train_df[features]
    y_train = train_df[target]
    
    if params is None:
        params = {'solver': 'liblinear', 'random_state': 42}
    model = LogisticRegression(**params)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, test_df, features, target):
    """
    Evaluates a trained model on the test dataset.
    
    Returns a dictionary containing:
      - accuracy, AUC, classification report, confusion matrix,
      - y_test, y_pred, and predicted probabilities.
    """
    X_test = test_df[features]
    y_test = test_df[target]
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
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
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.plot(fpr, tpr, label=label)

def compare_models(original_metrics, augmented_metrics):
    print("=== Original Training Model Metrics ===")
    print(f"Accuracy: {original_metrics['accuracy']:.3f}")
    print(f"AUC: {original_metrics['auc']:.3f}")
    print("Classification Report:")
    print(pd.DataFrame(original_metrics['classification_report']).transpose())
    print("Confusion Matrix:")
    print(original_metrics['confusion_matrix'])
    
    print("\n=== Augmented Training Model Metrics ===")
    print(f"Accuracy: {augmented_metrics['accuracy']:.3f}")
    print(f"AUC: {augmented_metrics['auc']:.3f}")
    print("Classification Report:")
    print(pd.DataFrame(augmented_metrics['classification_report']).transpose())
    print("Confusion Matrix:")
    print(augmented_metrics['confusion_matrix'])
    
    plt.figure(figsize=(8,6))
    plt.plot([0,1], [0,1], 'k--', label="Random")
    plot_roc_curve(original_metrics['y_test'], original_metrics['y_proba'], label="Original Model")
    plot_roc_curve(augmented_metrics['y_test'], augmented_metrics['y_proba'], label="Augmented Model")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve Comparison")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    # Example usage (if run standalone)
    pass
