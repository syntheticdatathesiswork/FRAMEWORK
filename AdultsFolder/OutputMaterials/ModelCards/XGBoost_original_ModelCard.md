# Model Card: Original XGBoost for Synthetic Data Augmentation

**Overview:**  
Name of relevant dataset is Adult Dataset, this ML model was trained to classify the target value of income

**Dataset Information:**  
- **Features:** age, workclass, fnlwgt, education_num, marital_status, relationship, race, sex, capital_gain, capital_loss, hours_per_week, native_country
- **Target:** income

**Pre-processing Details:**  
- **Data Pre-processing File:** DataPrepv1.py
- **Random State:** 43
- **Train/Test Split Ratio:** 0.25
- **Train Set Name:** original_trainBORDERLINE2.csv
- **Test Set Name:** test_setBORDERLINE2.csv

**Evaluation Metrics:**  
```json
{
    "accuracy": 0.8615197732487515,
    "auc": 0.9216974758382097,
    "classification_report": {
        "0": {
            "precision": 0.8897854077253219,
            "recall": 0.9310220944853601,
            "f1-score": 0.909936797752809,
            "support": 5567.0
        },
        "1": {
            "precision": 0.7575757575757576,
            "recall": 0.6514657980456026,
            "f1-score": 0.7005253940455342,
            "support": 1842.0
        },
        "accuracy": 0.8615197732487515,
        "macro avg": {
            "precision": 0.8236805826505398,
            "recall": 0.7912439462654813,
            "f1-score": 0.8052310958991715,
            "support": 7409.0
        },
        "weighted avg": {
            "precision": 0.8569159009665829,
            "recall": 0.8615197732487515,
            "f1-score": 0.8578736575680607,
            "support": 7409.0
        }
    },
    "confusion_matrix": [
        [
            5183,
            384
        ],
        [
            642,
            1200
        ]
    ]
}
```

**Intended Use:**  
Classify the target value of income as well as possible.

**Ethical and Bias Concerns:**  
Works with potentially sensitive data including race, sex, and country of origin.

**Date of Creation:** 2025-05-08