# Model Card: Augmented Logistic Regression for Synthetic Data Augmentation

**Overview:**  
Name of relevant dataset is Adult Dataset, this ML model was trained to classify the target value of income

**Dataset Information:**  
- **Features:** age, workclass, fnlwgt, education_num, marital_status, relationship, race, sex, capital_gain, capital_loss, hours_per_week, native_country
- **Target:** income

**Pre-processing Details:**  
- **Data Pre-processing File:** DataPrepv1.py
- **Random State:** 43
- **Train/Test Split Ratio:** 0.25
- **Train Set Name:** augmented_trainBORDERLINE2.csv
- **Test Set Name:** test_setBORDERLINE2.csv

**Evaluation Metrics:**  
```json
{
    "accuracy": 0.8234579565393441,
    "auc": 0.8558696284351305,
    "classification_report": {
        "0": {
            "precision": 0.8427490745211653,
            "recall": 0.9405424824860786,
            "f1-score": 0.8889643463497453,
            "support": 5567.0
        },
        "1": {
            "precision": 0.7232441471571907,
            "recall": 0.46959826275787186,
            "f1-score": 0.5694535878867676,
            "support": 1842.0
        },
        "accuracy": 0.8234579565393441,
        "macro avg": {
            "precision": 0.7829966108391779,
            "recall": 0.7050703726219752,
            "f1-score": 0.7292089671182564,
            "support": 7409.0
        },
        "weighted avg": {
            "precision": 0.8130381720775911,
            "recall": 0.8234579565393441,
            "f1-score": 0.8095286847100092,
            "support": 7409.0
        }
    },
    "confusion_matrix": [
        [
            5236,
            331
        ],
        [
            977,
            865
        ]
    ]
}
```

**Intended Use:**  
Classify the target value of income as well as possible.

**Ethical and Bias Concerns:**  
Works with potentially sensitive data including race, sex, and country of origin.

**Date of Creation:** 2025-05-08