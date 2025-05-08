# Model Card: Original Logistic Regression for Synthetic Data Augmentation

**Overview:**  
Name of relevant dataset is Forest Cover Type Dataset, this ML model was trained to classify the target value of Cover_Type

**Dataset Information:**  
- **Features:** Elevation, Aspect, Slope, Horizontal_Distance_To_Hydrology, Vertical_Distance_To_Hydrology, Horizontal_Distance_To_Roadways, Hillshade_9am, Hillshade_Noon, Hillshade_3pm, Horizontal_Distance_To_Fire_Points
- **Target:** Cover_Type

**Pre-processing Details:**  
- **Data Pre-processing File:** DataPrepMultiClassv1.py
- **Random State:** 42
- **Train/Test Split Ratio:** 0.25
- **Train Set Name:** original_trainVAEForestFINAL.csv
- **Test Set Name:** test_setVAEForestFINAL.csv

**Evaluation Metrics:**  
```json
{
    "accuracy": 0.6786328626894804,
    "auc": 0.8949177308722273,
    "classification_report": {
        "0": {
            "precision": 0.6650645783847982,
            "recall": 0.6910854432210417,
            "f1-score": 0.6778253756607753,
            "support": 51859.0
        },
        "1": {
            "precision": 0.6995467162392727,
            "recall": 0.7916000288163677,
            "f1-score": 0.7427319981344166,
            "support": 69405.0
        },
        "2": {
            "precision": 0.6054313099041534,
            "recall": 0.6362618914381645,
            "f1-score": 0.6204638472032742,
            "support": 8935.0
        },
        "3": {
            "precision": 0.875,
            "recall": 0.04081632653061224,
            "f1-score": 0.07799442896935933,
            "support": 686.0
        },
        "4": {
            "precision": 0.0,
            "recall": 0.0,
            "f1-score": 0.0,
            "support": 2269.0
        },
        "5": {
            "precision": 0.14673913043478262,
            "recall": 0.006228373702422145,
            "f1-score": 0.011949546359814118,
            "support": 4335.0
        },
        "6": {
            "precision": 0.3477157360406091,
            "recall": 0.027732793522267207,
            "f1-score": 0.051368578927634044,
            "support": 4940.0
        },
        "accuracy": 0.6786328626894804,
        "macro avg": {
            "precision": 0.4770710672862309,
            "recall": 0.31338926531869654,
            "f1-score": 0.3117619678936105,
            "support": 142429.0
        },
        "weighted avg": {
            "precision": 0.6417599816857029,
            "recall": 0.6786328626894804,
            "f1-score": 0.6501735686083443,
            "support": 142429.0
        }
    },
    "confusion_matrix": [
        [
            35839,
            15799,
            0,
            0,
            0,
            0,
            221
        ],
        [
            13391,
            54941,
            1013,
            0,
            3,
            27,
            30
        ],
        [
            3,
            3114,
            5685,
            4,
            0,
            129,
            0
        ],
        [
            0,
            23,
            634,
            28,
            0,
            1,
            0
        ],
        [
            43,
            2192,
            31,
            0,
            0,
            0,
            3
        ],
        [
            2,
            2276,
            2027,
            0,
            0,
            27,
            3
        ],
        [
            4610,
            193,
            0,
            0,
            0,
            0,
            137
        ]
    ]
}
```

**Intended Use:**  
Classify the target value of Cover_Type as well as possible.

**Ethical and Bias Concerns:**  
Works with data related to forest coverage which can potentially impact environmental policy.

**Date of Creation:** 2025-04-22