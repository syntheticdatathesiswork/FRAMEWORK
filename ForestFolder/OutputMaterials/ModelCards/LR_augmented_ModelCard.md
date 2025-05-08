# Model Card: Augmented Logistic Regression for Synthetic Data Augmentation

**Overview:**  
Name of relevant dataset is Forest Cover Type Dataset, this ML model was trained to classify the target value of Cover_Type

**Dataset Information:**  
- **Features:** Elevation, Aspect, Slope, Horizontal_Distance_To_Hydrology, Vertical_Distance_To_Hydrology, Horizontal_Distance_To_Roadways, Hillshade_9am, Hillshade_Noon, Hillshade_3pm, Horizontal_Distance_To_Fire_Points
- **Target:** Cover_Type

**Pre-processing Details:**  
- **Data Pre-processing File:** DataPrepMultiClassv1.py
- **Random State:** 42
- **Train/Test Split Ratio:** 0.25
- **Train Set Name:** augmented_trainVAEForestFINAL.csv
- **Test Set Name:** test_setVAEForestFINAL.csv

**Evaluation Metrics:**  
```json
{
    "accuracy": 0.6772426963609939,
    "auc": 0.8988548623715656,
    "classification_report": {
        "0": {
            "precision": 0.6630100184698047,
            "recall": 0.6852812433714495,
            "f1-score": 0.6739616916366394,
            "support": 51859.0
        },
        "1": {
            "precision": 0.6987996142523601,
            "recall": 0.7934730927166631,
            "f1-score": 0.7431331934445696,
            "support": 69405.0
        },
        "2": {
            "precision": 0.6267052023121388,
            "recall": 0.6067151650811415,
            "f1-score": 0.6165481944839352,
            "support": 8935.0
        },
        "3": {
            "precision": 0.47345132743362833,
            "recall": 0.3119533527696793,
            "f1-score": 0.37609841827768015,
            "support": 686.0
        },
        "4": {
            "precision": 0.0,
            "recall": 0.0,
            "f1-score": 0.0,
            "support": 2269.0
        },
        "5": {
            "precision": 0.21071428571428572,
            "recall": 0.04083044982698962,
            "f1-score": 0.06840579710144927,
            "support": 4335.0
        },
        "6": {
            "precision": 0.48717948717948717,
            "recall": 0.007692307692307693,
            "f1-score": 0.01514547628537266,
            "support": 4940.0
        },
        "accuracy": 0.6772426963609939,
        "macro avg": {
            "precision": 0.45140856219452935,
            "recall": 0.3494208016368901,
            "f1-score": 0.35618468160423516,
            "support": 142429.0
        },
        "weighted avg": {
            "precision": 0.6468327058641653,
            "recall": 0.6772426963609939,
            "f1-score": 0.6506143978651947,
            "support": 142429.0
        }
    },
    "confusion_matrix": [
        [
            35538,
            16294,
            1,
            0,
            0,
            0,
            26
        ],
        [
            13364,
            55071,
            812,
            3,
            0,
            141,
            14
        ],
        [
            1,
            2816,
            5421,
            178,
            0,
            519,
            0
        ],
        [
            0,
            23,
            446,
            214,
            0,
            3,
            0
        ],
        [
            35,
            2226,
            8,
            0,
            0,
            0,
            0
        ],
        [
            4,
            2135,
            1962,
            57,
            0,
            177,
            0
        ],
        [
            4659,
            243,
            0,
            0,
            0,
            0,
            38
        ]
    ]
}
```

**Intended Use:**  
Classify the target value of Cover_Type as well as possible.

**Ethical and Bias Concerns:**  
Works with data related to forest coverage which can potentially impact environmental policy.

**Date of Creation:** 2025-04-22