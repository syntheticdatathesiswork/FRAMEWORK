import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

def undersample_majority_classes(df, target='Cover_Type', majority_classes=[1, 2], target_ratio=0.3):
    """
    Undersample majority classes to improve minority class prediction
    
    Parameters:
        df: DataFrame with the data
        target: target column name
        majority_classes: list of class labels to undersample
        target_ratio: ratio of samples to keep (0.3 = keep 30% of samples)
    
    Returns:
        DataFrame with undersampled majority classes
    """
    result_dfs = []
    
    # Process each class
    for class_label in df[target].unique():
        class_df = df[df[target] == class_label]
        
        # Undersample if this is a majority class
        if class_label in majority_classes:
            # Randomly sample the specified ratio
            sampled_df = class_df.sample(frac=target_ratio, random_state=42)
            result_dfs.append(sampled_df)
        else:
            # Keep all samples for minority classes
            result_dfs.append(class_df)
    
    # Combine all the dataframes
    return pd.concat(result_dfs, axis=0).reset_index(drop=True)