import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.combine import SMOTEENN
from sklearn.neighbors import NearestNeighbors

def augment_dataframe_smoteenn(df, target, test_size=0.25, random_state=42, ratio_limit=1.0, tol=1e-6):
    """
    Splits the dataframe into training and test sets, then applies SMOTE-ENN
    to augment the training set using a combination of SMOTE and ENN for binary classification.
    Because ENN cleans both classes, synthetic samples are not separated by the algorithm;
    therefore, we label each row as synthetic (True) if it does not nearly match any original training row.
    
    A boolean column 'synthetic' is added to the augmented training set:
      - False indicates an original sample.
      - True indicates a synthetic sample.
    
    Parameters:
      df (pd.DataFrame): Input DataFrame.
      target (str): Target column name.
      test_size (float): Fraction for test split.
      random_state (int): Random seed.
      ratio_limit (float): Desired majority:minority ratio (e.g., 1.0 for balanced).
      tol (float): Tolerance used to match original rows.
    
    Returns:
      original_train (pd.DataFrame): Original training set before augmentation.
      augmented_train (pd.DataFrame): Augmented training set with the 'synthetic' column.
      test_set (pd.DataFrame): Test set.
    """
    # Split into train and test sets (stratified on target)
    train, test = train_test_split(df, test_size=test_size, random_state=random_state, stratify=df[target])
    original_train = train.copy()
    
    # Separate features and target for training data
    X_train = train.drop(columns=[target])
    y_train = train[target]
    
    # Scale the training features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
    
    # Determine class counts and set desired minority count
    counts = y_train.value_counts()
    majority_class = counts.idxmax()
    minority_class = counts.idxmin()
    majority_count = counts[majority_class]
    desired_minority_count = int(majority_count * ratio_limit)
    sampling_strategy = {minority_class: desired_minority_count}
    
    # Apply SMOTEENN (which performs SMOTE followed by ENN cleaning)
    smoteenn = SMOTEENN(sampling_strategy=sampling_strategy, random_state=random_state)
    X_res, y_res = smoteenn.fit_resample(X_train_scaled, y_train)
    
    # Build the augmented training DataFrame
    augmented_train = pd.DataFrame(X_res, columns=X_train.columns)
    augmented_train[target] = y_res
    
    # Label rows as synthetic if they do not nearly match any original training sample.
    synthetic_flag = []
    for i, row in augmented_train.iterrows():
        # Compute Euclidean distances to every row in the original training set
        diff = X_train_scaled - row[X_train.columns]
        dists = np.linalg.norm(diff, axis=1)
        if (dists < tol).any():
            synthetic_flag.append(False)
        else:
            synthetic_flag.append(True)
    augmented_train['synthetic'] = synthetic_flag
    
    # The test set is left unchanged.
    test_set = test.copy()
    
    return original_train, augmented_train, test_set

# Example usage:
if __name__ == '__main__':
    # Create a synthetic dataset for demonstration.
    data = {
        'feature1': np.random.randn(100),
        'feature2': np.random.randn(100),
        'class': np.concatenate([np.zeros(80), np.ones(20)])
    }
    df = pd.DataFrame(data)
    orig_train, aug_train, test_set = augment_dataframe_smoteenn(df, target='class', test_size=0.3, random_state=42, ratio_limit=1.0)
    
    print("Original Training set class distribution:")
    print(orig_train['class'].value_counts())
    print("\nAugmented Training set class distribution:")
    print(aug_train['class'].value_counts())
    print("\nTest set class distribution:")
    print(test_set['class'].value_counts())
    print("\nSample of augmented data with 'synthetic' flag:")
    print(aug_train.head())
