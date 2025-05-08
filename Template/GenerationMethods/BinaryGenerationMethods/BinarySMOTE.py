import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

def augment_dataframe_smote(df, target, test_size=0.25, random_state=42, ratio_limit=1.0):
    """
    Splits the dataframe into training and test sets, then applies SMOTE
    to oversample the minority class in the training set. If more synthetic samples
    are generated than needed, they are trimmed. A boolean column 'synthetic' is added:
      - False indicates an original sample.
      - True indicates a synthetic sample.
    
    Parameters:
      df (pd.DataFrame): Input DataFrame.
      target (str): Target column.
      test_size (float): Fraction for test split.
      random_state (int): Random seed.
      ratio_limit (float): Desired majority:minority ratio (e.g., 1.0 for balanced).
    
    Returns:
      original_train (pd.DataFrame): Original training set.
      augmented_train (pd.DataFrame): Augmented training set with 'synthetic' column.
      test_set (pd.DataFrame): Test set.
    """
    # Split into train and test sets (stratified on target)
    train, test = train_test_split(df, test_size=test_size, random_state=random_state, stratify=df[target])
    original_train = train.copy()
    
    # Separate features and target for training data
    X_train = train.drop(columns=[target])
    y_train = train[target]
    
    # Scale training features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
    
    # Determine class counts and set desired minority count
    counts = y_train.value_counts()
    majority_class = counts.idxmax()
    minority_class = counts.idxmin()
    majority_count = counts[majority_class]
    original_minority_count = counts[minority_class]
    desired_minority_count = int(majority_count * min(ratio_limit, 1.0))
    synthetic_needed = desired_minority_count - original_minority_count
    
    # If no synthetic samples are needed, add the synthetic flag and return.
    if synthetic_needed <= 0:
        augmented_train = train.copy()
        augmented_train['synthetic'] = False
        return original_train, augmented_train, test
    
    # Apply SMOTE with the custom sampling strategy
    smote = SMOTE(sampling_strategy={minority_class: desired_minority_count}, random_state=random_state)
    X_res, y_res = smote.fit_resample(X_train_scaled, y_train)
    
    # In SMOTE, original samples are assumed to be the first len(X_train_scaled) rows.
    num_original = len(X_train_scaled)
    synthetic_generated = (y_res == minority_class).sum() - original_minority_count
    
    if synthetic_generated > synthetic_needed:
        # Trim synthetic samples to exactly synthetic_needed.
        synthetic_X = X_res[num_original:]
        synthetic_y = y_res[num_original:]
        # Filter for minority class synthetic samples and keep only the first synthetic_needed.
        mask = (synthetic_y == minority_class)
        synthetic_X = synthetic_X[mask]
        synthetic_y = synthetic_y[mask]
        synthetic_X_trimmed = synthetic_X[:synthetic_needed]
        synthetic_y_trimmed = synthetic_y[:synthetic_needed]
        
        # Combine original with trimmed synthetic samples.
        X_aug = np.vstack([X_train_scaled, synthetic_X_trimmed])
        y_aug = np.concatenate([y_train, synthetic_y_trimmed])
        
        synthetic_flag = np.concatenate([np.full(len(X_train_scaled), False),
                                           np.full(len(synthetic_X_trimmed), True)])
    else:
        X_aug = X_res
        y_aug = y_res
        synthetic_flag = np.concatenate([np.full(len(X_train_scaled), False),
                                           np.full(synthetic_generated, True)])
    
    augmented_train = pd.DataFrame(X_aug, columns=X_train.columns)
    augmented_train[target] = y_aug
    augmented_train['synthetic'] = synthetic_flag
    
    # Prepare the test set (unchanged)
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
    orig_train, aug_train, test_set = augment_dataframe_smote(df, target='class', test_size=0.3, random_state=42, ratio_limit=1.0)
    
    print("Original Training set class distribution:")
    print(orig_train['class'].value_counts())
    print("\nAugmented Training set class distribution:")
    print(aug_train['class'].value_counts())
    print("\nTest set class distribution:")
    print(test_set['class'].value_counts())
    print("\nSample of augmented data with 'synthetic' flag:")
    print(aug_train.head())
