import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from imblearn.over_sampling import BorderlineSMOTE

def augment_dataframe_borderline_smote(df, target, test_size=0.2, random_state=42, ratio_limit=1.0, knn_value=5):
    """
    Splits the dataframe into training and test sets, then applies Borderline-SMOTE
    to oversample the minority class in the training set. After oversampling, if the 
    resulting minority-to-majority ratio exceeds the desired ratio_limit, extra synthetic 
    samples are removed (with a priority to remove points from dense areas first).

    A boolean column 'synthetic' is added to the augmented training set, where:
      - False indicates an original sample.
      - True indicates a synthetic sample.
    
    Parameters:
        df (pd.DataFrame): Input dataframe without outliers.
        target (str): The name of the target variable column.
        test_size (float): Fraction of the data to be used as test set.
        random_state (int): Seed for reproducibility.
        ratio_limit (float): Desired majority:minority ratio in the augmented training set.
                             For example, 1.0 means balanced classes.
        knn_value (int): The number of nearest neighbors to use in both the SMOTE algorithm
                         and for computing density during synthetic sample trimming.
    
    Returns:
        original_train (pd.DataFrame): Original training data before augmentation.
        augmented_train (pd.DataFrame): Training data after oversampling and synthetic sample trimming.
                                        Includes the 'synthetic' column.
        test_set (pd.DataFrame): The test dataset.
    """
    # Split the data into train and test sets (stratified on the target)
    train, test = train_test_split(df, test_size=test_size, random_state=random_state, stratify=df[target])
    original_train = train.copy()
    
    # Separate features and target from training data
    X_train = train.drop(columns=[target])
    y_train = train[target]
    
    # Determine which class is majority and which is minority
    counts = y_train.value_counts()
    majority_class = counts.idxmax()
    minority_class = counts.idxmin()
    
    N_majority = counts[majority_class]
    N_minority = counts[minority_class]
    
    # Calculate desired minority count based on ratio_limit
    desired_minority_count = int(N_majority * ratio_limit)
    synthetic_needed = desired_minority_count - N_minority

    # If no synthetic samples are needed, add the synthetic flag and return.
    if synthetic_needed <= 0:
        augmented_train = train.copy()
        augmented_train['synthetic'] = False
        return original_train, augmented_train, test

    # Apply Borderline-SMOTE on the training set.
    # Pass the knn_value to the BorderlineSMOTE algorithm.
    smote = BorderlineSMOTE(sampling_strategy=ratio_limit, random_state=random_state, kind='borderline-1', k_neighbors=knn_value)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    
    # Determine how many synthetic samples were generated.
    new_minority_count = (y_res == minority_class).sum()
    synthetic_generated = new_minority_count - N_minority
    
    # If more synthetic samples were generated than needed, trim them.
    if synthetic_generated > synthetic_needed:
        # The synthetic samples are assumed to start at index len(X_train)
        synthetic_X = X_res[len(X_train):]
        synthetic_y = y_res[len(X_train):]
        
        # Compute density measure using average distance to knn_value nearest synthetic neighbors.
        nbrs = NearestNeighbors(n_neighbors=knn_value)
        nbrs.fit(synthetic_X)
        distances, _ = nbrs.kneighbors(synthetic_X)
        avg_distances = distances.mean(axis=1)
        # Lower average distance means higher density.
        density = 1 / (avg_distances + 1e-8)
        
        # Sort synthetic indices by descending density (densest first).
        sorted_idx = np.argsort(-density)
        # Determine how many synthetic samples to remove.
        to_remove = synthetic_generated - synthetic_needed
        # Keep indices that are not among the top 'to_remove' densest ones.
        indices_to_keep = np.setdiff1d(np.arange(synthetic_generated), sorted_idx[:to_remove])
        
        # Get the trimmed synthetic samples.
        synthetic_X_trimmed = synthetic_X[indices_to_keep]
        synthetic_y_trimmed = synthetic_y[indices_to_keep]
        
        # Combine original training data with the trimmed synthetic samples.
        X_aug = np.vstack([X_train, synthetic_X_trimmed])
        y_aug = np.concatenate([y_train, synthetic_y_trimmed])
        
        # Create synthetic flag array: original samples get False, synthetic samples get True.
        synthetic_flags_original = np.full(len(X_train), False)
        synthetic_flags_trimmed = np.full(len(synthetic_X_trimmed), True)
        synthetic_flag = np.concatenate([synthetic_flags_original, synthetic_flags_trimmed])
    else:
        # No trimming needed; use all synthetic samples.
        X_aug = X_res
        y_aug = y_res
        synthetic_flags_original = np.full(len(X_train), False)
        synthetic_flags_synthetic = np.full(synthetic_generated, True)
        synthetic_flag = np.concatenate([synthetic_flags_original, synthetic_flags_synthetic])
    
    # Reconstruct the augmented training set as a DataFrame and add the synthetic column.
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
    df_no_outliers = pd.DataFrame(data)
    
    orig_train, aug_train, test_set = augment_dataframe_borderline_smote(
        df_no_outliers, target='class', test_size=0.3, random_state=42, ratio_limit=1.0, knn_value=5
    )
    
    print("Original Training set class distribution:")
    print(orig_train['class'].value_counts())
    print("\nAugmented Training set class distribution:")
    print(aug_train['class'].value_counts())
    print("\nTest set class distribution:")
    print(test_set['class'].value_counts())
    print("\nSample of augmented data with 'synthetic' flag:")
    print(aug_train.head())
