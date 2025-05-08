import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.combine import SMOTEENN
from sklearn.neighbors import NearestNeighbors
from collections import Counter

def augment_dataframe_multiclass_smoteenn(df, target, n_classes_to_augment=None, test_size=0.25, 
                                         random_state=42, ratio_limit=1.0, diminishing_factor=1.0, 
                                         tol=1e-6, min_samples_needed=6):
    """
    Applies SMOTE-ENN to augment multiple minority classes in a multi-class dataset.
    Because ENN cleans samples across all classes, synthetic samples are not separated by the algorithm;
    therefore, we label each row as synthetic (True) if it does not nearly match any original training row.
    
    Parameters:
      df (pd.DataFrame): Input DataFrame.
      target (str): Target column name.
      n_classes_to_augment (int, optional): Number of minority classes to augment, starting from the smallest.
                                           If None, all classes except the majority class will be augmented.
      test_size (float): Fraction for test split.
      random_state (int): Random seed.
      ratio_limit (float): Desired ratio relative to majority class (e.g., 1.0 for balanced).
      diminishing_factor (float): Factor to reduce synthetic samples for larger classes.
                                 1.0 means no reduction, 0.5 means half the synthetic samples for each step
                                 up in class size.
      tol (float): Tolerance used to match original rows.
      min_samples_needed (int): Minimum samples needed for a class to be augmented with SMOTE-ENN.
                               Classes with fewer samples will be skipped.
    
    Returns:
      original_train (pd.DataFrame): Original training set before augmentation.
      augmented_train (pd.DataFrame): Augmented training set with the 'synthetic' column.
      test_set (pd.DataFrame): Test set.
    """
    # Set random seed for reproducibility
    np.random.seed(random_state)
    
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
    
    # Determine class counts and sort by frequency (ascending)
    counts = y_train.value_counts()
    majority_class = counts.idxmax()
    majority_count = counts[majority_class]
    
    # Sort classes by frequency (ascending)
    sorted_classes = counts.sort_values().index.tolist()
    
    # Determine which classes to augment
    if n_classes_to_augment is None or n_classes_to_augment >= len(sorted_classes):
        # Augment all classes except the majority
        classes_to_augment = sorted_classes[:-1]  # Exclude the majority class
    else:
        # Augment only the n smallest classes
        classes_to_augment = sorted_classes[:n_classes_to_augment]
    
    print(f"Classes to augment: {classes_to_augment}")
    print(f"Original class distribution: {dict(counts)}")
    
    # If no classes to augment, return original data
    if not classes_to_augment:
        train['synthetic'] = False
        return original_train, train, test
    
    # Filter out classes with too few samples
    min_samples_for_smote = min_samples_needed  # Minimum required for SMOTE to work
    too_small_classes = [cls for cls in classes_to_augment if counts[cls] < min_samples_for_smote]
    if too_small_classes:
        print(f"Warning: Classes {too_small_classes} have fewer than {min_samples_for_smote} samples.")
        print(f"These classes will be skipped for SMOTE-ENN generation.")
        classes_to_augment = [cls for cls in classes_to_augment if cls not in too_small_classes]
        
        # If no classes remain to augment, return original data
        if not classes_to_augment:
            print("No classes with sufficient samples for augmentation.")
            train['synthetic'] = False
            return original_train, train, test
    
    # Calculate target counts for each class to augment
    sampling_strategy = {}
    
    for i, cls in enumerate(classes_to_augment):
        current_count = counts[cls]
        position_factor = (current_count / counts[classes_to_augment[0]]) if diminishing_factor != 1.0 else 1.0
        adjustment = position_factor ** diminishing_factor
        
        # Calculate target count with adjustment
        target_count = int(majority_count * ratio_limit * adjustment)
        target_count = max(target_count, current_count)  # Ensure we don't reduce class size
        
        # Store the target count for this class
        sampling_strategy[cls] = target_count
        
        print(f"Class {cls}: Current count = {current_count}, Target count = {target_count}, "
              f"Generating {target_count - current_count} synthetic samples")
    
    # Apply SMOTEENN with custom sampling strategy
    try:
        print("Applying SMOTE-ENN for selected classes...")
        smoteenn = SMOTEENN(
            sampling_strategy=sampling_strategy,
            random_state=random_state,
            # Use n_neighbors parameter that will work for smallest class
            smote_kwargs={'k_neighbors': min(5, min([counts[cls] for cls in classes_to_augment]) - 1)}
        )
        X_res, y_res = smoteenn.fit_resample(X_train_scaled, y_train)
        
        print("SMOTE-ENN completed successfully")
        print(f"Result class distribution: {dict(Counter(y_res))}")
        
    except Exception as e:
        print(f"SMOTE-ENN Error: {str(e)}")
        print("Returning original data without augmentation.")
        train['synthetic'] = False
        return original_train, train, test
    
    # Build the augmented training DataFrame
    augmented_train = pd.DataFrame(X_res, columns=X_train.columns)
    augmented_train[target] = y_res
    
    # Keep track of the original training data to identify synthetic samples
    original_X_scaled_array = X_train_scaled.values
    
    # Initialize an array to store synthetic flags
    synthetic_flag = np.ones(len(augmented_train), dtype=bool)  # Assume all are synthetic initially
    
    # For each row in the augmented data, check if it's in the original data
    print("Identifying synthetic samples...")
    
    # Build a KD-tree for faster nearest neighbor lookup
    if len(original_X_scaled_array) > 0:  # Make sure there's data to build a tree
        try:
            nbrs = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(original_X_scaled_array)
            distances, _ = nbrs.kneighbors(augmented_train[X_train.columns].values)
            
            # Determine which samples are original (very close to original samples)
            synthetic_flag = distances.flatten() > tol
            
        except Exception as e:
            print(f"Error in nearest neighbors calculation: {str(e)}")
            print("Using slower point-by-point comparison instead...")
            
            # Fallback to slower method if KD-tree fails
            for i, row in enumerate(augmented_train[X_train.columns].values):
                # Compute Euclidean distances to every row in the original training set
                dists = np.linalg.norm(original_X_scaled_array - row, axis=1)
                if (dists < tol).any():
                    synthetic_flag[i] = False
    
    # Add the synthetic flag to the augmented DataFrame
    augmented_train['synthetic'] = synthetic_flag
    
    # Verify the augmentation
    final_counts = augmented_train[target].value_counts()
    print(f"Final class distribution after augmentation: {dict(final_counts)}")
    
    # Get synthetic samples counts
    synthetic_mask = augmented_train['synthetic'] == True
    if synthetic_mask.any():
        synthetic_counts = augmented_train[synthetic_mask][target].value_counts()
        print(f"Synthetic samples by class: {dict(synthetic_counts)}")
    else:
        print("No synthetic samples were identified")
    
    # Get original samples counts
    original_mask = augmented_train['synthetic'] == False
    if original_mask.any():
        original_counts = augmented_train[original_mask][target].value_counts()
        print(f"Original samples by class: {dict(original_counts)}")
    
    # Prepare the test set (unchanged)
    test_set = test.copy()
    
    return original_train, augmented_train, test_set

# Example usage:
if __name__ == '__main__':
    # Create a synthetic multi-class dataset for demonstration
    np.random.seed(42)
    data = {
        'feature1': np.random.randn(200),
        'feature2': np.random.randn(200),
        'class': np.concatenate([np.zeros(100), np.ones(60), np.full(30, 2), np.full(10, 3)])
    }
    df = pd.DataFrame(data)
    
    # Apply multi-class SMOTE-ENN to the 2 smallest classes
    orig_train, aug_train, test_set = augment_dataframe_multiclass_smoteenn(
        df, 
        target='class', 
        n_classes_to_augment=2,  # Augment the 2 smallest classes
        test_size=0.25, 
        random_state=42, 
        ratio_limit=0.7,  # Target 70% of majority class size
        diminishing_factor=0.8  # Slight reduction for larger classes
    )
    
    print("\nOriginal Training set class distribution:")
    print(orig_train['class'].value_counts())
    
    print("\nAugmented Training set class distribution:")
    print(aug_train['class'].value_counts())
    
    print("\nTest set class distribution:")
    print(test_set['class'].value_counts())
    
    print("\nSample of augmented data with 'synthetic' flag:")
    print(aug_train.head())
    
    # Show synthetic samples for each class
    for cls in aug_train['class'].unique():
        synthetic_count = aug_train[(aug_train['class'] == cls) & (aug_train['synthetic'])].shape[0]
        if synthetic_count > 0:
            print(f"\nClass {cls} has {synthetic_count} synthetic samples")