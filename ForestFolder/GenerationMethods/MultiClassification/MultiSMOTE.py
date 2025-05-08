import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from collections import Counter

def augment_dataframe_multiclass_smote(df, target, n_classes_to_augment=None, test_size=0.25, 
                                       random_state=42, ratio_limit=1.0, diminishing_factor=1.0):
    """
    Splits the dataframe into training and test sets, then applies SMOTE
    to oversample multiple minority classes in the training set. If more synthetic samples
    are generated than needed, they are trimmed. A boolean column 'synthetic' is added:
      - False indicates an original sample.
      - True indicates a synthetic sample.
    
    Parameters:
      df (pd.DataFrame): Input DataFrame.
      target (str): Target column.
      n_classes_to_augment (int): Number of minority classes to augment, starting from the smallest.
                                 If None, all classes except the majority class will be augmented.
      test_size (float): Fraction for test split.
      random_state (int): Random seed.
      ratio_limit (float): Desired ratio relative to majority class (e.g., 1.0 for balanced).
      diminishing_factor (float): Factor to reduce synthetic samples for larger classes.
                                 1.0 means no reduction, 0.5 means half the synthetic samples for each step up in class size.
    
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
    
    # Determine class counts
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
    
    # Calculate target counts for each class to augment
    smote_sampling_strategy = {}
    synthetic_needed_by_class = {}
    
    for i, cls in enumerate(classes_to_augment):
        current_count = counts[cls]
        position_factor = (current_count / counts[classes_to_augment[0]]) if diminishing_factor != 1.0 else 1.0
        adjustment = position_factor ** diminishing_factor
        
        # Calculate target count with adjustment
        target_count = int(majority_count * ratio_limit * adjustment)
        target_count = max(target_count, current_count)  # Ensure we don't reduce class size
        
        # Store how many synthetic samples we need for this class
        synthetic_needed = target_count - current_count
        synthetic_needed_by_class[cls] = synthetic_needed
        
        # For SMOTE, we specify the total desired count
        smote_sampling_strategy[cls] = target_count
        
        print(f"Class {cls}: Current count = {current_count}, Target count = {target_count}, "
              f"Generating {synthetic_needed} synthetic samples")
    
    # Check if any synthetic samples are needed
    if sum(synthetic_needed_by_class.values()) <= 0:
        train['synthetic'] = False
        return original_train, train, test
    
    # Apply SMOTE with custom sampling strategy
    try:
        smote = SMOTE(sampling_strategy=smote_sampling_strategy, random_state=random_state)
        X_res, y_res = smote.fit_resample(X_train_scaled, y_train)
    except ValueError as e:
        print(f"SMOTE Error: {str(e)}")
        print("This might be due to too few samples in some classes.")
        print("Returning original data without augmentation.")
        train['synthetic'] = False
        return original_train, train, test
    
    # Convert to numpy arrays for easier manipulation
    X_orig_np = X_train_scaled.values
    y_orig_np = y_train.values
    
    # Initialize synthetic flag array (all False initially)
    synthetic_flag = np.zeros(len(X_res), dtype=bool)
    
    # Mark synthetic samples
    # The first len(X_train_scaled) samples are original, the rest are synthetic
    synthetic_flag[len(X_train_scaled):] = True
    
    # Create the augmented DataFrame
    augmented_train = pd.DataFrame(X_res, columns=X_train.columns)
    augmented_train[target] = y_res
    augmented_train['synthetic'] = synthetic_flag
    
    # Verify the augmentation
    final_counts = augmented_train[target].value_counts()
    print(f"Final class distribution after augmentation: {dict(final_counts)}")
    
    # Verify synthetic samples
    synthetic_counts = augmented_train[augmented_train['synthetic']][target].value_counts()
    print(f"Synthetic samples by class: {dict(synthetic_counts)}")
    
    # Calculate unscaled features for the test set
    test_set = test.copy()
    
    # Inverse transform the scaled features for the augmented training set
    X_aug_unscaled = scaler.inverse_transform(augmented_train.drop(columns=[target, 'synthetic']))
    augmented_train_unscaled = pd.DataFrame(X_aug_unscaled, columns=X_train.columns)
    augmented_train_unscaled[target] = augmented_train[target]
    augmented_train_unscaled['synthetic'] = augmented_train['synthetic']
    
    return original_train, augmented_train_unscaled, test_set

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
    
    # Apply multi-class SMOTE to the 2 smallest classes
    orig_train, aug_train, test_set = augment_dataframe_multiclass_smote(
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
    
    # Show synthetic samples for each augmented class
    for cls in aug_train['class'].unique():
        synthetic_count = aug_train[(aug_train['class'] == cls) & (aug_train['synthetic'])].shape[0]
        if synthetic_count > 0:
            print(f"\nClass {cls} has {synthetic_count} synthetic samples")