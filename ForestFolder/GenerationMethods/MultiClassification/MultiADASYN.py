import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import ADASYN, SMOTE, RandomOverSampler
from collections import Counter

def augment_dataframe_multiclass_adasyn(df, target, n_classes_to_augment=None, test_size=0.25, 
                                       random_state=42, ratio_limit=1.0, diminishing_factor=1.0,
                                       min_samples_needed=6):
    """
    Splits the dataframe into training and test sets, then applies ADASYN
    to oversample multiple minority classes in the training set. A boolean column 'synthetic' is added:
      - False indicates an original sample.
      - True indicates a synthetic sample.
    
    Parameters:
      df (pd.DataFrame): Input DataFrame.
      target (str): Target column.
      n_classes_to_augment (int, optional): Number of minority classes to augment, starting from the smallest.
                                 If None, all classes except the majority class will be augmented.
      test_size (float): Fraction for test split.
      random_state (int): Random seed.
      ratio_limit (float): Desired ratio relative to majority class (e.g., 1.0 for balanced).
      diminishing_factor (float): Factor to reduce synthetic samples for larger classes.
                                 1.0 means no reduction, 0.5 means half the synthetic samples for each step
                                 up in class size.
      min_samples_needed (int): Minimum number of samples needed for a class to be augmented with ADASYN.
                                Classes with fewer samples will use RandomOverSampler instead.
    
    Returns:
      original_train (pd.DataFrame): Original training set.
      augmented_train (pd.DataFrame): Augmented training set with 'synthetic' column.
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
    
    # Scale training features
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
    
    # Calculate target counts for each class
    adasyn_sampling_strategy = {}
    random_os_sampling_strategy = {}
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
        
        # Determine if we should use ADASYN or RandomOverSampler based on class size
        if current_count >= min_samples_needed:
            # For ADASYN, specify the total desired count
            adasyn_sampling_strategy[cls] = target_count
            print(f"Class {cls}: Current count = {current_count}, Target count = {target_count}, "
                  f"Generating {synthetic_needed} synthetic samples with ADASYN")
        else:
            # For RandomOverSampler, specify the total desired count
            random_os_sampling_strategy[cls] = target_count
            print(f"Class {cls}: Current count = {current_count}, Target count = {target_count}, "
                  f"Generating {synthetic_needed} synthetic samples with RandomOverSampler "
                  f"(too few samples for ADASYN)")
    
    # Check if any synthetic samples are needed
    if sum(synthetic_needed_by_class.values()) <= 0:
        print("No synthetic samples needed based on ratio_limit.")
        train['synthetic'] = False
        return original_train, train, test
    
    # Create a copy of the training data to track original samples
    original_X = X_train_scaled.copy()
    original_y = y_train.copy()
    synthetic_X_all = pd.DataFrame(columns=X_train.columns)
    synthetic_y_all = pd.Series(dtype=original_y.dtype)
    
    # Apply ADASYN for classes with enough samples
    if adasyn_sampling_strategy:
        try:
            print("Applying ADASYN for classes with sufficient samples...")
            adasyn = ADASYN(
                sampling_strategy=adasyn_sampling_strategy,
                random_state=random_state,
                n_neighbors=min(5, min([counts[cls] for cls in adasyn_sampling_strategy.keys()]) - 1)
            )
            X_adasyn, y_adasyn = adasyn.fit_resample(original_X, original_y)
            
            # Extract only the synthetic samples (exclude original samples)
            synthetic_mask = ~y_adasyn.index.isin(original_y.index)
            synthetic_X_adasyn = X_adasyn[synthetic_mask]
            synthetic_y_adasyn = y_adasyn[synthetic_mask]
            
            # Add to our collection of all synthetic samples
            synthetic_X_all = pd.concat([synthetic_X_all, synthetic_X_adasyn])
            synthetic_y_all = pd.concat([synthetic_y_all, synthetic_y_adasyn])
            
            print(f"ADASYN generated {len(synthetic_X_adasyn)} synthetic samples")
        except Exception as e:
            print(f"ADASYN Error: {str(e)}")
            print("Trying with SMOTE instead...")
            
            try:
                # Fallback to SMOTE if ADASYN fails
                smote = SMOTE(
                    sampling_strategy=adasyn_sampling_strategy,
                    random_state=random_state,
                    k_neighbors=min(5, min([counts[cls] for cls in adasyn_sampling_strategy.keys()]) - 1)
                )
                X_smote, y_smote = smote.fit_resample(original_X, original_y)
                
                # Extract only the synthetic samples
                synthetic_mask = ~y_smote.index.isin(original_y.index)
                synthetic_X_smote = X_smote[synthetic_mask]
                synthetic_y_smote = y_smote[synthetic_mask]
                
                # Add to our collection
                synthetic_X_all = pd.concat([synthetic_X_all, synthetic_X_smote])
                synthetic_y_all = pd.concat([synthetic_y_all, synthetic_y_smote])
                
                print(f"SMOTE generated {len(synthetic_X_smote)} synthetic samples as fallback")
            except Exception as e:
                print(f"SMOTE Error: {str(e)}")
                print("Will try RandomOverSampler for all classes...")
                # If both ADASYN and SMOTE fail, we'll handle it with RandomOverSampler below
    
    # Apply RandomOverSampler for classes with too few samples
    # Also include classes that were supposed to use ADASYN but failed
    classes_for_random = list(random_os_sampling_strategy.keys())
    
    # Check which classes were supposed to be handled by ADASYN but weren't
    if adasyn_sampling_strategy:
        for cls in adasyn_sampling_strategy:
            # Count how many synthetic samples we actually got for this class
            if not synthetic_y_all.empty:
                actual_generated = sum(synthetic_y_all == cls)
                needed = synthetic_needed_by_class[cls]
                
                if actual_generated < needed:
                    # If we didn't get enough samples with ADASYN/SMOTE, add to random oversampling
                    remaining_needed = needed - actual_generated
                    random_os_sampling_strategy[cls] = counts[cls] + remaining_needed
                    classes_for_random.append(cls)
                    print(f"Class {cls}: Only got {actual_generated}/{needed} samples with ADASYN/SMOTE, "
                          f"will generate {remaining_needed} more with RandomOverSampler")
    
    if random_os_sampling_strategy:
        try:
            print(f"Applying RandomOverSampler for classes: {classes_for_random}...")
            random_os = RandomOverSampler(
                sampling_strategy=random_os_sampling_strategy,
                random_state=random_state
            )
            
            # If we already have some synthetic samples, only oversample the original data
            # that belongs to classes in random_os_sampling_strategy
            classes_mask = original_y.isin(random_os_sampling_strategy.keys())
            X_to_oversample = original_X[classes_mask]
            y_to_oversample = original_y[classes_mask]
            
            X_random, y_random = random_os.fit_resample(X_to_oversample, y_to_oversample)
            
            # Extract only the synthetic samples
            synthetic_mask = ~y_random.index.isin(y_to_oversample.index)
            synthetic_X_random = X_random[synthetic_mask]
            synthetic_y_random = y_random[synthetic_mask]
            
            # Add to our collection
            synthetic_X_all = pd.concat([synthetic_X_all, synthetic_X_random])
            synthetic_y_all = pd.concat([synthetic_y_all, synthetic_y_random])
            
            print(f"RandomOverSampler generated {len(synthetic_X_random)} synthetic samples")
        except Exception as e:
            print(f"RandomOverSampler Error: {str(e)}")
    
    # If we couldn't generate any synthetic samples, return original data
    if synthetic_X_all.empty:
        print("Failed to generate any synthetic samples.")
        train['synthetic'] = False
        return original_train, train, test
    
    # Now let's trim the synthetic samples if we generated more than needed
    final_synthetic_X = pd.DataFrame(columns=X_train.columns)
    final_synthetic_y = pd.Series(dtype=original_y.dtype)
    
    for cls in classes_to_augment:
        needed = synthetic_needed_by_class[cls]
        class_synthetic_X = synthetic_X_all[synthetic_y_all == cls]
        class_synthetic_y = synthetic_y_all[synthetic_y_all == cls]
        generated = len(class_synthetic_X)
        
        print(f"Class {cls}: Generated {generated} synthetic samples (needed {needed})")
        
        if generated > needed:
            # If we generated more than needed, take a random subset
            indices = np.random.choice(generated, needed, replace=False)
            class_synthetic_X = class_synthetic_X.iloc[indices]
            class_synthetic_y = class_synthetic_y.iloc[indices]
            print(f"  Trimmed to {needed} samples")
        
        # Add to final synthetic data
        final_synthetic_X = pd.concat([final_synthetic_X, class_synthetic_X])
        final_synthetic_y = pd.concat([final_synthetic_y, class_synthetic_y])
    
    # Combine original and final synthetic data
    augmented_X = pd.concat([original_X, final_synthetic_X])
    augmented_y = pd.concat([original_y, final_synthetic_y])
    
    # Create the 'synthetic' flag column
    synthetic_flags = pd.Series(False, index=original_X.index)
    synthetic_flags = pd.concat([synthetic_flags, pd.Series(True, index=final_synthetic_X.index)])
    
    # Create the final augmented DataFrame
    augmented_train = augmented_X.copy()
    augmented_train[target] = augmented_y
    augmented_train['synthetic'] = synthetic_flags
    
    # Verify the augmentation
    final_counts = augmented_train[target].value_counts()
    print(f"Final class distribution after augmentation: {dict(final_counts)}")
    
    # Verify synthetic samples
    synthetic_counts = augmented_train[augmented_train['synthetic']][target].value_counts()
    print(f"Synthetic samples by class: {dict(synthetic_counts)}")
    
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
    
    # Apply multi-class ADASYN to the 2 smallest classes
    orig_train, aug_train, test_set = augment_dataframe_multiclass_adasyn(
        df, 
        target='class', 
        n_classes_to_augment=2,  # Augment the 2 smallest classes
        test_size=0.25, 
        random_state=42, 
        ratio_limit=0.7,  # Target 70% of majority class size
        diminishing_factor=0.8,  # Slight reduction for larger classes
        min_samples_needed=6  # Minimum samples needed for ADASYN
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