import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from imblearn.over_sampling import BorderlineSMOTE, SMOTE
from collections import Counter

def augment_dataframe_multiclass_borderline_smote(df, target, n_classes_to_augment=None, test_size=0.2, 
                                                 random_state=42, ratio_limit=1.0, knn_value=5,
                                                 diminishing_factor=1.0):
    """
    Applies Borderline-SMOTE to oversample multiple minority classes in multi-class datasets.
    Augments a specified number of the smallest classes, with options to control the augmentation ratio.
    
    Parameters:
        df (pd.DataFrame): Input dataframe without outliers.
        target (str): The name of the target variable column.
        n_classes_to_augment (int, optional): Number of minority classes to augment, starting from the smallest.
                                             If None, all classes except the majority class will be augmented.
        test_size (float): Fraction of the data to be used as test set.
        random_state (int): Seed for reproducibility.
        ratio_limit (float): Desired ratio relative to majority class (e.g., 1.0 for balanced).
        knn_value (int): The number of nearest neighbors to use for Borderline-SMOTE.
        diminishing_factor (float): Factor to reduce synthetic samples for larger classes.
                                   1.0 means no reduction, 0.5 means half the synthetic samples for each step
                                   up in class size.
    
    Returns:
        original_train (pd.DataFrame): Original training data before augmentation.
        augmented_train (pd.DataFrame): Training data after oversampling with 'synthetic' column.
        test_set (pd.DataFrame): The test dataset.
    """
    # Set random seed for reproducibility
    np.random.seed(random_state)
    
    # Split the data into train and test sets (stratified on the target)
    train, test = train_test_split(df, test_size=test_size, random_state=random_state, stratify=df[target])
    original_train = train.copy()
    
    # Separate features and target from training data
    X_train = train.drop(columns=[target])
    y_train = train[target]
    
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
    
    # Calculate target counts for each class to augment
    smote_sampling_strategy = {}
    synthetic_needed_by_class = {}
    
    # First check for classes with too few samples (less than 3 which is minimum for SMOTE)
    min_samples_for_smote = 6  # Minimum required for BorderlineSMOTE to work
    too_small_classes = [cls for cls in classes_to_augment if counts[cls] < min_samples_for_smote]
    if too_small_classes:
        print(f"Warning: Classes {too_small_classes} have fewer than {min_samples_for_smote} samples.")
        print(f"These classes will be skipped for SMOTE generation.")
        classes_to_augment = [cls for cls in classes_to_augment if cls not in too_small_classes]
        
        # If no classes remain to augment, return original data
        if not classes_to_augment:
            print("No classes with sufficient samples for augmentation.")
            train['synthetic'] = False
            return original_train, train, test
    
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
        
        # For BorderlineSMOTE, we specify the total desired count
        smote_sampling_strategy[cls] = target_count
        
        print(f"Class {cls}: Current count = {current_count}, Target count = {target_count}, "
              f"Generating {synthetic_needed} synthetic samples")
    
    # Check if any synthetic samples are needed
    if sum(synthetic_needed_by_class.values()) <= 0:
        print("No synthetic samples needed based on ratio_limit.")
        train['synthetic'] = False
        return original_train, train, test
    
    # Create a copy of the training data to track original samples
    original_X = X_train.values
    original_y = y_train.values
    
    # Try different SMOTE variants with decreasing requirements
    augmentation_success = False
    X_res = None
    y_res = None
    
    # Prepare an appropriate k_neighbors value
    min_class_size = min([counts[cls] for cls in classes_to_augment])
    # k must be smaller than the smallest class size
    safe_k = min(knn_value, min_class_size - 1)
    safe_k = max(1, safe_k)  # Ensure k is at least 1
    
    print(f"Using k_neighbors={safe_k} for SMOTE generation")
    
    # Try BorderlineSMOTE first
    try:
        smote = BorderlineSMOTE(
            sampling_strategy=smote_sampling_strategy, 
            random_state=random_state, 
            kind='borderline-1', 
            k_neighbors=safe_k
        )
        X_res, y_res = smote.fit_resample(X_train, y_train)
        augmentation_success = True
        print("Successfully generated synthetic samples using BorderlineSMOTE")
    except Exception as e:
        print(f"BorderlineSMOTE Error: {str(e)}")
        print("Trying with regular SMOTE instead...")
    
    # If BorderlineSMOTE failed, try regular SMOTE
    if not augmentation_success:
        try:
            smote = SMOTE(
                sampling_strategy=smote_sampling_strategy, 
                random_state=random_state,
                k_neighbors=safe_k
            )
            X_res, y_res = smote.fit_resample(X_train, y_train)
            augmentation_success = True
            print("Successfully generated synthetic samples using regular SMOTE")
        except Exception as e:
            print(f"SMOTE Error: {str(e)}")
    
    # If both SMOTE variants failed, try a very basic SMOTE with k=1
    if not augmentation_success:
        try:
            print("Trying with minimal SMOTE configuration (k=1)...")
            smote = SMOTE(
                sampling_strategy=smote_sampling_strategy, 
                random_state=random_state,
                k_neighbors=1
            )
            X_res, y_res = smote.fit_resample(X_train, y_train)
            augmentation_success = True
            print("Successfully generated synthetic samples using minimal SMOTE")
        except Exception as e:
            print(f"Minimal SMOTE Error: {str(e)}")
            print("Returning original data without augmentation.")
            train['synthetic'] = False
            return original_train, train, test
    
    # If we couldn't generate synthetic samples, return original data
    if not augmentation_success or X_res is None or y_res is None:
        print("Failed to generate synthetic samples with any SMOTE variant.")
        train['synthetic'] = False
        return original_train, train, test
    
    # Process the augmented dataset
    # First len(X_train) samples are original, the rest are synthetic
    synthetic_indices = list(range(len(X_train), len(X_res)))
    
    # Initialize arrays for the final augmented dataset
    final_X = original_X.copy()
    final_y = original_y.copy()
    synthetic_flags = np.zeros(len(original_X), dtype=bool)  # All original samples are not synthetic
    
    # Process each class that was augmented
    for cls in classes_to_augment:
        try:
            # Extract synthetic samples for this class
            synthetic_mask = (y_res[synthetic_indices] == cls)
            synthetic_class_indices = [synthetic_indices[i] for i, is_synth in enumerate(synthetic_mask) if is_synth]
            
            # Check if we have any synthetic samples for this class
            if len(synthetic_class_indices) == 0:
                print(f"No synthetic samples were generated for class {cls}")
                continue
            
            # Get the synthetic samples for this class
            synthetic_X_cls = X_res[synthetic_class_indices]
            synthetic_y_cls = y_res[synthetic_class_indices]
            
            # Get the target number needed for this class
            needed = synthetic_needed_by_class[cls]
            generated = len(synthetic_X_cls)
            
            print(f"Class {cls}: Generated {generated} synthetic samples (needed {needed})")
            
            # If we generated more than needed, trim using random selection
            # Density-based approach is skipped to avoid NearestNeighbors errors
            if generated > needed and needed > 0:
                # Use random selection which is more robust
                random_indices = np.random.choice(generated, needed, replace=False)
                synthetic_X_cls = synthetic_X_cls[random_indices]
                synthetic_y_cls = synthetic_y_cls[random_indices]
                print(f"    Trimmed to {needed} samples using random selection")
            
            # Add synthetic samples to final dataset
            if len(synthetic_X_cls) > 0:
                final_X = np.vstack([final_X, synthetic_X_cls])
                final_y = np.append(final_y, synthetic_y_cls)
                # Mark these samples as synthetic
                synthetic_flags = np.append(synthetic_flags, np.ones(len(synthetic_X_cls), dtype=bool))
        except Exception as e:
            print(f"Error processing synthetic samples for class {cls}: {str(e)}")
            print(f"Skipping this class.")
            continue
    
    # Reconstruct the augmented training dataset
    try:
        augmented_train = pd.DataFrame(final_X, columns=X_train.columns)
        augmented_train[target] = final_y
        augmented_train['synthetic'] = synthetic_flags
        
        # Verify the augmentation
        final_counts = augmented_train[target].value_counts()
        print(f"Final class distribution after augmentation: {dict(final_counts)}")
        
        # Get synthetic samples counts
        synthetic_mask = augmented_train['synthetic'] == True
        if synthetic_mask.any():
            synthetic_counts = augmented_train[synthetic_mask][target].value_counts()
            print(f"Synthetic samples by class: {dict(synthetic_counts)}")
        else:
            print("No synthetic samples were generated")
    except Exception as e:
        print(f"Error creating final augmented dataset: {str(e)}")
        print("Returning original data.")
        train['synthetic'] = False
        return original_train, train, test
    
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
    
    # Apply multi-class Borderline SMOTE to the 2 smallest classes
    orig_train, aug_train, test_set = augment_dataframe_multiclass_borderline_smote(
        df, 
        target='class', 
        n_classes_to_augment=2,  # Augment the 2 smallest classes
        test_size=0.25, 
        random_state=42, 
        ratio_limit=0.7,  # Target 70% of majority class size
        knn_value=5,
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