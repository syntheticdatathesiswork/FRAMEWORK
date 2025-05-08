import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

def create_multiple_resampled_datasets(
    df: pd.DataFrame,
    target: str,
    test_size: float = 0.2,
    random_state: int = 42
):
    """
    Splits df into train/test, then creates three versions of training data:
    original, random oversampled, and random undersampled.

    Parameters:
        df (pd.DataFrame): Input dataframe.
        target (str): Name of the target column.
        test_size (float): Fraction of data to reserve as test set.
        random_state (int): Seed for reproducibility.

    Returns:
        original_train (pd.DataFrame): Original training set.
        oversampled_train (pd.DataFrame): Training set after random oversampling to 1:1 ratio,
                                         with a boolean 'synthetic' column.
        undersampled_train (pd.DataFrame): Training set after random undersampling to 1:1 ratio.
        test_set (pd.DataFrame): Hold‐out test set.
    """
    # 1) Stratified train/test split
    train, test = train_test_split(
        df, test_size=test_size,
        stratify=df[target],
        random_state=random_state
    )
    original_train = train.copy()

    # 2) Separate features/target
    X_train = train.drop(columns=[target])
    y_train = train[target]

    # 3) Identify majority/minority classes
    counts = y_train.value_counts()
    maj_class, min_class = counts.idxmax(), counts.idxmin()
    maj_count, min_count = counts[maj_class], counts[min_class]

    # 4) Create a dictionary for balanced sampling strategy (1:1 ratio)
    sampling_strategy = {
        min_class: maj_count,  # For oversampling: increase minority to match majority
        maj_class: min_count   # For undersampling: decrease majority to match minority
    }

    # 5) Apply random oversampling
    ros = RandomOverSampler(
        sampling_strategy={min_class: maj_count},
        random_state=random_state
    )
    X_over, y_over = ros.fit_resample(X_train, y_train)

    # Add synthetic flag to identify real vs synthetic samples
    real_samples_mask = np.zeros(len(X_over), dtype=bool)
    real_samples_mask[:len(X_train)] = True
    
    # 6) Apply random undersampling
    rus = RandomUnderSampler(
        sampling_strategy={maj_class: min_count},
        random_state=random_state
    )
    X_under, y_under = rus.fit_resample(X_train, y_train)

    # 7) Rebuild DataFrames for different sampling approaches
    # Oversampled DataFrame with synthetic flag
    oversampled_train = pd.DataFrame(X_over, columns=X_train.columns)
    oversampled_train[target] = y_over
    oversampled_train['synthetic'] = ~real_samples_mask

    # Undersampled DataFrame
    undersampled_train = pd.DataFrame(X_under, columns=X_train.columns)
    undersampled_train[target] = y_under

    return original_train, oversampled_train, undersampled_train, test.copy()


# Example usage:
if __name__ == "__main__":
    # Create a demo dataset with imbalanced classes
    df_demo = pd.DataFrame({
        'num1': np.random.randn(100),
        'num2': np.random.randn(100),
        'cat1': np.random.choice(['A','B','C'], size=100),
        'cat2': np.random.choice(['X','Y'], size=100),
        'class': np.concatenate([np.zeros(80), np.ones(20)])
    })
    
    # Get all four datasets
    orig, over, under, test_set = create_multiple_resampled_datasets(
        df_demo,
        target='class',
        test_size=0.3,
        random_state=42
    )
    
    # Print the results
    print("Original training distribution:")
    print(orig['class'].value_counts())
    print("\nOversampled training distribution:")
    print(over['class'].value_counts())
    print("\nUndersampled training distribution:")
    print(under['class'].value_counts())
    print("\nTest distribution:")
    print(test_set['class'].value_counts())
    
    print("\nOriginal training sample:")
    print(orig.head())
    print("\nOversampled training sample (with synthetic flag):")
    print(over.head())
    print("\nUndersampled training sample:")
    print(under.head())