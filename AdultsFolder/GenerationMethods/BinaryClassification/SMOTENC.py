import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from imblearn.over_sampling import SMOTENC

def augment_dataframe_smotenc(
    df: pd.DataFrame,
    target: str,
    categorical_cols: list,
    test_size: float = 0.2,
    random_state: int = 42,
    ratio_limit: float = 1.0,
    knn_value: int = 5
):
    """
    Splits df into train/test, then applies SMOTENC on the training set using
    the explicit list of categorical_cols.  Excess synthetic minority samples
    are trimmed based on density.

    Parameters:
        df (pd.DataFrame): Input dataframe.
        target (str): Name of the target column.
        categorical_cols (list): List of column names to treat as categorical.
        test_size (float): Fraction of data to reserve as test set.
        random_state (int): Seed for reproducibility.
        ratio_limit (float): Desired minority:majority ratio (e.g. 1.0 for balance).
        knn_value (int): Number of neighbors for SMOTENC and density trimming.

    Returns:
        original_train (pd.DataFrame): Original training set.
        augmented_train (pd.DataFrame): Training set after SMOTENC + trimming,
                                        with a boolean 'synthetic' column.
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
    maj, min_ = counts.idxmax(), counts.idxmin()
    N_maj, N_min = counts[maj], counts[min_]

    # 4) Compute synthetic samples needed
    desired_min = int(N_maj * ratio_limit)
    synth_needed = desired_min - N_min
    if synth_needed <= 0:
        aug = train.copy()
        aug['synthetic'] = False
        return original_train, aug, test.copy()

    # 5) Map categorical column names to indices
    cat_indices = [X_train.columns.get_loc(col) for col in categorical_cols]

    # 6) Apply SMOTENC
    smote_nc = SMOTENC(
        categorical_features=cat_indices,
        sampling_strategy=ratio_limit,
        random_state=random_state,
        k_neighbors=knn_value
    )
    X_res, y_res = smote_nc.fit_resample(X_train, y_train)

    # 7) Determine how many synthetic created
    new_min = (y_res == min_).sum()
    synth_gen = new_min - N_min

    # 8) Trim excess synthetic by density if needed
    if synth_gen > synth_needed:
        synth_X = X_res[len(X_train):]
        synth_y = y_res[len(X_train):]

        nbrs = NearestNeighbors(n_neighbors=knn_value)
        nbrs.fit(synth_X)
        dists, _ = nbrs.kneighbors(synth_X)
        density = 1.0 / (dists.mean(axis=1) + 1e-8)

        drop_count = synth_gen - synth_needed
        drop_idx = np.argsort(-density)[:drop_count]
        keep_mask = np.ones(synth_gen, dtype=bool)
        keep_mask[drop_idx] = False

        X_aug = np.vstack([X_train, synth_X[keep_mask]])
        y_aug = np.concatenate([y_train, synth_y[keep_mask]])
        synth_flags = np.concatenate([
            np.zeros(len(X_train), dtype=bool),
            np.ones(keep_mask.sum(), dtype=bool)
        ])
    else:
        X_aug = X_res
        y_aug = y_res
        synth_flags = np.concatenate([
            np.zeros(len(X_train), dtype=bool),
            np.ones(synth_gen, dtype=bool)
        ])

    # 9) Rebuild DataFrame with 'synthetic' flag
    augmented_train = pd.DataFrame(X_aug, columns=X_train.columns)
    augmented_train[target] = y_aug
    augmented_train['synthetic'] = synth_flags

    return original_train, augmented_train, test.copy()


# Example usage:
if __name__ == "__main__":
    df_demo = pd.DataFrame({
        'num1': np.random.randn(100),
        'num2': np.random.randn(100),
        'cat1': np.random.choice(['A','B','C'], size=100),
        'cat2': np.random.choice(['X','Y'], size=100),
        'class': np.concatenate([np.zeros(80), np.ones(20)])
    })
    cats = ['cat1', 'cat2']
    orig, aug, tst = augment_dataframe_smotenc(
        df_demo,
        target='class',
        categorical_cols=cats,
        test_size=0.3,
        random_state=42,
        ratio_limit=1.0,
        knn_value=5
    )
    print("Original distribution:\n", orig['class'].value_counts())
    print("Augmented distribution:\n", aug['class'].value_counts())
    print("Test distribution:\n", tst['class'].value_counts())
    print("\nAugmented sample:\n", aug.head())
