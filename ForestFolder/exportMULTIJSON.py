import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ks_2samp, chi2_contingency
from sklearn.metrics import pairwise_distances, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.model_selection import train_test_split
import json
import inspect


def export_pipeline_config(dataset_name, features, train_test_ratio, randomState,
                           synthetic_method, augmentation_ratio, augmentation_file,
                           pipeline_name, validation_file, data_file_name,
                           evaluation_metrics, output_json='pipeline_config.json'):
    """
    Exports pipeline configuration details to a JSON file.
    
    Parameters:
      - dataset_name (str): Name of the dataset.
      - features (list): List of feature names used.
      - train_test_ratio (float): Train-test split ratio.
      - randomState (int): The random seed used for reproducibility.
      - synthetic_method (str): Synthetic generation method name.
      - augmentation_ratio (float): Augmentation ratio used.
      - augmentation_file (str): Filename where the augmentation function is defined.
      - pipeline_name (str): The synthetic data generation function used.
      - validation_file (str): Filename where the validation code is defined.
      - data_file_name (list): List of filenames for the saved datasets.
      - evaluation_metrics (dict): Dictionary of evaluation metrics.
      - output_json (str): Name of the JSON file to write the configuration.
    
    Returns:
      None
    """
    
    config = {
        "data": {
            "dataset_name": dataset_name,
            "features": features,
            "train_test_split_ratio": train_test_ratio,
            "data_files": data_file_name
        },
        "reproducibility": {
            "random_state": randomState
        },
        "synthetic_generation": {
            "method": synthetic_method,
            "augmentation_ratio": augmentation_ratio,
            "augmentation_file": augmentation_file,
            "pipeline_name": pipeline_name
        },
        "validation": {
            "validation_file": validation_file,
            "metrics": evaluation_metrics
        }
    }
    
    with open(output_json, 'w') as f:
        json.dump(config, f, indent=4)
    print(f"Configuration exported to {output_json}")


def discriminative_score(real, synthetic, features):
    """
    Computes a discriminative score by training a classifier to distinguish between
    real and synthetic samples. Accuracy near 0.5 indicates the two distributions are very similar.
    """
    real = real.copy()
    synthetic = synthetic.copy()
    real["is_synthetic"] = 0
    synthetic["is_synthetic"] = 1
    data = pd.concat([real, synthetic], axis=0).reset_index(drop=True)
    X = data[features]
    y = data["is_synthetic"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    return acc

def compute_mmd(X, Y, gamma=1.0):
    """
    Computes the Maximum Mean Discrepancy (MMD) between two sets of samples using an RBF kernel.
    Lower MMD indicates more similar distributions.
    """
    XX = rbf_kernel(X, X, gamma=gamma)
    YY = rbf_kernel(Y, Y, gamma=gamma)
    XY = rbf_kernel(X, Y, gamma=gamma)
    mmd = np.mean(XX) + np.mean(YY) - 2 * np.mean(XY)
    return mmd

def check_scaling(original, synthetic, continuous_features):
    """
    Compares the min, max, mean, and standard deviation of continuous features between original and synthetic data.
    """
    #print("### Scaling / Normalization Check ###")
    for feature in continuous_features:
        orig_min = original[feature].min()
        orig_max = original[feature].max()
        orig_mean = original[feature].mean()
        orig_std = original[feature].std()
        
        synth_min = synthetic[feature].min()
        synth_max = synthetic[feature].max()
        synth_mean = synthetic[feature].mean()
        synth_std = synthetic[feature].std()
        
        #print(f"\nFeature: {feature}")
        #print(f"Original: min={orig_min:.3f}, max={orig_max:.3f}, mean={orig_mean:.3f}, std={orig_std:.3f}")
        #print(f"Synthetic: min={synth_min:.3f}, max={synth_max:.3f}, mean={synth_mean:.3f}, std={synth_std:.3f}")

def remove_duplicates(original, synthetic):
    """
    Removes duplicates from synthetic data that are identical to the original data,
    and also removes duplicates within the synthetic set. Reports the counts.
    """
    # Find duplicates between synthetic and original.
    common = pd.merge(synthetic, original, how="inner")
    n_common = common.shape[0]
    #print(f"Duplicates found in synthetic data that match original: {n_common}")
    
    # Remove duplicates within synthetic data.
    n_before = synthetic.shape[0]
    synthetic_unique = synthetic.drop_duplicates()
    n_duplicates = n_before - synthetic_unique.shape[0]
    #print(f"Duplicates within synthetic data removed: {n_duplicates}")
    return synthetic_unique, n_common, n_duplicates

def analyze_outcome_distribution(df, target):
    """
    Analyzes and plots the distribution of the outcome for multi-class scenarios.
    """
    counts = df[target].value_counts()
    
    # Calculate imbalance ratio (largest class to smallest class)
    if len(counts) >= 2:
        largest_class = counts.max()
        smallest_class = counts.min()
        imbalance_ratio = largest_class / smallest_class
        #print(f"Imbalance ratio (largest:smallest): {round(imbalance_ratio, 2)}")
    
    #print(f"{target} class distribution:")
    #print(counts)
    
    plt.figure(figsize=(8, 5))
    counts.plot(kind="bar")
    plt.title(f"{target} Distribution")
    plt.xlabel(target)
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def validate_synthetic_data(original, synthetic, continuous_features, categorical_features,
                            distance_threshold=0.1, density_threshold=0.5, gamma=1.0, plot=False):
    """
    Validates synthetic data quality by comparing it with the original dataset.
    
    This function computes:
      - Summary statistics and KS tests for continuous features.
      - Chi-squared tests for categorical features.
      - A scaling/normalization check.
      - Duplicate removal counts.
      - Coverage, diversity, and density metrics.
      - A discriminative score using an auxiliary classifier.
      - Maximum Mean Discrepancy (MMD).
      
    All metrics are printed.
    
    Parameters:
      original (pd.DataFrame): Original data.
      synthetic (pd.DataFrame): Synthetic data.
      continuous_features (list): Continuous feature names.
      categorical_features (list): Categorical feature names.
      distance_threshold (float): For coverage metric.
      density_threshold (float): Radius for local density calculation.
      gamma (float): Gamma for the RBF kernel in MMD.
      plot (bool): If True, display plots.
      
    Returns:
      metrics (dict): Dictionary of computed metrics.
    """
    # Drop rows with missing values.
    required_cols = continuous_features + categorical_features
    original = original.dropna(subset=required_cols)
    synthetic = synthetic.dropna(subset=required_cols)
    
    # Remove duplicates.
    synthetic, n_common, n_dup = remove_duplicates(original, synthetic)
    
    metrics = {}
    
    # Scaling check.
    check_scaling(original, synthetic, continuous_features)
    
    #print("\n### Continuous Features Validation ###")
    metrics["continuous"] = {}
    for feature in continuous_features:
        #print(f"\nFeature: {feature}")
        orig_mean = original[feature].mean()
        orig_std = original[feature].std()
        synth_mean = synthetic[feature].mean()
        synth_std = synthetic[feature].std()
        #print(f"Original: mean={orig_mean:.3f}, std={orig_std:.3f}")
        #print(f"Synthetic: mean={synth_mean:.3f}, std={synth_std:.3f}")
        
        ks_stat, ks_p = ks_2samp(original[feature], synthetic[feature])
        #print(f"KS test: statistic={ks_stat:.3f}, p-value={ks_p:.3f}")
        #if ks_p > 0.05:
            #print("-> No significant difference")
        #else:
            #print("-> Significant difference detected")
            
        if plot:
            plt.figure(figsize=(6,4))
            plt.hist(original[feature], bins=30, alpha=0.5, label="Original")
            plt.hist(synthetic[feature], bins=30, alpha=0.5, label="Synthetic")
            plt.title(f"Histogram of {feature}")
            plt.legend()
            plt.show()
            
        metrics["continuous"][feature] = {
            "orig_mean": orig_mean,
            "orig_std": orig_std,
            "synth_mean": synth_mean,
            "synth_std": synth_std,
            "ks_stat": ks_stat,
            "ks_p": ks_p
        }
    
    #print("\n### Categorical Features Validation ###")
    metrics["categorical"] = {}
    for feature in categorical_features:
        #print(f"\nFeature: {feature}")
        orig_counts = original[feature].value_counts().sort_index()
        synth_counts = synthetic[feature].value_counts().sort_index()
        #print("Original counts:\n", orig_counts)
        #print("Synthetic counts:\n", synth_counts)
        
        counts_df = pd.DataFrame({"Original": orig_counts, "Synthetic": synth_counts}).fillna(0)
        chi2, p_value, dof, expected = chi2_contingency(counts_df)
        #print(f"Chi-squared test: statistic={chi2:.3f}, p-value={p_value:.3f}")
        #if p_value > 0.05:
            #print("-> No significant difference")
        #else:
            #print("-> Significant difference detected")
            
        if plot:
            counts_df.plot(kind="bar", figsize=(6,4))
            plt.title(f"Counts for {feature}")
            plt.show()
            
        metrics["categorical"][feature] = {
            "orig_counts": orig_counts.to_dict(),
            "synth_counts": synth_counts.to_dict(),
            "chi2_stat": chi2,
            "chi2_p": p_value
        }
    
    #print("\n### Coverage Metric ###")
    distances = pairwise_distances(original[continuous_features], synthetic[continuous_features])
    min_distances = distances.min(axis=1)
    coverage = np.mean(min_distances <= distance_threshold)
    #print(f"Coverage: {coverage*100:.2f}% of original samples have a synthetic neighbor within {distance_threshold}")
    metrics["coverage"] = coverage

    #print("\n### Diversity Metric ###")
    synth_distances = pairwise_distances(synthetic[continuous_features])
    i_upper = np.triu_indices_from(synth_distances, k=1)
    avg_distance = np.mean(synth_distances[i_upper])
    std_distance = np.std(synth_distances[i_upper])
    #print(f"Average pairwise distance among synthetic samples: {avg_distance:.3f}")
    #print(f"Standard deviation of pairwise distances: {std_distance:.3f}")
    metrics["diversity"] = {"avg_distance": avg_distance, "std_distance": std_distance}
    
    #print("\n### Density Metric ###")
    synth_distances = pairwise_distances(synthetic[continuous_features])
    neighbor_counts = (synth_distances <= density_threshold).sum(axis=1) - 1
    average_density = np.mean(neighbor_counts)
    #print(f"Average local density: {average_density:.3f} neighbors within a radius of {density_threshold}")
    # Only record the average density and density threshold, not the full neighbor_counts list.
    metrics["density"] = {"density_threshold": density_threshold, "average_density": average_density}
    
    #print("\n### Discriminative Score ###")
    disc_score = discriminative_score(original[continuous_features].copy(),
                                      synthetic[continuous_features].copy(),
                                      continuous_features)
    #print(f"Discriminative score (classifier accuracy): {disc_score:.3f}")
    metrics["discriminative_score"] = disc_score

    #print("\n### MMD Metric ###")
    mmd_value = compute_mmd(original[continuous_features].values,
                            synthetic[continuous_features].values,
                            gamma=gamma)
    #print(f"Maximum Mean Discrepancy (MMD): {mmd_value:.3f}")
    metrics["mmd"] = mmd_value

    if plot and len(continuous_features) == 2:
        feature_x, feature_y = continuous_features
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        sns.kdeplot(data=original, x=feature_x, y=feature_y, cmap="Reds", fill=True, thresh=0.05, alpha=0.5, label="Original")
        sns.kdeplot(data=synthetic, x=feature_x, y=feature_y, cmap="Blues", fill=True, thresh=0.05, alpha=0.5, label="Synthetic")
        plt.title("Density Diagram")
        plt.legend()

        plt.subplot(1, 2, 2)
        original_colors = ['green' if d <= distance_threshold else 'red' for d in min_distances]
        plt.scatter(original[feature_x], original[feature_y], c=original_colors, label="Original", edgecolor='k', alpha=0.7)
        plt.scatter(synthetic[feature_x], synthetic[feature_y], marker='x', color='blue', label="Synthetic", alpha=0.7)
        plt.title("Coverage Diagram")
        plt.xlabel(feature_x)
        plt.ylabel(feature_y)
        plt.legend()
        plt.tight_layout()
        plt.show()
    
    return metrics

def compute_evaluation_metrics(original, synthetic, continuous_features, categorical_features,
                               distance_threshold=0.1, density_threshold=0.5, gamma=1.0, plot=False):
    """
    Computes evaluation metrics for synthetic data by calling validate_synthetic_data
    with the specified parameters. This function automates the generation of a dictionary
    containing all important metrics computed from the data.
    
    Parameters:
      original (pd.DataFrame): Original dataset.
      synthetic (pd.DataFrame): Synthetic dataset.
      continuous_features (list): List of continuous feature names.
      categorical_features (list): List of categorical feature names.
      distance_threshold (float): Threshold for the coverage metric.
      density_threshold (float): Threshold for computing local density.
      gamma (float): Gamma parameter for the RBF kernel in MMD.
      plot (bool): If True, plots will be displayed.
    
    Returns:
      metrics (dict): Dictionary containing all computed evaluation metrics.
    """
    metrics = validate_synthetic_data(original, synthetic, continuous_features, categorical_features,
                                      distance_threshold=distance_threshold, density_threshold=density_threshold,
                                      gamma=gamma, plot=plot)
    return metrics

# Example usage when run as a standalone module.
if __name__ == "__main__":
    np.random.seed(42)
    # Simulated data for demonstration purposes.
    original = pd.DataFrame({
        "age": np.random.normal(40, 12, 500),
        "fnlwgt": np.random.normal(189000, 50000, 500),
        "education_num": np.random.randint(1, 16, 500),
        "capital_gain": np.abs(np.random.normal(5000, 15000, 500)),
        "capital_loss": np.abs(np.random.normal(0, 300, 500)),
        "hours_per_week": np.random.randint(20, 60, 500),
        "workclass": np.random.choice(["Private", "Self-emp", "Government"], 500),
        "income": np.random.choice([0, 1], 500, p=[0.75, 0.25])
    })
    synthetic = pd.DataFrame({
        "age": np.random.normal(45, 10, 150),
        "fnlwgt": np.random.normal(200000, 40000, 150),
        "education_num": np.random.randint(10, 16, 150),
        "capital_gain": np.abs(np.random.normal(10000, 12000, 150)),
        "capital_loss": np.abs(np.random.normal(100, 200, 150)),
        "hours_per_week": np.random.randint(30, 70, 150),
        "workclass": np.random.choice(["Private", "Government"], 150),
        "income": 1
    })
    continuous_features = ["age", "fnlwgt", "education_num", "capital_gain", "capital_loss", "hours_per_week"]
    categorical_features = ["workclass"]
    
    metrics = compute_evaluation_metrics(original, synthetic, continuous_features, categorical_features,
                                         distance_threshold=0.5, density_threshold=0.5, gamma=1.0, plot=True)
    
    print("\nComputed Evaluation Metrics:")
    print(metrics)
