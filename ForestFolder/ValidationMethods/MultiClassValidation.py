import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ks_2samp, chi2_contingency
from sklearn.metrics import pairwise_distances, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.model_selection import train_test_split

def discriminative_score(real, synthetic, features):
    """
    Computes a discriminative score by training a classifier to distinguish between
    real and synthetic samples. Returns the classifier accuracy—values near 0.5 indicate that 
    the two datasets are very similar.
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
    Compute the Maximum Mean Discrepancy (MMD) between two sets of samples X and Y using an RBF kernel.
    Lower values indicate more similar distributions.
    """
    XX = rbf_kernel(X, X, gamma=gamma)
    YY = rbf_kernel(Y, Y, gamma=gamma)
    XY = rbf_kernel(X, Y, gamma=gamma)
    mmd = np.mean(XX) + np.mean(YY) - 2 * np.mean(XY)
    return mmd

def analyze_target_distribution(df, target):
    """
    Analyze and plot the distribution of the target variable for a multi-class dataset.
    This function prints the counts and relative frequencies for each class and displays a bar chart.
    """
    counts = df[target].value_counts().sort_index()
    total = counts.sum()
    percentages = (counts / total * 100).round(1)
    
    print("Target counts:")
    print(counts)
    print("\nRelative frequencies:")
    print((counts / total).round(2))
    
    # Create color palette matching the second image
    colors = ['#1f77b4', '#ff7f0e', '#d62728', '#8c564b', '#e377c2', '#bcbd22', '#17becf']
    
    plt.figure(figsize=(10, 6))
    ax = counts.plot(kind="bar", color=colors[:len(counts)])
    
    # Add count numbers at the top of each bar
    for i, (count, percentage) in enumerate(zip(counts, percentages)):
        ax.text(i, count + 0.01*max(counts), str(count), ha='center')
        
        # Add percentage labels inside bars
        ax.text(i, count/2, f"{percentage}%", ha='center', color='white', fontweight='bold')
    
    plt.title("Target Distribution")
    plt.xlabel(target)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()

def validate_synthetic_data(original, synthetic, continuous_features, categorical_features,
                            distance_threshold=0.1, density_threshold=0.5, gamma=1.0, plot=False):
    """
    Validate synthetic data quality by comparing it with the original dataset.
    
    This function computes:
      - Summary statistics and KS tests for continuous features.
      - Chi-squared tests for categorical features.
      - Coverage, diversity, density, discriminative score, and Maximum Mean Discrepancy (MMD).
      
    Parameters:
      original (pd.DataFrame): Original data.
      synthetic (pd.DataFrame): Synthetic data.
      continuous_features (list): Names of continuous features.
      categorical_features (list): Names of categorical features.
      distance_threshold (float): Threshold for the coverage metric.
      density_threshold (float): Radius for local density evaluation in synthetic data.
      gamma (float): Gamma parameter for the RBF kernel in MMD.
      plot (bool): Whether to display plots.
      
    Returns:
      metrics (dict): A dictionary containing computed metrics.
    """
    # Drop rows with missing values in required columns.
    required_cols = continuous_features + categorical_features
    original = original.dropna(subset=required_cols)
    synthetic = synthetic.dropna(subset=required_cols)
    
    metrics = {}
    
    print("### Continuous Features Validation ###")
    metrics["continuous"] = {}
    for feature in continuous_features:
        print(f"\nFeature: {feature}")
        orig_mean = original[feature].mean()
        orig_std = original[feature].std()
        synth_mean = synthetic[feature].mean()
        synth_std = synthetic[feature].std()
        print(f"Original: mean={orig_mean:.3f}, std={orig_std:.3f}")
        print(f"Synthetic: mean={synth_mean:.3f}, std={synth_std:.3f}")
        
        ks_stat, ks_p = ks_2samp(original[feature], synthetic[feature])
        print(f"KS test: statistic={ks_stat:.3f}, p-value={ks_p:.3f}")
        if ks_p > 0.05:
            print("-> No significant difference in distribution")
        else:
            print("-> Significant difference detected!")
        
        print("Synthetic shape after dropna:", synthetic.shape)
        print("Any NA in mcg for synthetic?", synthetic[feature].isna().sum())
        print("Synthetic mcg describe:\n", synthetic[feature].describe())
        
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
    
    print("\n### Categorical Features Validation ###")
    metrics["categorical"] = {}
    for feature in categorical_features:
        print(f"\nFeature: {feature}")
        orig_counts = original[feature].value_counts().sort_index()
        synth_counts = synthetic[feature].value_counts().sort_index()
        print("Original counts:\n", orig_counts)
        print("Synthetic counts:\n", synth_counts)
        
        counts_df = pd.DataFrame({
            "Original": orig_counts,
            "Synthetic": synth_counts
        }).fillna(0)
        chi2, p_value, dof, expected = chi2_contingency(counts_df)
        print(f"Chi-squared test: statistic={chi2:.3f}, p-value={p_value:.3f}")
        if p_value > 0.05:
            print("-> No significant difference in categorical distribution")
        else:
            print("-> Significant difference detected!")
            
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
    
    print("\n### Coverage Metric ###")
    distances = pairwise_distances(original[continuous_features], synthetic[continuous_features])
    min_distances = distances.min(axis=1)
    coverage = np.mean(min_distances <= distance_threshold)
    print(f"Coverage: {coverage*100:.2f}% of original samples have a synthetic neighbor within {distance_threshold}")
    metrics["coverage"] = coverage
    
    print("\n### Diversity Metric ###")
    synth_distances = pairwise_distances(synthetic[continuous_features])
    i_upper = np.triu_indices_from(synth_distances, k=1)
    avg_distance = np.mean(synth_distances[i_upper])
    std_distance = np.std(synth_distances[i_upper])
    print(f"Average pairwise distance among synthetic samples: {avg_distance:.3f}")
    print(f"Standard deviation of pairwise distances: {std_distance:.3f}")
    metrics["diversity"] = {
        "avg_distance": avg_distance,
        "std_distance": std_distance
    }
    
    print("\n### Density Metric ###")
    synth_distances = pairwise_distances(synthetic[continuous_features])
    neighbor_counts = (synth_distances <= density_threshold).sum(axis=1) - 1
    average_density = np.mean(neighbor_counts)
    print(f"Average local density: {average_density:.3f} neighbors within a radius of {density_threshold}")
    metrics["density"] = {
        "density_threshold": density_threshold,
        "average_density": average_density,
        "neighbor_counts": neighbor_counts.tolist()
    }
    
    print("\n### Discriminative Score ###")
    disc_score = discriminative_score(original[continuous_features].copy(), synthetic[continuous_features].copy(), continuous_features)
    print(f"Discriminative score (classifier accuracy): {disc_score:.3f}")
    metrics["discriminative_score"] = disc_score
    
    print("\n### MMD Metric ###")
    mmd_value = compute_mmd(original[continuous_features].values, synthetic[continuous_features].values, gamma=gamma)
    print(f"Maximum Mean Discrepancy (MMD): {mmd_value:.3f}")
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

def validate_synthetic_data_per_class(original, synthetic, target, continuous_features, categorical_features,
                                      num_classes, distance_threshold=0.1, density_threshold=0.5, gamma=1.0, plot=False):
    """
    Validates synthetic data quality for a specified number of classes (starting from the smallest original class).
    
    Parameters:
      original (pd.DataFrame): Original dataset containing the target column.
      synthetic (pd.DataFrame): Synthetic dataset containing the target column.
      target (str): Name of the target column.
      continuous_features (list): List of continuous feature names.
      categorical_features (list): List of categorical feature names.
      num_classes (int): Number of classes to validate (starting from the smallest).
      distance_threshold (float): Threshold for the coverage metric.
      density_threshold (float): Radius for local density evaluation in synthetic data.
      gamma (float): Gamma parameter for the RBF kernel in MMD.
      plot (bool): Whether to display plots for each class.
      
    Returns:
      metrics_per_class (dict): Dictionary mapping each validated class to its computed metrics.
    """
    # Get class frequencies from the original data and sort in ascending order.
    class_counts = original[target].value_counts().sort_values()
    selected_classes = class_counts.index[:num_classes]
    
    metrics_per_class = {}
    for cls in selected_classes:
        print(f"\n\n### Validation for class: {cls} ###\n")
        # Filter both original and synthetic datasets for the current class.
        orig_class = original[original[target] == cls].copy()
        synth_class = synthetic[synthetic[target] == cls].copy()
        
        # Call the existing validation function for this class.
        metrics = validate_synthetic_data(
            original=orig_class,
            synthetic=synth_class,
            continuous_features=continuous_features,
            categorical_features=categorical_features,
            distance_threshold=distance_threshold,
            density_threshold=density_threshold,
            gamma=gamma,
            plot=plot
        )
        metrics_per_class[cls] = metrics
    return metrics_per_class

