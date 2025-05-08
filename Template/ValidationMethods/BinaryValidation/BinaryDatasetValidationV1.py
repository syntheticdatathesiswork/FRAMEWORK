"""
Synthetic Data Validation Module

This module provides comprehensive tools for validating synthetic data quality
by comparing it with original data across multiple dimensions:
- Statistical similarity
- Distribution matching
- Feature correlations
- Coverage and diversity
- Predictive utility

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ks_2samp, chi2_contingency, wasserstein_distance
from sklearn.metrics import pairwise_distances, accuracy_score, classification_report, roc_curve, auc
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


def discriminative_score(real, synthetic, features):
    """
    Computes a discriminative score by training a classifier to distinguish between
    real and synthetic samples. Accuracy near 0.5 indicates the two distributions are very similar.
    
    Parameters:
        real (pd.DataFrame): Original data
        synthetic (pd.DataFrame): Synthetic data
        features (list): Features to use for distinguishing
    
    Returns:
        acc (float): Classification accuracy
        auc_score (float): Area under ROC curve
        confusion (dict): Confusion matrix values
    """
    real = real.copy()
    synthetic = synthetic.copy()
    real["is_synthetic"] = 0
    synthetic["is_synthetic"] = 1
    data = pd.concat([real, synthetic], axis=0).reset_index(drop=True)
    X = data[features]
    y = data["is_synthetic"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    # Scale features for more reliable results
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train classifier
    clf = LogisticRegression(max_iter=1000, class_weight='balanced')
    clf.fit(X_train_scaled, y_train)
    
    # Get predictions and probabilities
    y_pred = clf.predict(X_test_scaled)
    y_proba = clf.predict_proba(X_test_scaled)[:, 1]
    
    # Calculate metrics
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    
    # Calculate AUC
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc_score = auc(fpr, tpr)
    
    # Extract confusion matrix values
    real_classified_as_real = report['0']['recall'] * sum(y_test == 0)
    synthetic_classified_as_synthetic = report['1']['recall'] * sum(y_test == 1)
    confusion = {
        'real_as_real': report['0']['recall'],
        'real_as_synthetic': 1 - report['0']['recall'],
        'synthetic_as_synthetic': report['1']['recall'],
        'synthetic_as_real': 1 - report['1']['recall']
    }
    
    return acc, auc_score, confusion


def compute_mmd(X, Y, gamma=1.0):
    """
    Computes the Maximum Mean Discrepancy (MMD) between two sets of samples using an RBF kernel.
    Lower MMD indicates more similar distributions.
    
    Parameters:
        X (array-like): First dataset
        Y (array-like): Second dataset
        gamma (float): Kernel bandwidth parameter
        
    Returns:
        float: MMD value
    """
    # Scale data for reliable MMD calculation
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    Y_scaled = scaler.transform(Y)
    
    XX = rbf_kernel(X_scaled, X_scaled, gamma=gamma)
    YY = rbf_kernel(Y_scaled, Y_scaled, gamma=gamma)
    XY = rbf_kernel(X_scaled, Y_scaled, gamma=gamma)
    mmd = np.mean(XX) + np.mean(YY) - 2 * np.mean(XY)
    return mmd


def earth_movers_distance(real, synthetic, features):
    """
    Computes Earth Mover's Distance (Wasserstein distance) for each feature.
    
    Parameters:
        real (pd.DataFrame): Original data
        synthetic (pd.DataFrame): Synthetic data
        features (list): Features to compare
        
    Returns:
        distances (dict): Feature-wise EMD values
    """
    distances = {}
    for feature in features:
        emd = wasserstein_distance(real[feature], synthetic[feature])
        distances[feature] = emd
    
    # Also calculate average EMD across all features
    distances['average'] = np.mean(list(distances.values()))
    return distances


def check_scaling(original, synthetic, continuous_features):
    """
    Compares the min, max, mean, and standard deviation of continuous features 
    between original and synthetic data.
    
    Parameters:
        original (pd.DataFrame): Original data
        synthetic (pd.DataFrame): Synthetic data
        continuous_features (list): Names of continuous features
        
    Returns:
        dict: Scaling comparison results
    """
    print("### Scaling / Normalization Check ###", flush=True)
    scaling_results = {}
    
    for feature in continuous_features:
        orig_min = original[feature].min()
        orig_max = original[feature].max()
        orig_mean = original[feature].mean()
        orig_std = original[feature].std()
        
        synth_min = synthetic[feature].min()
        synth_max = synthetic[feature].max()
        synth_mean = synthetic[feature].mean()
        synth_std = synthetic[feature].std()
        
        # Calculate relative differences
        min_diff_pct = abs((synth_min - orig_min) / (orig_max - orig_min + 1e-10)) * 100 if orig_max != orig_min else 0
        max_diff_pct = abs((synth_max - orig_max) / (orig_max - orig_min + 1e-10)) * 100 if orig_max != orig_min else 0
        mean_diff_pct = abs((synth_mean - orig_mean) / orig_mean) * 100 if orig_mean != 0 else 0
        std_diff_pct = abs((synth_std - orig_std) / orig_std) * 100 if orig_std != 0 else 0
        
        scaling_results[feature] = {
            'min_diff_pct': min_diff_pct,
            'max_diff_pct': max_diff_pct,
            'mean_diff_pct': mean_diff_pct,
            'std_diff_pct': std_diff_pct
        }
        
        print(f"\nFeature: {feature}", flush = True)
        print(f"Original: min={orig_min:.3f}, max={orig_max:.3f}, mean={orig_mean:.3f}, std={orig_std:.3f}", flush = True)
        print(f"Synthetic: min={synth_min:.3f}, max={synth_max:.3f}, mean={synth_mean:.3f}, std={synth_std:.3f}", flush = True)
        print(f"Relative differences: min={min_diff_pct:.1f}%, max={max_diff_pct:.1f}%, mean={mean_diff_pct:.1f}%, std={std_diff_pct:.1f}%", flush = True)
    
    return scaling_results


def extract_synthetic_samples(original_df, augmented_df, synthetic_column='synthetic'):
    """
    Extracts synthetic samples from augmented dataset using a column that indicates synthetic status.
    
    Parameters:
        original_df (pd.DataFrame): Original dataset
        augmented_df (pd.DataFrame): Augmented dataset (original + synthetic)
        synthetic_column (str): Name of column indicating if a sample is synthetic
        
    Returns:
        synthetic_df (pd.DataFrame): Just the synthetic samples
    """
    # If the synthetic indicator column exists in augmented data, use it
    if synthetic_column in augmented_df.columns:
        synthetic = augmented_df[augmented_df[synthetic_column] == 1].copy()
        synthetic.drop(columns=[synthetic_column], inplace=True, errors='ignore')
        print(f"Extracted {len(synthetic)} synthetic samples using '{synthetic_column}' indicator column")
        return synthetic
    
    # If no synthetic column is in the data, infer synthetic samples by their absence in original
    print(f"Warning: '{synthetic_column}' column not found in augmented dataset.")
    print("Identifying synthetic samples by comparing to original dataset...")
    
    # Get expected synthetic count
    expected_synthetic_count = len(augmented_df) - len(original_df)
    if expected_synthetic_count <= 0:
        print("Warning: Augmented dataset doesn't contain more samples than original dataset.")
        return pd.DataFrame(columns=augmented_df.columns)
    
    # Simple approach: If we're only looking at minority class samples, 
    # assume all extra rows in augmented data are synthetic
    synthetic = augmented_df.iloc[len(original_df):].copy()
    print(f"Extracted {len(synthetic)} synthetic samples as extra rows in augmented dataset")
    
    return synthetic


def analyze_class_distribution(df, target):
    """
    Analyzes and plots the distribution of the outcome classes.
    
    Parameters:
        df (pd.DataFrame): The dataset to analyze
        target (str): The name of the target column
        
    Returns:
        dict: Class distribution statistics
    """
    from IPython.display import display
    
    counts = df[target].value_counts().sort_index()
    total = len(df)
    
    # Calculate class proportions
    proportions = counts / total
    
    # Calculate class imbalance ratio (majority:minority)
    majority_class = counts.idxmax()
    minority_class = counts.idxmin()
    imbalance_ratio = counts[majority_class] / counts[minority_class]
    
    print(f"Class distribution:", flush=True)
    for cls, count in counts.items():
        print(f"  Class {cls}: {count} samples ({proportions[cls]*100:.1f}%)", flush=True)
    print(f"Class imbalance ratio (majority:minority): {imbalance_ratio:.2f}:1", flush=True)
    
    # Plot the distribution
    plt.figure(figsize=(8, 5))
    
    # Create custom colors list - blue for first bar, orange for second
    colors = ['#3274A1', '#E1812C']  # Blue and Orange
    if len(counts) == 1:
        colors = [colors[0]]  # Just use blue if only one class
    elif len(counts) > 2:
        # Extend the color list if more than 2 classes
        colors = colors + ['#3A923A', '#C03D3E', '#8B7DAF', '#738C73'][:len(counts)-2]
    
    ax = counts.plot(kind="bar", color=colors)
    plt.title("Class Distribution", fontsize=14)
    plt.xlabel(target, fontsize=12)
    plt.ylabel("Count", fontsize=12)
    plt.grid(axis='y', alpha=0.3)
    
    # Add count labels on top of bars
    for i, count in enumerate(counts):
        ax.text(i, count + (max(counts)*0.02), str(count), 
                ha='center', va='bottom', fontweight='bold')
    
    # Add percentage labels in the middle of bars
    for i, (cls, count) in enumerate(counts.items()):
        percentage = count / total * 100
        ax.text(i, count/2, f"{percentage:.1f}%", 
                ha='center', va='center', fontweight='bold', color='white')
    
    plt.tight_layout()
    
    # Display the plot instead of saving it
    display(plt.gcf())
    plt.close()
    
    return {
        'counts': counts.to_dict(),
        'proportions': proportions.to_dict(),
        'imbalance_ratio': imbalance_ratio,
        'majority_class': majority_class,
        'minority_class': minority_class
    }


def feature_correlation_analysis(original, synthetic, features, target=None):
    """
    Analyzes and compares the correlation structure between original and synthetic data.
    
    Parameters:
        original (pd.DataFrame): Original dataset
        synthetic (pd.DataFrame): Synthetic data
        features (list): Features to include in correlation analysis
        target (str, optional): Target variable name
        
    Returns:
        tuple: (original correlation matrix, synthetic correlation matrix, difference matrix)
    """
    # Select columns for correlation analysis
    cols_to_use = features.copy()
    if target is not None and target in original.columns:
        cols_to_use.append(target)
    
    # Compute correlation matrices
    orig_corr = original[cols_to_use].corr()
    synth_corr = synthetic[cols_to_use].corr()
    
    # Compute absolute differences between correlation matrices
    diff_corr = (orig_corr - synth_corr).abs()
    
    # Plot correlation matrices
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    sns.heatmap(orig_corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1, ax=axes[0], fmt='.2f')
    axes[0].set_title('Original Data Correlation')
    
    sns.heatmap(synth_corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1, ax=axes[1], fmt='.2f')
    axes[1].set_title('Synthetic Data Correlation')
    
    sns.heatmap(diff_corr, annot=True, cmap='YlOrRd', vmin=0, vmax=1, ax=axes[2], fmt='.2f')
    axes[2].set_title('Absolute Correlation Differences')
    
    plt.tight_layout()
    plt.savefig("correlation_comparison.png")
    plt.close()
    
    # Print average absolute correlation difference
    avg_diff = diff_corr.values[np.triu_indices_from(diff_corr.values, k=1)].mean()
    print(f"Average absolute correlation difference: {avg_diff:.4f}", flush=True)
    
    # Identify largest correlation differences
    if not diff_corr.empty:
        diff_df = diff_corr.unstack().reset_index()
        diff_df.columns = ['Feature1', 'Feature2', 'Difference']
        diff_df = diff_df[diff_df['Feature1'] != diff_df['Feature2']]
        diff_df = diff_df.sort_values('Difference', ascending=False).head(5)
        
        print("\nLargest correlation differences:", flush=True)
        for _, row in diff_df.iterrows():
            f1, f2, diff = row
            orig_val = orig_corr.loc[f1, f2]
            synth_val = synth_corr.loc[f1, f2]
            print(f"  {f1} vs {f2}: Original={orig_val:.3f}, Synthetic={synth_val:.3f}, Diff={diff:.3f}", flush=True)
    
    return orig_corr, synth_corr, diff_corr


def predictive_performance_comparison(original, synthetic, features, target, test_data=None):
    """
    Compares the predictive performance of models trained on original vs. synthetic data.
    
    Parameters:
        original (pd.DataFrame): Original training data
        synthetic (pd.DataFrame): Synthetic data
        features (list): Feature columns to use
        target (str): Target column name
        test_data (pd.DataFrame, optional): Separate test data. If None, will split original data.
        
    Returns:
        dict: Performance metrics for both models
    """
    if test_data is None:
        # Split original data into train/test sets
        original_train, original_test = train_test_split(
            original, test_size=0.3, random_state=42, stratify=original[target])
    else:
        original_train = original
        original_test = test_data
    
    # Prepare data
    X_train_orig = original_train[features]
    y_train_orig = original_train[target]
    X_train_synth = synthetic[features]
    y_train_synth = synthetic[target]
    X_test = original_test[features]
    y_test = original_test[target]
    
    # Scale features
    scaler = StandardScaler()
    X_train_orig_scaled = scaler.fit_transform(X_train_orig)
    X_train_synth_scaled = scaler.transform(X_train_synth)
    X_test_scaled = scaler.transform(X_test)
    
    # Train models
    models = {
        'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
        'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42)
    }
    
    results = {}
    for model_name, model in models.items():
        print(f"\n{model_name} Performance Comparison:", flush=True)
        
        # Train on original data
        model_orig = model.__class__(**model.get_params())
        model_orig.fit(X_train_orig_scaled, y_train_orig)
        y_pred_orig = model_orig.predict(X_test_scaled)
        orig_report = classification_report(y_test, y_pred_orig, output_dict=True)
        
        # Train on synthetic data
        model_synth = model.__class__(**model.get_params())
        model_synth.fit(X_train_synth_scaled, y_train_synth)
        y_pred_synth = model_synth.predict(X_test_scaled)
        synth_report = classification_report(y_test, y_pred_synth, output_dict=True)
        
        # Cross-validation on original
        cv_scores_orig = cross_val_score(model, X_train_orig_scaled, y_train_orig, cv=5, scoring='accuracy')
        
        # Output results
        print(f"  Original data - Test accuracy: {orig_report['accuracy']:.4f}", flush=True)
        print(f"  Synthetic data - Test accuracy: {synth_report['accuracy']:.4f}", flush=True)
        print(f"  Original data - CV accuracy: {cv_scores_orig.mean():.4f} (±{cv_scores_orig.std():.4f})", flush=True)
        
        for class_label in sorted(orig_report.keys()):
            if class_label not in ['accuracy', 'macro avg', 'weighted avg']:
                orig_f1 = orig_report[class_label]['f1-score']
                synth_f1 = synth_report[class_label]['f1-score']
                print(f"  Class {class_label} - F1 score: Original={orig_f1:.4f}, Synthetic={synth_f1:.4f}", flush=True)
        
        results[model_name] = {
            'original': {
                'accuracy': orig_report['accuracy'],
                'cv_accuracy': cv_scores_orig.mean(),
                'cv_std': cv_scores_orig.std(),
                'f1_by_class': {k: v['f1-score'] for k, v in orig_report.items() 
                               if k not in ['accuracy', 'macro avg', 'weighted avg']}
            },
            'synthetic': {
                'accuracy': synth_report['accuracy'],
                'f1_by_class': {k: v['f1-score'] for k, v in synth_report.items() 
                               if k not in ['accuracy', 'macro avg', 'weighted avg']}
            }
        }
    
    return results


# Move the visualization functions to module level instead of being nested inside another function
def plot_feature_distributions(original, synthetic, features, feature_type='continuous', show_plots=True):
    """
    Creates and displays improved visualizations for feature distributions.
    
    Parameters:
        original (pd.DataFrame): Original dataset
        synthetic (pd.DataFrame): Synthetic dataset
        features (list): Features to visualize
        feature_type (str): Type of features ('continuous' or 'categorical')
        show_plots (bool): Whether to display plots in addition to saving them
        
    Returns:
        list: Paths to saved plot files
    """
    from IPython.display import display
    
    plot_files = []
    
    # Set better aesthetics
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Create a figure for each feature
    for feature in features:
        if feature_type == 'continuous':
            # Create figure
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
            
            # Set custom colors with better contrast
            original_color = '#3274A1'  # Blue
            synthetic_color = '#E1812C'  # Orange
            
            # Histogram plot
            ax1.set_title(f"Histogram of {feature}", fontsize=14)
            
            # Plot histograms with semi-transparency
            _, bins, _ = ax1.hist(original[feature], bins=30, alpha=0.7, 
                                 color=original_color, label="Original")
            ax1.hist(synthetic[feature], bins=bins, alpha=0.7, 
                    color=synthetic_color, label="Synthetic")
            
            ax1.set_xlabel(feature, fontsize=12)
            ax1.set_ylabel("Count", fontsize=12)
            ax1.legend(fontsize=12)
            ax1.grid(True, alpha=0.3)
            
            # KDE plot
            ax2.set_title(f"Density Distribution of {feature}", fontsize=14)
            sns.kdeplot(original[feature], ax=ax2, color=original_color, 
                       label="Original", linewidth=2.5)
            sns.kdeplot(synthetic[feature], ax=ax2, color=synthetic_color, 
                       label="Synthetic", linewidth=2.5)
            
            ax2.set_xlabel(feature, fontsize=12)
            ax2.set_ylabel("Density", fontsize=12)
            ax2.legend(fontsize=12)
            ax2.grid(True, alpha=0.3)
            
            # Adjust spacing
            plt.tight_layout()
            
            # Save figure
            filename = f"feature_{feature}_distribution.png"
            plt.savefig(filename, dpi=100, bbox_inches='tight')
            plot_files.append(filename)
            
            # Display if requested
            if show_plots:
                display(plt.gcf())
            
            plt.close()
            
        elif feature_type == 'categorical':
            # For categorical features
            plt.figure(figsize=(12, 6))
            
            # Get value counts
            orig_counts = original[feature].value_counts(normalize=True).sort_index()
            synth_counts = synthetic[feature].value_counts(normalize=True).sort_index()
            
            # Create a DataFrame with aligned categories
            all_categories = sorted(set(orig_counts.index) | set(synth_counts.index))
            orig_aligned = [orig_counts.get(cat, 0) for cat in all_categories]
            synth_aligned = [synth_counts.get(cat, 0) for cat in all_categories]
            
            # Plot as bar chart with custom colors
            bar_width = 0.35
            x = np.arange(len(all_categories))
            
            plt.bar(x - bar_width/2, orig_aligned, bar_width, 
                   label='Original', color='#3274A1', alpha=0.8)
            plt.bar(x + bar_width/2, synth_aligned, bar_width, 
                   label='Synthetic', color='#E1812C', alpha=0.8)
            
            plt.title(f"Distribution of {feature}", fontsize=14)
            plt.xlabel(feature, fontsize=12)
            plt.ylabel("Frequency", fontsize=12)
            
            # Handle x-axis labels for categorical features
            if len(all_categories) > 10:
                plt.xticks(x, all_categories, rotation=45, ha='right', fontsize=10)
            else:
                plt.xticks(x, all_categories, fontsize=10)
                
            plt.legend(fontsize=12)
            plt.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            
            # Save figure
            filename = f"feature_{feature}_categorical.png"
            plt.savefig(filename, dpi=100, bbox_inches='tight')
            plot_files.append(filename)
            
            # Display if requested
            if show_plots:
                display(plt.gcf())
                
            plt.close()
    
    return plot_files


def plot_2d_feature_comparison(original, synthetic, features, show_plots=True):
    """
    Creates and displays improved 2D feature space visualization.
    
    Parameters:
        original (pd.DataFrame): Original dataset
        synthetic (pd.DataFrame): Synthetic dataset
        features (list): Features to visualize (at least 2)
        show_plots (bool): Whether to display plots in addition to saving them
        
    Returns:
        str: Path to saved plot file
    """
    from IPython.display import display
    
    if len(features) < 2:
        print("Need at least 2 features for 2D visualization")
        return None
    
    # Set better aesthetics
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    feature_x, feature_y = features[:2]
    
    # Scatter plot
    ax1.set_title(f"Feature Space: {feature_x} vs. {feature_y}", fontsize=14)
    
    # Plot with better markers and colors
    ax1.scatter(original[feature_x], original[feature_y], 
              s=50, marker='o', alpha=0.6, edgecolor='white', linewidth=0.5,
              label="Original", color='#3274A1')
    ax1.scatter(synthetic[feature_x], synthetic[feature_y], 
              s=50, marker='x', alpha=0.6, linewidth=1.5,
              label="Synthetic", color='#E1812C')
    
    ax1.set_xlabel(feature_x, fontsize=12)
    ax1.set_ylabel(feature_y, fontsize=12)
    ax1.legend(fontsize=12, markerscale=1.5)
    ax1.grid(True, alpha=0.3)
    
    # KDE plot
    ax2.set_title(f"Density Comparison: {feature_x} vs. {feature_y}", fontsize=14)
    
    # Create joint KDE plots with adjusted levels
    sns.kdeplot(x=original[feature_x], y=original[feature_y], 
               ax=ax2, fill=True, levels=5, alpha=0.5, 
               color='#3274A1', label="Original")
    sns.kdeplot(x=synthetic[feature_x], y=synthetic[feature_y], 
               ax=ax2, fill=True, levels=5, alpha=0.5, 
               color='#E1812C', label="Synthetic")
    
    ax2.set_xlabel(feature_x, fontsize=12)
    ax2.set_ylabel(feature_y, fontsize=12)
    ax2.legend(fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    filename = "2d_feature_comparison.png"
    plt.savefig(filename, dpi=100, bbox_inches='tight')
    
    # Display if requested
    if show_plots:
        display(plt.gcf())
        
    plt.close()
    
    return filename


def validate_synthetic_data(original_df, augmented_df, continuous_features, categorical_features, 
                         target=None, minority_class=1, synthetic_column='is_synthetic',
                         distance_threshold=0.1, density_threshold=0.5, gamma=1.0, 
                         plot=True, show_plots=True, test_data=None):
    """
    Validates synthetic data quality by comparing original minority class samples 
    with the synthetic samples used for augmentation.
    
    This function:
    1. Extracts the minority class from both original and augmented datasets
    2. Identifies synthetic samples using the synthetic indicator column or other methods
    3. Computes various quality metrics between original minority class and synthetic samples
    4. Creates and displays visualizations for feature distributions
    
    Parameters:
      original_df (pd.DataFrame): Original dataset before augmentation
      augmented_df (pd.DataFrame): Augmented dataset (original + synthetic samples)
      continuous_features (list): Continuous feature names
      categorical_features (list): Categorical feature names
      target (str): Target variable name
      minority_class (any): Value of the minority class in the target column
      synthetic_column (str): Name of column that indicates if a sample is synthetic
      distance_threshold (float): For coverage metric
      density_threshold (float): Radius for local density calculation
      gamma (float): Gamma for the RBF kernel in MMD
      plot (bool): If True, generate plots
      show_plots (bool): If True, display plots in addition to saving them
      test_data (pd.DataFrame, optional): Separate test data for predictive performance
      
    Returns:
      metrics (dict): Dictionary of computed metrics
    """
    # Set up better plotting aesthetics globally
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Filter to extract just the minority class samples
    if target is not None:
        print(f"Analyzing quality of synthetic samples for {target}={minority_class}", flush=True)
        original_minority = original_df[original_df[target] == minority_class].copy()
        augmented_minority = augmented_df[augmented_df[target] == minority_class].copy()
    else:
        # If no target is specified, use the entire datasets
        original_minority = original_df.copy()
        augmented_minority = augmented_df.copy()
    
    # Identify which samples are synthetic
    print("Identifying synthetic samples...", flush=True)
    synthetic_samples = extract_synthetic_samples(original_minority, augmented_minority, synthetic_column)
    
    print(f"Original minority class: {len(original_minority)} samples", flush=True)
    print(f"Augmented minority class: {len(augmented_minority)} samples", flush=True)
    print(f"Synthetic samples: {len(synthetic_samples)} samples", flush=True)
    
    # If no synthetic samples were found, raise an error
    if len(synthetic_samples) == 0:
        raise ValueError("No synthetic samples detected. The augmented dataset should contain synthetic samples.")
    
    # Required columns for validation
    required_cols = continuous_features + categorical_features
    
    # Drop rows with missing values
    original = original_minority.dropna(subset=required_cols).reset_index(drop=True)
    synthetic = synthetic_samples.dropna(subset=required_cols).reset_index(drop=True)
    
    metrics = {}
    metrics['sample_sizes'] = {
        'original_minority': len(original),
        'synthetic': len(synthetic),
        'augmentation_ratio': len(synthetic) / len(original) if len(original) > 0 else 0
    }
    
    # Compute basic statistics for each continuous feature
    print("\n### Continuous Features Validation ###", flush=True)
    metrics["continuous"] = {}
    
    for feature in continuous_features:
        print(f"\nFeature: {feature}", flush=True)
        orig_mean = original[feature].mean()
        orig_std = original[feature].std()
        synth_mean = synthetic[feature].mean()
        synth_std = synthetic[feature].std()
        
        print(f"Original: mean={orig_mean:.3f}, std={orig_std:.3f}", flush=True)
        print(f"Synthetic: mean={synth_mean:.3f}, std={synth_std:.3f}", flush=True)
        
        # Calculate relative differences
        mean_diff_pct = abs((synth_mean - orig_mean) / orig_mean) * 100 if orig_mean != 0 else 0
        std_diff_pct = abs((synth_std - orig_std) / orig_std) * 100 if orig_std != 0 else 0
        print(f"Relative differences: mean={mean_diff_pct:.1f}%, std={std_diff_pct:.1f}%", flush=True)
        
        # KS test
        ks_stat, ks_p = ks_2samp(original[feature], synthetic[feature])
        print(f"KS test: statistic={ks_stat:.3f}, p-value={ks_p:.3f}", flush=True)
        
        if ks_p > 0.05:
            print("-> No significant difference", flush=True)
        else:
            print("-> Significant difference detected", flush=True)
        
        # Store metrics
        metrics["continuous"][feature] = {
            "orig_mean": orig_mean,
            "orig_std": orig_std,
            "synth_mean": synth_mean,
            "synth_std": synth_std,
            "mean_diff_pct": mean_diff_pct,
            "std_diff_pct": std_diff_pct,
            "ks_stat": ks_stat,
            "ks_p": ks_p
        }
    
    # Compute statistics for categorical features
    print("\n### Categorical Features Validation ###", flush=True)
    metrics["categorical"] = {}
    
    for feature in categorical_features:
        print(f"\nFeature: {feature}", flush=True)
        orig_counts = original[feature].value_counts(normalize=True).sort_index()
        synth_counts = synthetic[feature].value_counts(normalize=True).sort_index()
        
        print("Original distribution:", flush=True)
        print(orig_counts, flush=True)
        print("\nSynthetic distribution:", flush=True)
        print(synth_counts, flush=True)
        
        # Chi-squared test
        # Create a DataFrame with aligned categories for chi-square test
        all_categories = sorted(set(orig_counts.index) | set(synth_counts.index))
        count_data = {
            'Original': [original[feature].value_counts().get(cat, 0) for cat in all_categories],
            'Synthetic': [synthetic[feature].value_counts().get(cat, 0) for cat in all_categories]
        }
        counts_df = pd.DataFrame(count_data, index=all_categories)
        
        # Calculate chi-square only if we have enough data
        if counts_df['Original'].sum() > 0 and counts_df['Synthetic'].sum() > 0:
            chi2, p_value, dof, expected = chi2_contingency(counts_df)
            print(f"Chi-squared test: statistic={chi2:.3f}, p-value={p_value:.3f}", flush=True)
            
            if p_value > 0.05:
                print("-> No significant difference", flush=True)
            else:
                print("-> Significant difference detected", flush=True)
        else:
            chi2, p_value = np.nan, np.nan
            print("-> Insufficient data for chi-squared test", flush=True)
        
        # Store metrics
        metrics["categorical"][feature] = {
            "chi2_stat": chi2,
            "chi2_p": p_value,
            "dof": dof if 'dof' in locals() else None,
            "orig_distribution": orig_counts.to_dict(),
            "synth_distribution": synth_counts.to_dict()
        }
    
    # Calculate Earth Mover's Distance (Wasserstein distance)
    print("\n### Earth Mover's Distance (Wasserstein) ###", flush=True)
    emd_values = {}
    
    for feature in continuous_features:
        emd = wasserstein_distance(original[feature], synthetic[feature])
        emd_values[feature] = emd
        print(f"EMD for {feature}: {emd:.3f}", flush=True)
    
    # Calculate average EMD
    emd_values['average'] = np.mean(list(emd_values.values()))
    print(f"Average EMD across features: {emd_values['average']:.3f}", flush=True)
    metrics["earth_movers_distance"] = emd_values
    
    # Calculate coverage (how well synthetic data covers original data)
    print("\n### Coverage Metric ###", flush=True)
    # Scale the data for better distance calculations
    scaler = StandardScaler()
    original_scaled = scaler.fit_transform(original[continuous_features])
    synthetic_scaled = scaler.transform(synthetic[continuous_features])
    
    distances = pairwise_distances(original_scaled, synthetic_scaled)
    min_distances = distances.min(axis=1)
    coverage = np.mean(min_distances <= distance_threshold)
    print(f"Coverage: {coverage*100:.2f}% of original samples have a synthetic neighbor within {distance_threshold}", flush=True)
    metrics["coverage"] = coverage
    
    # Calculate diversity (how diverse the synthetic data is)
    print("\n### Diversity Metric ###", flush=True)
    synth_distances = pairwise_distances(synthetic_scaled)
    np.fill_diagonal(synth_distances, np.inf)  # Exclude self-distances
    
    avg_distance = np.mean(synth_distances.min(axis=1))
    print(f"Average nearest neighbor distance among synthetic samples: {avg_distance:.3f}", flush=True)
    metrics["diversity"] = {"avg_nearest_distance": avg_distance}
    
    # Create and display plots if requested
    if plot:
        print("\n### Generating Visualizations ###", flush=True)
        
        # Plot continuous feature distributions
        print("Generating continuous feature visualizations...", flush=True)
        continuous_plots = plot_feature_distributions(
            original, synthetic, continuous_features, 
            feature_type='continuous', show_plots=show_plots
        )
        
        # Plot categorical feature distributions
        if categorical_features:
            print("Generating categorical feature visualizations...", flush=True)
            categorical_plots = plot_feature_distributions(
                original, synthetic, categorical_features, 
                feature_type='categorical', show_plots=show_plots
            )
        
        # Create 2D visualization if we have multiple continuous features
        if len(continuous_features) >= 2:
            print("Generating 2D feature space visualization...", flush=True)
            # Select features with highest variance for visualization
            feature_vars = original[continuous_features].var().sort_values(ascending=False)
            top_features = feature_vars.index[:2].tolist()
            
            two_d_plot = plot_2d_feature_comparison(
                original, synthetic, top_features, show_plots=show_plots
            )
    
    # Generate summary report
    print("\n### SYNTHETIC DATA VALIDATION SUMMARY ###", flush=True)
    print(f"Original samples: {len(original)}", flush=True)
    print(f"Synthetic samples: {len(synthetic)}", flush=True)
    
    # Statistical similarity summary
    stat_similarity = {}
    for feature, metrics_dict in metrics["continuous"].items():
        if metrics_dict["ks_p"] > 0.05:
            stat_similarity[feature] = "Similar"
        else:
            stat_similarity[feature] = "Different"
    
    similar_count = sum(1 for val in stat_similarity.values() if val == "Similar")
    print(f"Features with statistically similar distributions: {similar_count}/{len(continuous_features)}", flush=True)
    
    # Overall quality assessment
    quality_score = (coverage + similar_count/len(continuous_features)) / 2
    quality_rating = "Excellent" if quality_score > 0.8 else \
                    "Good" if quality_score > 0.6 else \
                    "Fair" if quality_score > 0.4 else "Poor"
    
    print(f"Overall quality rating: {quality_rating} (score: {quality_score:.2f})", flush=True)
    metrics["overall_quality"] = {"score": quality_score, "rating": quality_rating}
    
    return metrics