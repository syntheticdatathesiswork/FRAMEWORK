"""
Multi-Class Fairness Assessment Module for Synthetic Data

This module provides tools for evaluating and measuring the fairness
of synthetic data augmentation across multiple classes, with a focus on 
detecting and quantifying potential bias in sensitive attributes.

The module includes:
- Distribution-based fairness metrics (KL, JS, TVD) across multiple classes
- Multi-class subgroup fairness analysis
- Intersectional fairness analysis
- Conditional demographic parity assessment for multiple classes

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import entropy
from scipy.stats import chi2_contingency
import warnings
warnings.filterwarnings('ignore')


def compute_kl_divergence(p, q):
    """
    Compute KL divergence between two distributions p and q.
    
    Parameters:
        p (array-like): First probability distribution
        q (array-like): Second probability distribution
        
    Returns:
        float: KL divergence value
    """
    # Add a small constant to avoid division by zero
    p = np.array(p) + 1e-10
    q = np.array(q) + 1e-10
    
    # Normalize to ensure we have proper probability distributions
    p = p / p.sum()
    q = q / q.sum()
    
    # Calculate KL divergence
    return entropy(p, q)


def compute_js_divergence(p, q):
    """
    Compute Jensen-Shannon divergence between two distributions.
    More symmetric than KL divergence.
    
    Parameters:
        p (array-like): First probability distribution
        q (array-like): Second probability distribution
        
    Returns:
        float: JS divergence value
    """
    p = np.array(p) + 1e-10
    q = np.array(q) + 1e-10
    
    # Normalize
    p = p / p.sum()
    q = q / q.sum()
    
    # Calculate midpoint distribution
    m = 0.5 * (p + q)
    
    # Calculate JS divergence
    return 0.5 * (entropy(p, m) + entropy(q, m))


def compute_tvd(p, q):
    """
    Compute Total Variation Distance between two distributions.
    
    Parameters:
        p (array-like): First probability distribution
        q (array-like): Second probability distribution
        
    Returns:
        float: TVD value
    """
    p = np.array(p) + 1e-10
    q = np.array(q) + 1e-10
    
    # Normalize
    p = p / p.sum()
    q = q / q.sum()
    
    # Calculate TVD
    return 0.5 * np.sum(np.abs(p - q))


def multiclass_fairness_bias_check(original, augmented, sensitive_features, target=None, classes=None, subgroup_analysis=True):
    """
    Performs comprehensive fairness and bias checks by comparing the distribution of sensitive attributes
    between original and augmented datasets across multiple target classes.
    Reports distributions and computes multiple divergence metrics.
    
    Parameters:
      original (pd.DataFrame): The original dataset.
      augmented (pd.DataFrame): The augmented dataset (original + synthetic).
      sensitive_features (list): List of sensitive attribute column names (e.g., "race", "sex", "native_country").
      target (str, optional): Target variable name for subgroup analysis.
      classes (list, optional): List of target classes to analyze. If None, all unique values are used.
      subgroup_analysis (bool): Whether to perform subgroup analysis by target variable.
    
    Returns:
      dict: Fairness metrics and analysis results
    """
    print("### Multi-Class Fairness and Bias Checks ###\n", flush=True)
    results = {'overall': {}, 'subgroups': {}, 'by_class': {}}
    
    # If classes not specified but target is, get all unique classes
    if classes is None and target is not None:
        classes = sorted(original[target].unique())
        print(f"Analyzing {len(classes)} target classes: {classes}\n", flush=True)
    
    # 1. Overall distribution analysis for each sensitive feature (across all classes)
    for feature in sensitive_features:
        print(f"Sensitive Feature: {feature}", flush=True)
        
        # Calculate distributions
        orig_dist = original[feature].value_counts(normalize=True).sort_index()
        aug_dist = augmented[feature].value_counts(normalize=True).sort_index()
        
        # Get all unique categories across both datasets
        all_categories = sorted(set(orig_dist.index).union(aug_dist.index))
        
        # Create aligned distributions with all categories
        orig_aligned = np.array([orig_dist.get(cat, 0) for cat in all_categories])
        aug_aligned = np.array([aug_dist.get(cat, 0) for cat in all_categories])
        
        # Compute divergence metrics
        kl_div = compute_kl_divergence(orig_aligned, aug_aligned)
        js_div = compute_js_divergence(orig_aligned, aug_aligned)
        tvd = compute_tvd(orig_aligned, aug_aligned)
        
        # Prepare visualization data
        visual_data = pd.DataFrame({
            'Category': all_categories,
            'Original': orig_aligned,
            'Augmented': aug_aligned,
            'Absolute Difference': np.abs(orig_aligned - aug_aligned)
        })
        
        # Print distributions and metrics
        print("Original distribution:", flush=True)
        print(orig_dist, flush=True)
        print("\nAugmented distribution:", flush=True)
        print(aug_dist, flush=True)
        print("\nDistribution metrics:", flush=True)
        print(f"KL Divergence: {kl_div:.4f}", flush=True)
        print(f"Jensen-Shannon Divergence: {js_div:.4f}", flush=True)
        print(f"Total Variation Distance: {tvd:.4f}", flush=True)
        
        # Evaluate fairness quality based on TVD
        if tvd < 0.05:
            fairness_quality = "Excellent"
        elif tvd < 0.1:
            fairness_quality = "Good"
        elif tvd < 0.2:
            fairness_quality = "Fair"
        else:
            fairness_quality = "Poor"
        
        print(f"Fairness quality for {feature}: {fairness_quality}", flush=True)
        
        # Create visualization
        plt.figure(figsize=(10, 6))
        bar_width = 0.35
        x = np.arange(len(all_categories))
        
        plt.bar(x - bar_width/2, orig_aligned, bar_width, label='Original', color='skyblue')
        plt.bar(x + bar_width/2, aug_aligned, bar_width, label='Augmented', color='salmon')
        
        plt.title(f"Distribution of {feature} (TVD: {tvd:.4f})")
        plt.xlabel(feature)
        plt.ylabel("Frequency")
        plt.xticks(x, all_categories, rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"fairness_{feature}_distribution.png")
        plt.close()
        
        # Store results
        results['overall'][feature] = {
            'kl_divergence': kl_div,
            'js_divergence': js_div,
            'total_variation_distance': tvd,
            'fairness_quality': fairness_quality,
            'categories': all_categories,
            'original_distribution': orig_aligned.tolist(),
            'augmented_distribution': aug_aligned.tolist()
        }
        
        print("-" * 40, flush=True)
    
    # 2. Class-specific analysis (analyze each class separately)
    if target is not None and classes is not None:
        print("\n### Class-Specific Fairness Analysis ###", flush=True)
        
        for target_class in classes:
            print(f"\nAnalyzing {target} = {target_class}:", flush=True)
            results['by_class'][str(target_class)] = {}
            
            # Filter data by target class
            orig_class = original[original[target] == target_class]
            aug_class = augmented[augmented[target] == target_class]
            
            # Skip if we don't have enough data
            if len(orig_class) < 10 or len(aug_class) < 10:
                print(f"  Not enough data for analysis (Original: {len(orig_class)}, Augmented: {len(aug_class)})", flush=True)
                continue
            
            print(f"  Original samples: {len(orig_class)}", flush=True)
            print(f"  Augmented samples: {len(aug_class)}", flush=True)
            
            # Class-specific TVD for each sensitive feature
            class_tvds = {}
            
            for feature in sensitive_features:
                # Calculate distributions
                orig_dist = orig_class[feature].value_counts(normalize=True).sort_index()
                aug_dist = aug_class[feature].value_counts(normalize=True).sort_index()
                
                # Get all unique categories across both datasets
                all_categories = sorted(set(orig_dist.index).union(aug_dist.index))
                
                # Create aligned distributions with all categories
                orig_aligned = np.array([orig_dist.get(cat, 0) for cat in all_categories])
                aug_aligned = np.array([aug_dist.get(cat, 0) for cat in all_categories])
                
                # Compute TVD
                tvd = compute_tvd(orig_aligned, aug_aligned)
                class_tvds[feature] = tvd
                
                # Evaluate fairness quality based on TVD
                if tvd < 0.05:
                    fairness_quality = "Excellent"
                elif tvd < 0.1:
                    fairness_quality = "Good"
                elif tvd < 0.2:
                    fairness_quality = "Fair"
                else:
                    fairness_quality = "Poor"
                
                print(f"  Feature {feature}: TVD = {tvd:.4f} ({fairness_quality})", flush=True)
                
                # Store results
                results['by_class'][str(target_class)][feature] = {
                    'total_variation_distance': tvd,
                    'fairness_quality': fairness_quality,
                    'categories': all_categories,
                    'original_distribution': orig_aligned.tolist(),
                    'augmented_distribution': aug_aligned.tolist()
                }
            
            # Class-level summary
            avg_tvd = np.mean(list(class_tvds.values()))
            if avg_tvd < 0.05:
                class_fairness_quality = "Excellent"
            elif avg_tvd < 0.1:
                class_fairness_quality = "Good"
            elif avg_tvd < 0.2:
                class_fairness_quality = "Fair"
            else:
                class_fairness_quality = "Poor"
            
            print(f"  Overall class fairness quality: {class_fairness_quality} (avg TVD: {avg_tvd:.4f})", flush=True)
            
            # Store class summary
            results['by_class'][str(target_class)]['summary'] = {
                'average_tvd': avg_tvd,
                'fairness_quality': class_fairness_quality
            }
    
    # 3. Subgroup fairness analysis (if target is provided)
    if subgroup_analysis and target is not None and classes is not None:
        print("\n### Subgroup Fairness Analysis ###", flush=True)
        
        for feature in sensitive_features:
            print(f"\nSubgroup Analysis for {feature} by {target}:", flush=True)
            results['subgroups'][feature] = {}
            
            for target_class in classes:
                print(f"\n  Target Class: {target} = {target_class}", flush=True)
                
                # Filter data by target class
                orig_filtered = original[original[target] == target_class]
                aug_filtered = augmented[augmented[target] == target_class]
                
                # Skip if we don't have enough data
                if len(orig_filtered) < 10 or len(aug_filtered) < 10:
                    print(f"  Not enough data for analysis (Original: {len(orig_filtered)}, Augmented: {len(aug_filtered)})", flush=True)
                    continue
                
                # Calculate distributions for this subgroup
                orig_subgroup_dist = orig_filtered[feature].value_counts(normalize=True).sort_index()
                aug_subgroup_dist = aug_filtered[feature].value_counts(normalize=True).sort_index()
                
                # Get all categories for this subgroup
                subgroup_categories = sorted(set(orig_subgroup_dist.index).union(aug_subgroup_dist.index))
                
                # Create aligned distributions
                orig_subgroup_aligned = np.array([orig_subgroup_dist.get(cat, 0) for cat in subgroup_categories])
                aug_subgroup_aligned = np.array([aug_subgroup_dist.get(cat, 0) for cat in subgroup_categories])
                
                # Compute metrics
                subgroup_tvd = compute_tvd(orig_subgroup_aligned, aug_subgroup_aligned)
                
                print(f"  Total Variation Distance: {subgroup_tvd:.4f}", flush=True)
                
                # Store subgroup results
                results['subgroups'][feature][str(target_class)] = {
                    'total_variation_distance': subgroup_tvd,
                    'categories': subgroup_categories,
                    'original_distribution': orig_subgroup_aligned.tolist(),
                    'augmented_distribution': aug_subgroup_aligned.tolist()
                }
    
    # 4. Intersectional fairness analysis (if we have multiple sensitive features)
    if len(sensitive_features) > 1:
        print("\n### Intersectional Fairness Analysis ###", flush=True)
        results['intersectional'] = {}
        
        # Analyze pairs of sensitive features
        for i, feature1 in enumerate(sensitive_features):
            for feature2 in sensitive_features[i+1:]:
                print(f"\nIntersectional Analysis for {feature1} × {feature2}:", flush=True)
                results['intersectional'][f"{feature1}_{feature2}"] = {}
                
                # If target is provided, do class-specific intersectional analysis
                if target is not None and classes is not None:
                    for target_class in classes:
                        print(f"  Class: {target} = {target_class}", flush=True)
                        
                        # Filter data by target class
                        orig_class = original[original[target] == target_class]
                        aug_class = augmented[augmented[target] == target_class]
                        
                        # Skip if we don't have enough data
                        if len(orig_class) < 10 or len(aug_class) < 10:
                            print(f"  Not enough data for analysis (Original: {len(orig_class)}, Augmented: {len(aug_class)})", flush=True)
                            continue
                        
                        # Create composite feature
                        orig_class['temp_intersect'] = orig_class[feature1].astype(str) + " & " + orig_class[feature2].astype(str)
                        aug_class['temp_intersect'] = aug_class[feature1].astype(str) + " & " + aug_class[feature2].astype(str)
                        
                        # Calculate distributions
                        orig_intersect_dist = orig_class['temp_intersect'].value_counts(normalize=True).sort_index()
                        aug_intersect_dist = aug_class['temp_intersect'].value_counts(normalize=True).sort_index()
                        
                        # Get all intersectional categories
                        intersect_categories = sorted(set(orig_intersect_dist.index).union(aug_intersect_dist.index))
                        
                        # Create aligned distributions
                        orig_intersect_aligned = np.array([orig_intersect_dist.get(cat, 0) for cat in intersect_categories])
                        aug_intersect_aligned = np.array([aug_intersect_dist.get(cat, 0) for cat in intersect_categories])
                        
                        # Compute metrics
                        intersect_tvd = compute_tvd(orig_intersect_aligned, aug_intersect_aligned)
                        
                        print(f"    Total Variation Distance: {intersect_tvd:.4f}", flush=True)
                        
                        # Store results
                        results['intersectional'][f"{feature1}_{feature2}"][str(target_class)] = {
                            'total_variation_distance': intersect_tvd,
                            'categories': intersect_categories,
                            'original_distribution': orig_intersect_aligned.tolist(),
                            'augmented_distribution': aug_intersect_aligned.tolist()
                        }
                        
                        # Clean up temporary column
                        orig_class.drop('temp_intersect', axis=1, inplace=True)
                        aug_class.drop('temp_intersect', axis=1, inplace=True)
                else:
                    # If no target, do overall intersectional analysis
                    # Create composite feature
                    original['temp_intersect'] = original[feature1].astype(str) + " & " + original[feature2].astype(str)
                    augmented['temp_intersect'] = augmented[feature1].astype(str) + " & " + augmented[feature2].astype(str)
                    
                    # Calculate distributions
                    orig_intersect_dist = original['temp_intersect'].value_counts(normalize=True).sort_index()
                    aug_intersect_dist = augmented['temp_intersect'].value_counts(normalize=True).sort_index()
                    
                    # Get all intersectional categories
                    intersect_categories = sorted(set(orig_intersect_dist.index).union(aug_intersect_dist.index))
                    
                    # Create aligned distributions
                    orig_intersect_aligned = np.array([orig_intersect_dist.get(cat, 0) for cat in intersect_categories])
                    aug_intersect_aligned = np.array([aug_intersect_dist.get(cat, 0) for cat in intersect_categories])
                    
                    # Compute metrics
                    intersect_tvd = compute_tvd(orig_intersect_aligned, aug_intersect_aligned)
                    
                    print(f"Total Variation Distance: {intersect_tvd:.4f}", flush=True)
                    
                    # Only show top differences for brevity
                    diff_df = pd.DataFrame({
                        'Intersectional Group': intersect_categories,
                        'Original': orig_intersect_aligned,
                        'Augmented': aug_intersect_aligned,
                        'Absolute Difference': np.abs(orig_intersect_aligned - aug_intersect_aligned)
                    }).sort_values('Absolute Difference', ascending=False)
                    
                    print("\nTop 5 largest distribution differences:", flush=True)
                    for idx, row in diff_df.head(5).iterrows():
                        print(f"  {row['Intersectional Group']}: Original={row['Original']:.3f}, Augmented={row['Augmented']:.3f}, Diff={row['Absolute Difference']:.3f}", flush=True)
                    
                    # Store results
                    results['intersectional'][f"{feature1}_{feature2}"]['overall'] = {
                        'total_variation_distance': intersect_tvd,
                        'categories': intersect_categories,
                        'original_distribution': orig_intersect_aligned.tolist(),
                        'augmented_distribution': aug_intersect_aligned.tolist()
                    }
                    
                    # Clean up temporary column
                    original.drop('temp_intersect', axis=1, inplace=True)
                    augmented.drop('temp_intersect', axis=1, inplace=True)
    
    # 5. Summary of fairness analysis
    print("\n### Fairness Analysis Summary ###", flush=True)
    
    # Overall summary
    overall_tvd = np.mean([results['overall'][f]['total_variation_distance'] for f in sensitive_features])
    print(f"Overall dataset TVD across all sensitive features: {overall_tvd:.4f}", flush=True)
    
    if overall_tvd < 0.05:
        overall_quality = "Excellent"
    elif overall_tvd < 0.1:
        overall_quality = "Good"
    elif overall_tvd < 0.2:
        overall_quality = "Fair"
    else:
        overall_quality = "Poor"
    
    print(f"Overall dataset fairness quality: {overall_quality}", flush=True)
    
    # Class-specific summary
    if 'by_class' in results and results['by_class']:
        print("\nFairness quality by class:", flush=True)
        class_qualities = {}
        for cls, metrics in results['by_class'].items():
            if 'summary' in metrics:
                tvd = metrics['summary']['average_tvd']
                quality = metrics['summary']['fairness_quality']
                class_qualities[cls] = quality
                print(f"  {target} = {cls}: {quality} (TVD = {tvd:.4f})", flush=True)
        
        # Detect problematic classes
        problematic_classes = [cls for cls, quality in class_qualities.items() 
                              if quality in ["Fair", "Poor"]]
        
        if problematic_classes:
            print("\nClasses needing attention:")
            for cls in problematic_classes:
                print(f"  {target} = {cls}: {class_qualities[cls]}")
    
    # Store summary
    results['summary'] = {
        'average_tvd': overall_tvd,
        'overall_fairness_quality': overall_quality
    }
    
    if 'by_class' in results and results['by_class']:
        results['summary']['class_fairness'] = {
            cls: results['by_class'][cls]['summary'] 
            for cls in results['by_class'] 
            if 'summary' in results['by_class'][cls]
        }
    
    return results


def multiclass_conditional_demographic_parity(original, augmented, sensitive_feature, target, classes=None):
    """
    Measures conditional demographic parity by comparing conditional probabilities
    of each target class given sensitive attributes between original and augmented data.
    
    Parameters:
        original (pd.DataFrame): Original dataset
        augmented (pd.DataFrame): Augmented dataset (original + synthetic)
        sensitive_feature (str): Name of sensitive attribute column
        target (str): Name of target column
        classes (list, optional): List of target classes to analyze. If None, all unique values are used.
        
    Returns:
        tuple: (dictionary of metric values by class, results dictionary)
    """
    print(f"### Multi-Class Conditional Demographic Parity for {sensitive_feature} ###", flush=True)
    
    # If classes not specified, get all unique classes
    if classes is None:
        classes = sorted(original[target].unique())
    
    # Initialize results
    metrics_by_class = {}
    results = {'by_class': {}, 'overall': {}}
    
    # For each sensitive attribute value
    for sens_value in sorted(set(original[sensitive_feature].unique()) | set(augmented[sensitive_feature].unique())):
        print(f"\nSensitive attribute {sensitive_feature}={sens_value}:", flush=True)
        
        # Get subsets
        orig_subset = original[original[sensitive_feature] == sens_value]
        aug_subset = augmented[augmented[sensitive_feature] == sens_value]
        
        # Skip if not enough data
        if len(orig_subset) < 10 or len(aug_subset) < 10:
            print(f"  Insufficient data (Original: {len(orig_subset)}, Augmented: {len(aug_subset)})", flush=True)
            continue
        
        # Calculate class distribution for this sensitive value
        orig_class_dist = orig_subset[target].value_counts(normalize=True).sort_index()
        aug_class_dist = aug_subset[target].value_counts(normalize=True).sort_index()
        
        # Print distributions
        print("  Original class distribution:", flush=True)
        for cls, prob in orig_class_dist.items():
            print(f"    Class {cls}: {prob:.4f}", flush=True)
        
        print("  Augmented class distribution:", flush=True)
        for cls, prob in aug_class_dist.items():
            print(f"    Class {cls}: {prob:.4f}", flush=True)
        
        # For each class, calculate conditional probability differences
        diffs_by_class = {}
        for cls in classes:
            orig_prob = orig_class_dist.get(cls, 0)
            aug_prob = aug_class_dist.get(cls, 0)
            diff = abs(orig_prob - aug_prob)
            diffs_by_class[str(cls)] = diff
            print(f"  Class {cls} - Original: {orig_prob:.4f}, Augmented: {aug_prob:.4f}, Diff: {diff:.4f}", flush=True)
            
            # Store in class results
            if str(cls) not in results['by_class']:
                results['by_class'][str(cls)] = {'diffs_by_attribute': {}}
            
            if sens_value not in results['by_class'][str(cls)]['diffs_by_attribute']:
                results['by_class'][str(cls)]['diffs_by_attribute'][sens_value] = diff
        
        # Store by sensitive value
        if sens_value not in results['overall']:
            results['overall'][sens_value] = {'class_distributions': {}}
        
        results['overall'][sens_value]['class_distributions'] = {
            'original': orig_class_dist.to_dict(),
            'augmented': aug_class_dist.to_dict(),
            'differences': diffs_by_class
        }
    
    # Calculate overall metrics for each class
    print("\nSummary of conditional probability differences by class:", flush=True)
    for cls in classes:
        if str(cls) in results['by_class'] and 'diffs_by_attribute' in results['by_class'][str(cls)]:
            diffs = list(results['by_class'][str(cls)]['diffs_by_attribute'].values())
            avg_diff = np.mean(diffs) if diffs else np.nan
            max_diff = np.max(diffs) if diffs else np.nan
            
            print(f"Class {cls} - Avg Diff: {avg_diff:.4f}, Max Diff: {max_diff:.4f}", flush=True)
            
            # Update class results
            results['by_class'][str(cls)]['avg_diff'] = avg_diff
            results['by_class'][str(cls)]['max_diff'] = max_diff
            
            # Store in metrics dictionary
            metrics_by_class[str(cls)] = avg_diff
    
    # Calculate overall metric across all classes
    avg_all_diffs = np.mean(list(metrics_by_class.values())) if metrics_by_class else np.nan
    print(f"\nAverage conditional probability difference across all classes: {avg_all_diffs:.4f}", flush=True)
    
    # Store overall metric
    results['overall_metric'] = avg_all_diffs
    
    # Create visualization
    plt.figure(figsize=(12, 6))
    
    # If we have metrics for multiple classes
    if metrics_by_class:
        # Plot average differences by class
        classes_list = list(metrics_by_class.keys())
        diffs_list = [metrics_by_class[cls] for cls in classes_list]
        
        bars = plt.bar(classes_list, diffs_list, color='skyblue')
        
        # Add reference line
        plt.axhline(y=0.1, color='red', linestyle='--', label='Fair threshold (0.1)')
        
        # Color bars based on fairness
        for i, diff in enumerate(diffs_list):
            if diff > 0.2:
                bars[i].set_color('salmon')  # Poor fairness
            elif diff > 0.1:
                bars[i].set_color('orange')  # Fair fairness
            else:
                bars[i].set_color('lightgreen')  # Good fairness
        
        plt.title(f"Conditional Probability Differences by Class for {sensitive_feature}")
        plt.xlabel("Class")
        plt.ylabel("Average Absolute Difference")
        plt.xticks(rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"multiclass_conditional_parity_{sensitive_feature}.png")
        plt.close()
    
    return metrics_by_class, results


def multiclass_fairness_assessment(original_df, augmented_df, sensitive_features, target, classes=None):
    """
    Performs a comprehensive fairness assessment on augmented data (original + synthetic) for multiple classes.
    Analyzes each class to ensure synthetic data augmentation doesn't introduce bias across target classes.
    
    Parameters:
        original_df (pd.DataFrame): Original unaugmented dataset
        augmented_df (pd.DataFrame): Augmented dataset (original + synthetic data)
        sensitive_features (list): List of sensitive attribute columns
        target (str): Target variable name
        classes (list, optional): List of target classes to analyze. If None, all unique values are used.
        
    Returns:
        dict: Comprehensive fairness metrics
    """
    # If classes not specified, use all unique values in the target column
    if classes is None:
        classes = sorted(original_df[target].unique())
    
    print(f"Analyzing fairness for {len(classes)} classes: {classes}", flush=True)
    
    # Verify we have data to work with
    if len(original_df) == 0 or len(augmented_df) == 0:
        raise ValueError("No data available. Please check your datasets.")
    
    # Initialize results
    results = {}
    
    # Store basic information
    results['dataset_info'] = {
        'original_samples': len(original_df),
        'augmented_samples': len(augmented_df),
        'synthetic_samples': len(augmented_df) - len(original_df),
        'target_column': target,
        'classes': classes,
        'sensitive_features': sensitive_features
    }
    
    # Print class distribution in original and augmented datasets
    print("\nClass distribution in original dataset:", flush=True)
    orig_class_counts = original_df[target].value_counts().sort_index()
    for cls, count in orig_class_counts.items():
        print(f"  Class {cls}: {count} samples ({count/len(original_df)*100:.1f}%)", flush=True)
    
    print("\nClass distribution in augmented dataset:", flush=True)
    aug_class_counts = augmented_df[target].value_counts().sort_index()
    for cls, count in aug_class_counts.items():
        print(f"  Class {cls}: {count} samples ({count/len(augmented_df)*100:.1f}%)", flush=True)
    
    # Store class distribution
    results['class_distribution'] = {
        'original': orig_class_counts.to_dict(),
        'augmented': aug_class_counts.to_dict()
    }
    
    # 1. Distribution-based fairness (marginal distributions)
    print("\n--- Multi-Class Distribution-Based Fairness Analysis ---", flush=True)
    distribution_results = multiclass_fairness_bias_check(
        original_df, augmented_df, sensitive_features, target, classes, subgroup_analysis=True)
    results['distribution_metrics'] = distribution_results
    
    # 2. Conditional fairness metrics for each sensitive feature
    print("\n--- Multi-Class Conditional Fairness Metrics ---", flush=True)
    results['conditional_metrics'] = {}
    
    for feature in sensitive_features:
        metrics_by_class, cond_results = multiclass_conditional_demographic_parity(
            original_df, augmented_df, feature, target, classes)
        results['conditional_metrics'][feature] = cond_results
    
    # 3. Overall fairness summary across all classes and features
    print("\n--- Overall Multi-Class Fairness Summary ---", flush=True)
    
    # Calculate aggregate fairness scores:
    # 1. First by class
    class_fairness_scores = {}
    for cls in classes:
        cls_str = str(cls)
        
        # Distribution TVD by class (if available)
        if ('by_class' in distribution_results and 
            cls_str in distribution_results['by_class'] and 
            'summary' in distribution_results['by_class'][cls_str]):
            
            dist_score = distribution_results['by_class'][cls_str]['summary']['average_tvd']
        else:
            dist_score = 1.0  # Default high value if not available
        
        # Conditional probability differences by class (average across features)
        cond_scores = []
        for feature in sensitive_features:
            if (feature in results['conditional_metrics'] and 
                'by_class' in results['conditional_metrics'][feature] and
                cls_str in results['conditional_metrics'][feature]['by_class'] and
                'avg_diff' in results['conditional_metrics'][feature]['by_class'][cls_str]):
                
                cond_scores.append(results['conditional_metrics'][feature]['by_class'][cls_str]['avg_diff'])
        
        cond_score = np.mean(cond_scores) if cond_scores else 1.0
        
        # Combine for overall class score (lower is better)
        class_score = 0.5 * dist_score + 0.5 * cond_score
        
        # Interpret the class score
        if class_score < 0.05:
            class_quality = "Excellent"
        elif class_score < 0.1:
            class_quality = "Good"
        elif class_score < 0.2:
            class_quality = "Fair"
        else:
            class_quality = "Poor"
        
        print(f"Class {cls} fairness score: {class_score:.4f} ({class_quality})", flush=True)
        
        class_fairness_scores[cls_str] = {
            'score': class_score,
            'quality': class_quality,
            'distribution_score': dist_score,
            'conditional_score': cond_score
        }
    
    # 2. Calculate overall fairness score across all classes
    overall_score = np.mean([score['score'] for score in class_fairness_scores.values()])
    
    # Interpret the overall score
    if overall_score < 0.05:
        overall_quality = "Excellent"
    elif overall_score < 0.1:
        overall_quality = "Good"
    elif overall_score < 0.2:
        overall_quality = "Fair"
    else:
        overall_quality = "Poor"
    
    print(f"Overall multi-class fairness score: {overall_score:.4f} ({overall_quality})", flush=True)
    
    # List classes that need attention
    problematic_classes = []
    for cls, metrics in class_fairness_scores.items():
        if metrics['quality'] in ["Fair", "Poor"]:
            problematic_classes.append(cls)
    
    if problematic_classes:
        print("\nClasses needing attention:", flush=True)
        for cls in problematic_classes:
            print(f"  - Class {cls}: {class_fairness_scores[cls]['quality']} (score: {class_fairness_scores[cls]['score']:.4f})", flush=True)
    
    # List features that need attention
    if overall_quality in ["Fair", "Poor"]:
        print("\nFeatures needing attention:", flush=True)
        for feature in sensitive_features:
            feature_quality = distribution_results['overall'][feature]['fairness_quality']
            if feature_quality in ["Fair", "Poor"]:
                tvd = distribution_results['overall'][feature]['total_variation_distance']
                print(f"  - {feature}: TVD = {tvd:.4f} ({feature_quality})", flush=True)
    
    # Store summary results
    results['overall_fairness'] = {
        'score': overall_score,
        'quality': overall_quality,
        'class_fairness': class_fairness_scores
    }
    
    return results


def print_multiclass_fairness_report(results):
    """
    Prints a simplified fairness report for multi-class fairness assessment,
    highlighting only the key metrics and insights.
    
    Parameters:
        results (dict): The fairness assessment results
    """
    # Print basic statistics (with safe defaults)
    dataset_info = results.get('dataset_info', {})
    original_count = dataset_info.get('original_samples', 0)
    augmented_count = dataset_info.get('augmented_samples', 0)
    synthetic_count = dataset_info.get('synthetic_samples', 0)
    target_column = dataset_info.get('target_column', 'target')
    classes = dataset_info.get('classes', [])
    
    print("\n===== MULTI-CLASS FAIRNESS ASSESSMENT SUMMARY =====")
    print(f"Original samples: {original_count}")
    print(f"Augmented samples: {augmented_count}")
    print(f"Synthetic samples added: {synthetic_count}")
    print(f"Target column: {target_column}")
    print(f"Classes analyzed: {classes}")
    
    # Overall fairness score (with safe access)
    overall_fairness = results.get('overall_fairness', {})
    fairness_score = overall_fairness.get('score', 0)
    fairness_quality = overall_fairness.get('quality', 'Unknown')
    print(f"\nOVERALL FAIRNESS: {fairness_quality} (score: {fairness_score:.4f})")
    
    # Class-specific fairness
    class_fairness = overall_fairness.get('class_fairness', {})
    
    if class_fairness:
        print("\nFAIRNESS BY CLASS:")
        for cls, metrics in class_fairness.items():
            score = metrics.get('score', 0)
            quality = metrics.get('quality', 'Unknown')
            print(f"  • Class {cls}: {quality} (score: {score:.4f})")
    
    # Distribution metrics for sensitive attributes
    distribution_metrics = results.get('distribution_metrics', {})
    overall_metrics = distribution_metrics.get('overall', {})
    
    if overall_metrics:
        print("\nSENSITIVE ATTRIBUTE ANALYSIS:")
        for feature, metrics in overall_metrics.items():
            tvd = metrics.get('total_variation_distance', 0)
            quality = metrics.get('fairness_quality', 'Unknown')
            print(f"  • {feature}: {quality} (TVD: {tvd:.4f})")
    
    # Highlight potential problems
    problematic_classes = [cls for cls, metrics in class_fairness.items() 
                          if metrics.get('quality', '') in ['Fair', 'Poor']]
    problematic_features = [feature for feature, metrics in overall_metrics.items() 
                           if metrics.get('fairness_quality', '') in ['Fair', 'Poor']]
    
    if problematic_classes or problematic_features:
        print("\nPOTENTIAL FAIRNESS CONCERNS:")
        
        if problematic_classes:
            print("  Classes requiring attention:")
            for cls in problematic_classes:
                print(f"  • Class {cls}: {class_fairness[cls]['quality']}")
        
        if problematic_features:
            print("  Sensitive attributes requiring attention:")
            for feature in problematic_features:
                print(f"  • {feature}: {overall_metrics[feature]['fairness_quality']}")
    
    # Recommendations based on fairness quality
    print("\nRECOMMENDATIONS:")
    if fairness_quality in ["Excellent", "Good"]:
        print("  The synthetic data preserves fairness across classes and sensitive attributes")
        print("  Safe to use for augmenting the dataset")
    else:
        print("  There are fairness concerns with the synthetic data")
        
        if problematic_classes:
            print("  • Focus on improving synthetic data quality for specific classes:")
            for cls in problematic_classes[:3]:  # Show top 3 problematic classes
                print(f"    - Class {cls}")
        
        if problematic_features:
            print("  • Address distribution issues in these sensitive attributes:")
            for feature in problematic_features[:3]:  # Show top 3 problematic features
                print(f"    - {feature}")
        
        print("Consider adjusting your synthetic data generation process to better preserve distributions")
        print("For classes with poor fairness scores, consider alternative augmentation approaches")