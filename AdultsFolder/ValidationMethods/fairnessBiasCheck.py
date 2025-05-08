"""
Fairness Assessment Module for Synthetic Data

This module provides tools for evaluating and measuring the fairness
of synthetic data augmentation, with a focus on detecting and quantifying 
potential bias in sensitive attributes.

The module includes:
- Distribution-based fairness metrics (KL, JS, TVD)
- Subgroup fairness analysis
- Intersectional fairness analysis
- Conditional demographic parity assessment

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


def fairness_bias_check(original, augmented, sensitive_features, target=None, subgroup_analysis=True):
    """
    Performs comprehensive fairness and bias checks by comparing the distribution of sensitive attributes
    between original and augmented datasets. Reports distributions and computes multiple divergence metrics.
    
    Parameters:
      original (pd.DataFrame): The original dataset.
      augmented (pd.DataFrame): The augmented dataset (original + synthetic).
      sensitive_features (list): List of sensitive attribute column names (e.g., "race", "sex", "native_country").
      target (str, optional): Target variable name for subgroup analysis.
      subgroup_analysis (bool): Whether to perform subgroup analysis by target variable.
    
    Returns:
      dict: Fairness metrics and analysis results
    """
    print("### Fairness and Bias Checks ###\n", flush=True)
    results = {'overall': {}, 'subgroups': {}}
    
    # 1. Overall distribution analysis for each sensitive feature
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
    
    # 2. Subgroup fairness analysis (if target is provided)
    if subgroup_analysis and target is not None and target in original.columns and target in augmented.columns:
        print("\n### Subgroup Fairness Analysis ###", flush=True)
        
        # Get target classes
        target_classes = sorted(original[target].unique())
        
        for feature in sensitive_features:
            print(f"\nSubgroup Analysis for {feature} by {target}:", flush=True)
            results['subgroups'][feature] = {}
            
            for target_class in target_classes:
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
    
    # 3. Intersectional fairness analysis (if we have multiple sensitive features)
    if len(sensitive_features) > 1:
        print("\n### Intersectional Fairness Analysis ###", flush=True)
        results['intersectional'] = {}
        
        # Analyze pairs of sensitive features
        for i, feature1 in enumerate(sensitive_features):
            for feature2 in sensitive_features[i+1:]:
                print(f"\nIntersectional Analysis for {feature1} × {feature2}:", flush=True)
                
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
                results['intersectional'][f"{feature1}_{feature2}"] = {
                    'total_variation_distance': intersect_tvd,
                    'categories': intersect_categories,
                    'original_distribution': orig_intersect_aligned.tolist(),
                    'augmented_distribution': aug_intersect_aligned.tolist()
                }
                
                # Clean up temporary column
                original.drop('temp_intersect', axis=1, inplace=True)
                augmented.drop('temp_intersect', axis=1, inplace=True)
    
    # 4. Summary of fairness analysis
    print("\n### Fairness Analysis Summary ###", flush=True)
    overall_tvd = np.mean([results['overall'][f]['total_variation_distance'] for f in sensitive_features])
    print(f"Average Total Variation Distance across all sensitive features: {overall_tvd:.4f}", flush=True)
    
    if overall_tvd < 0.05:
        print("Overall fairness quality: Excellent", flush=True)
    elif overall_tvd < 0.1:
        print("Overall fairness quality: Good", flush=True)
    elif overall_tvd < 0.2:
        print("Overall fairness quality: Fair", flush=True)
    else:
        print("Overall fairness quality: Poor", flush=True)
    
    for feature in sensitive_features:
        tvd = results['overall'][feature]['total_variation_distance']
        print(f"  {feature}: TVD = {tvd:.4f} ({results['overall'][feature]['fairness_quality']})", flush=True)
    
    results['summary'] = {
        'average_tvd': overall_tvd,
        'overall_fairness_quality': "Excellent" if overall_tvd < 0.05 else 
                                   "Good" if overall_tvd < 0.1 else 
                                   "Fair" if overall_tvd < 0.2 else "Poor"
    }
    
    return results


def conditional_demographic_parity(original, augmented, sensitive_feature, target):
    """
    Measures conditional demographic parity by comparing conditional probabilities
    of the target variable given sensitive attributes between original and augmented data.
    
    Parameters:
        original (pd.DataFrame): Original dataset
        augmented (pd.DataFrame): Augmented dataset (original + synthetic)
        sensitive_feature (str): Name of sensitive attribute column
        target (str): Name of target column
        
    Returns:
        tuple: (metric value, results dictionary)
    """
    print(f"### Conditional Demographic Parity for {sensitive_feature} ###", flush=True)
    
    # Calculate conditional probabilities in original data
    orig_cond_probs = {}
    for value in original[sensitive_feature].unique():
        subset = original[original[sensitive_feature] == value]
        if len(subset) > 0:
            orig_cond_probs[value] = subset[target].mean()
    
    # Calculate conditional probabilities in augmented data
    aug_cond_probs = {}
    for value in augmented[sensitive_feature].unique():
        subset = augmented[augmented[sensitive_feature] == value]
        if len(subset) > 0:
            aug_cond_probs[value] = subset[target].mean()
    
    # Compare the conditional probabilities
    all_values = sorted(set(orig_cond_probs.keys()) | set(aug_cond_probs.keys()))
    differences = []
    
    print("Conditional probabilities of positive outcome:", flush=True)
    for value in all_values:
        orig_prob = orig_cond_probs.get(value, np.nan)
        aug_prob = aug_cond_probs.get(value, np.nan)
        
        if not np.isnan(orig_prob) and not np.isnan(aug_prob):
            diff = abs(orig_prob - aug_prob)
            differences.append(diff)
            print(f"  {sensitive_feature}={value}: Original={orig_prob:.4f}, Augmented={aug_prob:.4f}, Diff={diff:.4f}", flush=True)
        else:
            print(f"  {sensitive_feature}={value}: Original={orig_prob:.4f}, Augmented={aug_prob:.4f}, Diff=N/A", flush=True)
    
    # Calculate overall metric
    if differences:
        avg_diff = np.mean(differences)
        max_diff = np.max(differences)
        print(f"Average absolute difference: {avg_diff:.4f}", flush=True)
        print(f"Maximum absolute difference: {max_diff:.4f}", flush=True)
        
        # Visualize
        plt.figure(figsize=(10, 6))
        width = 0.35
        x = np.arange(len(all_values))
        
        orig_probs = [orig_cond_probs.get(val, 0) for val in all_values]
        aug_probs = [aug_cond_probs.get(val, 0) for val in all_values]
        
        plt.bar(x - width/2, orig_probs, width, label='Original', color='skyblue')
        plt.bar(x + width/2, aug_probs, width, label='Augmented', color='salmon')
        
        plt.title(f"Conditional Probabilities by {sensitive_feature}")
        plt.xlabel(sensitive_feature)
        plt.ylabel(f"P({target}=1 | {sensitive_feature})")
        plt.xticks(x, all_values, rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"conditional_parity_{sensitive_feature}.png")
        plt.close()
        
        return avg_diff, {
            'conditional_probabilities': {
                'original': orig_cond_probs,
                'augmented': aug_cond_probs
            },
            'differences': {str(value): diff for value, diff in zip(all_values, differences) if not np.isnan(diff)},
            'average_difference': avg_diff,
            'max_difference': max_diff
        }
    else:
        print("Insufficient data for comparison", flush=True)
        return None, {}


def fairness_assessment(original_df, augmented_df, sensitive_features, target, minority_class=1):
    """
    Performs a comprehensive fairness assessment on augmented data (original + synthetic).
    Specifically focuses on the minority class to ensure synthetic data augmentation doesn't introduce bias.
    
    Parameters:
        original_df (pd.DataFrame): Original unaugmented dataset
        augmented_df (pd.DataFrame): Augmented dataset (original + synthetic data)
        sensitive_features (list): List of sensitive attribute columns
        target (str): Target variable name
        minority_class (any, optional): Value of the minority class in the target column
        
    Returns:
        dict: Comprehensive fairness metrics
    """
    # Filter to extract just the minority class from both datasets
    print(f"Analyzing fairness for class {target}={minority_class}", flush=True)
    original = original_df[original_df[target] == minority_class].copy()
    augmented = augmented_df[augmented_df[target] == minority_class].copy()
    
    print(f"Original minority class samples: {len(original)}", flush=True)
    print(f"Augmented minority class samples: {len(augmented)}", flush=True)
    
    # Calculate how many synthetic samples were added
    synthetic_count = len(augmented) - len(original)
    if synthetic_count <= 0:
        print("Warning: Augmented dataset doesn't contain more samples than original dataset.", flush=True)
    else:
        print(f"Synthetic samples added: {synthetic_count}", flush=True)
    
    # Verify we have data to work with
    if len(original) == 0 or len(augmented) == 0:
        raise ValueError("No data available for the specified minority class. Please check your target and minority_class values.")
    
    results = {}
    
    # 1. Distribution-based fairness (marginal distributions)
    print("\n--- Distribution-Based Fairness Analysis ---", flush=True)
    distribution_results = fairness_bias_check(
        original, augmented, sensitive_features, target, subgroup_analysis=True)
    results['distribution_metrics'] = distribution_results
    
    # 2. Conditional fairness metrics
    print("\n--- Conditional Fairness Metrics ---", flush=True)
    results['conditional_metrics'] = {}
    
    for feature in sensitive_features:
        avg_diff, cond_results = conditional_demographic_parity(
            original, augmented, feature, target)
        results['conditional_metrics'][feature] = cond_results
    
    # 3. Overall fairness summary
    print("\n--- Overall Fairness Summary ---", flush=True)
    
    # Calculate an aggregate fairness score (lower is better)
    distribution_score = distribution_results['summary']['average_tvd']
    
    if 'conditional_metrics' in results and any(results['conditional_metrics'].values()):
        conditional_scores = [results['conditional_metrics'][f].get('average_difference', 1.0) 
                             for f in sensitive_features 
                             if f in results['conditional_metrics'] and results['conditional_metrics'][f]]
        conditional_score = np.mean(conditional_scores) if conditional_scores else 1.0
        overall_score = 0.5 * distribution_score + 0.5 * conditional_score
    else:
        overall_score = distribution_score
    
    # Interpret the score
    if overall_score < 0.05:
        fairness_quality = "Excellent"
    elif overall_score < 0.1:
        fairness_quality = "Good"
    elif overall_score < 0.2:
        fairness_quality = "Fair"
    else:
        fairness_quality = "Poor"
    
    print(f"Overall fairness score: {overall_score:.4f} ({fairness_quality})", flush=True)
    
    # List features that need attention
    if fairness_quality in ["Fair", "Poor"]:
        print("\nFeatures needing attention:", flush=True)
        for feature in sensitive_features:
            feature_quality = distribution_results['overall'][feature]['fairness_quality']
            if feature_quality in ["Fair", "Poor"]:
                tvd = distribution_results['overall'][feature]['total_variation_distance']
                print(f"  - {feature}: TVD = {tvd:.4f} ({feature_quality})", flush=True)
                
                # Find most problematic categories
                if 'categories' in distribution_results['overall'][feature]:
                    cats = distribution_results['overall'][feature]['categories']
                    orig_dist = distribution_results['overall'][feature]['original_distribution']
                    aug_dist = distribution_results['overall'][feature]['augmented_distribution']
                    
                    diffs = [abs(o - s) for o, s in zip(orig_dist, aug_dist)]
                    sorted_indices = np.argsort(diffs)[::-1][:3]  # Top 3 differences
                    
                    print(f"    Largest discrepancies in categories:")
                    for idx in sorted_indices:
                        cat = cats[idx]
                        diff = diffs[idx]
                        if diff > 0.05:  # Only show meaningful differences
                            print(f"      {cat}: diff = {diff:.4f}")
    
    results['overall_fairness'] = {
        'score': overall_score,
        'quality': fairness_quality
    }
    
    return results

def print_simplified_fairness_report(results):
    """
    Prints a simplified fairness report highlighting only the key metrics and insights.
    Handles missing keys gracefully to avoid errors.
    
    Parameters:
        results (dict): The fairness assessment results
    """
    # Print basic statistics (with safe defaults)
    sample_sizes = results.get('sample_sizes', {})
    original_count = sample_sizes.get('original_minority', 0)
    augmented_count = sample_sizes.get('augmented_minority', 0)
    synthetic_count = augmented_count - original_count
    
    print("\n===== FAIRNESS ASSESSMENT SUMMARY =====")
    print(f"Original minority samples: {original_count}")
    print(f"Augmented minority samples: {augmented_count}")
    print(f"Synthetic samples added: {synthetic_count}")
    
    # Overall fairness score (with safe access)
    overall_fairness = results.get('overall_fairness', {})
    fairness_score = overall_fairness.get('score', 0)
    fairness_quality = overall_fairness.get('quality', 'Unknown')
    print(f"\nOVERALL FAIRNESS: {fairness_quality} (score: {fairness_score:.4f})")
    
    # Distribution metrics for sensitive attributes
    distribution_metrics = results.get('distribution_metrics', {})
    overall_metrics = distribution_metrics.get('overall', {})
    
    if overall_metrics:
        print("\nSENSITIVE ATTRIBUTE ANALYSIS:")
        for feature, metrics in overall_metrics.items():
            tvd = metrics.get('total_variation_distance', 0)
            quality = metrics.get('fairness_quality', 'Unknown')
            print(f"  • {feature}: {quality} (TVD: {tvd:.4f})")
    else:
        print("\nSENSITIVE ATTRIBUTE ANALYSIS: No data available")
    
    # Intersectional fairness (highlighting only concerning combinations)
    print("\nINTERSECTIONAL FAIRNESS:")
    concerns = []
    threshold = 0.1  # Consider intersections with TVD > 0.1 as potentially concerning
    
    # Safely check for intersectional data
    intersectional = results.get('intersectional', {})
    if intersectional:
        for combo, metrics in intersectional.items():
            tvd = metrics.get('total_variation_distance', 0)
            if tvd > threshold:
                concerns.append((combo, tvd))
        
        if concerns:
            print("  Potential concerns in attribute combinations:")
            for combo, tvd in sorted(concerns, key=lambda x: x[1], reverse=True):
                print(f"  • {combo.replace('_', ' × ')}: TVD = {tvd:.4f}")
        else:
            print("  No significant concerns in attribute combinations")
    else:
        print("  No intersectional analysis data available")
    
    # Recommendations based on fairness quality
    print("\nRECOMMENDATIONS:")
    if fairness_quality in ["Excellent", "Good"]:
        print("  ✓ The synthetic data preserves fairness across sensitive attributes")
        print("  ✓ Safe to use for augmenting the minority class")
    else:
        print("  ⚠ There are fairness concerns with the synthetic data")
        
        # Detailed recommendations for specific issues
        for feature, metrics in overall_metrics.items():
            if metrics.get('fairness_quality', '') in ["Fair", "Poor"]:
                print(f"  • Address distribution issues in '{feature}' (TVD: {metrics.get('total_variation_distance', 0):.4f})")
        
        # Recommendations for intersectional issues
        if concerns:
            print("  • Review generation process to better preserve intersectional distributions")
