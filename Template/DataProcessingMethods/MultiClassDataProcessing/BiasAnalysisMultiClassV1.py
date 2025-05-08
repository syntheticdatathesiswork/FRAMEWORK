import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from matplotlib.colors import LinearSegmentedColormap

def analyse_multiclass_attribute(df, attr, target_col, target_classes=None, target_name='FEATURE TARGET'):
    """
    Analyse distribution statistics for a single attribute across multiple target classes.
    
    Parameters:
    -----------
    df : pandas DataFrame
        The dataset to analyse
    attr : str
        The attribute to analyse
    target_col : str
        The target column name
    target_classes : list or None, default=None
        List of classes to analyze. If None, all unique values in target_col will be used.
    target_name : str, default='Target'
        Human-readable name for the target variable (used in plot labels)
        
    Returns:
    --------
    tuple: (DataFrame with distribution analysis, dict of class representation ratios, overall class rates)
    """
    # If target_classes not specified, use all unique values
    if target_classes is None:
        target_classes = df[target_col].unique()
    
    # Overall distribution of the attribute
    overall_dist = df[attr].value_counts(normalize=True).reset_index()
    overall_dist.columns = [attr, 'proportion']
    
    # Calculate counts
    counts = df[attr].value_counts().reset_index()
    counts.columns = [attr, 'count']
    result = overall_dist.merge(counts, on=attr)
    
    # Calculate overall rates for each target class
    overall_class_rates = {}
    for cls in target_classes:
        overall_class_rates[cls] = (df[target_col] == cls).mean()
    
    # Distribution by each target class
    class_distributions = {}
    class_representation = {}
    
    for cls in target_classes:
        # Get distribution of this attribute for this class
        class_dist = df[df[target_col] == cls][attr].value_counts(normalize=True).reset_index()
        class_dist.columns = [attr, f'class_{cls}_proportion']
        result = result.merge(class_dist, on=attr, how='left')
        class_distributions[cls] = class_dist
        
        # Calculate class rate for each group
        group_class_rates = df.groupby(attr)[target_col].apply(
            lambda x: (x == cls).mean()
        ).reset_index()
        group_class_rates.columns = [attr, f'class_{cls}_rate']
        result = result.merge(group_class_rates, on=attr, how='left')
        
        # Calculate representation ratio for this class
        result[f'class_{cls}_representation'] = result[f'class_{cls}_rate'] / overall_class_rates[cls]
        class_representation[cls] = result[[attr, f'class_{cls}_representation']].copy()
    
    # Fill NaN values with 0
    result = result.fillna(0)
        
    return result, class_representation, overall_class_rates

def analyse_multiclass_demographics(df, sensitive_attrs, target_col, target_classes=None, 
                                   output_dir='demographic_analysis', target_name='TARGET FEATURE',
                                   class_labels=None, representation_thresholds=(0.8, 1.2)):
    """
    Perform comprehensive demographic distribution analysis on a dataset with a multi-class target.
    
    Parameters:
    -----------
    df : pandas DataFrame
        The dataset to analyse
    sensitive_attrs : list
        List of sensitive attributes to analyse (e.g., ['race', 'gender', 'native_country'])
    target_col : str
        Name of the target column (multi-class classification target)
    target_classes : list or None, default=None
        List of classes to analyze. If None, all unique values in target_col will be used.
    output_dir : str, default='demographic_analysis'
        Directory where output files and visualizations will be saved
    target_name : str, default='Target'
        Human-readable name for the target variable (used in plot labels)
    class_labels : dict or None, default=None
        Dictionary mapping target class values to human-readable labels
        E.g., {0: 'Rejected', 1: 'Approved', 2: 'Waitlisted'}
    representation_thresholds : tuple, default=(0.8, 1.2)
        Thresholds for under and over representation (low, high)
    
    Returns:
    --------
    dict : Dictionary containing analysis results for each attribute
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # If target_classes not specified, use all unique values
    if target_classes is None:
        target_classes = sorted(df[target_col].unique())
    
    # Create class labels if not provided
    if class_labels is None:
        class_labels = {cls: f'Class {cls}' for cls in target_classes}
    
    # Get under/over representation thresholds
    under_threshold, over_threshold = representation_thresholds
    
    # Store results
    results = {}
    
    # Analyze each attribute and create visualizations
    for attr in sensitive_attrs:
        print(f"\nAnalyzing {attr}...")
        
        # Get distribution analysis
        dist_data, class_repr, class_rates = analyse_multiclass_attribute(
            df, attr, target_col, target_classes, target_name
        )
        
        # Sort by count for better visualization
        dist_data = dist_data.sort_values('count', ascending=False)
        
        # Save numerical analysis to CSV
        dist_data.to_csv(f"{output_dir}/{attr}_distribution.csv", index=False)
        
        # Create plots
        
        # 1. Overall distribution
        plt.figure(figsize=(12, 6))
        sns.barplot(x=attr, y='proportion', data=dist_data)
        plt.title(f'Distribution of {attr}')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{attr}_distribution.png")
        plt.close()
        
        # 2. Distribution by target class
        plt.figure(figsize=(14, 8))
        
        # Create a merged dataframe for seaborn
        plot_data_pieces = []
        
        for cls in target_classes:
            class_data = pd.DataFrame({
                attr: dist_data[attr],
                'proportion': dist_data[f'class_{cls}_proportion'],
                'target_class': [class_labels[cls]] * len(dist_data)
            })
            plot_data_pieces.append(class_data)
        
        plot_data = pd.concat(plot_data_pieces, ignore_index=True)
        
        sns.barplot(x=attr, y='proportion', hue='target_class', data=plot_data)
        plt.title(f'Distribution of {attr} by {target_name}')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{attr}_by_target.png")
        plt.close()
        
        # 3. Class rates by attribute value
        plt.figure(figsize=(14, 8))
        
        # Create data for rate plot
        rate_data_pieces = []
        
        for cls in target_classes:
            rate_data = pd.DataFrame({
                attr: dist_data[attr],
                'rate': dist_data[f'class_{cls}_rate'],
                'target_class': [class_labels[cls]] * len(dist_data)
            })
            rate_data_pieces.append(rate_data)
        
        rate_data = pd.concat(rate_data_pieces, ignore_index=True)
        
        sns.barplot(x=attr, y='rate', hue='target_class', data=rate_data)
        plt.title(f'{target_name} Rates by {attr}')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{attr}_class_rates.png")
        plt.close()
        
        # 4. Representation ratios (one plot for each class)
        for cls in target_classes:
            plt.figure(figsize=(12, 6))
            ratio_col = f'class_{cls}_representation'
            bars = plt.bar(dist_data[attr], dist_data[ratio_col], color='lightgreen')
            
            # Color bars based on ratio value
            for i, ratio in enumerate(dist_data[ratio_col]):
                if ratio < under_threshold:
                    bars[i].set_color('salmon')
                elif ratio > over_threshold:
                    bars[i].set_color('royalblue')
            
            plt.axhline(y=1.0, color='red', linestyle='--', label='Parity')
            plt.title(f'Representation Ratio for {class_labels[cls]} by {attr}\n(Ratio of group rate to overall rate)')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(f"{output_dir}/{attr}_{class_labels[cls].replace(' ', '_')}_representation.png")
            plt.close()
        
        # 5. Heatmap of representation ratios
        plt.figure(figsize=(14, len(dist_data) * 0.4 + 2))
        
        # Create heatmap data
        heatmap_data = []
        for cls in target_classes:
            ratio_col = f'class_{cls}_representation'
            heatmap_data.append(dist_data[ratio_col].values)
        
        # Custom colormap (red for underrepresentation, blue for overrepresentation)
        cmap = LinearSegmentedColormap.from_list(
            'custom_diverging',
            ['#FF5555', '#FFFFFF', '#5555FF'],
            N=256
        )
        
        # Create heatmap
        hm = sns.heatmap(
            np.array(heatmap_data),
            annot=True,
            fmt='.2f',
            cmap=cmap,
            center=1.0,
            vmin=min(0.5, under_threshold),
            vmax=max(1.5, over_threshold),
            xticklabels=dist_data[attr],
            yticklabels=[class_labels[cls] for cls in target_classes],
            cbar_kws={'label': 'Representation Ratio'}
        )
        
        plt.title(f'Representation Ratios by {attr} and {target_name}')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{attr}_representation_heatmap.png")
        plt.close()
        
        # Store results
        results[attr] = dist_data
    
    # Generate a summary report
    with open(f"{output_dir}/demographic_summary.md", 'w') as f:
        f.write(f"# {target_name} Multi-Class Demographic Distribution Analysis\n\n")
        f.write(f"Analysis date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}\n\n")
        f.write(f"Total records analysed: {len(df)}\n\n")
        
        # Class distribution table
        f.write(f"## Overall {target_name} Distribution\n\n")
        f.write("| Class | Label | Count | Percentage |\n")
        f.write("|-------|-------|-------|------------|\n")
        
        for cls in target_classes:
            count = (df[target_col] == cls).sum()
            percentage = count / len(df) * 100
            f.write(f"| {cls} | {class_labels[cls]} | {count} | {percentage:.2f}% |\n")
        f.write("\n")
        
        # Individual attribute analysis
        for attr in sensitive_attrs:
            dist_data = results[attr]
            
            f.write(f"## {attr.title()} Analysis\n\n")
            
            for cls in target_classes:
                cls_label = class_labels[cls]
                ratio_col = f'class_{cls}_representation'
                
                f.write(f"### {cls_label} Representation in {attr}\n\n")
                
                # Most underrepresented groups
                underrep = dist_data[dist_data[ratio_col] < under_threshold].sort_values(ratio_col)
                if not underrep.empty:
                    f.write(f"#### Underrepresented Groups (ratio < {under_threshold})\n\n")
                    f.write(f"| {attr} | Representation Ratio | {cls_label} Rate | Count | % of Data |\n")
                    f.write("|-----|---------------------|------------|-------|----------|\n")
                    
                    for _, row in underrep.iterrows():
                        f.write(f"| {row[attr]} | {row[ratio_col]:.2f} | {row[f'class_{cls}_rate']:.2%} | {row['count']} | {row['proportion']:.2%} |\n")
                    
                    f.write("\n")
                
                # Most overrepresented groups
                overrep = dist_data[dist_data[ratio_col] > over_threshold].sort_values(ratio_col, ascending=False)
                if not overrep.empty:
                    f.write(f"#### Overrepresented Groups (ratio > {over_threshold})\n\n")
                    f.write(f"| {attr} | Representation Ratio | {cls_label} Rate | Count | % of Data |\n")
                    f.write("|-----|---------------------|------------|-------|----------|\n")
                    
                    for _, row in overrep.iterrows():
                        f.write(f"| {row[attr]} | {row[ratio_col]:.2f} | {row[f'class_{cls}_rate']:.2%} | {row['count']} | {row['proportion']:.2%} |\n")
                    
                    f.write("\n")
    
    print(f"Multi-class demographic analysis complete! Results saved to {output_dir}")
    return results
