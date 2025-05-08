import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def analyse_attribute(df, attr, target_col, positive_class=1, target_name='TARGET FEATURE'):
    """
    Analyse distribution statistics for a single attribute.
    
    Parameters:
    -----------
    df : pandas DataFrame
        The dataset to analyse
    attr : str
        The attribute to analyse
    target_col : str
        The target column name
    positive_class : int or str, default=1
        The value that represents the positive class in the target column
    target_name : str, default='Target'
        Human-readable name for the target variable (used in plot labels)
        
    Returns:
    --------
    pandas DataFrame with distribution analysis
    """
    # Overall distribution
    overall_dist = df[attr].value_counts(normalize=True).reset_index()
    overall_dist.columns = [attr, 'proportion']
    
    # Calculate counts too
    counts = df[attr].value_counts().reset_index()
    counts.columns = [attr, 'count']
    overall_dist = overall_dist.merge(counts, on=attr)
    
    # Distribution by target class
    positive_class_dist = df[df[target_col] == positive_class][attr].value_counts(normalize=True).reset_index()
    positive_class_dist.columns = [attr, 'positive_class_proportion']
    
    negative_class_dist = df[df[target_col] != positive_class][attr].value_counts(normalize=True).reset_index()
    negative_class_dist.columns = [attr, 'negative_class_proportion']
    
    # Merge distributions
    result = overall_dist.merge(positive_class_dist, on=attr, how='left')
    result = result.merge(negative_class_dist, on=attr, how='left')
    
    # Calculate representation ratio (positive class rate / overall positive class rate)
    overall_positive_rate = (df[target_col] == positive_class).mean()
    
    # Calculate positive class rate for each group
    group_rates = df.groupby(attr)[target_col].apply(
        lambda x: (x == positive_class).mean()
    ).reset_index()
    group_rates.columns = [attr, 'positive_class_rate']
    
    result = result.merge(group_rates, on=attr, how='left')
    
    # Calculate representation ratio
    result['representation_ratio'] = result['positive_class_rate'] / overall_positive_rate
    
    return result, overall_positive_rate

def analyse_demographics(df, sensitive_attrs, target_col, output_dir='demographic_analysis', 
                         positive_class=1, target_name='TARGET FEATURE', 
                         positive_class_label='Positive Class', 
                         negative_class_label='Negative Class',
                         under_representation_threshold=0.8,
                         over_representation_threshold=1.2):
    """
    Perform comprehensive demographic distribution analysis on a dataset.
    
    Parameters:
    -----------
    df : pandas DataFrame
        The dataset to analyse
    sensitive_attrs : list
        List of sensitive attributes to analyse (e.g., ['race', 'gender', 'native_country'])
    target_col : str
        Name of the target column (binary classification target)
    output_dir : str, default='demographic_analysis'
        Directory where output files and visualizations will be saved
    positive_class : int or str, default=1
        The value that represents the positive class in the target column
    target_name : str, default='Target'
        Human-readable name for the target variable (used in plot labels)
    positive_class_label : str, default='Positive Class'
        Label to use for the positive class in visualizations
    negative_class_label : str, default='Negative Class'
        Label to use for the negative class in visualizations
    under_representation_threshold : float, default=0.8
        Threshold below which a group is considered underrepresented
    over_representation_threshold : float, default=1.2
        Threshold above which a group is considered overrepresented
    
    Returns:
    --------
    dict : Dictionary containing analysis results for each attribute
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Store results
    results = {}
    
    # Analyse each attribute and create visualizations
    for attr in sensitive_attrs:
        print(f"\nAnalyzing {attr}...")
        
        # Get distribution analysis
        dist_data, overall_positive_rate = analyse_attribute(
            df, attr, target_col, positive_class, target_name
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
        plt.figure(figsize=(14, 7))
        
        # Create a merged dataframe for seaborn
        plot_data = pd.DataFrame({
            attr: np.concatenate([dist_data[attr], dist_data[attr]]),
            'proportion': np.concatenate([
                dist_data['positive_class_proportion'], 
                dist_data['negative_class_proportion']
            ]),
            'target_class': [positive_class_label] * len(dist_data) + [negative_class_label] * len(dist_data)
        })
        
        sns.barplot(x=attr, y='proportion', hue='target_class', data=plot_data)
        plt.title(f'Distribution of {attr} by {target_name}')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{attr}_by_target.png")
        plt.close()
        
        # 3. Representation ratio
        plt.figure(figsize=(12, 6))
        bars = plt.bar(dist_data[attr], dist_data['representation_ratio'], color='lightgreen')
        
        # Color bars based on ratio value
        for i, ratio in enumerate(dist_data['representation_ratio']):
            if ratio < under_representation_threshold:
                bars[i].set_color('salmon')
            elif ratio > over_representation_threshold:
                bars[i].set_color('royalblue')
        
        plt.axhline(y=1.0, color='red', linestyle='--', label='Parity')
        plt.title(f'Representation Ratio for {attr}\n(Ratio of group positive rate to overall positive rate)')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{attr}_representation_ratio.png")
        plt.close()
        
        # Store results
        results[attr] = dist_data
    
    # Generate a summary report
    with open(f"{output_dir}/demographic_summary.md", 'w') as f:
        f.write(f"# {target_name} Demographic Distribution Analysis\n\n")
        f.write(f"Analysis date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}\n\n")
        f.write(f"Total records analysed: {len(df)}\n")
        f.write(f"Overall positive class rate: {overall_positive_rate:.2%}\n\n")
        
        for attr in sensitive_attrs:
            dist_data = results[attr]
            
            f.write(f"## {attr.title()} Analysis\n\n")
            
            # Most underrepresented groups
            underrep = dist_data[dist_data['representation_ratio'] < under_representation_threshold].sort_values('representation_ratio')
            if not underrep.empty:
                f.write(f"### Underrepresented Groups (ratio < {under_representation_threshold})\n\n")
                f.write("| Group | Representation Ratio | Positive Class Rate | Count | % of Data |\n")
                f.write("|-------|---------------------|---------------------|-------|----------|\n")
                
                for _, row in underrep.iterrows():
                    f.write(f"| {row[attr]} | {row['representation_ratio']:.2f} | {row['positive_class_rate']:.2%} | {row['count']} | {row['proportion']:.2%} |\n")
                
                f.write("\n")
            
            # Most overrepresented groups
            overrep = dist_data[dist_data['representation_ratio'] > over_representation_threshold].sort_values('representation_ratio', ascending=False)
            if not overrep.empty:
                f.write(f"### Overrepresented Groups (ratio > {over_representation_threshold})\n\n")
                f.write("| Group | Representation Ratio | Positive Class Rate | Count | % of Data |\n")
                f.write("|-------|---------------------|---------------------|-------|----------|\n")
                
                for _, row in overrep.iterrows():
                    f.write(f"| {row[attr]} | {row['representation_ratio']:.2f} | {row['positive_class_rate']:.2%} | {row['count']} | {row['proportion']:.2%} |\n")
                
                f.write("\n")
    
    print(f"Demographic analysis complete! Results saved to {output_dir}")
    return results
