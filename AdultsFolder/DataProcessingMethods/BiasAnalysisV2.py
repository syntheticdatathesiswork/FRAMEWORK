import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def analyse_attribute(df, attr, target_col):
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
    
    # Distribution by income class
    high_income_dist = df[df[target_col] == 1][attr].value_counts(normalize=True).reset_index()
    high_income_dist.columns = [attr, 'high_income_proportion']
    
    low_income_dist = df[df[target_col] == 0][attr].value_counts(normalize=True).reset_index()
    low_income_dist.columns = [attr, 'low_income_proportion']
    
    # Merge distributions
    result = overall_dist.merge(high_income_dist, on=attr, how='left')
    result = result.merge(low_income_dist, on=attr, how='left')
    
    # Calculate representation ratio (high income rate / overall high income rate)
    overall_high_income_rate = df[target_col].mean()
    
    # Calculate high income rate for each group
    group_rates = df.groupby(attr)[target_col].mean().reset_index()
    group_rates.columns = [attr, 'high_income_rate']
    
    result = result.merge(group_rates, on=attr, how='left')
    
    # Calculate representation ratio
    result['representation_ratio'] = result['high_income_rate'] / overall_high_income_rate
    
    return result

def analyse_demographics(df, sensitive_attrs, target_col='income', output_dir='demographic_analysis'):
    """
    Perform comprehensive demographic distribution analysis on a dataset.
    
    Parameters:
    -----------
    df : pandas DataFrame
        The dataset to analyse
    sensitive_attrs : list
        List of sensitive attributes to analyse (e.g., ['race', 'gender', 'native_country'])
    target_col : str, default='income'
        Name of the target column (should be binary, 1 for positive class)
    output_dir : str, default='demographic_analysis'
        Directory where output files and visualizations will be saved
    
    Returns:
    --------
    dict : Dictionary containing analysis results for each attribute
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Ensure target is binary (1 for >50K, 0 for <=50K)
    if df[target_col].dtype == 'object':
        df = df.copy()
        df[target_col] = df[target_col].apply(lambda x: 1 if '>50K' in str(x) else 0)
    
    # Store results
    results = {}
    
    # Analyse each attribute and create visualizations
    for attr in sensitive_attrs:
        print(f"\nAnalyzing {attr}...")
        
        # Get distribution analysis
        dist_data = analyse_attribute(df, attr, target_col)
        
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
        
        # 2. Distribution by income class
        plt.figure(figsize=(14, 7))
        
        # Create a merged dataframe for seaborn
        plot_data = pd.DataFrame({
            attr: np.concatenate([dist_data[attr], dist_data[attr]]),
            'proportion': np.concatenate([dist_data['high_income_proportion'], dist_data['low_income_proportion']]),
            'income_class': ['High Income (>50K)'] * len(dist_data) + ['Low Income (≤50K)'] * len(dist_data)
        })
        
        sns.barplot(x=attr, y='proportion', hue='income_class', data=plot_data)
        plt.title(f'Distribution of {attr} by Income Class')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{attr}_by_income.png")
        plt.close()
        
        # 3. Representation ratio
        plt.figure(figsize=(12, 6))
        bars = plt.bar(dist_data[attr], dist_data['representation_ratio'], color='lightgreen')
        
        # Color bars based on ratio value
        for i, ratio in enumerate(dist_data['representation_ratio']):
            if ratio < 0.8:
                bars[i].set_color('salmon')
            elif ratio > 1.2:
                bars[i].set_color('royalblue')
        
        plt.axhline(y=1.0, color='red', linestyle='--', label='Parity')
        plt.title(f'Representation Ratio for {attr}\n(Ratio of group high income rate to overall high income rate)')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{attr}_representation_ratio.png")
        plt.close()
        
        # Store results
        results[attr] = dist_data
    
    # Generate a summary report
    with open(f"{output_dir}/demographic_summary.md", 'w') as f:
        f.write("# Demographic Distribution Analysis\n\n")
        f.write(f"Analysis date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}\n\n")
        f.write(f"Total records analysed: {len(df)}\n")
        f.write(f"Overall positive class rate: {df[target_col].mean():.2%}\n\n")
        
        for attr in sensitive_attrs:
            dist_data = results[attr]
            
            f.write(f"## {attr.title()} Analysis\n\n")
            
            # Most underrepresented groups
            underrep = dist_data[dist_data['representation_ratio'] < 0.8].sort_values('representation_ratio')
            if not underrep.empty:
                f.write("### Underrepresented Groups (ratio < 0.8)\n\n")
                f.write("| Group | Representation Ratio | High Income Rate | Count | % of Data |\n")
                f.write("|-------|---------------------|------------------|-------|----------|\n")
                
                for _, row in underrep.iterrows():
                    f.write(f"| {row[attr]} | {row['representation_ratio']:.2f} | {row['high_income_rate']:.2%} | {row['count']} | {row['proportion']:.2%} |\n")
                
                f.write("\n")
            
            # Most overrepresented groups
            overrep = dist_data[dist_data['representation_ratio'] > 1.2].sort_values('representation_ratio', ascending=False)
            if not overrep.empty:
                f.write("### Overrepresented Groups (ratio > 1.2)\n\n")
                f.write("| Group | Representation Ratio | High Income Rate | Count | % of Data |\n")
                f.write("|-------|---------------------|------------------|-------|----------|\n")
                
                for _, row in overrep.iterrows():
                    f.write(f"| {row[attr]} | {row['representation_ratio']:.2f} | {row['high_income_rate']:.2%} | {row['count']} | {row['proportion']:.2%} |\n")
                
                f.write("\n")
    
    print(f"Demographic analysis complete! Results saved to {output_dir}")
    return results