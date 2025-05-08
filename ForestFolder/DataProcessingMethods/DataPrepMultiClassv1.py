import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

def remove_outliers(df_input, features, factor=1.5, class_feature="target"):
    """
    Removes outliers for numeric features within each class.
    
    Parameters:
    df_input (DataFrame): Input dataframe
    features (list): List of numeric features to check for outliers
    factor (float): Multiplier for IQR to determine outlier bounds
    class_feature (str): The feature used to group the data into classes
    """
    df_out = df_input.copy()
    # Get unique classes
    classes = df_out[class_feature].unique()
    
    # Create a mask to track which rows to keep
    keep_mask = pd.Series(True, index=df_out.index)
    
    for cls in classes:
        class_mask = df_out[class_feature] == cls
        
        for feature in features:
            if feature == "hours_per_week":
                lower_bound = df_out.loc[class_mask, feature].quantile(0.01)
                upper_bound = df_out.loc[class_mask, feature].quantile(0.99)
            else:
                Q1 = df_out.loc[class_mask, feature].quantile(0.15)
                Q3 = df_out.loc[class_mask, feature].quantile(0.85)
                IQR = Q3 - Q1
                lower_bound = Q1 - factor * IQR
                upper_bound = Q3 + factor * IQR
            
            # Count before removing outliers
            before_count = class_mask.sum()
            
            # Update the keep mask for this feature and class
            feature_mask = (df_out.loc[class_mask, feature] >= lower_bound) & (df_out.loc[class_mask, feature] <= upper_bound)
            class_indices = df_out[class_mask].index
            keep_indices = class_indices[feature_mask]
            
            # Update the main mask
            keep_mask.loc[class_mask] = keep_mask.loc[class_mask] & feature_mask.values
            
            # Calculate how many were removed in this class for this feature
            after_count = feature_mask.sum()
            removed_count = before_count - after_count
            
            print(f"{class_feature}={cls}, {feature}: Removed {removed_count} outliers")
    
    # Apply the final mask to keep only non-outlier rows
    final_df = df_out[keep_mask]
    print(f"Total rows before: {len(df_out)}, after: {len(final_df)}, removed: {len(df_out) - len(final_df)}")
    
    return final_df

def prepare_data_pipeline(df, list_features_specialise_outliers, numeric_features, dist_features, target):
    """
    Performs full data preparation on the input DataFrame for multi-class classification, including:
      1. Cleaning missing values.
      2. Encoding categorical features.
      3. Outlier analysis and removal.
      4. Generating diagnostic plots (class distribution, feature distributions by class, correlation, 
         pairplot, feature importance, cluster analysis with KMeans, DBSCAN, Hierarchical clustering,
         and dimensionality reduction with PCA and t-SNE).
      5. Statistical tests (ANOVA, Chi-squared).
    
    Parameters:
      df (pd.DataFrame): Input DataFrame.
      list_features_specialise_outliers (list): Features to check for outliers.
      numeric_features (list): Numeric features for analysis.
      dist_features (list): Features to show distributions.
      target (str): Name of the target column.
    
    Returns:
      df_no_outliers (pd.DataFrame): The final cleaned DataFrame.
    """
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans, DBSCAN
    from sklearn.manifold import TSNE
    from sklearn.ensemble import RandomForestClassifier
    from scipy import stats
    from scipy.cluster.hierarchy import dendrogram, linkage
    from scipy.stats import ks_2samp, chi2_contingency
    
    # ---------------
    # Helper Functions for Adaptive Plotting
    # ---------------
    def create_adaptive_grid(features, df_data, plot_function, title_prefix="Distribution of", 
                            max_cols=4, figsize_unit=(5, 4)):
        """
        Creates an adaptive grid of subplots based on number of features
        
        Parameters:
        features (list): List of features to plot
        df_data (DataFrame): Data to plot
        plot_function (function): Function to call for each subplot
        title_prefix (str): Prefix for subplot titles
        max_cols (int): Maximum number of columns
        figsize_unit (tuple): Base size for each subplot
        
        Returns:
        fig: Matplotlib figure
        """
        n_features = len(features)
        n_cols = min(max_cols, n_features)
        n_rows = (n_features + n_cols - 1) // n_cols  # Ceiling division
        
        fig = plt.figure(figsize=(figsize_unit[0]*n_cols, figsize_unit[1]*n_rows))
        
        for i, feature in enumerate(features):
            ax = plt.subplot(n_rows, n_cols, i+1)
            plot_function(df_data, feature, ax)
            ax.set_title(f"{title_prefix} {feature}")
        
        plt.tight_layout()
        return fig
    
    def histplot_func(data, feature, ax):
        """Function to create a histogram with KDE"""
        sns.histplot(data[feature], kde=True, bins=20, ax=ax)
    
    def class_kdeplot_func(data, feature, ax, target_col, class_cmap):
        """Function to create KDE plots for each class"""
        for class_val in sorted(data[target_col].unique()):
            sns.kdeplot(
                data[data[target_col] == class_val][feature],
                ax=ax,
                label=f'Class {class_val}',
                color=class_cmap(class_val % 10)  # Use modulo for larger class counts
            )
        ax.legend()
    
    # ----------------------------
    # 1. Missing Value Cleaning
    # ----------------------------
    print("Rows before cleaning empty values:", len(df))
    df_clean = df.dropna()
    print("Rows after cleaning empty values:", len(df_clean))
    print("")
    
    print("\nDataFrame head:")
    print(df_clean.head())
    
    # ----------------------------
    # 2. Encode Categorical Features
    # ----------------------------
    dataset_values = list(df.columns)
    categorical_cols = df_clean.select_dtypes(include=['object']).columns.tolist()
    print("Categorical columns found:", categorical_cols)

    # Encode target column with LabelEncoder for multi-class support
    if target in categorical_cols:
        # Create a specific encoder for the target column
        target_encoder = LabelEncoder()
        df_clean[target] = target_encoder.fit_transform(df_clean[target].astype(str))
        print(f"\nTarget class mapping: {dict(zip(target_encoder.classes_, target_encoder.transform(target_encoder.classes_)))}")
        # Remove target from categorical columns for feature encoding
        categorical_cols.remove(target)
    
    encoder_dict = {}  # To store label mappings.
    for col in categorical_cols:
        le = LabelEncoder()
        df_clean[col] = le.fit_transform(df_clean[col].astype(str))
        encoder_dict[col] = dict(zip(le.classes_, le.transform(le.classes_)))
    
    print("\nEncoder mappings:")
    for col, mapping in encoder_dict.items():
        print(f"{col}: {mapping}")
    
    print("\nEncoded DataFrame head:")
    print(df_clean.head())

    # ----------------------------
    # 3. Outlier Analysis and Removal
    # ----------------------------
    print("\nOutlier Cleaning")
    df_no_outliers = remove_outliers(df_clean, list_features_specialise_outliers, class_feature=target)
    print("Rows after cleaning outliers:", len(df_no_outliers))
    print("")
    print("\nEncoded DataFrame head (after outlier removal):")
    print(df_no_outliers.head())
    
    # ----------------------------
    # 4. Class Distribution Analysis
    # ----------------------------
    plt.figure(figsize=(10, 6))
    class_counts = df_no_outliers[target].value_counts().sort_index()

    # Create a custom color palette matching the density plots
    colors = plt.cm.tab10(np.linspace(0, 1, len(class_counts)))

    ax = sns.barplot(x=class_counts.index, y=class_counts.values, palette=colors)
    plt.title(f"Class Distribution: {target}")
    plt.xlabel(f"{target} Class")
    plt.ylabel("Count")

    # Add value labels on top of each bar
    for i, count in enumerate(class_counts.values):
        ax.text(i, count, str(count), ha='center', va='bottom')

    plt.savefig(f"class_distribution_{target}.png", bbox_inches="tight")
    plt.close()

    # Get the number of unique classes
    n_classes = df_no_outliers[target].nunique()
    # Use the same colormap for multi-class visualization to maintain consistency
    class_cmap = plt.cm.tab10
    
    # ----------------------------
    # 5. Feature Distribution Analysis for Multi-Class
    # ----------------------------
    # Use all numeric features for distribution analysis
    all_numeric_features = df_no_outliers.select_dtypes(include=['float64', 'int64']).columns.tolist()
    if target in all_numeric_features:
        all_numeric_features.remove(target)
    
    # 5.1 Combined distribution - with adaptive grid
    fig = create_adaptive_grid(all_numeric_features, df_no_outliers, histplot_func)
    plt.savefig("feature_distributions_combined.png", bbox_inches="tight")
    plt.close()
    
    # 5.2 Per-class distribution with KDE plots - one figure per feature
    for feature in all_numeric_features:
        plt.figure(figsize=(12, 8))
        for class_val in sorted(df_no_outliers[target].unique()):
            sns.kdeplot(
                df_no_outliers[df_no_outliers[target] == class_val][feature],
                label=f'Class {class_val}',
                color=class_cmap(class_val % 10)  # Use modulo for larger class counts
            )
        plt.title(f"Distribution of {feature} by Class")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"feature_distribution_{feature}_by_class.png", bbox_inches="tight")
        plt.close()
    
    # 5.3 Multi-class feature distribution grid (improved visualization)
    # For up to 5 important features, shows distribution across all classes
    important_features = dist_features.copy()
    if len(important_features) > 5:
        important_features = important_features[:5]  # Limit to top 5 for clarity
        
    fig, axes = plt.subplots(len(important_features), 1, figsize=(12, 4*len(important_features)))
    if len(important_features) == 1:
        axes = [axes]  # Ensure axes is indexable when there's only one feature
        
    for i, feature in enumerate(important_features):
        for class_val in sorted(df_no_outliers[target].unique()):
            sns.kdeplot(
                df_no_outliers[df_no_outliers[target] == class_val][feature],
                ax=axes[i],
                label=f'Class {class_val}',
                color=class_cmap(class_val % 10)
            )
        axes[i].set_title(f"Distribution of {feature} by Class")
        axes[i].legend()
    
    plt.tight_layout()
    plt.savefig("multi_class_feature_distributions.png", bbox_inches="tight")
    plt.close()

    # ----------------------------
    # 6. Correlation Analysis
    # ----------------------------
    plt.figure(figsize=(12, 10))
    corr_matrix = df_no_outliers.corr()
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Correlation Matrix of Dataset (Numeric Features)")
    plt.savefig("correlation_matrix.png", bbox_inches="tight")
    plt.close()

    # ----------------------------
    # 7. Pairwise Feature Relationships (cleaner version)
    # ----------------------------
    # Select most important numeric features to avoid overcrowding
    important_features = dist_features.copy()
    if len(important_features) > 6:
        # Only use up to 6 features for pairplot to keep it clean
        important_features = important_features[:6]

    print(f"\nCreating pairplot with {len(important_features)} features")
    pairplot_df = df_no_outliers[important_features + [target]].copy()

    # Create a proper palette dictionary for seaborn
    # This converts the colormap to a list of colors that seaborn can use
    n_classes = df_no_outliers[target].nunique()
    palette_colors = [plt.cm.tab10(i % 10) for i in range(n_classes)]
    class_palette = {i: palette_colors[i % len(palette_colors)] for i in sorted(df_no_outliers[target].unique())}

    try:
        g = sns.pairplot(pairplot_df, vars=important_features, hue=target,
                         palette=class_palette, plot_kws={'alpha': 0.6}, diag_kind="kde")
        g.fig.suptitle("Pairwise Feature Relationships", y=1.02)
        plt.savefig("pairplot_features_clean.png", bbox_inches="tight")
        plt.close()
    except Exception as e:
        print(f"Warning: Could not create pairplot due to error: {e}")
        print("Skipping pairplot generation and continuing with the analysis.")

    # ----------------------------
    # 8. Feature Importance Analysis (all features)
    # ----------------------------
    # Use all features except target
    X_all_features = df_no_outliers.drop(columns=[target])
    y = df_no_outliers[target]
    
    # For feature importance, we can use all features (categorical + numeric)
    print("\nTraining Random Forest for feature importance...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_all_features, y)
    importances = rf.feature_importances_
    
    # Sort features by importance
    feature_names = X_all_features.columns
    indices = np.argsort(importances)[::-1]
    
    # Plot feature importance - handle large feature sets with rotation
    plt.figure(figsize=(12, 8))
    plt.title("Feature Importances from RandomForest")
    
    # If we have too many features, only show top 20
    if len(indices) > 20:
        indices = indices[:20]
        plt.title("Top 20 Feature Importances from RandomForest")
    
    sns.barplot(x=[feature_names[i] for i in indices], y=importances[indices])
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig("feature_importances_all.png", bbox_inches="tight")
    plt.close()

    # ----------------------------
    # 9. Cluster Analysis - KMeans with Multi-Class
    # ----------------------------
    # Scale data for clustering (important!)
    print("\nPerforming clustering analysis...")
    X_numeric = df_no_outliers[numeric_features]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_numeric)
    
    # PCA for visualization
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    
    # KMeans clustering - now set to the number of classes for comparison
    kmeans = KMeans(n_clusters=n_classes, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    
    # Plot KMeans clusters
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, 
                        cmap="tab10", alpha=0.6, s=50)
    plt.colorbar(scatter, label="Cluster")
    plt.title(f"KMeans Clustering ({n_classes} clusters) on PCA-reduced Data")
    plt.xlabel(f"Principal Component 1 ({pca.explained_variance_ratio_[0]:.2%} variance)")
    plt.ylabel(f"Principal Component 2 ({pca.explained_variance_ratio_[1]:.2%} variance)")
    plt.savefig("kmeans_clusters.png", bbox_inches="tight")
    plt.close()
    
    # Plot true class labels on PCA
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, 
                        cmap="tab10", alpha=0.6, s=50)
    plt.colorbar(scatter, label=target)
    plt.title(f"PCA-reduced Data with True {target} Labels")
    plt.xlabel(f"Principal Component 1 ({pca.explained_variance_ratio_[0]:.2%} variance)")
    plt.ylabel(f"Principal Component 2 ({pca.explained_variance_ratio_[1]:.2%} variance)")
    plt.savefig("pca_true_labels.png", bbox_inches="tight")
    plt.close()
    
    # Compare clusters with true labels
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap="tab10", alpha=0.6, s=50)
    plt.title("KMeans Clusters")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    
    plt.subplot(1, 2, 2)
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap="tab10", alpha=0.6, s=50)
    plt.title(f"True {target} Classes")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    
    plt.tight_layout()
    plt.savefig("clusters_vs_true_labels.png", bbox_inches="tight")
    plt.close()

    # Sample the dataset if it's too large for t-SNE and DBSCAN
    print(f"\nDataset size: {len(df_no_outliers)} rows")
    max_sample_size = 10000  # t-SNE can be slow with large datasets
    if len(df_no_outliers) > max_sample_size:
        print(f"Sampling {max_sample_size} rows for t-SNE and DBSCAN analysis...")
        sample_indices = np.random.choice(len(df_no_outliers), max_sample_size, replace=False)
        X_scaled_sample = X_scaled[sample_indices]
        y_sample = y.iloc[sample_indices]
    else:
        X_scaled_sample = X_scaled
        y_sample = y

    # ----------------------------
    # 10. t-SNE Visualization for Multi-Class
    # ----------------------------
    print("\nRunning t-SNE visualization...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    X_tsne = tsne.fit_transform(X_scaled_sample)
    
    # Plot t-SNE with true labels
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_sample, 
                        cmap="tab10", alpha=0.6, s=50)
    plt.colorbar(scatter, label=target)
    plt.title(f"t-SNE Visualization with True {target} Classes")
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.savefig("tsne_true_labels.png", bbox_inches="tight")
    plt.close()

    # ----------------------------
    # 11. DBSCAN Clustering
    # ----------------------------
    print("\nRunning DBSCAN clustering...")
    # DBSCAN is sensitive to the eps parameter
    # Use a heuristic to determine eps
    from sklearn.neighbors import NearestNeighbors
    k = min(len(X_scaled_sample) - 1, 5)  # number of neighbors to consider
    nn = NearestNeighbors(n_neighbors=k)
    nn.fit(X_scaled_sample)
    distances, indices = nn.kneighbors(X_scaled_sample)
    
    # Use mean of the k-th distances as eps
    eps = np.mean(distances[:, -1]) * 0.5
    
    # Run DBSCAN
    dbscan = DBSCAN(eps=eps, min_samples=5)
    dbscan_clusters = dbscan.fit_predict(X_scaled_sample)
    
    # Get PCA for visualization of sampled data
    X_pca_sample = pca.transform(X_scaled_sample)
    
    # Plot DBSCAN clusters
    plt.figure(figsize=(10, 8))
    # Handle noise points (-1 label) with a special color
    unique_clusters = np.unique(dbscan_clusters)
    if -1 in unique_clusters:
        mask_noise = dbscan_clusters == -1
        # Plot noise points first with grey color
        plt.scatter(X_pca_sample[mask_noise, 0], X_pca_sample[mask_noise, 1], 
                  c='grey', marker='x', alpha=0.6, s=50, label='Noise')
        # Then plot cluster points
        for cluster in unique_clusters[unique_clusters != -1]:
            mask = dbscan_clusters == cluster
            plt.scatter(X_pca_sample[mask, 0], X_pca_sample[mask, 1], 
                      alpha=0.6, s=50, label=f'Cluster {cluster}')
        plt.legend()
    else:
        # If no noise, use standard coloring
        scatter = plt.scatter(X_pca_sample[:, 0], X_pca_sample[:, 1], c=dbscan_clusters, 
                            cmap="tab10", alpha=0.6, s=50)
        plt.colorbar(scatter, label="Cluster")
    
    plt.title(f"DBSCAN Clustering on PCA-reduced Data (eps={eps:.3f})")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.savefig("dbscan_clusters.png", bbox_inches="tight")
    plt.close()

    # ----------------------------
    # 12. Hierarchical Clustering for Multi-Class
    # ----------------------------
    print("\nRunning hierarchical clustering on sample...")
    # Sample subset if dataset is large (for performance)
    hier_sample_size = min(1000, len(X_scaled_sample))
    indices = np.random.choice(len(X_scaled_sample), hier_sample_size, replace=False)
    X_hier_sample = X_scaled_sample[indices]
    y_hier_sample = y_sample.iloc[indices] if hasattr(y_sample, 'iloc') else y_sample[indices]
    
    # Compute linkage matrix
    Z = linkage(X_hier_sample, method='ward')
    
    # Plot dendrogram
    plt.figure(figsize=(12, 8))
    plt.title("Hierarchical Clustering Dendrogram")
    dendrogram(Z, truncate_mode='level', p=5)
    plt.xlabel("Sample index")
    plt.ylabel("Distance")
    plt.savefig("hierarchical_clustering.png", bbox_inches="tight")
    plt.close()
    
    # Cut the dendrogram to get n_classes clusters (for comparison with true labels)
    from scipy.cluster.hierarchy import fcluster
    hier_clusters = fcluster(Z, t=n_classes, criterion='maxclust')
    
    # Plot hierarchical clusters on PCA projection
    X_pca_hier = pca.transform(X_hier_sample)
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_pca_hier[:, 0], X_pca_hier[:, 1], 
                        c=hier_clusters, cmap="tab10", alpha=0.6, s=50)
    plt.colorbar(scatter, label="Cluster")
    plt.title(f"Hierarchical Clustering ({n_classes} clusters) on PCA-reduced Data")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.savefig("hierarchical_clusters_pca.png", bbox_inches="tight")
    plt.close()

    # ----------------------------
    # 13. Statistical Tests for Multi-Class
    # ----------------------------
    print("\nRunning statistical tests...")
    # 13.1 ANOVA Test for Multi-Class
    from scipy import stats

    anova_results = {}
    for feature in numeric_features:
        # Create groups by class
        groups = [df_no_outliers[df_no_outliers[target] == cls][feature].values for cls in sorted(df_no_outliers[target].unique())]
        # Filter out empty groups
        groups = [g for g in groups if len(g) > 0]
        
        # Perform ANOVA test
        if len(groups) >= 2:  # Need at least 2 groups for ANOVA
            f_stat, p_value = stats.f_oneway(*groups)
            anova_results[feature] = {
                'f_statistic': f_stat,
                'p_value': p_value,
                'different_means': p_value < 0.05
            }
    
    # Plot ANOVA test results
    if anova_results:
        plt.figure(figsize=(12, 6))
        features = list(anova_results.keys())
        p_values = [anova_results[f]['p_value'] for f in features]
        f_statistics = [anova_results[f]['f_statistic'] for f in features]
        
        plt.subplot(1, 2, 1)
        bars = plt.bar(features, p_values)
        plt.axhline(y=0.05, color='r', linestyle='--', label='Significance level (0.05)')
        plt.xticks(rotation=45, ha='right')
        plt.title('ANOVA Test: p-values')
        plt.ylabel('p-value')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.bar(features, f_statistics)
        plt.xticks(rotation=45, ha='right')
        plt.title('ANOVA Test: F-Statistics')
        plt.ylabel('F Statistic')
        
        plt.tight_layout()
        plt.savefig("anova_test_results.png", bbox_inches="tight")
        plt.close()
        
        # Print ANOVA test results
        print("\nANOVA Test Results:")
        for feature, result in anova_results.items():
            conclusion = "At least one class has a different mean" if result['different_means'] else "No significant difference between class means"
            print(f"{feature}: F={result['f_statistic']:.4f}, p-value={result['p_value']:.4f} => {conclusion}")
    
    # 13.2 Chi-squared Tests for Categorical Features remain the same
    chi2_results = {}
    categorical_cols = df_no_outliers.select_dtypes(include=['object']).columns.tolist()
    if not categorical_cols:
        # Check for binary features that might be categorical in nature
        binary_cols = []
        for col in df_no_outliers.columns:
            if col != target and col not in numeric_features:
                unique_vals = df_no_outliers[col].nunique()
                if unique_vals <= 10:  # Assume columns with few unique values are categorical
                    binary_cols.append(col)
        
        if binary_cols:
            print(f"\nFound {len(binary_cols)} binary/categorical columns for Chi-squared tests")
            categorical_cols = binary_cols
    
    for feature in categorical_cols:
        if feature == target:
            continue
            
        # Create contingency table
        contingency = pd.crosstab(df_no_outliers[feature], df_no_outliers[target])
        
        # Perform Chi-squared test
        chi2, p, dof, expected = chi2_contingency(contingency)
        chi2_results[feature] = {
            'chi2': chi2,
            'p_value': p,
            'dof': dof,
            'dependent': p < 0.05
        }
    
    # Plot Chi-squared test results
    if chi2_results:  # Only if there are categorical features
        plt.figure(figsize=(12, 6))
        features = list(chi2_results.keys())
        p_values = [chi2_results[f]['p_value'] for f in features]
        chi2_values = [chi2_results[f]['chi2'] for f in features]
        
        # If too many features, limit to top 20 by chi-square value
        if len(features) > 20:
            # Sort by chi-square value
            sorted_indices = np.argsort(chi2_values)[::-1][:20]
            features = [features[i] for i in sorted_indices]
            p_values = [p_values[i] for i in sorted_indices]
            chi2_values = [chi2_values[i] for i in sorted_indices]
            plt.title('Chi-squared Test Results (Top 20 Features)')
        
        plt.subplot(1, 2, 1)
        bars = plt.bar(features, p_values)
        plt.axhline(y=0.05, color='r', linestyle='--', label='Significance level (0.05)')
        plt.xticks(rotation=45, ha='right')
        plt.title('Chi-squared Test: p-values')
        plt.ylabel('p-value')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.bar(features, chi2_values)
        plt.xticks(rotation=45, ha='right')
        plt.title('Chi-squared Test: Statistics')
        plt.ylabel('Chi² Statistic')
        
        plt.tight_layout()
        plt.savefig("chi2_test_results.png", bbox_inches="tight")
        plt.close()
        
        # Print Chi-squared test results
        print("\nChi-squared Test Results:")
        for feature, result in chi2_results.items():
            conclusion = "Dependent on target" if result['dependent'] else "Independent of target"
            print(f"{feature}: Chi²={result['chi2']:.4f}, p-value={result['p_value']:.4f}, dof={result['dof']} => {conclusion}")

    # ----------------------------
    # 14. Class Separation Analysis
    # ----------------------------
    # Create a confusion matrix-like plot to show how well the clusters separate classes
    from sklearn.metrics import confusion_matrix
    
    # Get confusion matrix between KMeans clusters and true classes
    conf_matrix = confusion_matrix(y, clusters)
    
    # Normalize by row (true class)
    cm_norm = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_norm, annot=True, cmap="Blues", fmt=".2f",
              xticklabels=[f"Cluster {i}" for i in range(n_classes)],
              yticklabels=[f"Class {i}" for i in range(n_classes)])
    plt.ylabel('True Class')
    plt.xlabel('KMeans Cluster')
    plt.title('Normalized Confusion Matrix: True Classes vs KMeans Clusters')
    plt.tight_layout()
    plt.savefig("class_cluster_confusion_matrix.png", bbox_inches="tight")
    plt.close()

    print("\nData preparation and analysis complete! All visualizations saved.")
    return df_no_outliers
    