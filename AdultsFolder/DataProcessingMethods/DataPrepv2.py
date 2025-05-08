import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def remove_outliers(df_input, features, factor=1.5, class_feature="income"):
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

def prepare_data_pipeline_for_smotenc(df, list_features_specialise_outliers, numeric_features, dist_features, target):
    """
    Performs full data preparation on the input DataFrame, including:
      1. Cleaning missing values.
      2. KEEPING categorical features as-is (for SMOTENC).
      3. Outlier analysis and removal.
      4. Generating diagnostic plots (class distribution, feature distributions by class, correlation, 
         pairplot, feature importance, cluster analysis with KMeans, DBSCAN, Hierarchical clustering,
         and dimensionality reduction with PCA and t-SNE).
      5. Statistical tests (Kolmogorov-Smirnov, Chi-squared).
    
    Returns:
      df_no_outliers (pd.DataFrame): The final cleaned DataFrame with preserved categorical features.
    """
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans, DBSCAN
    from sklearn.manifold import TSNE
    from sklearn.ensemble import RandomForestClassifier
    from scipy import stats
    from scipy.cluster.hierarchy import dendrogram, linkage
    from scipy.stats import ks_2samp, chi2_contingency
    
    # ----------------------------
    # 1. Missing Value Cleaning
    # ----------------------------
    print("Rows before cleaning empty values:", len(df))
    df_clean = df.dropna()
    print("Rows after cleaning empty values:", len(df_clean))
    df_clean = df_clean.drop_duplicates(keep='first')
    print("Rows after dropping duplicates:", len(df_clean))
    print("")
    
    print("\nDataFrame head:")
    print(df_clean.head())
    
    # ----------------------------
    # 2. Identify Categorical and Numeric Features
    # ----------------------------
    dataset_values = list(df.columns)
    categorical_cols = df_clean.select_dtypes(include=['object']).columns.tolist()
    print("Categorical columns found:", categorical_cols)
    
    # Handle target column - SMOTENC expects binary target (0,1) but we'll keep categorical features as-is
    # Only standardize the target values, not encoding the categorical features
    if target in df_clean.columns:
        # Assuming target has values like '<=50K' and '>50K'
        # Just standardize to 0 and 1 for the target
        df_clean[target] = df_clean[target].str.strip().map({'<=50K': 0, '>50K': 1})
    
    # Store categorical feature columns for SMOTENC (excluding target)
    if target in categorical_cols:
        categorical_cols.remove(target)
    
    # Print categorical values (important for SMOTENC)
    print("\nCategorical feature values (for SMOTENC):")
    for col in categorical_cols:
        unique_values = df_clean[col].unique()
        print(f"{col}: {unique_values[:5]}{'...' if len(unique_values) > 5 else ''}")
    
    # ----------------------------
    # 3. Outlier Analysis and Removal
    # ----------------------------
    print("\nOutlier Cleaning")
    df_no_outliers = remove_outliers(df_clean, list_features_specialise_outliers)
    print("Rows after cleaning outliers:", len(df_no_outliers))
    print("")
    print("\nDataFrame head (after outlier removal):")
    print(df_no_outliers.head())

    # ----------------------------
    # 4. Class Distribution Analysis
    # ----------------------------
    plt.figure(figsize=(6,4))
    plt.grid(axis='y', alpha=0.3)

    # Get the counts for each class
    counts = df_no_outliers[target].value_counts().sort_index()
    total = len(df_no_outliers)

    # Create a bar plot with custom colors
    ax = plt.bar([0, 1], counts, color=['skyblue', 'salmon'])

    # Add count labels on top of bars
    for i, v in enumerate(counts):
        plt.text(i, v + 50, str(v), ha='center')

    # Add percentage labels in the middle of bars
    for i, v in enumerate(counts):
        percentage = f"{v/total*100:.1f}%"
        plt.text(i, v/2, percentage, ha='center', color='white', fontweight='bold')

    plt.title("Class Distribution")
    plt.xlabel("income")
    plt.ylabel("Count")
    plt.xticks([0, 1])
    plt.tight_layout()
    plt.savefig("class_distribution_income.png", bbox_inches="tight")
    plt.close()

    # ----------------------------
    # 5. Feature Distribution Analysis (3 ways: combined, class 0, class 1)
    # ----------------------------
    # Use all numeric features for distribution analysis
    all_numeric_features = df_no_outliers.select_dtypes(include=['float64', 'int64']).columns.tolist()
    
    # 5.1 Combined distribution
    fig = plt.figure(figsize=(20, 15))
    for i, feature in enumerate(all_numeric_features):
        plt.subplot(4, 4, i+1)
        sns.histplot(df_no_outliers[feature], kde=True, bins=20)
        plt.title(f"Distribution of {feature}")
        plt.tight_layout()
    plt.savefig("feature_distributions_combined.png", bbox_inches="tight")
    plt.close()
    
    # 5.2 Class 0 distribution
    fig = plt.figure(figsize=(20, 15))
    for i, feature in enumerate(all_numeric_features):
        plt.subplot(4, 4, i+1)
        sns.histplot(df_no_outliers[df_no_outliers[target] == 0][feature], 
                    kde=True, bins=20, color='blue')
        plt.title(f"Distribution of {feature} (Income <=50K)")
        plt.tight_layout()
    plt.savefig("feature_distributions_class0.png", bbox_inches="tight")
    plt.close()
    
    # 5.3 Class 1 distribution
    fig = plt.figure(figsize=(20, 15))
    for i, feature in enumerate(all_numeric_features):
        plt.subplot(4, 4, i+1)
        sns.histplot(df_no_outliers[df_no_outliers[target] == 1][feature], 
                    kde=True, bins=20, color='orange')
        plt.title(f"Distribution of {feature} (Income >50K)")
        plt.tight_layout()
    plt.savefig("feature_distributions_class1.png", bbox_inches="tight")
    plt.close()

    # 5.4 KDE overlaid for both classes (better for comparison)
    fig = plt.figure(figsize=(20, 15))
    for i, feature in enumerate(all_numeric_features):
        plt.subplot(4, 4, i+1)
        sns.kdeplot(df_no_outliers[df_no_outliers[target] == 0][feature], 
                  label='<=50K', color='blue')
        sns.kdeplot(df_no_outliers[df_no_outliers[target] == 1][feature], 
                  label='>50K', color='orange')
        plt.title(f"Distribution of {feature} by Income")
        plt.legend()
        plt.tight_layout()
    plt.savefig("feature_distributions_class_comparison.png", bbox_inches="tight")
    plt.close()

    # ----------------------------
    # 6. Correlation Analysis
    # ----------------------------
    # For correlation analysis, use only numeric features
    plt.figure(figsize=(12, 10))
    corr_matrix = df_no_outliers[all_numeric_features].corr()
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Correlation Matrix of Dataset (Numeric Features)")
    plt.savefig("correlation_matrix_income.png", bbox_inches="tight")
    plt.close()

    # ----------------------------
    # 7. Pairwise Feature Relationships (cleaner version)
    # ----------------------------
    # Select most important numeric features to avoid overcrowding
    important_features = all_numeric_features.copy()
    if len(important_features) > 6:
        # Only use up to 6 features for pairplot to keep it clean
        important_features = important_features[:6]
    
    plt.figure(figsize=(12, 10))
    sns.pairplot(df_no_outliers, vars=important_features, hue=target, 
                plot_kws={'alpha': 0.6}, diag_kind="kde")
    plt.suptitle("Pairwise Feature Relationships in Dataset", y=1.02)
    plt.savefig("pairplot_features_clean.png", bbox_inches="tight")
    plt.close()

    # ----------------------------
    # 8. Feature Importance Analysis
    # ----------------------------
    # For feature importance with RandomForest, we need to encode categorical features temporarily
    # Create a copy of the dataframe to avoid altering the original
    df_for_rf = df_no_outliers.copy()
    
    # Temporarily encode categorical features for RandomForest
    from sklearn.preprocessing import LabelEncoder
    
    encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df_for_rf[col] = le.fit_transform(df_for_rf[col].astype(str))
        encoders[col] = le
    
    # Extract features and target
    X_all_features = df_for_rf.drop(columns=[target])
    y = df_for_rf[target]
    
    # Train RandomForest for feature importance
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_all_features, y)
    importances = rf.feature_importances_
    
    # Sort features by importance
    feature_names = X_all_features.columns
    indices = np.argsort(importances)[::-1]
    
    # Plot feature importance
    plt.figure(figsize=(12, 8))
    plt.title("Feature Importances from RandomForest")
    sns.barplot(x=[feature_names[i] for i in indices], y=importances[indices])
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig("feature_importances_all.png", bbox_inches="tight")
    plt.close()

    # ----------------------------
    # 9. Cluster Analysis - KMeans
    # ----------------------------
    # Create a copy of the data for clustering (we will only use numeric features)
    X_numeric = df_no_outliers[numeric_features].copy()
    
    # Scale data for clustering (important!)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_numeric)
    
    # PCA for visualization
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    
    # KMeans clustering
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    
    # Plot KMeans clusters
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, 
                        cmap="viridis", alpha=0.6, s=50)
    plt.colorbar(scatter, label="Cluster")
    plt.title("KMeans Clustering (2 clusters) on PCA-reduced Data")
    plt.xlabel(f"Principal Component 1 ({pca.explained_variance_ratio_[0]:.2%} variance)")
    plt.ylabel(f"Principal Component 2 ({pca.explained_variance_ratio_[1]:.2%} variance)")
    plt.savefig("kmeans_clusters_fixed.png", bbox_inches="tight")
    plt.close()
    
    # Plot true income labels on PCA
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, 
                        cmap="coolwarm", alpha=0.6, s=50)
    plt.colorbar(scatter, label="Income")
    plt.title("PCA-reduced Data with True Income Labels")
    plt.xlabel(f"Principal Component 1 ({pca.explained_variance_ratio_[0]:.2%} variance)")
    plt.ylabel(f"Principal Component 2 ({pca.explained_variance_ratio_[1]:.2%} variance)")
    plt.savefig("pca_true_labels_fixed.png", bbox_inches="tight")
    plt.close()
    
    # Compare clusters with true labels
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap="viridis", alpha=0.6, s=50)
    plt.title("KMeans Clusters")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    
    plt.subplot(1, 2, 2)
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap="coolwarm", alpha=0.6, s=50)
    plt.title("True Income Labels")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    
    plt.tight_layout()
    plt.savefig("clusters_vs_true_labels.png", bbox_inches="tight")
    plt.close()

    # ----------------------------
    # 10. t-SNE Visualization
    # ----------------------------
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    X_tsne = tsne.fit_transform(X_scaled)
    
    # Plot t-SNE with true labels
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, 
                        cmap="coolwarm", alpha=0.6, s=50)
    plt.colorbar(scatter, label="Income")
    plt.title("t-SNE Visualization with True Income Labels")
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.savefig("tsne_true_labels.png", bbox_inches="tight")
    plt.close()

    # ----------------------------
    # 11. DBSCAN Clustering
    # ----------------------------
    # DBSCAN is sensitive to the eps parameter
    # Use a heuristic to determine eps
    from sklearn.neighbors import NearestNeighbors
    k = min(len(X_scaled) - 1, 5)  # number of neighbors to consider
    nn = NearestNeighbors(n_neighbors=k)
    nn.fit(X_scaled)
    distances, indices = nn.kneighbors(X_scaled)
    
    # Use mean of the k-th distances as eps
    eps = np.mean(distances[:, -1]) * 0.5
    
    # Run DBSCAN
    dbscan = DBSCAN(eps=eps, min_samples=5)
    dbscan_clusters = dbscan.fit_predict(X_scaled)
    
    # Plot DBSCAN clusters
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=dbscan_clusters, 
                        cmap="tab10", alpha=0.6, s=50)
    plt.colorbar(scatter, label="Cluster")
    plt.title(f"DBSCAN Clustering on PCA-reduced Data (eps={eps:.3f})")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.savefig("dbscan_clusters.png", bbox_inches="tight")
    plt.close()

    # ----------------------------
    # 12. Hierarchical Clustering
    # ----------------------------
    # Sample subset if dataset is large (for performance)
    sample_size = min(1000, len(X_scaled))
    indices = np.random.choice(len(X_scaled), sample_size, replace=False)
    X_sample = X_scaled[indices]
    y_sample = y.iloc[indices]
    
    # Compute linkage matrix
    Z = linkage(X_sample, method='ward')
    
    # Plot dendrogram
    plt.figure(figsize=(12, 8))
    plt.title("Hierarchical Clustering Dendrogram")
    dendrogram(Z, truncate_mode='level', p=5)
    plt.xlabel("Sample index")
    plt.ylabel("Distance")
    plt.savefig("hierarchical_clustering.png", bbox_inches="tight")
    plt.close()
    
    # Cut the dendrogram to get 2 clusters (for comparison with true labels)
    from scipy.cluster.hierarchy import fcluster
    hier_clusters = fcluster(Z, t=2, criterion='maxclust')
    
    # Plot hierarchical clusters on PCA projection
    X_pca_sample = pca.transform(X_sample)
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_pca_sample[:, 0], X_pca_sample[:, 1], 
                        c=hier_clusters, cmap="viridis", alpha=0.6, s=50)
    plt.colorbar(scatter, label="Cluster")
    plt.title("Hierarchical Clustering (2 clusters) on PCA-reduced Data")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.savefig("hierarchical_clusters_pca.png", bbox_inches="tight")
    plt.close()

    # ----------------------------
    # 13. Statistical Tests
    # ----------------------------
    # 13.1 Kolmogorov-Smirnov Tests for Numeric Features
    ks_results = {}
    for feature in numeric_features:
        # Get data for each class
        data_0 = df_no_outliers[df_no_outliers[target] == 0][feature]
        data_1 = df_no_outliers[df_no_outliers[target] == 1][feature]
        
        # Perform KS test
        statistic, p_value = ks_2samp(data_0, data_1)
        ks_results[feature] = {
            'statistic': statistic,
            'p_value': p_value,
            'different_distribution': p_value < 0.05
        }
    
    # Plot KS test results
    plt.figure(figsize=(12, 6))
    features = list(ks_results.keys())
    p_values = [ks_results[f]['p_value'] for f in features]
    statistics = [ks_results[f]['statistic'] for f in features]
    
    plt.subplot(1, 2, 1)
    bars = plt.bar(features, p_values)
    plt.axhline(y=0.05, color='r', linestyle='--', label='Significance level (0.05)')
    plt.xticks(rotation=45, ha='right')
    plt.title('Kolmogorov-Smirnov Test: p-values')
    plt.ylabel('p-value')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.bar(features, statistics)
    plt.xticks(rotation=45, ha='right')
    plt.title('Kolmogorov-Smirnov Test: Statistics')
    plt.ylabel('KS Statistic')
    
    plt.tight_layout()
    plt.savefig("ks_test_results.png", bbox_inches="tight")
    plt.close()
    
    # Print KS test results
    print("\nKolmogorov-Smirnov Test Results:")
    for feature, result in ks_results.items():
        conclusion = "Different distributions" if result['different_distribution'] else "Similar distributions"
        print(f"{feature}: Statistic={result['statistic']:.4f}, p-value={result['p_value']:.4f} => {conclusion}")
    
    # 13.2 Chi-squared Tests for Categorical Features
    chi2_results = {}
    for feature in categorical_cols:
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
    if categorical_cols:  # Only if there are categorical features
        plt.figure(figsize=(12, 6))
        features = list(chi2_results.keys())
        p_values = [chi2_results[f]['p_value'] for f in features]
        chi2_values = [chi2_results[f]['chi2'] for f in features]
        
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
    # 14. Prepare and Return SMOTENC-Ready Data
    # ----------------------------
    # Return the dataframe with categorical features preserved
    print("\nPreparation for SMOTENC complete.")
    print(f"Categorical features preserved: {categorical_cols}")
    print(f"Numeric features: {numeric_features}")
    print(f"Total features: {len(categorical_cols) + len(numeric_features)}")
    print(f"Dataset shape: {df_no_outliers.shape}")
    
    return df_no_outliers, categorical_cols  # Return both the dataframe and the list of categorical columns

# Example usage for SMOTENC preparation:
"""
# Define your columns and parameters
list_features_specialise_outliers = ['age', 'fnlwgt', 'education_num', 'capital_gain', 'capital_loss', 'hours_per_week']
numeric_features = ['age', 'fnlwgt', 'education_num', 'capital_gain', 'capital_loss', 'hours_per_week']
dist_features = numeric_features
target = 'income'

# Load the data
df = pd.read_csv('adult.csv')

# Prepare data for SMOTENC
df_prepared, categorical_features = prepare_data_pipeline_for_smotenc(
    df, 
    list_features_specialise_outliers, 
    numeric_features, 
    dist_features, 
    target
)

# Now you can use SMOTENC for oversampling:
from imblearn.over_sampling import SMOTENC

# Get categorical feature indices
categorical_features_indices = [df_prepared.columns.get_loc(col) for col in categorical_features]

# Apply SMOTENC
smote_nc = SMOTENC(categorical_features=categorical_features_indices, random_state=42)
X = df_prepared.drop(columns=[target])
y = df_prepared[target]
X_resampled, y_resampled = smote_nc.fit_resample(X, y)

# Output balanced dataset
df_balanced = pd.concat([pd.DataFrame(X_resampled, columns=X.columns), 
                         pd.Series(y_resampled, name=target)], axis=1)
print(f"Original class distribution:\n{y.value_counts()}")
print(f"Resampled class distribution:\n{y_resampled.value_counts()}")
"""