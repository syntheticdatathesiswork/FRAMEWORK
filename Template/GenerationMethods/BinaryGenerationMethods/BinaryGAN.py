import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, LeakyReLU, BatchNormalization
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam

# ----------------------------
# GAN Building Blocks - FIXED VERSION
# ----------------------------

def build_generator(noise_dim, output_dim):
    """Build generator using Sequential API"""
    model = Sequential()
    model.add(Input(shape=(noise_dim,)))
    # Layer 1
    model.add(Dense(128))
    model.add(LeakyReLU(negative_slope=0.2))
    model.add(BatchNormalization())
    # Layer 2
    model.add(Dense(256))
    model.add(LeakyReLU(negative_slope=0.2))
    model.add(BatchNormalization())
    # Layer 3
    model.add(Dense(256))
    model.add(LeakyReLU(negative_slope=0.2))
    model.add(BatchNormalization())
    # Layer 4
    model.add(Dense(128))
    model.add(LeakyReLU(negative_slope=0.2))
    model.add(BatchNormalization())
    # Layer 5
    model.add(Dense(128))
    model.add(LeakyReLU(negative_slope=0.2))
    model.add(BatchNormalization())
    # Output layer with tanh activation so outputs lie in [-1, 1]
    model.add(Dense(output_dim, activation='tanh'))
    return model

def build_discriminator(input_dim):
    """Build discriminator and feature extractor using Functional API"""
    # Define input
    inputs = Input(shape=(input_dim,))
    
    # Layer 1
    x = Dense(256)(inputs)
    x = LeakyReLU(negative_slope=0.2)(x)
    
    # Layer 2
    x = Dense(256)(x)
    x = LeakyReLU(negative_slope=0.2)(x)
    
    # Layer 3
    x = Dense(128)(x)
    x = LeakyReLU(negative_slope=0.2)(x)
    
    # Layer 4
    x = Dense(128)(x)
    x = LeakyReLU(negative_slope=0.2)(x)
    
    # Layer 5
    x = Dense(64)(x)
    x = LeakyReLU(negative_slope=0.2)(x)
    
    # Feature layer - we'll use this for feature matching
    features = Dense(32)(x)
    features_activated = LeakyReLU(negative_slope=0.2)(features)
    
    # Output layer with sigmoid activation
    outputs = Dense(1, activation='sigmoid')(features_activated)
    
    # Create the full model
    discriminator = Model(inputs=inputs, outputs=outputs, name='discriminator')
    
    # Create feature extractor model sharing the same inputs
    feature_extractor = Model(inputs=inputs, outputs=features_activated, name='feature_extractor')
    
    return discriminator, feature_extractor

def generate_synthetic_samples_for_class(class_data, n_samples, epochs=2000, batch_size=64, noise_dim=100, lambda_fm=10.0, n_critic=5):
    """
    Trains a GAN on class_data and generates n_samples synthetic samples.
    Data is scaled with MinMaxScaler to [-1,1] and the generator uses tanh.
    A feature matching loss is added to encourage generated samples to mimic the real data's feature statistics.
    
    FIXED: Uses properly built discriminator and feature extractor using Functional API
    """
    # Scale data to [-1,1]
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaled_data = scaler.fit_transform(class_data)
    n_features = scaled_data.shape[1]
    
    # Set up optimizers
    d_optimizer = Adam(0.0002, 0.5)
    g_optimizer = Adam(0.0002, 0.5)
    
    # Build discriminator and feature extractor using Functional API - FIXED
    discriminator, feature_extractor = build_discriminator(n_features)
    discriminator.compile(loss='binary_crossentropy', optimizer=d_optimizer, metrics=['accuracy'])
    
    # Build generator
    generator = build_generator(noise_dim, n_features)
    
    # Define loss functions
    bce_loss = tf.keras.losses.BinaryCrossentropy()
    mse_loss = tf.keras.losses.MeanSquaredError()
    
    X_train = scaled_data.copy()
    
    # Check batch size doesn't exceed data size
    batch_size = min(batch_size, X_train.shape[0])
    
    # Adjust batch size if too small
    if batch_size < 4:
        batch_size = min(4, X_train.shape[0])
        print(f"Adjusted batch size to {batch_size} due to small dataset size")
    
    for epoch in range(epochs):
        # Update the discriminator n_critic times
        for _ in range(n_critic):
            # Generate random indices for data batch
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            real_samples = X_train[idx]
            
            # Generate noise and fake samples
            noise = np.random.normal(0, 1, (batch_size, noise_dim))
            fake_samples = generator.predict(noise, verbose=0)
            
            # Prepare labels with label smoothing
            valid = np.ones((batch_size, 1)) * 0.9  # label smoothing
            fake = np.zeros((batch_size, 1))
            
            # Train discriminator on real and fake samples
            d_loss_real = discriminator.train_on_batch(real_samples, valid)
            d_loss_fake = discriminator.train_on_batch(fake_samples, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
        
        # Update generator using adversarial loss + feature matching loss
        noise = np.random.normal(0, 1, (batch_size, noise_dim))
        with tf.GradientTape() as tape:
            # Generate fake samples
            generated_samples = generator(noise, training=True)
            # Get discriminator predictions
            y_pred = discriminator(generated_samples, training=False)
            # Adversarial loss
            adv_loss = bce_loss(np.ones((batch_size, 1)) * 0.9, y_pred)
            
            # Feature matching - extract features from real and generated samples
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            real_samples = X_train[idx]
            real_features = feature_extractor(real_samples, training=False)
            fake_features = feature_extractor(generated_samples, training=False)
            
            # Feature matching loss
            fm_loss = mse_loss(tf.reduce_mean(real_features, axis=0),
                               tf.reduce_mean(fake_features, axis=0))
            
            # Total generator loss
            g_loss = adv_loss + lambda_fm * fm_loss
        
        # Apply gradients to update generator
        grads = tape.gradient(g_loss, generator.trainable_weights)
        g_optimizer.apply_gradients(zip(grads, generator.trainable_weights))
        
        # Print progress
        if epoch % 100 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch}/{epochs} - D loss: {d_loss[0]:.4f}, Adv loss: {adv_loss.numpy():.4f}, FM loss: {fm_loss.numpy():.4f}, G total loss: {g_loss.numpy():.4f}")
    
    # Generate synthetic samples after training
    print("Generating final synthetic samples...")
    noise = np.random.normal(0, 1, (n_samples, noise_dim))
    synthetic_scaled = generator.predict(noise, verbose=0)
    synthetic_samples = scaler.inverse_transform(synthetic_scaled)
    return synthetic_samples


# ----------------------------
# Binary Augmentation Function
# ----------------------------

def augment_binary_gan(df, target, test_size=0.2, random_state=42, 
                       ratio_limit=1.0, gan_epochs=2000, gan_batch_size=64, 
                       noise_dim=100, lambda_fm=10.0, n_critic=5,
                       output_dir='./output', save_results=True, plot_results=True):
    """
    Binary classification augmentation using GAN with feature matching loss.
    
    Parameters:
        df: DataFrame containing the pre-processed data
        target: Target column name (binary classification column, 0/1 or binary values)
        test_size: Fraction of data to use for testing
        random_state: Random seed for reproducibility
        ratio_limit: Maximum ratio to balance classes (1.0 = full balance)
        gan_epochs: Number of training epochs for GAN
        gan_batch_size: Batch size for training
        noise_dim: Dimensionality of the noise input to generator
        lambda_fm: Weight for feature matching loss
        n_critic: Number of discriminator updates per generator update
        output_dir: Directory to save results
        save_results: Whether to save results to files
        plot_results: Whether to plot and save visualizations
        
    Returns:
        original_train: Original training data
        augmented_train: Augmented training data 
        test_set: Test data
    """
    if save_results:
        os.makedirs(output_dir, exist_ok=True)
    
    # Validate binary classification task
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in DataFrame")
    
    # Ensure target has binary values
    unique_values = df[target].unique()
    if len(unique_values) != 2:
        raise ValueError(f"Target column should contain exactly 2 unique values for binary classification. Found: {unique_values}")
        
    # Convert to 0/1 if needed
    if not all(val in [0, 1] for val in unique_values):
        print(f"Warning: Target column values are not 0/1. Found: {unique_values}")
        mapping = {unique_values[0]: 0, unique_values[1]: 1}
        df = df.copy()
        df[target] = df[target].map(mapping)
        print(f"Converted values to 0/1 using mapping: {mapping}")
    
    # Split the data with stratification to maintain class balance
    train, test = train_test_split(df, test_size=test_size, random_state=random_state, stratify=df[target])
    original_train = train.copy()
    
    # Identify minority and majority classes
    class_counts = train[target].value_counts()
    print(f"Class distribution in training data: {class_counts.to_dict()}")
    
    # Determine minority class (assume class 1 is minority unless counts show otherwise)
    minority_class = 1 if class_counts[1] <= class_counts[0] else 0
    majority_class = 1 - minority_class  # The other class
    
    minority_count = class_counts[minority_class]
    majority_count = class_counts[majority_class]
    
    print(f"Minority class ({minority_class}): {minority_count} samples")
    print(f"Majority class ({majority_class}): {majority_count} samples")
    print(f"Imbalance ratio: {majority_count/minority_count:.2f}:1")
    
    # Calculate how many synthetic samples to generate
    desired_by_ratio = int(majority_count * ratio_limit)
    synthetic_needed = desired_by_ratio - minority_count
    synthetic_needed = max(0, synthetic_needed)  # Ensure non-negative
    
    if synthetic_needed <= 0:
        print(f"No augmentation needed based on current parameters.")
        train['synthetic'] = False
        return original_train.reset_index(drop=True), train.reset_index(drop=True), test.reset_index(drop=True)
    
    print(f"Generating {synthetic_needed} synthetic samples for minority class {minority_class}")
    print(f"  - Current minority count: {minority_count}")
    print(f"  - Current majority count: {majority_count}")
    print(f"  - Target minority count with ratio_limit={ratio_limit}: {desired_by_ratio}")
    print(f"  - Final minority count after augmentation: {minority_count + synthetic_needed}")
    
    # Get feature data for minority class
    minority_data = train[train[target] == minority_class].drop(columns=[target]).values
    
    # Check if we have enough minority samples for GAN training
    min_required_samples = 4  # Absolute minimum for GAN training
    if len(minority_data) < min_required_samples:
        print(f"Warning: Not enough minority samples ({len(minority_data)}) for effective GAN training.")
        print("Minimum required samples for GAN is 4. Returning original data without augmentation.")
        train['synthetic'] = False
        return original_train.reset_index(drop=True), train.reset_index(drop=True), test.reset_index(drop=True)
    
    # Handle batch size based on available data
    effective_batch_size = min(gan_batch_size, len(minority_data))
    if effective_batch_size != gan_batch_size:
        print(f"Adjusted batch size from {gan_batch_size} to {effective_batch_size} due to limited data")
    
    # Generate synthetic samples for the minority class
    try:
        synthetic_samples = generate_synthetic_samples_for_class(
            minority_data, 
            n_samples=synthetic_needed, 
            epochs=gan_epochs, 
            batch_size=effective_batch_size, 
            noise_dim=noise_dim, 
            lambda_fm=lambda_fm,
            n_critic=n_critic
        )
        
        # Create DataFrame for synthetic samples
        synthetic_df = pd.DataFrame(synthetic_samples, columns=train.drop(columns=[target]).columns)
        synthetic_df[target] = minority_class
        synthetic_df['synthetic'] = True
        
        # Flag original rows as non-synthetic
        train['synthetic'] = False
        
        # Combine original and synthetic data
        augmented_train = pd.concat([train, synthetic_df], axis=0).reset_index(drop=True)
        
    except Exception as e:
        print(f"Error during GAN training or sample generation: {str(e)}")
        print("Returning original data without augmentation.")
        train['synthetic'] = False
        return original_train.reset_index(drop=True), train.reset_index(drop=True), test.reset_index(drop=True)
    
    # Print summary
    print(f"\nAugmented data summary:")
    print(f"  - Original samples: {len(train)}")
    print(f"  - Synthetic minority samples: {len(synthetic_df)}")
    print(f"  - Total samples: {len(augmented_train)}")
    
    # Show final class distribution
    aug_counts = augmented_train[target].value_counts()
    print(f"Final class distribution:")
    for cls, count in aug_counts.items():
        orig_count = class_counts.get(cls, 0)
        print(f"  Class {cls}: {count} samples (was {orig_count})")
    
    print(f"New class ratio: 1:{aug_counts[majority_class]/aug_counts[minority_class]:.2f}")
    
    # Save results
    if save_results:
        original_train.to_csv(f"{output_dir}/gan_original_train.csv", index=False)
        augmented_train.to_csv(f"{output_dir}/gan_augmented_train.csv", index=False)
        test.to_csv(f"{output_dir}/gan_test_set.csv", index=False)
        print(f"Saved results to {output_dir}/")
    
    # Create visualizations
    if plot_results:
        # Plot class distribution before and after
        plt.figure(figsize=(10, 6))
        
        # Before augmentation
        before_counts = [class_counts.get(0, 0), class_counts.get(1, 0)]
        
        # After augmentation
        after_counts = [
            augmented_train[augmented_train[target] == 0].shape[0],
            augmented_train[augmented_train[target] == 1].shape[0]
        ]
        
        x = np.arange(2)
        width = 0.35
        
        plt.bar(x - width/2, before_counts, width, label='Before Augmentation')
        plt.bar(x + width/2, after_counts, width, label='After Augmentation')
        
        plt.xlabel('Class')
        plt.ylabel('Count')
        plt.title('Class Distribution Before and After Augmentation')
        plt.xticks(x, ['Class 0', 'Class 1'])
        plt.legend()
        plt.savefig(f'{output_dir}/gan_class_distribution.png')
        plt.close()
        
        # Plot feature distributions for minority class
        orig_minority = train[train[target] == minority_class]
        feature_cols = [col for col in train.columns if col != target and col != 'synthetic']
        
        for feature in feature_cols:
            # Skip non-numeric features
            if not pd.api.types.is_numeric_dtype(train[feature]):
                continue
                
            plt.figure(figsize=(12, 6))
            
            # Use KDE plots for smoother visualization
            sns.kdeplot(
                data=orig_minority,
                x=feature,
                fill=True,
                alpha=0.5,
                label="Original Minority",
                color="steelblue"
            )
            
            sns.kdeplot(
                data=synthetic_df,
                x=feature,
                fill=True,
                alpha=0.5,
                label="Synthetic Minority",
                color="coral"
            )
            
            # Also add histograms with transparency
            plt.hist(
                orig_minority[feature], 
                bins=30, 
                alpha=0.3, 
                density=True,
                color="steelblue"
            )
            
            plt.hist(
                synthetic_df[feature], 
                bins=30, 
                alpha=0.3, 
                density=True,
                color="coral"
            )
            
            plt.title(f'Distribution of {feature} for Minority Class ({minority_class})')
            plt.legend()
            plt.tight_layout()
            plt.savefig(f'{output_dir}/gan_{feature}_distribution.png')
            plt.close()
        
        # Plot correlation heatmaps
        numeric_features = orig_minority.select_dtypes(include=['number']).columns
        numeric_features = [col for col in numeric_features if col != target and col != 'synthetic']
        
        if len(numeric_features) > 1:
            plt.figure(figsize=(12, 10))
            
            plt.subplot(1, 2, 1)
            orig_corr = orig_minority[numeric_features].corr()
            sns.heatmap(orig_corr, annot=False, cmap='coolwarm')
            plt.title('Original Minority Data Correlation')
            
            plt.subplot(1, 2, 2)
            synth_corr = synthetic_df[numeric_features].corr()
            sns.heatmap(synth_corr, annot=False, cmap='coolwarm')
            plt.title('Synthetic Minority Data Correlation')
            
            plt.tight_layout()
            plt.savefig(f'{output_dir}/gan_correlation_comparison.png')
            plt.close()
            
            # Plot correlation difference
            plt.figure(figsize=(10, 8))
            corr_diff = abs(orig_corr - synth_corr)
            sns.heatmap(corr_diff, annot=True, cmap='Reds')
            plt.title('Correlation Matrix Absolute Difference (Original vs Synthetic)')
            plt.tight_layout()
            plt.savefig(f'{output_dir}/gan_correlation_difference.png')
            plt.close()
        
        # 2D feature plot if possible
        if len(numeric_features) >= 2:
            # Select 2 features with highest variance
            feature_vars = orig_minority[numeric_features].var().sort_values(ascending=False)
            top_features = feature_vars.index[:2].tolist()
            
            plt.figure(figsize=(10, 8))
            plt.scatter(
                orig_minority[top_features[0]], 
                orig_minority[top_features[1]],
                alpha=0.6, 
                label="Original Minority",
                color="steelblue",
                edgecolor='w',
                s=50
            )
            
            plt.scatter(
                synthetic_df[top_features[0]], 
                synthetic_df[top_features[1]],
                alpha=0.6, 
                label="Synthetic Minority",
                color="coral",
                marker='x',
                s=50
            )
            
            plt.title(f'2D Feature Comparison: {top_features[0]} vs {top_features[1]}')
            plt.xlabel(top_features[0])
            plt.ylabel(top_features[1])
            plt.legend()
            plt.savefig(f'{output_dir}/gan_2d_feature_comparison.png')
            plt.close()
    
    return original_train.reset_index(drop=True), augmented_train, test.reset_index(drop=True)

# Example usage
if __name__ == "__main__":
    # Load your pre-processed dataset
    # This assumes you already have a clean DataFrame with binary classification target
    df = pd.read_csv('your_binary_dataset.csv')
    
    # Set target column (must be binary 0/1 or will be converted)
    target_column = 'your_target_column'
    
    # Run the GAN augmentation for binary classification
    original_train, augmented_train, test_set = augment_binary_gan(
        df=df, 
        target=target_column,
        test_size=0.2,
        random_state=42,
        ratio_limit=1.0,
        gan_epochs=2000,
        gan_batch_size=64,
        noise_dim=100,
        lambda_fm=10.0,
        output_dir='./binary_gan_output'
    )
    
    print(f"Binary GAN augmentation completed successfully!")
    print(f"Original training samples: {len(original_train)}")
    print(f"Augmented training samples: {len(augmented_train)}")
    print(f"Test samples: {len(test_set)}")
    
    # Get class distribution after augmentation
    aug_distribution = augmented_train.groupby(target_column).size()
    print("Final class distribution:")
    for cls, count in aug_distribution.items():
        print(f"  Class {cls}: {count} samples ({count/len(augmented_train)*100:.1f}%)")
    
    # Save final output
    augmented_train.to_csv("./binary_gan_output/final_augmented_dataset.csv", index=False)