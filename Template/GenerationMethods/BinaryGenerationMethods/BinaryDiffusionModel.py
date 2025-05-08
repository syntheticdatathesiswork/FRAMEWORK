import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Concatenate, Dropout, Add, Activation
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
import matplotlib.pyplot as plt
import seaborn as sns
import os

def compute_beta_max_vector(data, method="variance", k=0.05):
    """
    Computes per-feature beta_max values based on a variability measure of each feature.
    
    Parameters:
      data: numpy array of shape (n_samples, n_features); typically after scaling.
      method: 'variance', 'iqr', or 'snr' (signal-to-noise ratio).
      k: multiplier constant to scale the variability measure.
      
    Returns:
      beta_max_vector: numpy array of shape (n_features,) with feature-specific beta_max values.
    """
    if method == "variance":
        # Use variance as the variability measure
        var = np.var(data, axis=0)
        beta_max_vector = k * var
    elif method == "iqr":
        # Use interquartile range (IQR)
        Q1 = np.percentile(data, 25, axis=0)
        Q3 = np.percentile(data, 75, axis=0)
        iqr = Q3 - Q1
        beta_max_vector = k * iqr
    elif method == "snr":
        # Use inverse of signal-to-noise ratio: lower SNR means higher noise
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        snr = np.where(std == 0, 1e-8, mean / std)
        beta_max_vector = k / snr
    else:
        raise ValueError("Method not recognized. Choose 'variance', 'iqr', or 'snr'.")
    return beta_max_vector

def build_diffusion_model(input_dim, weight_decay=1e-6, time_embedding_dim=16, dropout_rate=0.3):
    """
    Improved Diffusion Model with Increased Capacity, Residual Connections, and Dropout Layers.
    """
    # Inputs for data and time step
    x_input = Input(shape=(input_dim,), name="x_input")
    t_input = Input(shape=(1,), name="t_input")
    
    # Create a time embedding
    t_emb = Dense(
        time_embedding_dim,
        activation='relu',
        kernel_regularizer=regularizers.l2(weight_decay),
        name="t_embedding"
    )(t_input)
    
    # Concatenate the data with the time embedding
    concat = Concatenate(name="concat")([x_input, t_emb])
    
    # Initial dense layer to set the dimension for residual blocks
    h = Dense(256, activation='relu', kernel_regularizer=regularizers.l2(weight_decay))(concat)
    h = Dropout(dropout_rate)(h)
    
    # Residual Block 1
    block_input = h  # Save the input for the residual connection
    h = Dense(256, activation='relu', kernel_regularizer=regularizers.l2(weight_decay))(block_input)
    h = Dropout(dropout_rate)(h)
    h = Dense(256, activation='linear', kernel_regularizer=regularizers.l2(weight_decay))(h)
    # Add skip connection and apply activation
    h = Add()([h, block_input])
    h = Activation('relu')(h)
    
    # Residual Block 2
    block_input = h  # Save the input for the residual connection
    h = Dense(256, activation='relu', kernel_regularizer=regularizers.l2(weight_decay))(block_input)
    h = Dropout(dropout_rate)(h)
    h = Dense(256, activation='linear', kernel_regularizer=regularizers.l2(weight_decay))(h)
    # Add skip connection and apply activation
    h = Add()([h, block_input])
    h = Activation('relu')(h)
    
    # Final layers before output
    h = Dense(128, activation='relu', kernel_regularizer=regularizers.l2(weight_decay))(h)
    h = Dropout(dropout_rate)(h)
    
    # Output layer predicts the noise with the same dimension as the input data
    out = Dense(input_dim, activation='linear', kernel_regularizer=regularizers.l2(weight_decay))(h)
    
    model = Model(inputs=[x_input, t_input], outputs=out)
    return model

def generate_synthetic_samples_for_class_diffusion(
    class_data,
    n_samples,
    epochs=100,
    batch_size=32,
    diffusion_steps=10,
    weight_decay=1e-6,
    early_stopping_patience=20,
    temperature=1.0,
    aux_loss_weight=10.0,
    beta_max_vector=None
):
    """
    Synthetic Sample Generation with Auxiliary Loss
    ------------------------------------------------
    Trains a diffusion model on class_data and generates n_samples synthetic samples. 
    The data are normalized using MinMaxScaler, and after training, 
    the model produces synthetic outputs in normalized space that are then 
    un-normalized to the original scale.

    A PER-FEATURE noise schedule is used here, so each feature i can follow its own alpha_t.
    An auxiliary loss term penalizes denoised samples (computed from the predicted noise) 
    that fall outside the [0, 1] range.

    Parameters:
      class_data: numpy array of shape (n_original, n_features)
      n_samples: number of synthetic samples to generate.
      epochs: maximum training epochs.
      batch_size: training batch size.
      diffusion_steps: number of diffusion steps.
      weight_decay: L2 regularization factor.
      early_stopping_patience: epochs to wait before early stopping.
      temperature: scaling factor during generation.
      aux_loss_weight: weight for the auxiliary loss.
      beta_max_vector: Optional numpy array of shape (n_features,) containing per-feature beta_max values.
      
    Returns:
      synthetic_samples: synthetic samples in the original scale.
      synthetic_samples_norm: synthetic samples in normalized space.
    """
    # Normalize the class data
    scaler = RobustScaler()
    norm_class_data = scaler.fit_transform(class_data)
    n_features = norm_class_data.shape[1]

    # Build the diffusion model
    model = build_diffusion_model(n_features, weight_decay=weight_decay)
    optimizer = Adam(0.00025)  # Adjust learning rate as desired

    # ----------------------------------------------------------------
    # Create a PER-FEATURE COSINE NOISE SCHEDULE using a chosen beta_max
    # ----------------------------------------------------------------
    # If a custom beta_max_vector is provided, use it; otherwise compute from data variability.
    if beta_max_vector is None:
        beta_max = compute_beta_max_vector(norm_class_data, method="variance", k=0.05).astype(np.float32)
    else:
        beta_max = beta_max_vector.astype(np.float32)

    # For each feature i, generate alpha_bar[t, i] using a cosine schedule.
    alpha_bar_f = np.zeros((diffusion_steps, n_features), dtype=np.float32)
    s = 0.008  # smoothing constant

    for i in range(n_features):
        for t in range(diffusion_steps):
            # fraction of steps in [0, 1]
            fraction = t / (diffusion_steps - 1)  
            # Cosine schedule
            cosine_term = np.cos(((fraction + s) / (1.0 + s)) * (np.pi / 2.0)) ** 2
            alpha_bar_f[t, i] = (1.0 - beta_max[i]) + beta_max[i] * cosine_term

        # Force alpha_bar_f[0, i] to 1.0 at t=0
        alpha_bar_f[0, i] = 1.0

    # Compute alpha_f(t, i) = alpha_bar_f(t, i) / alpha_bar_f(t-1, i)
    alpha_f = np.zeros_like(alpha_bar_f)
    alpha_f[0, :] = 1.0
    for t in range(1, diffusion_steps):
        alpha_f[t] = alpha_bar_f[t] / alpha_bar_f[t - 1]

    # Convert to tf.constant for easy batch gather
    alpha_bar_tf = tf.constant(alpha_bar_f, dtype=tf.float32)  # (diffusion_steps, n_features)
    alpha_tf     = tf.constant(alpha_f,     dtype=tf.float32)  # (diffusion_steps, n_features)

    # Training loop
    X = norm_class_data
    best_loss = float('inf')
    patience = 0

    for epoch in range(epochs):
        # Sample a random batch
        idx = np.random.randint(0, X.shape[0], batch_size)
        x_batch = X[idx]  # shape (batch_size, n_features)
        t_rand = np.random.randint(0, diffusion_steps, size=batch_size)
        noise_batch = np.random.normal(0, 1, x_batch.shape)
        
        # Convert to tensors
        x_batch_tf = tf.convert_to_tensor(x_batch, dtype=tf.float32)
        noise_tf = tf.convert_to_tensor(noise_batch, dtype=tf.float32)
        t_rand_tf = tf.convert_to_tensor(t_rand, dtype=tf.int32)
        
        with tf.GradientTape() as tape:
            # Gather time-dependent parameters for each sample in the batch
            # alpha_bar_t, alpha_t will be shape (batch_size, n_features)
            alpha_bar_t = tf.gather(alpha_bar_tf, t_rand_tf, axis=0)
            alpha_t     = tf.gather(alpha_tf,     t_rand_tf, axis=0)

            sqrt_alpha_bar_t = tf.sqrt(alpha_bar_t)                      # (batch_size, n_features)
            sqrt_one_minus_alpha_bar_t = tf.sqrt(1.0 - alpha_bar_t)
            sqrt_alpha_t = tf.sqrt(alpha_t)

            # Forward process: add noise (per-feature)
            x_noisy = sqrt_alpha_bar_t * x_batch_tf + sqrt_one_minus_alpha_bar_t * noise_tf

            # Predict the noise
            # t_input_tf shape: (batch_size, 1)
            t_input_tf = tf.reshape(t_rand_tf, (-1, 1))
            predicted_noise = model([x_noisy, t_input_tf], training=True)

            # MSE: difference between predicted noise and the actual noise added
            mse_loss = tf.reduce_mean(tf.square(predicted_noise - noise_tf))

            # Compute the denoised sample from the predicted noise
            x_denoised = (x_noisy - sqrt_one_minus_alpha_bar_t * predicted_noise) / sqrt_alpha_t

            # Auxiliary loss: penalize out-of-bound values [0, 1]
            lower_violation = tf.square(tf.maximum(0.0, -x_denoised))
            upper_violation = tf.square(tf.maximum(0.0, x_denoised - 1.0))
            aux_penalty = tf.reduce_mean(lower_violation + upper_violation)
            
            total_loss = mse_loss + aux_loss_weight * aux_penalty
        
        # Compute and apply gradients
        gradients = tape.gradient(total_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        
        # Early stopping logic
        if total_loss < best_loss:
            best_loss = total_loss
            patience = 0
        else:
            patience += 1
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{epochs} - Total Loss: {total_loss:.4f} - MSE: {mse_loss:.4f} - Aux: {aux_penalty:.4f}")

        if patience >= early_stopping_patience:
            print(f"Early stopping triggered at epoch {epoch} with total loss {total_loss:.4f}")
            break

    # -------------------------------------------------
    # Generation phase: start with pure noise and denoise
    # -------------------------------------------------
    synthetic_samples_norm = []
    for _ in range(n_samples):
        # Start from pure noise
        x = np.random.normal(0, 1, (1, n_features)).astype(np.float32)
        for t in reversed(range(diffusion_steps)):
            # Per-feature alpha_bar, alpha
            alpha_bar_t = alpha_bar_f[t]  # shape (n_features,)
            alpha_t     = alpha_f[t]      # shape (n_features,)

            sqrt_one_minus_alpha_bar_t = np.sqrt(1.0 - alpha_bar_t)   # shape (n_features,)
            sqrt_alpha_t               = np.sqrt(alpha_t)

            # Predict noise
            t_value = np.array([[t]], dtype=np.float32)
            predicted_noise = model.predict([x, t_value])  # shape (1, n_features)

            # Apply temperature
            predicted_noise *= temperature
            
            # Reverse step (per-feature)
            # x_{t-1} = ( x_t - sqrt(1 - alpha_bar_t)*noise ) / sqrt(alpha_t)
            x = (x - sqrt_one_minus_alpha_bar_t * predicted_noise) / sqrt_alpha_t

        synthetic_samples_norm.append(x.squeeze(axis=0))  # shape (n_features,)

    synthetic_samples_norm = np.array(synthetic_samples_norm)
    # Un-normalize the synthetic samples
    synthetic_samples = scaler.inverse_transform(synthetic_samples_norm)
    return synthetic_samples, synthetic_samples_norm

def augment_binary_diffusion(
    df,
    target,
    test_size=0.2,
    random_state=42,
    ratio_limit=1.0,
    diffusion_epochs=100,
    diffusion_batch_size=32,
    diffusion_steps=10,
    weight_decay=1e-6,
    early_stopping_patience=20,
    temperature=1.0,
    aux_loss_weight=10.0,
    beta_max_vector=None,
    output_dir='./output',
    save_results=True,
    plot_results=True
):
    """
    Binary Classification Augmentation Using the Diffusion Model
    ------------------------------------------------------------
    For binary classification tasks where the minority class needs augmentation.
    Automatically identifies the minority class and generates synthetic samples using
    a diffusion model with per-feature noise scheduling.
    
    Parameters:
        df: DataFrame containing the pre-processed data
        target: Target column name (binary classification column, 0/1)
        test_size: Fraction of data to use for testing
        random_state: Random seed for reproducibility
        ratio_limit: Maximum ratio to balance classes (1.0 = full balance)
        diffusion_epochs: Number of training epochs for diffusion model
        diffusion_batch_size: Batch size for training
        diffusion_steps: Number of diffusion steps for generation
        weight_decay: L2 regularization factor
        early_stopping_patience: Epochs to wait before early stopping
        temperature: Temperature for sampling (lower = less random)
        aux_loss_weight: Weight for the auxiliary loss (bounding synthetic outputs)
        beta_max_vector: Optional custom noise schedule vector
        output_dir: Directory to save results
        save_results: Whether to save results to files
        plot_results: Whether to plot and save visualizations
        
    Returns:
        original_train: Original training data
        augmented_train: Augmented training data 
        test_set: Test data
        out_of_bounds_flag: True if any synthetic normalized output is outside [0,1]
    """
    if save_results:
        os.makedirs(output_dir, exist_ok=True)
    
    # Validate binary classification task
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in DataFrame")
    
    # Ensure target has binary values (0/1)
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
    class_diff = majority_count - minority_count
    target_count = int(majority_count * ratio_limit)
    synthetic_needed = min(target_count - minority_count, int(minority_count * ratio_limit))
    synthetic_needed = max(0, synthetic_needed)  # Ensure non-negative
    
    print(f"Generating {synthetic_needed} synthetic samples for minority class {minority_class}")
    print(f"  - Current minority count: {minority_count}")
    print(f"  - Current majority count: {majority_count}")
    print(f"  - Target count with ratio_limit={ratio_limit}: {target_count}")
    print(f"  - Final minority count after augmentation: {minority_count + synthetic_needed}")
    
    # If no augmentation needed, return original data
    if synthetic_needed <= 0:
        print("No augmentation needed based on parameters.")
        train['synthetic'] = False
        return original_train.reset_index(drop=True), train.reset_index(drop=True), test.reset_index(drop=True), False
    
    # Get feature data for minority class
    feature_cols = [col for col in train.columns if col != target]
    minority_data = train[train[target] == minority_class][feature_cols].values
    
    # Generate synthetic samples for minority class
    synthetic_samples, synthetic_samples_norm = generate_synthetic_samples_for_class_diffusion(
        class_data=minority_data,
        n_samples=synthetic_needed,
        epochs=diffusion_epochs,
        batch_size=diffusion_batch_size,
        diffusion_steps=diffusion_steps,
        weight_decay=weight_decay,
        early_stopping_patience=early_stopping_patience,
        temperature=temperature,
        aux_loss_weight=aux_loss_weight,
        beta_max_vector=beta_max_vector
    )
    
    # Check if any normalized synthetic values are out-of-bound
    out_of_bounds_flag = False
    if np.any(synthetic_samples_norm < 0) or np.any(synthetic_samples_norm > 1):
        out_of_bounds_flag = True
        print("Warning: Some synthetic samples are out of the normalized [0,1] range.")
    
    # Create DataFrame for synthetic samples
    synthetic_df = pd.DataFrame(synthetic_samples, columns=feature_cols)
    synthetic_df[target] = minority_class
    synthetic_df['synthetic'] = True
    
    # Flag original rows as non-synthetic
    train['synthetic'] = False
    
    # Combine original and synthetic data
    augmented_train = pd.concat([train, synthetic_df], axis=0).reset_index(drop=True)
    
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
        original_train.to_csv(f"{output_dir}/diffusion_original_train.csv", index=False)
        augmented_train.to_csv(f"{output_dir}/diffusion_augmented_train.csv", index=False)
        test.to_csv(f"{output_dir}/diffusion_test_set.csv", index=False)
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
        plt.savefig(f'{output_dir}/diffusion_class_distribution.png')
        plt.close()
        
        # Plot feature distributions for minority class
        orig_minority = train[train[target] == minority_class]
        
        for feature in feature_cols:
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
            plt.savefig(f'{output_dir}/diffusion_{feature}_distribution.png')
            plt.close()
        
        # Plot correlation heatmaps
        numeric_features = orig_minority.select_dtypes(include=['number']).columns
        numeric_features = [col for col in numeric_features if col != target and col != 'synthetic']
        
        if len(numeric_features) > 0:
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
            plt.savefig(f'{output_dir}/diffusion_correlation_comparison.png')
            plt.close()
            
            # Plot correlation difference
            plt.figure(figsize=(10, 8))
            corr_diff = abs(orig_corr - synth_corr)
            sns.heatmap(corr_diff, annot=True, cmap='Reds')
            plt.title('Correlation Matrix Absolute Difference (Original vs Synthetic)')
            plt.tight_layout()
            plt.savefig(f'{output_dir}/diffusion_correlation_difference.png')
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
            plt.savefig(f'{output_dir}/diffusion_2d_feature_comparison.png')
            plt.close()
    
    return original_train.reset_index(drop=True), augmented_train, test.reset_index(drop=True), out_of_bounds_flag

# Example usage
if __name__ == "__main__":
    # Load your pre-processed dataset
    # This assumes you already have a clean DataFrame with binary classification target
    df = pd.read_csv('your_binary_dataset.csv')
    
    # Set target column (must be binary 0/1)
    target_column = 'your_target_column'
    
    # Run the diffusion augmentation for binary classification
    original_train, augmented_train, test_set, out_of_bounds = augment_binary_diffusion(
        df=df, 
        target=target_column,
        test_size=0.2,
        random_state=42,
        ratio_limit=1.0,
        diffusion_epochs=100,
        diffusion_batch_size=32,
        diffusion_steps=10,
        weight_decay=1e-6,
        early_stopping_patience=20,
        temperature=1.0,
        aux_loss_weight=10.0,
        output_dir='./binary_diffusion_output'
    )
    
    if not out_of_bounds:
        print("Binary diffusion augmentation completed successfully!")
        print(f"Original training samples: {len(original_train)}")
        print(f"Augmented training samples: {len(augmented_train)}")
        print(f"Test samples: {len(test_set)}")
        
        # Get class distribution after augmentation
        aug_distribution = augmented_train.groupby(target_column).size()
        print("Final class distribution:")
        for cls, count in aug_distribution.items():
            print(f"  Class {cls}: {count} samples ({count/len(augmented_train)*100:.1f}%)")
        
        # Save final output
        augmented_train.to_csv("./binary_diffusion_output/final_augmented_dataset.csv", index=False)
    else:
        print("Warning: Some synthetic samples were generated outside the normalized range.")
        print("This might indicate poor model fitting or numerical instability.")
        print("Consider adjusting hyperparameters or using a different approach.")