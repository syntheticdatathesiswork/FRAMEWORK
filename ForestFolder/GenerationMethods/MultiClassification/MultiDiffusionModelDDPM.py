"""
TabDDPM-based Synthetic Data Generation for Tabular Data (Refined with Per-Feature Beta_max)

This module provides functions to build and train a TabDDPM model for generating synthetic tabular data.
It includes:
  - A sinusoidal time embedding layer.
  - A TabDDPM model with residual-style dense layers.
  - A function (compute_beta_max_vector) to compute per-feature beta_max values based on data variability.
  - A training routine using a per-feature cosine noise schedule.
  - DataFrame augmentation functions that allow supplying a custom beta_max_vector.

Adjust hyperparameters and model architecture as needed.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Concatenate, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers

def get_sinusoidal_embedding_layer(time_embedding_dim):
    """
    Returns a Lambda layer that computes a sinusoidal embedding for t_input.
    """
    def sinusoidal_embedding(x):
        # x shape: (batch, 1)
        half_dim = time_embedding_dim // 2
        log_timescale_increment = tf.math.log(10000.0) / (half_dim - 1)
        inv_timescales = tf.exp(tf.range(half_dim, dtype=tf.float32) * -log_timescale_increment)
        emb = tf.cast(x, tf.float32) * inv_timescales  # (batch, half_dim)
        emb = tf.concat([tf.sin(emb), tf.cos(emb)], axis=-1)  # (batch, time_embedding_dim)
        return emb
    return Lambda(sinusoidal_embedding, name="t_sinusoidal_embedding")

def build_tabddpm_model(input_dim, weight_decay=1e-6, time_embedding_dim=32):
    """
    Builds a TabDDPM model for tabular data.
    This model uses a sinusoidal time embedding and a series of dense layers with a residual-style connection.
    """
    x_input = Input(shape=(input_dim,), name="x_input")
    t_input = Input(shape=(1,), name="t_input")
    
    # Compute sinusoidal embedding and project it
    t_emb = get_sinusoidal_embedding_layer(time_embedding_dim)(t_input)
    t_proj = Dense(time_embedding_dim, activation='relu',
                   kernel_regularizer=regularizers.l2(weight_decay),
                   name="t_projection")(t_emb)
    
    # Concatenate input data with time embedding projection
    concat = Concatenate(name="concat")([x_input, t_proj])
    
    # Deep network with residual-style branch
    h = Dense(256, activation='relu', kernel_regularizer=regularizers.l2(weight_decay))(concat)
    h_res = Dense(256, activation='relu', kernel_regularizer=regularizers.l2(weight_decay))(h)
    h2 = Dense(256, activation='relu', kernel_regularizer=regularizers.l2(weight_decay))(h_res)
    h_cat = Concatenate(name="residual_concat")([h2, h_res])
    out = Dense(input_dim, activation='linear',
                kernel_regularizer=regularizers.l2(weight_decay),
                name="output")(h_cat)
    
    model = Model(inputs=[x_input, t_input], outputs=out)
    return model

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

def generate_synthetic_samples_for_class_tabddpm(
    class_data,
    n_samples,
    epochs=100,
    batch_size=32,
    diffusion_steps=10,
    weight_decay=1e-6,
    early_stopping_patience=20,
    temperature=1.0,
    aux_loss_weight=10.0,
    beta_max_vector=None  # Optional: numpy array of shape (n_features,)
):
    """
    Generates synthetic samples for a given class using the TabDDPM model.
    
    The class_data is normalized with MinMaxScaler, and a per-feature cosine noise schedule is computed.
    Optionally, if beta_max_vector is provided, it is used for the noise schedule; otherwise a fixed value is used.
    
    Returns:
      synthetic_samples: synthetic samples in the original scale.
      synthetic_samples_norm: synthetic samples in normalized [0,1] space.
    """
    scaler = MinMaxScaler()
    norm_class_data = scaler.fit_transform(class_data)
    n_features = norm_class_data.shape[1]
    
    model = build_tabddpm_model(n_features, weight_decay=weight_decay)
    optimizer = Adam(0.0005)
    
    # Use provided beta_max_vector or default to fixed value 0.01
    if beta_max_vector is None:
        beta_max = np.full(n_features, 0.01, dtype=np.float32)
    else:
        beta_max = beta_max_vector.astype(np.float32)
    
    # Create per-feature cosine noise schedule
    alpha_bar_f = np.zeros((diffusion_steps, n_features), dtype=np.float32)
    s = 0.008
    for i in range(n_features):
        for t in range(diffusion_steps):
            fraction = t / (diffusion_steps - 1)
            cosine_term = np.cos(((fraction + s) / (1.0 + s)) * (np.pi / 2.0)) ** 2
            alpha_bar_f[t, i] = (1.0 - beta_max[i]) + beta_max[i] * cosine_term
        alpha_bar_f[0, i] = 1.0
    alpha_f = np.zeros_like(alpha_bar_f)
    alpha_f[0, :] = 1.0
    for t in range(1, diffusion_steps):
        alpha_f[t] = alpha_bar_f[t] / alpha_bar_f[t - 1]
    
    alpha_bar_tf = tf.constant(alpha_bar_f, dtype=tf.float32)
    alpha_tf = tf.constant(alpha_f, dtype=tf.float32)
    
    X = norm_class_data
    best_loss = float('inf')
    patience = 0
    
    # Training loop
    for epoch in range(epochs):
        idx = np.random.randint(0, X.shape[0], batch_size)
        x_batch = X[idx]
        t_rand = np.random.randint(0, diffusion_steps, size=batch_size)
        noise_batch = np.random.normal(0, 1, x_batch.shape)
        
        x_batch_tf = tf.convert_to_tensor(x_batch, dtype=tf.float32)
        noise_tf = tf.convert_to_tensor(noise_batch, dtype=tf.float32)
        t_rand_tf = tf.convert_to_tensor(t_rand, dtype=tf.int32)
        
        with tf.GradientTape() as tape:
            alpha_bar_t = tf.gather(alpha_bar_tf, t_rand_tf, axis=0)
            alpha_t = tf.gather(alpha_tf, t_rand_tf, axis=0)
            
            sqrt_alpha_bar_t = tf.sqrt(alpha_bar_t)
            sqrt_one_minus_alpha_bar_t = tf.sqrt(1.0 - alpha_bar_t)
            sqrt_alpha_t = tf.sqrt(alpha_t)
            
            x_noisy = sqrt_alpha_bar_t * x_batch_tf + sqrt_one_minus_alpha_bar_t * noise_tf
            t_input_tf = tf.reshape(t_rand_tf, (-1, 1))
            predicted_noise = model([x_noisy, t_input_tf], training=True)
            
            mse_loss = tf.reduce_mean(tf.square(predicted_noise - noise_tf))
            x_denoised = (x_noisy - sqrt_one_minus_alpha_bar_t * predicted_noise) / sqrt_alpha_t
            
            lower_violation = tf.square(tf.maximum(0.0, -x_denoised))
            upper_violation = tf.square(tf.maximum(0.0, x_denoised - 1.0))
            aux_penalty = tf.reduce_mean(lower_violation + upper_violation)
            
            total_loss = mse_loss + aux_loss_weight * aux_penalty
        
        gradients = tape.gradient(total_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        
        if total_loss < best_loss:
            best_loss = total_loss
            patience = 0
        else:
            patience += 1
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{epochs} - Total Loss: {total_loss:.4f} - MSE: {mse_loss:.4f} - Aux: {aux_penalty:.4f}")
        
        if patience >= early_stopping_patience:
            print(f"Early stopping triggered at epoch {epoch} with loss {total_loss:.4f}")
            break
    
    # Generation phase: reverse diffusion starting from pure noise
    synthetic_samples_norm = []
    for _ in range(n_samples):
        x = np.random.normal(0, 1, (1, n_features)).astype(np.float32)
        for t in reversed(range(diffusion_steps)):
            alpha_bar_t = alpha_bar_f[t]  # shape (n_features,)
            alpha_t = alpha_f[t]          # shape (n_features,)
            sqrt_one_minus_alpha_bar_t = np.sqrt(1.0 - alpha_bar_t)
            sqrt_alpha_t = np.sqrt(alpha_t)
            
            t_value = np.array([[t]], dtype=np.float32)
            predicted_noise = model.predict([x, t_value])
            predicted_noise *= temperature
            x = (x - sqrt_one_minus_alpha_bar_t * predicted_noise) / sqrt_alpha_t
        synthetic_samples_norm.append(x.squeeze(axis=0))
    
    synthetic_samples_norm = np.array(synthetic_samples_norm)
    synthetic_samples = scaler.inverse_transform(synthetic_samples_norm)
    return synthetic_samples, synthetic_samples_norm

def augment_dataframe_tabddpm(
    df,
    target,
    test_size=0.2,
    random_state=42,
    n_classes_to_augment=2,
    ratio_limit=1.0,
    diminishing_factor=2.0,
    diffusion_epochs=100,
    diffusion_batch_size=32,
    diffusion_steps=10,
    weight_decay=1e-4,
    early_stopping_patience=20,
    temperature=1.0,
    aux_loss_weight=10.0,
    beta_max_vector=None  # Optional: custom per-feature beta_max vector computed via compute_beta_max_vector
):
    """
    Augments a DataFrame using the TabDDPM model.
    Splits the DataFrame into training and test sets, removes singleton classes,
    and generates synthetic samples for the smallest classes.
    Synthetic rows are flagged with a 'synthetic' marker.
    
    Returns:
      original_train: Original training data before augmentation.
      augmented_train: Training data after augmentation.
      test_set: Test dataset.
      out_of_bounds_flag: True if any normalized synthetic sample is outside [0,1].
    """
    train, test = train_test_split(df, test_size=test_size, random_state=random_state, stratify=df[target])
    original_train = train.copy()
    
    class_counts = train[target].value_counts()
    singleton_classes = class_counts[class_counts == 1].index.tolist()
    if singleton_classes:
        train = train[~train[target].isin(singleton_classes)]
        original_train = train.copy()
    
    counts = train[target].value_counts()
    largest_class_count = counts.max()
    
    classes_sorted = counts.sort_values().index.tolist()
    classes_to_augment = classes_sorted[:n_classes_to_augment]
    print("Classes to augment:", classes_to_augment)
    
    overall_flag = False
    synthetic_list = []
    
    for cls in classes_to_augment:
        current_count = counts[cls]
        desired_by_ratio = int(largest_class_count * ratio_limit)
        synthetic_needed = int((desired_by_ratio - current_count) * (current_count / largest_class_count)**diminishing_factor)
        if synthetic_needed <= 0:
            print(f"Class {cls}: No augmentation needed (current: {current_count}, desired: {desired_by_ratio})")
            continue
        
        print(f"Class {cls}: Generating {synthetic_needed} synthetic samples (current: {current_count}, desired: {desired_by_ratio})")
        class_data = train[train[target] == cls].drop(columns=[target]).values
        
        synthetic_samples, synthetic_samples_norm = generate_synthetic_samples_for_class_tabddpm(
            class_data,
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
        
        if np.any(synthetic_samples_norm < 0) or np.any(synthetic_samples_norm > 1):
            overall_flag = True
        
        synthetic_df = pd.DataFrame(synthetic_samples, columns=train.drop(columns=[target]).columns)
        synthetic_df[target] = cls
        synthetic_df['synthetic'] = True
        synthetic_list.append(synthetic_df)
    
    train['synthetic'] = False
    if synthetic_list:
        synthetic_df_all = pd.concat(synthetic_list, axis=0)
        augmented_train = pd.concat([train, synthetic_df_all], axis=0).reset_index(drop=True)
    else:
        augmented_train = train.copy()
    
    return original_train.reset_index(drop=True), augmented_train, test.reset_index(drop=True), overall_flag

if __name__ == "__main__":
    # Example usage:
    # Load your DataFrame, for example:
    # df = pd.read_csv("your_data.csv")
    # Scale your data (if not already scaled)
    # scaler = MinMaxScaler()
    # scaled_features = scaler.fit_transform(df.drop(columns=["class"]))
    # Compute beta_max_vector based on a chosen method (variance, iqr, or snr)
    # custom_beta_max = compute_beta_max_vector(scaled_features, method="variance", k=0.05)
    # Now, call the augmentation function with diffusion_steps=200 if desired:
    # orig_train, aug_train, test_set, flag = augment_dataframe_tabddpm(
    #     df, target="class", diffusion_steps=200, beta_max_vector=custom_beta_max
    # )
    print("TabDDPM-based synthetic data generation module (refined) loaded. Integrate into your pipeline.")
