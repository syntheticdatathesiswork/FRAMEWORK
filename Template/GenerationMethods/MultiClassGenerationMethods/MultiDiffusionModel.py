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

def augment_dataframe_diffusion(
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
    weight_decay=1e-6,
    early_stopping_patience=20,
    temperature=1.0,
    aux_loss_weight=10.0,
    beta_max_vector=None
):
    """
    DataFrame Augmentation Using the Diffusion Model (with per-feature schedule)
    ----------------------------------------------------------------------------
    Splits the dataframe into training and test sets, removes singleton classes, 
    and generates synthetic samples for the smallest classes using the diffusion model.
    
    Checks if any synthetic normalized sample drifts outside the [0,1] range.

    Returns:
      original_train: Original training data before augmentation.
      augmented_train: Training data after diffusion-based augmentation (with a 'synthetic' flag).
      test_set: The test dataset.
      out_of_bounds_flag: True if any synthetic normalized output is outside [0,1], else False.
    """
    # Split the data
    train, test = train_test_split(df, test_size=test_size, random_state=random_state, stratify=df[target])
    original_train = train.copy()

    # Remove singleton classes
    class_counts = train[target].value_counts()
    singleton_classes = class_counts[class_counts == 1].index.tolist()
    if singleton_classes:
        train = train[~train[target].isin(singleton_classes)]
        original_train = train.copy()

    counts = train[target].value_counts()
    largest_class_count = counts.max()

    # Select the smallest classes to augment
    classes_sorted = counts.sort_values().index.tolist()
    classes_to_augment = classes_sorted[:n_classes_to_augment]
    print("Classes to augment:", classes_to_augment)

    overall_flag = False  # Flag for out-of-bound normalized outputs
    synthetic_list = []

    for cls in classes_to_augment:
        current_count = counts[cls]
        desired_by_ratio = int(largest_class_count * ratio_limit)
        synthetic_needed = int(
            (desired_by_ratio - current_count) *
            (current_count / largest_class_count)**diminishing_factor
        )
        if synthetic_needed <= 0:
            print(f"Class {cls}: No augmentation needed (current: {current_count}, desired_by_ratio: {desired_by_ratio})")
            continue

        print(f"Class {cls}: Generating {synthetic_needed} synthetic samples (current count: {current_count}, desired_by_ratio: {desired_by_ratio})")

        # Get class training data (features only)
        class_data = train[train[target] == cls].drop(columns=[target]).values

        # Generate synthetic samples, passing the beta_max_vector if provided.
        synthetic_samples, synthetic_samples_norm = generate_synthetic_samples_for_class_diffusion(
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

        # Check if any normalized synthetic values are out-of-bound
        if np.any(synthetic_samples_norm < 0) or np.any(synthetic_samples_norm > 1):
            overall_flag = True

        synthetic_df = pd.DataFrame(
            synthetic_samples,
            columns=train.drop(columns=[target]).columns
        )
        synthetic_df[target] = cls
        synthetic_df['synthetic'] = True
        synthetic_list.append(synthetic_df)

    # Flag original rows as non-synthetic
    train['synthetic'] = False

    if synthetic_list:
        synthetic_df_all = pd.concat(synthetic_list, axis=0)
        augmented_train = pd.concat([train, synthetic_df_all], axis=0).reset_index(drop=True)
    else:
        augmented_train = train.copy()

    return original_train.reset_index(drop=True), augmented_train, test.reset_index(drop=True), overall_flag
