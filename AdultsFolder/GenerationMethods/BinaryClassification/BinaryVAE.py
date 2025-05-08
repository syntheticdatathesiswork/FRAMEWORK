import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import traceback
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Concatenate, Dropout
from tensorflow.keras.layers import BatchNormalization, Layer
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping
import os
import warnings
warnings.filterwarnings('ignore')

# ======= VAE Model Functions =======

def match_feature_means(synthetic_df, original_df, features_to_match, target_col=None):
    """
    Adjust means of specified features in synthetic data to match original data.
    
    Parameters:
    -----------
    synthetic_df : pandas.DataFrame
        DataFrame containing synthetic data
    original_df : pandas.DataFrame
        DataFrame containing original data
    features_to_match : list
        List of feature names to match means for
    target_col : str, optional
        If provided, means will be matched class by class
        
    Returns:
    --------
    pandas.DataFrame
        Synthetic DataFrame with adjusted features
    """
    result_df = synthetic_df.copy()
    
    # If target_col is provided, match means class by class
    if target_col is not None and target_col in synthetic_df.columns and target_col in original_df.columns:
        for cls in synthetic_df[target_col].unique():
            # Get data for this class
            orig_class_data = original_df[original_df[target_col] == cls]
            syn_class_data = synthetic_df[synthetic_df[target_col] == cls]
            
            # Skip if not enough data
            if len(orig_class_data) < 5 or len(syn_class_data) < 5:
                continue
            
            # Get indices for this class in synthetic data
            syn_indices = syn_class_data.index
            
            # Match each feature
            for feature in features_to_match:
                if feature in synthetic_df.columns and feature in original_df.columns:
                    # Calculate means
                    orig_mean = orig_class_data[feature].mean()
                    syn_mean = syn_class_data[feature].mean()
                    
                    # Calculate the mean difference
                    mean_diff = orig_mean - syn_mean
                    
                    # Apply shift to match means
                    result_df.loc[syn_indices, feature] += mean_diff
                    
                    print(f"Class {cls}, Feature {feature}: Shifted by {mean_diff:.2f} " 
                          f"(Original mean: {orig_mean:.2f}, Synthetic mean before: {syn_mean:.2f}, "
                          f"Synthetic mean after: {result_df.loc[syn_indices, feature].mean():.2f})")
    else:
        # Match means globally
        for feature in features_to_match:
            if feature in synthetic_df.columns and feature in original_df.columns:
                # Calculate means
                orig_mean = original_df[feature].mean()
                syn_mean = synthetic_df[feature].mean()
                
                # Calculate the mean difference
                mean_diff = orig_mean - syn_mean
                
                # Apply shift to match means
                result_df[feature] += mean_diff
                
                print(f"Feature {feature}: Shifted by {mean_diff:.2f} "
                      f"(Original mean: {orig_mean:.2f}, Synthetic mean before: {syn_mean:.2f}, "
                      f"Synthetic mean after: {result_df[feature].mean():.2f})")
    
    return result_df

def apply_copula_matching(synthetic_df, original_df, target_col):
    """Match joint distributions using Gaussian copulas"""
    from scipy.stats import norm
    
    matched_df = synthetic_df.copy()
    feature_cols = [col for col in synthetic_df.columns 
                   if col != target_col and col != 'synthetic'
                   and pd.api.types.is_numeric_dtype(synthetic_df[col])]
    
    # Process the minority class only
    cls = 1  # Binary case: assume 1 is the minority class
    orig_class_df = original_df[original_df[target_col] == cls]
    syn_class_df = synthetic_df[synthetic_df[target_col] == cls]
    
    if len(orig_class_df) > 10 and len(syn_class_df) > 10:
        syn_indices = syn_class_df.index
        
        # 1. Transform original data to standard normal using empirical CDF
        orig_normal = pd.DataFrame(index=orig_class_df.index)
        for col in feature_cols:
            # Convert to ranks, then to normal quantiles
            ranks = orig_class_df[col].rank() / (len(orig_class_df) + 1)
            orig_normal[col] = norm.ppf(ranks)
        
        # 2. Compute correlation matrix of transformed original data
        orig_corr = orig_normal.corr()
        
        # 3. Transform synthetic data to standard normal
        syn_normal = pd.DataFrame(index=syn_class_df.index)
        for col in feature_cols:
            ranks = syn_class_df[col].rank() / (len(syn_class_df) + 1)
            syn_normal[col] = norm.ppf(ranks)
        
        # 4. Apply Cholesky decomposition to impose correlation structure
        L_orig = np.linalg.cholesky(np.clip(orig_corr.values, -0.999, 0.999))
        L_syn = np.linalg.cholesky(np.clip(syn_normal.corr().values, -0.999, 0.999))
        
        # 5. Transform synthetic data to match original correlation
        Z = syn_normal.values @ np.linalg.inv(L_syn)
        X_corr = Z @ L_orig
        
        # 6. Convert back to uniform, then to original scale
        for i, col in enumerate(feature_cols):
            unif = norm.cdf(X_corr[:, i])
            # Map uniform back to original scale using quantiles
            orig_quantiles = np.quantile(orig_class_df[col].dropna(), np.linspace(0, 1, 1000))
            matched_df.loc[syn_indices, col] = np.interp(unif, np.linspace(0, 1, 1000), orig_quantiles)
    
    return matched_df

def apply_quantile_matching(synthetic_data, original_data, columns):
    """Match the quantiles of synthetic data to original data"""
    matched_data = synthetic_data.copy()
    
    for col in columns:
        # Get synthetic data values and ranks
        syn_values = synthetic_data[col].values
        syn_ranks = synthetic_data[col].rank(pct=True).values
        
        # Create a quantile function from original data
        orig_quantiles = np.quantile(original_data[col].dropna(), np.linspace(0, 1, 1000))
        
        # Map each synthetic point to the corresponding quantile in original data
        matched_values = np.interp(syn_ranks, np.linspace(0, 1, 1000), orig_quantiles)
        matched_data[col] = matched_values
    
    return matched_data

def apply_moment_matching(synthetic_data, original_data, columns):
    """Match mean and standard deviation of synthetic data to original data"""
    matched_data = synthetic_data.copy()
    
    for col in columns:
        # Get statistics from original data
        orig_mean = original_data[col].mean()
        orig_std = original_data[col].std()
        
        # Get statistics from synthetic data
        syn_mean = synthetic_data[col].mean()
        syn_std = synthetic_data[col].std()
        
        # Apply transformation: first standardize, then rescale
        if syn_std > 0:
            matched_data[col] = ((synthetic_data[col] - syn_mean) / syn_std) * orig_std + orig_mean
    
    return matched_data

def feature_matching_loss(original_data, generated_data):
    """Compute the difference between feature statistics"""
    orig_mean = tf.reduce_mean(original_data, axis=0)
    gen_mean = tf.reduce_mean(generated_data, axis=0)
    mean_loss = tf.reduce_mean(tf.square(orig_mean - gen_mean))
    
    orig_std = tf.math.reduce_std(original_data, axis=0)
    gen_std = tf.math.reduce_std(generated_data, axis=0)
    std_loss = tf.reduce_mean(tf.square(orig_std - gen_std))
    
    return mean_loss + std_loss

class KLDivergenceLayer(Layer):
    def __init__(self, beta_initial=0.01, beta_final=1.0, beta_steps=50000, **kwargs):
        super(KLDivergenceLayer, self).__init__(**kwargs)
        self.beta_initial = beta_initial
        self.beta_final = beta_final
        self.beta_steps = beta_steps
        self.step = 0
        
    def call(self, inputs):
        z_mean, z_log_var = inputs
        kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
        kl_loss = -0.5 * tf.reduce_mean(kl_loss, axis=1)
        
        # More gradual annealing schedule
        if self.step < self.beta_steps:
            # Use sigmoid-like annealing for smoother transition
            progress = self.step / self.beta_steps
            beta = self.beta_initial + (self.beta_final - self.beta_initial) * (
                1 / (1 + np.exp(-10 * (progress - 0.5)))
            )
        else:
            beta = self.beta_final
        
        self.step += 1
        self.add_loss(beta * tf.reduce_mean(kl_loss))
        
        # Return the raw KL loss for monitoring
        return kl_loss

class SamplingLayer(Layer):
    """Sampling layer for VAE"""
    def __init__(self, **kwargs):
        super(SamplingLayer, self).__init__(**kwargs)
        
    def call(self, inputs):
        z_mean, z_log_var = inputs
        # Clip log_var for numerical stability
        z_log_var_clipped = tf.clip_by_value(z_log_var, -20, 20)
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var_clipped) * epsilon

def combined_loss(y_true, y_pred):
    # Increase the minimum loss threshold to prevent convergence
    reconst_loss = tf.reduce_mean(tf.square(y_true - y_pred))
    feature_match_loss = feature_matching_loss(y_true, y_pred)
    min_loss = 0.1  # Increase from 0.01 to 0.1
    total_loss = reconst_loss + 0.5 * feature_match_loss + min_loss
    return total_loss

def build_enhanced_vae(input_dim, latent_dim=32, hidden_dims=[256, 128, 64], 
                      condition_dim=None, dropout_rate=0.2, weight_decay=1e-5):
    """Build an enhanced VAE model with more capacity and better regularization"""
    
    # Define regularizer
    regularizer = regularizers.l2(weight_decay)
    
    # Define encoder inputs
    x_input = Input(shape=(input_dim,), name='encoder_input')
    
    # Handle conditional input if needed
    if condition_dim is not None:
        condition_input = Input(shape=(condition_dim,), name='condition_input')
        # Use embedding for condition to get better representation
        h = Concatenate()([x_input, condition_input])
    else:
        h = x_input
    
    # Encoder network with residual connections
    skip_connections = []
    for i, dim in enumerate(hidden_dims):
        h_prev = h
        h = Dense(dim, activation='selu', kernel_regularizer=regularizer, 
                 kernel_initializer='lecun_normal', name=f'encoder_dense_{i}')(h)
        h = BatchNormalization(name=f'encoder_bn_{i}')(h)
        h = Dropout(dropout_rate, name=f'encoder_dropout_{i}')(h)
        
        # Add skip connection if dimensions match
        if i > 0 and hidden_dims[i-1] == dim:
            h = tf.keras.layers.Add()([h, skip_connections[-1]])
        
        skip_connections.append(h)
    
    # Latent space parameters with proper initialization
    z_mean = Dense(latent_dim, name='z_mean', 
                  kernel_initializer=tf.keras.initializers.HeNormal())(h)
    z_log_var = Dense(latent_dim, name='z_log_var',
                     kernel_initializer=tf.keras.initializers.HeNormal())(h)
    
    # Add KL divergence loss with slower annealing
    kl_loss = KLDivergenceLayer(beta_initial=0.0, beta_final=1.0, beta_steps=10000, 
                               name='kl_loss')([z_mean, z_log_var])
    
    # Sampling layer
    z = SamplingLayer(name='z_sampling')([z_mean, z_log_var])
    
    # Prepare decoder input (add condition if needed)
    if condition_dim is not None:
        decoder_input = Concatenate(name='decoder_input')([z, condition_input])
    else:
        decoder_input = z
    
    # Decoder network with wider layers
    h_decoder = decoder_input
    decoder_skip_connections = []
    
    for i, dim in enumerate(reversed(hidden_dims)):
        h_decoder_prev = h_decoder
        h_decoder = Dense(dim, activation='selu', kernel_regularizer=regularizer,
                         kernel_initializer='lecun_normal', name=f'decoder_dense_{i}')(h_decoder)
        h_decoder = BatchNormalization(name=f'decoder_bn_{i}')(h_decoder)
        h_decoder = Dropout(dropout_rate, name=f'decoder_dropout_{i}')(h_decoder)
        
        # Add skip connection if dimensions match
        if i > 0 and list(reversed(hidden_dims))[i-1] == dim:
            h_decoder = tf.keras.layers.Add()([h_decoder, decoder_skip_connections[-1]])
        
        decoder_skip_connections.append(h_decoder)
    
    # Output layer - use sigmoid for bounded output
    decoder_output = Dense(input_dim, activation='sigmoid', name='decoder_output')(h_decoder)
    
    # Define models
    if condition_dim is not None:
        vae = Model([x_input, condition_input], decoder_output, name='vae')
        encoder = Model([x_input, condition_input], [z_mean, z_log_var, z], name='encoder')
        
        # Define decoder model separately
        latent_input = Input(shape=(latent_dim,), name='decoder_latent_input')
        condition_input_decoder = Input(shape=(condition_dim,), name='decoder_condition_input')
        decoder_inputs = Concatenate()([latent_input, condition_input_decoder])
        
        # Reuse decoder layers - rebuild the decoder part
        decoder_outputs = decoder_inputs
        for i, dim in enumerate(reversed(hidden_dims)):
            decoder_outputs = vae.get_layer(f'decoder_dense_{i}')(decoder_outputs)
            decoder_outputs = vae.get_layer(f'decoder_bn_{i}')(decoder_outputs)
            decoder_outputs = vae.get_layer(f'decoder_dropout_{i}')(decoder_outputs)
        decoder_outputs = vae.get_layer('decoder_output')(decoder_outputs)
        
        decoder = Model([latent_input, condition_input_decoder], decoder_outputs, name='decoder')
    else:
        vae = Model(x_input, decoder_output, name='vae')
        encoder = Model(x_input, [z_mean, z_log_var, z], name='encoder')
        
        # Decoder model
        latent_input = Input(shape=(latent_dim,), name='decoder_latent_input')
        decoder_outputs = latent_input
        for i, dim in enumerate(reversed(hidden_dims)):
            decoder_outputs = vae.get_layer(f'decoder_dense_{i}')(decoder_outputs)
            decoder_outputs = vae.get_layer(f'decoder_bn_{i}')(decoder_outputs)
            decoder_outputs = vae.get_layer(f'decoder_dropout_{i}')(decoder_outputs)
        decoder_outputs = vae.get_layer('decoder_output')(decoder_outputs)
        
        decoder = Model(latent_input, decoder_outputs, name='decoder')
    
    return encoder, decoder, vae

def train_enhanced_vae(data, labels=None, num_classes=None, batch_size=32, epochs=500, 
                      latent_dim=32, hidden_dims=[256, 128, 64], learning_rate=1e-3,
                      early_stopping_patience=50, validation_split=0.15, verbose=1):
    """Train an enhanced VAE model with better learning dynamics"""
    try:
        n_samples, n_features = data.shape
        is_conditional = labels is not None and num_classes is not None
        
        # For binary case, ensure num_classes is 2
        if is_conditional:
            num_classes = 2
            print(f"Using {num_classes} classes for binary classification task")
        
        # Build the enhanced VAE model
        encoder, decoder, vae = build_enhanced_vae(
            input_dim=n_features,
            latent_dim=latent_dim,
            hidden_dims=hidden_dims,
            condition_dim=num_classes if is_conditional else None,
            dropout_rate=0.2,
            weight_decay=1e-5
        )
        
        # Learning rate schedule for better convergence
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=1e-3,
            decay_steps=1000,
            decay_rate=0.98,
            staircase=True
        )
        
        # Compile the model with the enhanced loss function
        optimizer = Adam(learning_rate=learning_rate)
        vae.compile(optimizer=optimizer, loss=combined_loss)
        
        # Callbacks for better training
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=early_stopping_patience,
                restore_best_weights=True,
                verbose=verbose
            )
        ]
        
        # Train the model
        if is_conditional:
            # Convert labels to one-hot
            y_onehot = tf.keras.utils.to_categorical(labels, num_classes=num_classes)
            history = vae.fit(
                [data, y_onehot], data,
                batch_size=batch_size,
                epochs=epochs,
                validation_split=validation_split,
                callbacks=callbacks,
                verbose=verbose,
                shuffle=True
            )
        else:
            history = vae.fit(
                data, data,
                batch_size=batch_size,
                epochs=epochs,
                validation_split=validation_split,
                callbacks=callbacks,
                verbose=verbose,
                shuffle=True
            )
        
        # Load the best model weights if available
        try:
            vae.load_weights('best_vae_model.h5')
            print("Loaded best model weights from checkpoint")
        except:
            print("Could not load best model weights, using final weights")
        
        return {'encoder': encoder, 'decoder': decoder, 'vae': vae, 
                'history': history, 'num_classes': num_classes}
    
    except Exception as e:
        print(f"Error in training enhanced VAE model: {str(e)}")
        traceback.print_exc()
        raise

def generate_samples_with_distribution_matching(model_dict, original_data, n_samples, 
                                              class_label=1, num_classes=2, 
                                              temperature=0.8, matching_factor=0.3):
    """Generate samples from trained VAE with distribution matching for binary classification"""
    try:
        decoder = model_dict['decoder']
        encoder = model_dict['encoder']
        latent_dim = decoder.input_shape[0][1] if isinstance(decoder.input, list) else decoder.input_shape[1]
        
        # Use the num_classes from model_dict if available
        if 'num_classes' in model_dict:
            num_classes = model_dict['num_classes']
        
        # For binary classification, enforce num_classes=2
        num_classes = 2
            
        # Get statistics from original data's latent space - focus on minority class
        if isinstance(original_data, tuple) and len(original_data) == 2:
            data, labels = original_data
            # Filter to only use the minority class data (class_label=1)
            class_mask = labels == class_label
            class_data = data[class_mask]
        else:
            class_data = original_data
        
        # Generate one-hot for original data
        y_onehot_orig = np.zeros((len(class_data), num_classes))
        y_onehot_orig[:, class_label] = 1
        
        # Get latent representation of original data
        z_mean, z_log_var, _ = encoder.predict([class_data, y_onehot_orig])
        
        # Calculate mean and std of original latent space
        orig_latent_mean = np.mean(z_mean, axis=0)
        orig_latent_std = np.std(z_mean, axis=0)
        
        # Sample from latent space with temperature and distribution matching
        z_random = np.random.normal(0, temperature, size=(n_samples, latent_dim))
        # Mix random samples with the original distribution statistics
        z = (1 - matching_factor) * z_random + matching_factor * (
            orig_latent_mean + z_random * orig_latent_std
        )
        
        # Create one-hot encoding for generation - all samples are the minority class
        y_onehot = np.zeros((n_samples, num_classes))
        y_onehot[:, class_label] = 1
        
        # Generate with condition
        samples = decoder.predict([z, y_onehot])
        
        # Ensure output is clipped to valid range
        samples = np.clip(samples, 0, 1)
        
        return samples
        
    except Exception as e:
        print(f"Error in enhanced sample generation: {str(e)}")
        traceback.print_exc()
        return np.random.uniform(0, 1, (n_samples, decoder.output_shape[1]))

# ======= Main Augmentation Function for Binary Classification =======

def augment_binary_vae(
    df, 
    target,
    test_size=0.25, 
    random_state=42, 
    ratio_limit=0.5,
    vae_epochs=500,
    vae_batch_size=32,
    latent_dim=32,
    hidden_dims=[256, 128, 64],
    temperature=0.7,
    matching_factor=0.3,
    early_stopping_patience=50,
    output_dir='./output',
    save_results=True,
    plot_results=True
):
    """
    VAE-based augmentation specifically for binary classification, focused on augmenting the minority class.
    
    Parameters:
        df: DataFrame containing the pre-processed data
        target: Target column name (binary classification column, 0/1)
        test_size: Fraction of data to use for testing
        random_state: Random seed for reproducibility
        ratio_limit: Maximum ratio of synthetic to original samples (0.5 = max 50% synthetic)
        vae_epochs: Number of training epochs for VAE model
        vae_batch_size: Batch size for training
        latent_dim: Dimensionality of the latent space
        hidden_dims: List of hidden layer dimensions
        temperature: Temperature for sampling (lower = less random)
        matching_factor: Factor to control distribution matching (0-1, higher = closer match)
        early_stopping_patience: Patience for early stopping
        output_dir: Directory to save results
        save_results: Whether to save results to files
        plot_results: Whether to plot and save visualizations
        
    Returns:
        original_train: Original training data
        augmented_train: Augmented training data
        test_set: Test data
        success_flag: Boolean indicating success
    """
    if save_results:
        os.makedirs(output_dir, exist_ok=True)
    
    success_flag = False
    
    try:
        print("Starting VAE-based augmentation for binary classification...")
        
        # Ensure Pandas DataFrame
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")
        
        # Validate target column
        if target not in df.columns:
            raise ValueError(f"Target column '{target}' not found in DataFrame")
        
        # Ensure target has binary values (0/1)
        unique_values = df[target].unique()
        if len(unique_values) != 2 or not all(val in [0, 1] for val in unique_values):
            print(f"Warning: Target column should contain binary values (0/1). Found: {unique_values}")
            # Convert to 0/1 if needed
            if len(unique_values) == 2:
                mapping = {unique_values[0]: 0, unique_values[1]: 1}
                df[target] = df[target].map(mapping)
                print(f"Converted values to 0/1 using mapping: {mapping}")
        
        # Define feature columns
        feature_cols = [col for col in df.columns if col != target]
        X = df[feature_cols].values
        y = df[target].values
        
        # Binary classification settings
        num_classes = 2
        
        # Split data into train and test sets with stratification
        train_data, test_data, train_labels, test_labels = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Scale data to [0, 1] range for VAE
        scaler = MinMaxScaler()
        train_data_scaled = scaler.fit_transform(train_data)
        test_data_scaled = scaler.transform(test_data)
        
        print(f"Training data shape: {train_data_scaled.shape}")
        print(f"Test data shape: {test_data_scaled.shape}")
        
        # Get class distribution
        unique_classes, class_counts = np.unique(train_labels, return_counts=True)
        class_distribution = dict(zip(unique_classes, class_counts))
        print(f"Class distribution in training data: {class_distribution}")
        
        # Identify minority class
        minority_class = 1  # In binary classification, assume 1 is positive/minority class
        majority_class = 0
        
        if class_counts[0] < class_counts[1]:
            print("Note: Class 0 has fewer samples than class 1. Treating class 0 as minority.")
            minority_class = 0
            majority_class = 1
        
        # Calculate minority and majority class counts
        minority_count = class_counts[minority_class]
        majority_count = class_counts[majority_class]
        
        print(f"Minority class ({minority_class}): {minority_count} samples")
        print(f"Majority class ({majority_class}): {majority_count} samples")
        print(f"Imbalance ratio: {majority_count/minority_count:.2f}:1")
        
        # Train the enhanced VAE model on the entire dataset
        print("Training VAE model...")
        vae_results = train_enhanced_vae(
            data=train_data_scaled,
            labels=train_labels,
            num_classes=num_classes,
            batch_size=vae_batch_size,
            epochs=vae_epochs,
            latent_dim=latent_dim,
            hidden_dims=hidden_dims,
            learning_rate=5e-5,
            early_stopping_patience=early_stopping_patience,
            validation_split=0.15,
            verbose=1
        )
        
        # Plot training history
        if plot_results and hasattr(vae_results['history'], 'history') and 'loss' in vae_results['history'].history:
            plt.figure(figsize=(10, 5))
            plt.plot(vae_results['history'].history['loss'], label='Training Loss')
            if 'val_loss' in vae_results['history'].history:
                plt.plot(vae_results['history'].history['val_loss'], label='Validation Loss')
            plt.title('VAE Model Training History')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.savefig(f'{output_dir}/vae_training_history.png')
            plt.close()
        
        # Calculate number of synthetic samples to generate for minority class
        # Goal: balance the classes while respecting ratio_limit
        max_synthetic = int(minority_count * ratio_limit)
        class_diff = majority_count - minority_count
        synthetic_needed = min(class_diff, max_synthetic)
        
        print(f"Generating {synthetic_needed} synthetic samples for minority class {minority_class}")
        print(f"  - Current minority count: {minority_count}")
        print(f"  - Current majority count: {majority_count}")
        print(f"  - Difference: {class_diff}")
        print(f"  - Limited by ratio constraint: max {max_synthetic} new samples")
        print(f"  - Target minority count after augmentation: {minority_count + synthetic_needed}")
        
        # Create minority class subset for distribution matching
        minority_mask = train_labels == minority_class
        minority_data = train_data_scaled[minority_mask]
        
        # Generate synthetic samples for minority class
        synthetic_samples_norm = generate_samples_with_distribution_matching(
            model_dict=vae_results,
            original_data=(train_data_scaled, train_labels),  # Pass full dataset with labels
            n_samples=synthetic_needed,
            class_label=minority_class,
            num_classes=num_classes,
            temperature=temperature,
            matching_factor=matching_factor
        )
        
        # Ensure values are in [0, 1] range
        synthetic_samples_norm = np.clip(synthetic_samples_norm, 0, 1)
        
        # Prepare original training data DataFrame
        original_train_df = pd.DataFrame(scaler.inverse_transform(train_data_scaled), columns=feature_cols)
        original_train_df[target] = train_labels
        original_train_df['synthetic'] = False
        
        # Prepare test data DataFrame
        test_df = pd.DataFrame(scaler.inverse_transform(test_data_scaled), columns=feature_cols)
        test_df[target] = test_labels
        
        # Process synthetic samples
        if synthetic_needed > 0:
            print(f"Processing {synthetic_needed} synthetic samples for minority class...")
            
            # Debug information about the synthetic samples
            print(f"Debug - Synthetic samples shape before inverse transform: {synthetic_samples_norm.shape}")
            print(f"Debug - Synthetic samples range: [{np.min(synthetic_samples_norm)}, {np.max(synthetic_samples_norm)}]")
            
            # Ensure values are in [0, 1] range before inverse transform
            synthetic_samples_norm = np.clip(synthetic_samples_norm, 0, 1)
            
            # Apply additional smoothing to make distributions more natural
            if synthetic_samples_norm.shape[0] > 10:
                # Add small amount of noise to break up artificial clusters
                noise_level = 0.02  # Small noise to avoid major distribution changes
                noise = np.random.normal(0, noise_level, synthetic_samples_norm.shape)
                synthetic_samples_norm = np.clip(synthetic_samples_norm + noise, 0, 1)
            
            # Inverse-transform to original scale
            try:
                synthetic_samples = scaler.inverse_transform(synthetic_samples_norm)
                
                # Check for NaN or Inf values and replace with reasonable values
                synthetic_samples = np.nan_to_num(synthetic_samples, nan=0.0, posinf=1e5, neginf=-1e5)
                
                # Create synthetic DataFrame
                synthetic_df = pd.DataFrame(synthetic_samples, columns=feature_cols)
                
                # All synthetic samples are from the minority class
                synthetic_df[target] = minority_class
                synthetic_df['synthetic'] = True
                
                # Apply quantile matching to ensure realistic distributions
                print("Applying quantile matching to synthetic samples...")
                for col in feature_cols:
                    # Get synthetic data values and ranks
                    syn_values = synthetic_df[col].values
                    syn_ranks = synthetic_df[col].rank(pct=True).values
                    
                    # Create a quantile function from original minority class data
                    minority_df = original_train_df[original_train_df[target] == minority_class]
                    orig_values = minority_df[col].dropna().values
                    
                    if len(orig_values) > 0:
                        orig_quantiles = np.quantile(orig_values, np.linspace(0, 1, 1000))
                        
                        # Map each synthetic point to the corresponding quantile in original data
                        matched_values = np.interp(syn_ranks, np.linspace(0, 1, 1000), orig_quantiles)
                        synthetic_df[col] = matched_values
                
                # Combine original and synthetic data
                augmented_train_df = pd.concat([original_train_df, synthetic_df], axis=0, ignore_index=True)
                
                # Print DataFrame info for debugging
                print("Original DataFrame info:")
                print(original_train_df.info())
                print("\nSynthetic DataFrame info:")
                print(synthetic_df.info())
                print("\nAugmented DataFrame info:")
                print(augmented_train_df.info())
                
                # Print class distribution after augmentation
                print("\nClass distribution after augmentation:")
                aug_class_counts = augmented_train_df.groupby(target).size()
                for cls, count in aug_class_counts.items():
                    orig_count = class_distribution.get(cls, 0)
                    print(f"  Class {cls}: {count} samples (was {orig_count})")
                
                print(f"\nAugmented data summary:")
                print(f"  - Original samples: {len(original_train_df)}")
                print(f"  - Synthetic minority samples: {len(synthetic_df)}")
                print(f"  - Total samples: {len(augmented_train_df)}")
                print(f"  - New class ratio: 1:{aug_class_counts[majority_class]/aug_class_counts[minority_class]:.2f}")
                
            except Exception as e:
                print(f"Error during inverse transform or DataFrame creation: {str(e)}")
                print("Falling back to original data only")
                augmented_train_df = original_train_df.copy()
        else:
            # No synthetic samples needed
            print("No synthetic samples were generated.")
            augmented_train_df = original_train_df.copy()
        
        # Save results
        if save_results:
            original_train_df.to_csv(f"{output_dir}/vae_original_train.csv", index=False)
            augmented_train_df.to_csv(f"{output_dir}/vae_augmented_train.csv", index=False)
            test_df.to_csv(f"{output_dir}/vae_test_set.csv", index=False)
            
            print(f"Saved results to {output_dir}/")
        
        # Enhanced visualizations
        if plot_results and 'synthetic_df' in locals() and len(synthetic_df) > 0:
            # Plot class distribution before and after
            plt.figure(figsize=(10, 6))
            
            # Before augmentation
            before_counts = [class_distribution.get(0, 0), class_distribution.get(1, 0)]
            
            # After augmentation
            after_counts = [
                augmented_train_df[augmented_train_df[target] == 0].shape[0],
                augmented_train_df[augmented_train_df[target] == 1].shape[0]
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
            plt.savefig(f'{output_dir}/class_distribution.png')
            plt.close()
            
            # Plot feature distributions with KDE for smoother visualization
            for feature in feature_cols:
                plt.figure(figsize=(12, 6))
                
                # Get minority class data only
                orig_minority = original_train_df[original_train_df[target] == minority_class]
                
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
                plt.savefig(f'{output_dir}/vae_{feature}_distribution.png')
                plt.close()
            
            # Plot correlation heatmaps
            plt.figure(figsize=(12, 10))
            
            # Extract only numeric features from minority class
            numeric_features = orig_minority.select_dtypes(include=['number']).columns
            numeric_features = [col for col in numeric_features if col != target and col != 'synthetic']
            
            if len(numeric_features) > 0:
                plt.subplot(1, 2, 1)
                orig_corr = orig_minority[numeric_features].corr()
                sns.heatmap(orig_corr, annot=False, cmap='coolwarm')
                plt.title('Original Minority Data Correlation')
                
                plt.subplot(1, 2, 2)
                synth_corr = synthetic_df[numeric_features].corr()
                sns.heatmap(synth_corr, annot=False, cmap='coolwarm')
                plt.title('Synthetic Minority Data Correlation')
                
                plt.tight_layout()
                plt.savefig(f'{output_dir}/vae_correlation_comparison.png')
                plt.close()
                
                # Plot correlation difference to identify discrepancies
                plt.figure(figsize=(10, 8))
                corr_diff = abs(orig_corr - synth_corr)
                sns.heatmap(corr_diff, annot=True, cmap='Reds')
                plt.title('Correlation Matrix Absolute Difference (Original vs Synthetic)')
                plt.tight_layout()
                plt.savefig(f'{output_dir}/vae_correlation_difference.png')
                plt.close()
            
            # Plot 2D visualizations of feature pairs if possible
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
                plt.savefig(f'{output_dir}/vae_2d_feature_comparison.png')
                plt.close()
        
        success_flag = True
        return original_train_df, augmented_train_df, test_df, success_flag
        
    except Exception as e:
        print(f"Error in VAE binary augmentation: {str(e)}")
        traceback.print_exc()
        if not 'original_train_df' in locals():
            original_train_df = pd.DataFrame()
        if not 'augmented_train_df' in locals():
            augmented_train_df = pd.DataFrame()
        if not 'test_df' in locals():
            test_df = pd.DataFrame()
        
        return original_train_df, augmented_train_df, test_df, False


# Example usage
if __name__ == "__main__":
    # Load your pre-processed dataset
    # This assumes you already have a clean DataFrame with binary classification target
    df = pd.read_csv('your_binary_dataset.csv')
    
    # Set target column (must be binary 0/1)
    target_column = 'your_target_column'
    
    # Run the VAE augmentation for binary classification
    original_train, augmented_train, test_set, success = augment_binary_vae(
        df=df, 
        target=target_column,
        test_size=0.25,
        random_state=42,
        ratio_limit=0.5,
        vae_epochs=500,
        vae_batch_size=32,
        latent_dim=32,
        hidden_dims=[256, 128, 64],
        temperature=0.7,
        matching_factor=0.3,
        early_stopping_patience=50,
        output_dir='./binary_vae_output'
    )
    
    if success:
        print("Binary augmentation completed successfully!")
        print(f"Original training samples: {len(original_train)}")
        print(f"Augmented training samples: {len(augmented_train)}")
        print(f"Test samples: {len(test_set)}")
        
        # Get class distribution after augmentation
        aug_distribution = augmented_train.groupby(target_column).size()
        print("Final class distribution:")
        for cls, count in aug_distribution.items():
            print(f"  Class {cls}: {count} samples ({count/len(augmented_train)*100:.1f}%)")
        
        # Save final output
        augmented_train.to_csv("./binary_vae_output/final_augmented_dataset.csv", index=False)
    else:
        print("Augmentation failed. Check the error messages above.")