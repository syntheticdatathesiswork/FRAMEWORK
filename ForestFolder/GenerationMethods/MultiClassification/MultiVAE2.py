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

# Integration code - add this to your augment_dataframe_vae_enhanced function
# Right after creating synthetic_df but before applying other matching techniques:

def apply_copula_matching(synthetic_df, original_df, target_col='Cover_Type'):
    """Match joint distributions using Gaussian copulas"""
    from scipy.stats import norm
    
    matched_df = synthetic_df.copy()
    feature_cols = [col for col in synthetic_df.columns 
                   if col != target_col and col != 'synthetic'
                   and pd.api.types.is_numeric_dtype(synthetic_df[col])]
    
    # Process each class separately
    for cls in synthetic_df[target_col].unique():
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

def apply_class_conditional_matching(synthetic_df, original_df, target_col='Cover_Type'):
    """Match distributions separately for each class"""
    matched_df = synthetic_df.copy()
    feature_cols = [col for col in synthetic_df.columns 
                   if col != target_col and col != 'synthetic' 
                   and pd.api.types.is_numeric_dtype(synthetic_df[col])]
    
    # Process each class separately
    for cls in synthetic_df[target_col].unique():
        # Get samples for this class
        orig_class_df = original_df[original_df[target_col] == cls]
        syn_class_df = synthetic_df[synthetic_df[target_col] == cls]
        
        if len(orig_class_df) > 10 and len(syn_class_df) > 10:
            # Get indices for this class in the synthetic data
            syn_indices = syn_class_df.index
            
            # Perform matching for this class
            class_matched = apply_quantile_matching(syn_class_df, orig_class_df, feature_cols)
            
            # Update the matched dataframe
            matched_df.loc[syn_indices, feature_cols] = class_matched[feature_cols].values
    
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
                      latent_dim=32, hidden_dims=[256, 128, 64], learning_rate=1e-3,  # Changed from 5e-5
                      early_stopping_patience=50, validation_split=0.15, verbose=1):
    """Train an enhanced VAE model with better learning dynamics"""
    try:
        n_samples, n_features = data.shape
        is_conditional = labels is not None and num_classes is not None
        
        # Ensure num_classes is large enough for the labels
        if is_conditional:
            max_label = int(np.max(labels))
            # Adjust num_classes if needed
            num_classes = max(num_classes, max_label + 1)
            print(f"Using {num_classes} classes for one-hot encoding (max label: {max_label})")
        
        # Build the enhanced VAE model
        encoder, decoder, vae = build_enhanced_vae(
            input_dim=n_features,
            latent_dim=latent_dim,
            hidden_dims=hidden_dims,
            condition_dim=num_classes if is_conditional else None,
            dropout_rate=0.2,  # Increased dropout
            weight_decay=1e-5  # Adjusted weight decay
        )
        
        # Learning rate schedule for better convergence
        # Replace the current lr_schedule with this:
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=1e-3,  # Much higher starting rate
            decay_steps=1000,            # Slower decay
            decay_rate=0.98,             # Much slower decay rate
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
            )#,
            # Add learning rate reduction on plateau
            #tf.keras.callbacks.ReduceLROnPlateau(
            #    monitor='val_loss',
            #    factor=0.5,
            #    patience=10,
            #    min_lr=1e-6,
            #    verbose=verbose
            #),
            # Add model checkpoint
            #tf.keras.callbacks.ModelCheckpoint(
            #    filepath='best_vae_model.h5',
            #    monitor='val_loss',
            #    save_best_only=True,
            #    verbose=0
            #)
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
                shuffle=True  # Ensure good shuffling
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
        
        # Load the best model weights
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
                                              class_label=None, num_classes=None, 
                                              temperature=0.8, matching_factor=0.3):
    """Generate samples from trained VAE with distribution matching"""
    try:
        decoder = model_dict['decoder']
        encoder = model_dict['encoder']
        latent_dim = decoder.input_shape[0][1] if isinstance(decoder.input, list) else decoder.input_shape[1]
        
        # Use the num_classes from model_dict if available
        if 'num_classes' in model_dict:
            num_classes = model_dict['num_classes']
            
        # Get statistics from original data's latent space
        if class_label is not None and num_classes is not None:
            # Filter original data by class if needed
            if isinstance(original_data, tuple) and len(original_data) == 2:
                data, labels = original_data
                if isinstance(class_label, (int, np.integer)):
                    class_mask = labels == class_label
                    class_data = data[class_mask]
                else:
                    # Handle multiple classes
                    class_mask = np.isin(labels, class_label)
                    class_data = data[class_mask]
            else:
                class_data = original_data
            
            # Generate one-hot for original data
            if isinstance(class_label, (int, np.integer)):
                y_onehot_orig = np.zeros((len(class_data), num_classes))
                y_onehot_orig[:, class_label] = 1
            else:
                # Multiple classes
                if 'labels' in locals():
                    orig_labels = labels[class_mask]
                    y_onehot_orig = tf.keras.utils.to_categorical(
                        orig_labels, num_classes=num_classes
                    )
                else:
                    # Create a placeholder if we don't have labels
                    y_onehot_orig = np.zeros((len(class_data), num_classes))
                    if isinstance(class_label, (list, np.ndarray)):
                        for cl in class_label:
                            y_onehot_orig[:, cl] = 1/len(class_label)
                    else:
                        y_onehot_orig[:, class_label] = 1
            
            # Get latent representation of original data
            z_mean, z_log_var, _ = encoder.predict([class_data, y_onehot_orig])
        else:
            # Get latent representation without class conditioning
            z_mean, z_log_var, _ = encoder.predict(original_data)
        
        # Calculate mean and std of original latent space
        orig_latent_mean = np.mean(z_mean, axis=0)
        orig_latent_std = np.std(z_mean, axis=0)
        
        # Sample from latent space with temperature and distribution matching
        z_random = np.random.normal(0, temperature, size=(n_samples, latent_dim))
        # Mix random samples with the original distribution statistics
        z = (1 - matching_factor) * z_random + matching_factor * (
            orig_latent_mean + z_random * orig_latent_std
        )
        
        # Generate samples
        if class_label is not None and num_classes is not None:
            # Create one-hot encoding for generation
            if isinstance(class_label, (int, np.integer)):
                y_onehot = np.zeros((n_samples, num_classes))
                y_onehot[:, class_label] = 1
            else:
                # Handle array of labels
                class_labels = np.array([cl % num_classes if cl >= num_classes else cl for cl in class_label])
                y_onehot = tf.keras.utils.to_categorical(
                    class_labels, num_classes=num_classes
                )
                if len(y_onehot.shape) == 1:
                    y_onehot = np.tile(y_onehot, (n_samples, 1))
                elif y_onehot.shape[0] < n_samples:
                    repetitions = int(np.ceil(n_samples / y_onehot.shape[0]))
                    y_onehot = np.tile(y_onehot, (repetitions, 1))[:n_samples]
            
            # Generate with condition
            samples = decoder.predict([z, y_onehot])
        else:
            # Generate without condition
            samples = decoder.predict(z)
        
        # Ensure output is clipped to valid range
        samples = np.clip(samples, 0, 1)
        
        return samples
        
    except Exception as e:
        print(f"Error in enhanced sample generation: {str(e)}")
        traceback.print_exc()
        return np.random.uniform(0, 1, (n_samples, decoder.output_shape[1]))

# ======= Main Augmentation Function =======

# Example usage
if __name__ == "__main__":
    # Load your pre-processed dataset
    # This assumes you already have a clean DataFrame ready for augmentation
    df = pd.read_csv('your_preprocessed_dataset.csv')
    
    # Set target column
    target_column = 'your_target_column'
    
    # Run the enhanced VAE augmentation
    original_train, augmented_train, test_set, success = augment_dataframe_vae_enhanced(
        df=df, 
        target=target_column,
        test_size=0.25,
        random_state=42,
        n_classes_to_augment=4,
        ratio_limit=0.5,
        diminishing_factor=0.65,
        vae_epochs=500,
        vae_batch_size=32,
        latent_dim=32,
        hidden_dims=[256, 128, 64],
        temperature=0.7,
        matching_factor=0.3,
        early_stopping_patience=50,
        output_dir='./enhanced_vae_output'
    )
    
    if success:
        print("Enhanced augmentation completed successfully!")
        print(f"Original training samples: {len(original_train)}")
        print(f"Augmented training samples: {len(augmented_train)}")
        print(f"Test samples: {len(test_set)}")
        
        # Get and round numeric features
        numeric_features = augmented_train.select_dtypes(include=['number']).columns
        
        # Exclude the target and synthetic columns if they exist and are numeric
        columns_to_exclude = []
        if target_column in numeric_features:
            columns_to_exclude.append(target_column)
        if 'synthetic' in numeric_features:
            columns_to_exclude.append('synthetic')
            
        # Filter numeric features to exclude certain columns
        if columns_to_exclude:
            numeric_features = [col for col in numeric_features if col not in columns_to_exclude]
        
        # Round numeric features
        if numeric_features:
            augmented_train[numeric_features] = augmented_train[numeric_features].round(2)
        
        # Save outputs
        augmented_train.to_csv("./enhanced_vae_output/final_augmented_dataset.csv", index=False)
    else:
        print("Augmentation failed. Check the error messages above.")

def augment_dataframe_vae_enhanced(
    df, 
    target=None,
    test_size=0.25, 
    random_state=42, 
    n_classes_to_augment=4, 
    ratio_limit=0.5,
    diminishing_factor=0.65,
    vae_epochs=500,           # Increased from 200
    vae_batch_size=32,        # Smaller batch size for better learning
    latent_dim=32,            # Increased from 16
    hidden_dims=[256, 128, 64], # Larger network
    temperature=0.7,          # Slightly reduced temperature
    matching_factor=0.3,      # New parameter for distribution matching
    early_stopping_patience=50, # Increased from 30
    output_dir='./output',
    save_results=True,
    plot_results=True
):
    """
    Enhanced version of the VAE-based augmentation function with better distribution matching.
    
    Parameters:
        df: DataFrame containing the pre-processed data
        target: Target column name (for classification)
        test_size: Fraction of data to use for testing
        random_state: Random seed for reproducibility
        n_classes_to_augment: Number of classes to augment (focusing on minority classes)
        ratio_limit: Maximum ratio of synthetic to original samples (0.5 = max 50% synthetic)
        diminishing_factor: Factor to reduce synthetic samples for larger classes (0-1)
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
        print("Starting enhanced VAE-based augmentation...")
        
        # Ensure Pandas DataFrame
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")
        
        # Define feature columns
        if target is None:
            # Assume regression task without a specific target
            feature_cols = df.columns.tolist()
            X = df.values
            y = None
            num_classes = None
        else:
            # Classification with target column
            if target not in df.columns:
                raise ValueError(f"Target column '{target}' not found in DataFrame")
            
            feature_cols = [col for col in df.columns if col != target]
            X = df[feature_cols].values
            y = df[target].values
            # Get information about classes
            num_classes = len(np.unique(y))
            max_label = int(np.max(y))
            min_label = int(np.min(y))
            print(f"Target values range: [{min_label}, {max_label}], unique classes: {num_classes}")
            if max_label >= num_classes:
                print(f"Warning: max_label {max_label} >= num_classes {num_classes}, adjusting num_classes to {max_label + 1}")
                num_classes = max_label + 1
        
        # Split data into train and test sets
        if y is None:
            # Regression or unsupervised case
            train_data, test_data = train_test_split(
                X, test_size=test_size, random_state=random_state
            )
            train_labels = None
            test_labels = None
        else:
            # Classification case
            train_data, test_data, train_labels, test_labels = train_test_split(
                X, y, test_size=test_size, random_state=random_state, 
                stratify=y if len(np.unique(y)) > 1 else None
            )
        
        # Scale data to [0, 1] range for VAE
        scaler = MinMaxScaler()
        train_data_scaled = scaler.fit_transform(train_data)
        test_data_scaled = scaler.transform(test_data)
        
        print(f"Training data shape: {train_data_scaled.shape}")
        print(f"Test data shape: {test_data_scaled.shape}")
        
        # Train the enhanced VAE model
        print("Training enhanced VAE model...")
        vae_results = train_enhanced_vae(
            data=train_data_scaled,
            labels=train_labels,
            num_classes=num_classes,
            batch_size=vae_batch_size,
            epochs=vae_epochs,
            latent_dim=latent_dim,
            hidden_dims=hidden_dims,
            learning_rate=5e-5,  # Lower learning rate for better convergence
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
        
        # Generate synthetic samples with improved method
        synthetic_samples_all = []
        synthetic_labels_all = []
        
        if train_labels is not None:
            # Classification: augment specific classes
            unique_classes, class_counts = np.unique(train_labels, return_counts=True)
            
            # Create a sorted list of classes by count (smallest first)
            class_count_pairs = list(zip(unique_classes, class_counts))
            class_count_pairs.sort(key=lambda x: x[1])  # Sort by count (ascending)
            
            # Get the n smallest classes
            n_classes_to_augment = min(n_classes_to_augment, len(unique_classes))
            classes_to_augment = [cls for cls, _ in class_count_pairs[:n_classes_to_augment]]
            
            print(f"Class distribution: {dict(zip(unique_classes, class_counts))}")
            print(f"Classes to augment (from smallest to largest): {classes_to_augment}")
            
            # Check for class index range issues
            max_class_idx = max(unique_classes)
            if max_class_idx >= num_classes:
                print(f"Warning: max class index {max_class_idx} >= num_classes {num_classes}")
                num_classes = max_class_idx + 1
            
            # For tracking the augmentation plan
            augmentation_plan = []
            
            # Process each class to be augmented
            for idx, (cls, count) in enumerate(class_count_pairs[:n_classes_to_augment]):
                current_count = count
                
                # Apply diminishing factor
                current_diminishing_factor = diminishing_factor ** idx
                
                # Calculate maximum allowed synthetic samples
                max_synthetic = int(current_count * ratio_limit)
                
                # Find largest class as reference
                largest_class_count = class_count_pairs[-1][1]
                
                # Calculate balanced target count
                theoretical_target = int(largest_class_count * current_diminishing_factor)
                
                # Determine target count
                target_count = min(theoretical_target, largest_class_count)
                
                # Calculate synthetic samples needed, respecting the ratio_limit
                synthetic_needed_raw = target_count - current_count
                synthetic_needed = min(synthetic_needed_raw, max_synthetic)
                synthetic_needed = max(0, synthetic_needed)  # Ensure non-negative
                
                actual_target = current_count + synthetic_needed
                
                # Track the augmentation plan
                augmentation_plan.append({
                    'class': cls,
                    'current_count': current_count,
                    'diminishing_factor': current_diminishing_factor,
                    'max_allowed_by_ratio': max_synthetic,
                    'balanced_target': theoretical_target,
                    'final_target': target_count,
                    'synthetic_needed': synthetic_needed
                })
                
                if synthetic_needed > 0:
                    print(f"Class {cls}: Generating {synthetic_needed} synthetic samples")
                    print(f"  - Current count: {current_count}")
                    print(f"  - Diminishing factor: {current_diminishing_factor:.2f}")
                    print(f"  - Theoretical target count: {theoretical_target}")
                    print(f"  - Limited by ratio constraint: max {max_synthetic} new samples")
                    print(f"  - Final target count: {actual_target}")
                    
                    # Create class-specific subset for distribution matching
                    class_mask = train_labels == cls
                    class_data = train_data_scaled[class_mask]
                    
                    # Generate synthetic samples with distribution matching
                    synthetic_samples_norm = generate_samples_with_distribution_matching(
                        model_dict=vae_results,
                        original_data=class_data,  # Pass original data of this class
                        n_samples=synthetic_needed,
                        class_label=cls,
                        num_classes=num_classes,
                        temperature=temperature,
                        matching_factor=matching_factor  # Distribution matching parameter
                    )
                    
                    # Ensure values are in [0, 1] range
                    synthetic_samples_norm = np.clip(synthetic_samples_norm, 0, 1)
                    
                    # Store generated samples
                    synthetic_samples_all.append(synthetic_samples_norm)
                    synthetic_labels_all.extend([cls] * synthetic_needed)
        else:
            # Regression or unsupervised case
            synthetic_needed = int(train_data_scaled.shape[0] * ratio_limit)
            
            if synthetic_needed > 0:
                print(f"Generating {synthetic_needed} synthetic samples")
                
                # Generate synthetic samples with distribution matching
                synthetic_samples_norm = generate_samples_with_distribution_matching(
                    model_dict=vae_results,
                    original_data=train_data_scaled,  # Pass all original data
                    n_samples=synthetic_needed,
                    temperature=temperature,
                    matching_factor=matching_factor
                )
                
                # Ensure values are in [0, 1] range
                synthetic_samples_norm = np.clip(synthetic_samples_norm, 0, 1)
                
                # Store generated samples
                synthetic_samples_all.append(synthetic_samples_norm)
        
        # Prepare original training data DataFrame
        if target is None:
            original_train_df = pd.DataFrame(scaler.inverse_transform(train_data_scaled), columns=feature_cols)
        else:
            original_train_df = pd.DataFrame(scaler.inverse_transform(train_data_scaled), columns=feature_cols)
            original_train_df[target] = train_labels
        
        original_train_df['synthetic'] = False
        
        # Prepare test data DataFrame
        if target is None:
            test_df = pd.DataFrame(scaler.inverse_transform(test_data_scaled), columns=feature_cols)
        else:
            test_df = pd.DataFrame(scaler.inverse_transform(test_data_scaled), columns=feature_cols)
            test_df[target] = test_labels
        
        # Inside your augment_dataframe_vae_enhanced function
        # After this section where synthetic samples are generated and combined:

        if len(synthetic_samples_all) > 0:
            if len(synthetic_samples_all) == 1:
                synthetic_samples_combined = synthetic_samples_all[0]
            else:
                synthetic_samples_combined = np.vstack(synthetic_samples_all)

            # Print debug information about the synthetic samples
            print(f"Debug - Synthetic samples shape before inverse transform: {synthetic_samples_combined.shape}")
            print(f"Debug - Synthetic samples range: [{np.min(synthetic_samples_combined)}, {np.max(synthetic_samples_combined)}]")

            # Ensure values are in [0, 1] range before inverse transform
            synthetic_samples_combined = np.clip(synthetic_samples_combined, 0, 1)

            # Apply additional smoothing to make distributions more natural
            if synthetic_samples_combined.shape[0] > 10:
                # Add small amount of noise to break up artificial clusters
                noise_level = 0.02  # Small noise to avoid major distribution changes
                noise = np.random.normal(0, noise_level, synthetic_samples_combined.shape)
                synthetic_samples_combined = np.clip(synthetic_samples_combined + noise, 0, 1)

            # Inverse-transform to original scale with careful handling
            try:
                synthetic_samples = scaler.inverse_transform(synthetic_samples_combined)

                # Check for NaN or Inf values and replace with reasonable values
                synthetic_samples = np.nan_to_num(synthetic_samples, nan=0.0, posinf=1e5, neginf=-1e5)

                # Create synthetic DataFrame
                synthetic_df = pd.DataFrame(synthetic_samples, columns=feature_cols)

                if train_labels is not None:
                    synthetic_df[target] = synthetic_labels_all

                synthetic_df['synthetic'] = True

                # Apply copula matching
                # Apply quantile matching
                print("Applying quantile matching to synthetic samples...")
                for col in feature_cols:
                    # Get synthetic data values and ranks
                    syn_values = synthetic_df[col].values
                    syn_ranks = synthetic_df[col].rank(pct=True).values

                    # Create a quantile function from original data
                    orig_values = original_train_df[col].dropna().values
                    if len(orig_values) > 0:
                        orig_quantiles = np.quantile(orig_values, np.linspace(0, 1, 1000))

                        # Map each synthetic point to the corresponding quantile in original data
                        matched_values = np.interp(syn_ranks, np.linspace(0, 1, 1000), orig_quantiles)
                        synthetic_df[col] = matched_values

                        
                # ADD THE MEAN MATCHING CODE HERE
                # Important features to match means for
                #features_to_match = ['Elevation']  # Add any other features here
                print("")
                # Apply mean matching to ensure distributions are properly aligned
                #print("Applying mean matching to important features...")
                #synthetic_df = match_feature_means(
                #    synthetic_df, 
                #    original_train_df, 
                #    features_to_match=features_to_match,
                #    target_col=target
                #)
                # END OF MEAN MATCHING CODE
                
                
                # Combine original and synthetic data
                augmented_train_df = pd.concat([original_train_df, synthetic_df], axis=0, ignore_index=True)

                # Print DataFrame info for debugging
                print("Original DataFrame info:")
                print(original_train_df.info())
                print("\nSynthetic DataFrame info:")
                print(synthetic_df.info())
                print("\nAugmented DataFrame info:")
                print(augmented_train_df.info())

            except Exception as e:
                print(f"Error during inverse transform or DataFrame creation: {str(e)}")
                print("Falling back to original data only")
                augmented_train_df = original_train_df.copy()
            
            # Print summary
            print(f"Augmented data summary:")
            print(f"  - Original samples: {len(original_train_df)}")
            print(f"  - Synthetic samples: {len(synthetic_df) if 'synthetic_df' in locals() else 0}")
            print(f"  - Total samples: {len(augmented_train_df)}")
            
            if train_labels is not None:
                before_counts = dict(zip(unique_classes, class_counts))
                after_counts = dict(augmented_train_df.groupby(target).size())
                
                print(f"Class distribution before: {before_counts}")
                print(f"Class distribution after: {after_counts}")
        else:
            # No synthetic samples generated
            augmented_train_df = original_train_df.copy()
            print("No synthetic samples were generated.")
        
        # Save results
        if save_results:
            original_train_df.to_csv(f"{output_dir}/vae_original_train.csv", index=False)
            augmented_train_df.to_csv(f"{output_dir}/vae_augmented_train.csv", index=False)
            test_df.to_csv(f"{output_dir}/vae_test_set.csv", index=False)
            
            print(f"Saved results to {output_dir}/")
        
        # Enhanced visualizations
        if plot_results and len(synthetic_samples_all) > 0:
            synthetic_df = augmented_train_df[augmented_train_df['synthetic'] == True]
            
            # Plot feature distributions with KDE for smoother visualization
            for feature in feature_cols:
                plt.figure(figsize=(12, 6))
                
                # Use KDE plots for smoother visualization
                sns.kdeplot(
                    data=original_train_df,
                    x=feature,
                    fill=True,
                    alpha=0.5,
                    label="Original",
                    color="steelblue"
                )
                
                sns.kdeplot(
                    data=synthetic_df,
                    x=feature,
                    fill=True,
                    alpha=0.5,
                    label="Synthetic",
                    color="coral"
                )
                
                # Also add histograms with transparency
                plt.hist(
                    original_train_df[feature], 
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
                
                plt.title(f'Distribution of {feature}')
                plt.legend()
                plt.tight_layout()
                plt.savefig(f'{output_dir}/vae_{feature}_distribution.png')
                plt.close()
            
            # Plot target distribution if classification
            if target is not None:
                plt.figure(figsize=(10, 6))
                target_counts = augmented_train_df.groupby([target, 'synthetic']).size().unstack()
                target_counts.plot(kind='bar')
                plt.title(f'Distribution of {target}')
                plt.savefig(f'{output_dir}/vae_{target}_distribution.png')
                plt.close()
            
            # Plot correlation heatmaps
            plt.figure(figsize=(12, 10))
            
            plt.subplot(1, 2, 1)
            orig_corr = original_train_df[feature_cols].corr()
            sns.heatmap(orig_corr, annot=False, cmap='coolwarm')
            plt.title('Original Data Correlation')
            
            plt.subplot(1, 2, 2)
            synth_corr = synthetic_df[feature_cols].corr()
            sns.heatmap(synth_corr, annot=False, cmap='coolwarm')
            plt.title('Synthetic Data Correlation')
            
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
            
            # Additional VAE-specific visualization: plot reconstructions
            if len(train_data_scaled) > 10:
                n_samples = min(10, len(train_data_scaled))
                idx = np.random.choice(len(train_data_scaled), n_samples, replace=False)
                samples = train_data_scaled[idx]
                
                try:
                    # Use reconstructions
                    if train_labels is not None:
                        sample_labels = train_labels[idx]
                        # Ensure labels are within bounds
                        corrected_labels = np.array([min(l, num_classes-1) for l in sample_labels])
                        sample_labels_onehot = tf.keras.utils.to_categorical(corrected_labels, num_classes=num_classes)
                        reconstructions = vae_results['vae'].predict([samples, sample_labels_onehot])
                    else:
                        reconstructions = vae_results['vae'].predict(samples)
                    
                    # Create a plot showing sample features
                    n_features = min(10, samples.shape[1])  # Show at most 10 features
                    
                    plt.figure(figsize=(20, 10))
                    for i in range(n_samples):
                        # Original
                        plt.subplot(2, n_samples, i + 1)
                        plt.bar(range(n_features), samples[i, :n_features])
                        plt.title(f"Original {i}")
                        plt.ylim(0, 1)
                        
                        # Reconstructed
                        plt.subplot(2, n_samples, n_samples + i + 1)
                        plt.bar(range(n_features), reconstructions[i, :n_features])
                        plt.title(f"Reconstructed {i}")
                        plt.ylim(0, 1)
                    
                    plt.tight_layout()
                    plt.savefig(f'{output_dir}/vae_reconstructions.png')
                    plt.close()
                except Exception as e:
                    print(f"Error generating reconstruction visualization: {e}")
        
        success_flag = True
        return original_train_df, augmented_train_df, test_df, success_flag
        
    except Exception as e:
        print(f"Error in enhanced VAE augmentation: {str(e)}")
        traceback.print_exc()
        if not 'original_train_df' in locals():
            original_train_df = pd.DataFrame()
        if not 'augmented_train_df' in locals():
            augmented_train_df = pd.DataFrame()
        if not 'test_df' in locals():
            test_df = pd.DataFrame()
        
        return original_train_df, augmented_train_df, test_df, False