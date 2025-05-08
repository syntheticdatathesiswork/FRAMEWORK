import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, LeakyReLU, BatchNormalization
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam

# ----------------------------
# GAN Building Blocks - FIXED
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
    
    FIXED: Now properly builds discriminator and feature extractor with Functional API
    """
    # Scale data to [-1,1]
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaled_data = scaler.fit_transform(class_data)
    n_features = scaled_data.shape[1]
    
    # Check if batch size is too large for the dataset
    if batch_size > len(class_data):
        batch_size = max(4, len(class_data))  # Ensure at least 4 samples per batch if possible
        print(f"Adjusted batch size to {batch_size} due to small dataset size")
    
    # Set up optimizers
    d_optimizer = Adam(0.0002, 0.5)
    g_optimizer = Adam(0.0002, 0.5)
    
    # Build discriminator and feature extractor with Functional API - FIXED
    discriminator, feature_extractor = build_discriminator(n_features)
    discriminator.compile(loss='binary_crossentropy', optimizer=d_optimizer, metrics=['accuracy'])
    
    # Build generator
    generator = build_generator(noise_dim, n_features)
    
    # Define loss functions
    bce_loss = tf.keras.losses.BinaryCrossentropy()
    mse_loss = tf.keras.losses.MeanSquaredError()
    
    X_train = scaled_data.copy()
    
    for epoch in range(epochs):
        # Update the discriminator n_critic times
        for _ in range(n_critic):
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            real_samples = X_train[idx]
            noise = np.random.normal(0, 1, (batch_size, noise_dim))
            fake_samples = generator.predict(noise, verbose=0)  # Set verbose=0 to reduce output
            
            valid = np.ones((batch_size, 1)) * 0.9  # label smoothing
            fake = np.zeros((batch_size, 1))
            
            d_loss_real = discriminator.train_on_batch(real_samples, valid)
            d_loss_fake = discriminator.train_on_batch(fake_samples, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
        
        # Update generator using adversarial loss + feature matching loss
        noise = np.random.normal(0, 1, (batch_size, noise_dim))
        with tf.GradientTape() as tape:
            generated_samples = generator(noise, training=True)
            y_pred = discriminator(generated_samples, training=False)
            adv_loss = bce_loss(np.ones((batch_size, 1)) * 0.9, y_pred)
            
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            real_samples = X_train[idx]
            real_features = feature_extractor(real_samples, training=False)
            fake_features = feature_extractor(generated_samples, training=False)
            fm_loss = mse_loss(tf.reduce_mean(real_features, axis=0),
                               tf.reduce_mean(fake_features, axis=0))
            
            g_loss = adv_loss + lambda_fm * fm_loss
        
        grads = tape.gradient(g_loss, generator.trainable_weights)
        g_optimizer.apply_gradients(zip(grads, generator.trainable_weights))
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch} - D loss: {d_loss[0]:.4f}, Adv loss: {adv_loss.numpy():.4f}, FM loss: {fm_loss.numpy():.4f}, G total loss: {g_loss.numpy():.4f}")
    
    # Generate synthetic samples after training
    print(f"Generating {n_samples} synthetic samples...")
    noise = np.random.normal(0, 1, (n_samples, noise_dim))
    synthetic_scaled = generator.predict(noise, verbose=0)
    synthetic_samples = scaler.inverse_transform(synthetic_scaled)
    return synthetic_samples


# ----------------------------
# Main Augmentation Function
# ----------------------------

def augment_dataframe_gan(df, target, test_size=0.2, random_state=42, 
                          n_classes_to_augment=2, ratio_limit=1.0, 
                          diminishing_factor=2.0,
                          gan_epochs=2000, gan_batch_size=64, noise_dim=100, lambda_fm=10.0):
    """
    Splits the dataframe into training and test sets, removes singleton classes,
    and uses a GAN with feature matching loss to generate synthetic samples for selected minority classes.
    """
    train, test = train_test_split(df, test_size=test_size, random_state=random_state, stratify=df[target])
    original_train = train.copy()
    
    class_counts = train[target].value_counts()
    singleton_classes = class_counts[class_counts == 1].index.tolist()
    if singleton_classes:
        print(f"Removing singleton classes: {singleton_classes}")
        train = train[~train[target].isin(singleton_classes)]
        original_train = train.copy()
    
    counts = train[target].value_counts()
    largest_class_count = counts.max()
    
    classes_sorted = counts.sort_values().index.tolist()
    n_classes_to_augment = min(n_classes_to_augment, len(classes_sorted))
    classes_to_augment = classes_sorted[:n_classes_to_augment]
    print("Classes to augment:", classes_to_augment)
    
    synthetic_list = []
    for cls in classes_to_augment:
        current_count = counts[cls]
        desired_by_ratio = int(largest_class_count * ratio_limit)
        synthetic_needed = int((desired_by_ratio - current_count) * (current_count / largest_class_count)**diminishing_factor)
        synthetic_needed = max(0, synthetic_needed)  # Ensure non-negative
        
        if synthetic_needed <= 0:
            print(f"Class {cls}: No augmentation needed.")
            continue
        
        print(f"Class {cls}: Generating {synthetic_needed} synthetic samples (current: {current_count}, target: {desired_by_ratio}).")
        class_data = train[train[target] == cls].drop(columns=[target]).values
        
        # Check if we have enough data for this class
        if len(class_data) < 4:  # Minimum required for meaningful GAN training
            print(f"Class {cls}: Not enough samples ({len(class_data)}) for GAN training. Skipping.")
            continue
            
        try:
            synthetic_samples = generate_synthetic_samples_for_class(
                class_data, 
                n_samples=synthetic_needed, 
                epochs=gan_epochs, 
                batch_size=min(gan_batch_size, len(class_data)),  # Ensure batch size doesn't exceed data size
                noise_dim=noise_dim, 
                lambda_fm=lambda_fm
            )
            
            synthetic_df = pd.DataFrame(synthetic_samples, columns=train.drop(columns=[target]).columns)
            synthetic_df[target] = cls
            synthetic_df['synthetic'] = True
            synthetic_list.append(synthetic_df)
            
        except Exception as e:
            print(f"Error generating synthetic samples for class {cls}: {str(e)}")
            print("Continuing with next class...")
            continue
    
    train['synthetic'] = False
    if synthetic_list:
        print("Combining synthetic samples with original data...")
        synthetic_df_all = pd.concat(synthetic_list, axis=0)
        augmented_train = pd.concat([train, synthetic_df_all], axis=0).reset_index(drop=True)
        
        # Print distribution after augmentation
        print("\nClass distribution after augmentation:")
        after_counts = augmented_train[target].value_counts().sort_index()
        for cls, count in after_counts.items():
            orig = counts.get(cls, 0)
            print(f"  Class {cls}: {count} (was {orig})")
    else:
        print("No synthetic samples were generated. Returning original data.")
        augmented_train = train.copy()
    
    return original_train.reset_index(drop=True), augmented_train, test.reset_index(drop=True)