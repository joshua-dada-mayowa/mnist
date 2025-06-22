import tensorflow as tf
import numpy as np
import os
# import argparse # No longer needed if args are defined within the notebook

from model import ConditionalVAE

def train_vae_model(epochs, batch_size, learning_rate, model_path):
    """
    Trains the Conditional VAE model on the MNIST dataset.
    """
    print("Loading MNIST dataset...")
    # Load and preprocess data
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.astype('float32') / 255.0
    x_train = np.expand_dims(x_train, -1)
    x_test = x_test.astype('float32') / 255.0
    x_test = np.expand_dims(x_test, -1)

    print(f"Training data shape: {x_train.shape}")
    print(f"Training labels shape: {y_train.shape}")

    # Build model
    print("Building model...")
    vae = ConditionalVAE()
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    # Build the model with dummy data
    dummy_input = tf.random.normal((1, 28, 28, 1))
    dummy_label = tf.constant([0], dtype=tf.int32)
    vae([dummy_input, dummy_label])

    print(f"Model built. Total parameters: {vae.count_params()}")

    @tf.function
    def train_step(x_batch, y_batch):
        with tf.GradientTape() as tape:
            reconstructed, mu, log_var = vae([x_batch, y_batch], training=True)

            # Reconstruction loss
            reconstruction_loss = tf.reduce_mean(
                tf.keras.losses.binary_crossentropy(x_batch, reconstructed)
            ) * 28 * 28

            # KL divergence loss
            kl_loss = -0.5 * tf.reduce_mean(
                1 + log_var - tf.square(mu) - tf.exp(log_var)
            )

            # Total loss
            total_loss = reconstruction_loss + kl_loss

        gradients = tape.gradient(total_loss, vae.trainable_variables)
        optimizer.apply_gradients(zip(gradients, vae.trainable_variables))
        return total_loss, reconstruction_loss, kl_loss

    # Create dataset
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = train_dataset.shuffle(60000).batch(batch_size)

    # Training loop
    print(f"Starting model training for {epochs} epochs...")
    print("-" * 50)

    for epoch in range(epochs):
        epoch_loss = 0
        epoch_recon_loss = 0
        epoch_kl_loss = 0
        num_batches = 0

        for x_batch, y_batch in train_dataset:
            total_loss, recon_loss, kl_loss = train_step(x_batch, y_batch)
            epoch_loss += total_loss
            epoch_recon_loss += recon_loss
            epoch_kl_loss += kl_loss
            num_batches += 1

        avg_loss = epoch_loss / num_batches
        avg_recon_loss = epoch_recon_loss / num_batches
        avg_kl_loss = epoch_kl_loss / num_batches

        print(f'Epoch {epoch + 1:3d}/{epochs} | '
              f'Total Loss: {avg_loss:8.4f} | '
              f'Recon Loss: {avg_recon_loss:8.4f} | '
              f'KL Loss: {avg_kl_loss:8.4f}')

    print("-" * 50)
    print("Training completed!")

    # Save model weights
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    vae.save_weights(model_path)
    print(f"Model weights saved to {model_path}")

    # Test generation
    print("\nTesting generation...")
    for digit in range(10):
        generated = vae.generate(digit, num_samples=1)
        print(f"Generated digit {digit}: Shape {generated.shape}")

    print("Model training and testing completed successfully!")
    
# Define your desired arguments directly in the notebook
epochs = 50
batch_size = 128
learning_rate = 1e-4
# Change the file extension to end with .weights.h5
model_path = "models/cvae_mnist_weights.weights.h5"

# Call the training function with the defined arguments
train_vae_model(epochs, batch_size, learning_rate, model_path)