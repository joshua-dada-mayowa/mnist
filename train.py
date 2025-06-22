import tensorflow as tf
import numpy as np
import os
import argparse
from cvae_model import ConditionalVAE

def train_vae_model(epochs, batch_size, learning_rate, model_path):
    """
    Trains the Conditional VAE model on the MNIST dataset.
    """
    # Load and preprocess data
    (x_train, y_train), (_, _) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.astype('float32') / 255.0
    x_train = np.expand_dims(x_train, -1)

    # Build model
    vae = ConditionalVAE()
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

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
    print("Starting model training...")
    for epoch in range(epochs):
        epoch_loss = 0
        num_batches = 0

        for x_batch, y_batch in train_dataset:
            total_loss, _, _ = train_step(x_batch, y_batch)
            epoch_loss += total_loss
            num_batches += 1

        avg_loss = epoch_loss / num_batches
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}')

    # Save model weights
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    vae.save_weights(model_path)
    print(f"Model weights saved to {model_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Conditional VAE on MNIST.")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=128, help="Training batch size.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--model_path", type=str, default="models/cvae_mnist_weights.h5", help="Path to save the model weights.")
    args = parser.parse_args()

    train_vae_model(args.epochs, args.batch_size, args.lr, args.model_path)