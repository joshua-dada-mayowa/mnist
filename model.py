import tensorflow as tf
from tensorflow.keras import layers, Model

class ConditionalVAE(Model):
    """
    Conditional Variational Autoencoder for MNIST digit generation.
    This architecture is conditioned on the digit label, allowing for
    the generation of specific digits.
    """

    def __init__(self, latent_dim=50, num_classes=10):
        super(ConditionalVAE, self).__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes

        # --- Encoder ---
        self.encoder_conv1 = layers.Conv2D(32, 3, activation='relu', strides=2, padding='same')
        self.encoder_conv2 = layers.Conv2D(64, 3, activation='relu', strides=2, padding='same')
        self.encoder_flatten = layers.Flatten()
        self.encoder_dense = layers.Dense(512, activation='relu')

        # --- Label Conditioning ---
        self.label_embedding = layers.Embedding(num_classes, 50)
        self.label_dense = layers.Dense(512, activation='relu')

        # --- Latent Space ---
        self.fc_mu = layers.Dense(latent_dim)
        self.fc_log_var = layers.Dense(latent_dim)

        # --- Decoder ---
        self.decoder_input_dense = layers.Dense(512, activation='relu')
        self.decoder_dense = layers.Dense(7 * 7 * 64, activation='relu')
        self.decoder_reshape = layers.Reshape((7, 7, 64))
        self.decoder_conv1 = layers.Conv2DTranspose(64, 3, activation='relu', strides=2, padding='same')
        self.decoder_conv2 = layers.Conv2DTranspose(32, 3, activation='relu', strides=2, padding='same')
        self.decoder_conv3 = layers.Conv2DTranspose(1, 3, activation='sigmoid', padding='same')

    def encode(self, x, labels):
        """Encodes input images and labels into the latent space parameters."""
        x = self.encoder_conv1(x)
        x = self.encoder_conv2(x)
        x = self.encoder_flatten(x)
        x = self.encoder_dense(x)

        label_emb = self.label_embedding(labels)
        label_emb = self.label_dense(label_emb)

        combined = layers.Concatenate()([x, label_emb])

        mu = self.fc_mu(combined)
        log_var = self.fc_log_var(combined)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        """Performs the reparameterization trick to sample from the latent space."""
        eps = tf.random.normal(shape=tf.shape(mu))
        return mu + tf.exp(0.5 * log_var) * eps

    def decode(self, z, labels):
        """Decodes a latent vector and a label into a reconstructed image."""
        label_emb = self.label_embedding(labels)
        label_emb = self.label_dense(label_emb)

        combined = layers.Concatenate()([z, label_emb])

        x = self.decoder_input_dense(combined)
        x = self.decoder_dense(x)
        x = self.decoder_reshape(x)
        x = self.decoder_conv1(x)
        x = self.decoder_conv2(x)
        x = self.decoder_conv3(x)
        return x

    def call(self, inputs, training=None):
        """The forward pass of the model."""
        x, labels = inputs
        mu, log_var = self.encode(x, labels)
        z = self.reparameterize(mu, log_var)
        reconstructed = self.decode(z, labels)
        return reconstructed, mu, log_var

    def generate(self, label, num_samples=1):
        """Generates new images for a given label."""
        z = tf.random.normal((num_samples, self.latent_dim))
        labels_tensor = tf.constant([label] * num_samples, dtype=tf.int32)
        return self.decode(z, labels_tensor)