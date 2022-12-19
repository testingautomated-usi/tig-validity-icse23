"""
## Setup
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import backend as K
from scipy.io import loadmat

"""
## Create a sampling layer
"""


class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


"""
## Build the encoder
"""

latent_dim = 512
img_dim = 32
img_chn = 3

encoder_inputs = tf.keras.Input(shape=(img_dim, img_dim, img_chn))
x = layers.Conv2D(8, (5, 5), padding='same', strides=(2, 2), activation='relu')(encoder_inputs)
x = layers.Conv2D(16, (5, 5), padding='same', activation='relu')(x)
x = layers.Conv2D(32, (5, 5), padding='same', strides=(2, 2), activation='relu')(x)
x = layers.Conv2D(64, (5, 5), padding='same', activation='relu')(x)
x = layers.Conv2D(64, (5, 5), padding='same', strides=(2, 2), activation='relu')(x)

# shape info needed to build decoder model
shape = K.int_shape(x)
# generate latent vector Q(z|X)
x = layers.Flatten()(x)

z_mean = layers.Dense(latent_dim, name="z_mean")(x)
z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
z = Sampling()([z_mean, z_log_var])
encoder = tf.keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
encoder.summary()

"""
## Build the decoder
"""

latent_inputs = tf.keras.Input(shape=(latent_dim,))
x = layers.Dense(shape[1] * shape[2] * shape[3], activation="relu")(latent_inputs)
x = layers.Reshape((shape[1], shape[2], shape[3]))(x)

x = layers.Conv2DTranspose(64, (5, 5), padding='same', strides=(2, 2), activation='relu')(x)
x = layers.Conv2DTranspose(64, (5, 5), padding='same', activation='relu')(x)
x = layers.Conv2DTranspose(32, (5, 5), padding='same', strides=(2, 2), activation='relu')(x)
x = layers.Conv2DTranspose(16, (5, 5), padding='same', activation='relu')(x)
x = layers.Conv2DTranspose(8, (5, 5), padding='same', strides=(2, 2), activation='relu')(x)
pos_mean = layers.Conv2DTranspose(3, (5, 5), padding='same', name='pos_mean')(x)

decoder = tf.keras.Model(latent_inputs, pos_mean, name="decoder")
decoder.summary()

"""
## Define the VAE as a `Model` with a custom `train_step`
"""


class VAE(tf.keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    #TODO: mse works better
                    tf.keras.losses.mean_squared_error(data, reconstruction), axis=(-1, -2, -3)
                    #tf.keras.losses.binary_crossentropy(data, reconstruction), axis=(1, 2)
                )
            )

            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }


"""
## Train the VAE
"""
CLASS = None


# Load SVHN data
SVHN_DATA = #TODO: insert svhn data
train_raw = loadmat(SVHN_DATA)
train_data = np.array(train_raw['X'])
train_data = np.moveaxis(train_data, -1, 0)

train_labels = train_raw['y']
train_labels[train_labels == 10] = 0
train_labels = np.array(train_labels)

svhn_images = train_data
svhn_labels = train_labels

x_train = np.reshape(svhn_images, [-1, img_dim, img_dim, img_chn])

x_train = x_train.astype('float32') / 255

vae_name = "trained/svhn_vae_stocco"
vae_name = vae_name + "_all_classes"

vae = VAE(encoder, decoder)
vae.compile(optimizer=tf.keras.optimizers.Adam())
vae.fit(x_train, epochs=200, batch_size=128)

vae.encoder.save(vae_name+"/encoder")
vae.decoder.save(vae_name+"/decoder")