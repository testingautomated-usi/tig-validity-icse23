"""
## Setup
"""

import os
import tensorflow_datasets as tfds

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Layer, Conv2D, Input, Flatten, Dense, Conv2DTranspose, Reshape
from tensorflow.keras import backend as K

"""
## Create a sampling layer
"""


class Sampling(Layer):
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

img_dim = 128
img_chn = 3
latent_dim = 1024

encoder_inputs = tf.keras.Input(shape=(img_dim, img_dim, img_chn))
x = Conv2D(16, (3, 3), padding='same', strides=(2, 2), activation='relu')(encoder_inputs)
x = Conv2D(32, (3, 3), padding='same', activation='relu')(x)
x = Conv2D(64, (3, 3), padding='same', strides=(2, 2), activation='relu')(x)
x = Conv2D(128, (3, 3), padding='same', activation='relu')(x)
x = Conv2D(128, (3, 3), padding='same', strides=(2, 2), activation='relu')(x)

# shape info needed to build decoder model
shape = K.int_shape(x)
# generate latent vector Q(z|X)
x = Flatten()(x)

z_mean = Dense(latent_dim, name="z_mean")(x)
z_log_var = Dense(latent_dim, name="z_log_var")(x)
z = Sampling()([z_mean, z_log_var])
encoder = tf.keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
encoder.summary()

"""
## Build the decoder
"""

latent_inputs = tf.keras.Input(shape=(latent_dim,))
x = Dense(shape[1] * shape[2] * shape[3], activation="relu")(latent_inputs)
x = Reshape((shape[1], shape[2], shape[3]))(x)

x = Conv2DTranspose(128, (3, 3), padding='same', strides=(2, 2), activation='relu')(x)
x = Conv2DTranspose(128, (3, 3), padding='same', activation='relu')(x)
x = Conv2DTranspose(64, (3, 3), padding='same', strides=(2, 2), activation='relu')(x)
x = Conv2DTranspose(32, (3, 3), padding='same', activation='relu')(x)
x = Conv2DTranspose(16, (3, 3), padding='same', strides=(2, 2), activation='relu')(x)
#pos_mean = Conv2DTranspose(3, (3, 3), activation='sigmoid', padding='same', name='pos_mean')(x)
pos_mean = Conv2DTranspose(3, (3, 3), padding='same', name='pos_mean')(x)

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
                    tf.keras.losses.mean_squared_error(data, reconstruction), axis=(-1, -2, -3)
                    #tf.keras.losses.mean_squared_error(data, reconstruction), axis = (0, 1, 2)
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

def preprocess_tfds(imagenet_val):
    x_test = imagenet_val.map(lambda x, _: x).prefetch(tf.data.experimental.AUTOTUNE)
    x_test_resized = x_test.map(lambda x: tf.image.resize(x, [img_dim, img_dim]),
                                num_parallel_calls=tf.data.experimental.AUTOTUNE)
    x_test_resized = x_test_resized.map(lambda x: x / np.float32(255),
                        num_parallel_calls=tf.data.experimental.AUTOTUNE)
    x_test_resized = x_test_resized.prefetch(tf.data.experimental.AUTOTUNE)
    y_test_iter = imagenet_val.map(lambda _, y: y).as_numpy_iterator()
    y_test = np.fromiter(y_test_iter, dtype=int)
    return x_test_resized, y_test


DATASET_DIR = #TODO
dataset_dir = DATASET_DIR  # directory where you downloaded the tar files to
temp_dir = r'temp'  # a temporary directory where the data will be stored intermediately

download_config = tfds.download.DownloadConfig(
    extract_dir=os.path.join(temp_dir, 'extracted'),
    manual_dir=dataset_dir
)

tfds.builder("imagenet2012").download_and_prepare(download_config=download_config)

imagenet_data = tfds.load('imagenet2012', shuffle_files=False, as_supervised=True)

#x_test_resized, y_test = preprocess_tfds(imagenet_data['validation'])
x_test_resized, y_test = preprocess_tfds(imagenet_data['train'])

data = x_test_resized
data = data.batch(128)



"""
## Train the VAE
"""

vae = VAE(encoder, decoder)
vae.compile(optimizer=keras.optimizers.Adam())
vae.fit(data, epochs=50, batch_size=128)

#50 epochs loss: 18286.0241 - reconstruction_loss: 16334.5820 - kl_loss: 1951.0100
vae.encoder.save("trained/imagenet_selforacle_long/encoder")
vae.decoder.save("trained/imagenet_selforacle_long/decoder")