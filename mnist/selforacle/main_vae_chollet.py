"""
Title: Variational AutoEncoder
Author: [fchollet](https://twitter.com/fchollet)
Date created: 2020/05/03
Last modified: 2020/05/03
Description: Convolutional Variational AutoEncoder (VAE) trained on MNIST digits.
"""

"""
## Setup
"""

import numpy as np
import tensorflow as tf
#from tensorflow import keras
from tensorflow.keras import layers
import tensorflow.keras.backend as K

"""
## Create a belin of sampling layer
"""


class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


img_dim = 28
img_chn = 1

"""
## Build the encoder
"""

original_dim = img_dim*img_dim*img_chn
latent_dim = 200
intermediate_dims = np.array([400])

#encoder_inputs = keras.Input(shape=(img_dim, img_dim, img_chn))
#flat_inputs = layers.Reshape((original_dim,))(encoder_inputs)
encoder_inputs = tf.keras.Input(shape=(original_dim,))
#x = layers.Dense(intermediate_dims[0], activation="relu")(flat_inputs)
x = layers.Dense(intermediate_dims[0], activation="relu")(encoder_inputs)
x = layers.Dense(intermediate_dims[0], activation="relu")(x)
z_mean = layers.Dense(latent_dim, name='z_mean')(x)
z_log_var = layers.Dense(latent_dim, name='z_log_var')(x)
z = Sampling()([z_mean, z_log_var])
encoder = tf.keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
encoder.summary()

"""
## Build the decoder
"""

intermediate_dims = np.flipud(intermediate_dims)
latent_inputs = tf.keras.Input(shape=(latent_dim,))
x = layers.Dense(intermediate_dims[0], activation='relu')(latent_inputs)
x = layers.Dense(intermediate_dims[0], activation='relu')(x)
#pos_mean_flatten = layers.Dense(original_dim, name='pos_mean', activation="sigmoid")(x)
#pos_mean = layers.Reshape([img_dim , img_dim , img_chn])(pos_mean_flatten)
#pos_log_var_flatten = layers.Dense(original_dim, name='pos_log_var', activation="sigmoid")(x)
#pos_log_var = layers.Reshape([img_dim , img_dim , img_chn])(pos_log_var_flatten)

#pos_mean = layers.Dense(original_dim, name='pos_mean', activation="sigmoid")(x)
#pos_log_var = layers.Dense(original_dim, name='pos_log_var', activation="sigmoid")(x)

pos_mean = layers.Dense(original_dim, name='pos_mean', activation='sigmoid')(x)

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
            outputs = self.decoder(z)

            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    # TODO: mse is worse
                    tf.keras.losses.binary_crossentropy(data, outputs), axis=(-1)
                    #tf.keras.losses.mean_squared_error(data, outputs), axis=(-1)
                )
            )
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=(-1)))
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
#CLASS = 5

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

#mnist_digits = np.concatenate([x_train, x_test], axis=0)
mnist_digits = x_train

vae_name = "trained/mnist_vae_stocco"
if CLASS is not None:
    #mnist_labels = np.concatenate([y_train, y_test], axis=0)
    mnist_labels = y_train
    idxs = np.argwhere(mnist_labels == CLASS)
    mnist_digits = mnist_digits[idxs]
    vae_name = vae_name + "_" + str(CLASS)
    BATCH_SIZE = 8
else:
    vae_name = vae_name + "_all_classes"
    BATCH_SIZE = 128

mnist_digits = np.expand_dims(mnist_digits, -1).astype("float32") / 255

mnist_digits = tf.reshape(tensor=mnist_digits, shape=(-1, original_dim,))

vae = VAE(encoder, decoder)
#optimizer = tf.keras.optimizers.Adam(1e-4)
#vae.compile(optimizer=optimizer)
vae.compile(optimizer="adam")
vae.fit(mnist_digits, epochs=50, batch_size=BATCH_SIZE)

vae.encoder.save(vae_name+"/encoder")
vae.decoder.save(vae_name+"/decoder")