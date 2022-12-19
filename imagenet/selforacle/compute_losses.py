import numpy as np
import os
from tensorflow.keras import backend as K
import tensorflow as tf
import tensorflow_datasets as tfds
from imagenet.anomaly_detectors.utils import preprocess_tfds

img_dim = 128
img_chn = 3


# Logic for calculating reconstruction probability
def reconstruction_probability(dec, z_mean, z_log_var, X):
    """
    :param decoder: decoder model
    :param z_mean: encoder predicted mean value
    :param z_log_var: encoder predicted sigma square value
    :param X: input data
    :return: reconstruction probability of input
            calculated over L samples from z_mean and z_log_var distribution
    """
    sampled_zs = sampling([z_mean, z_log_var])
    reconstruction = dec(sampled_zs)
    reconstruction_loss = tf.reduce_mean(
        tf.reduce_sum(
            tf.keras.losses.mean_squared_error(X, reconstruction), axis=(-1, -2, -3)
        )
    )
    return reconstruction_loss


# Calculates and returns probability density of test input
def calculate_density(x_target_orig, enc, dec):
    x_target = np.clip(x_target_orig, 0, 1)
    z_mean, z_log_var, _ = enc(x_target)
    reconstructed_prob_x_target = reconstruction_probability(dec, z_mean, z_log_var, x_target)
    return reconstructed_prob_x_target


def sampling(args):
    """Reparameterization trick by sampling from an isotropic unit Gaussian.
    # Arguments
        args (tensor): mean and log of variance of Q(z|X)
    # Returns
        z (tensor): sampled latent vector
    """
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean = 0 and std = 1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon


def main():
    if not os.path.exists("losses"):
        os.makedirs("losses")

    CLASS = None

    DATASET = "IMAGENET"

    VAE = "imagenet_selforacle_long"
    VAE_TYPE = "all_classes"

    decoder = tf.keras.models.load_model("trained/" + VAE + "/decoder", compile=False)
    encoder = tf.keras.models.load_model("trained/" + VAE + "/encoder", compile=False)

    print("Imagenet test set")


    DATASET_DIR = #TODO: update path
    dataset_dir = r'C:\Users\Nabaut\Downloads'
    temp_dir = r'temp'  # a temporary directory where the data will be stored intermediately
    download_config = tfds.download.DownloadConfig(
        extract_dir=os.path.join(temp_dir, 'extracted'),
        manual_dir=dataset_dir
    )
    tfds.builder("imagenet2012").download_and_prepare(download_config=download_config)
    imagenet_data = tfds.load('imagenet2012', shuffle_files=False, as_supervised=True)
    x_test_resized, y_test = preprocess_tfds(imagenet_data['validation'], img_dim)

        data = x_test_resized.batch(1)

    rec_losses = []
    for batch in data:
        #batch = np.expand_dims(batch,0)
        rec_loss = calculate_density(batch, encoder, decoder)
        rec_losses.append(rec_loss)
    rec_loss_summary = np.vstack(rec_losses)

    np.save('losses/stocco_rec_losses_' + DATASET + '_' + VAE_TYPE + '.npy', rec_loss_summary)

if __name__ == "__main__":
    main()