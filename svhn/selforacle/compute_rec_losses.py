import numpy as np
from tensorflow.keras import backend as K
import tensorflow as tf
from scipy.io import loadmat


image_size = 32
image_chn = 3
input_shape = (image_size, image_size, image_chn)


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
            # tf.keras.losses.binary_crossentropy(data, reconstruction), axis=(1, 2)
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
    ROOT_VAE = "svhn_vae_stocco"
    VAE_TYPE = "all_classes"
    VAE = ROOT_VAE + "_" + VAE_TYPE
    DATASET = "SVHN"

    decoder = tf.keras.models.load_model("trained/" + VAE + "/decoder", compile=False)
    encoder = tf.keras.models.load_model("trained/" + VAE + "/encoder", compile=False)

    # Load SVHN data
    print("SVHN test set")
    SVHN_DATA  = #TODO: insert SVHN data
    test_raw = loadmat(SVHN_DATA)
    test_data = np.array(test_raw['X'])
    test_data = np.moveaxis(test_data, -1, 0)

    #svhn_images = np.concatenate([test_data, train_data], axis=0)
    svhn_images = test_data
    svhn_images = np.reshape(svhn_images, [-1, image_size, image_size, image_chn])
    data = svhn_images.astype('float32') / 255

    rec_losses = []
    for batch in data:
        batch = np.expand_dims(batch,0)
        rec_loss = calculate_density(batch, encoder, decoder)
        rec_losses.append(rec_loss)
    rec_loss_summary = np.vstack(rec_losses)
    np.save('losses/stocco_rec_losses_' + DATASET + '_' + VAE_TYPE + '.npy', rec_loss_summary)


if __name__ == "__main__":
    main()