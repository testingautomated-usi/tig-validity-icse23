import os
import numpy as np
from tensorflow.keras import backend as K
import tensorflow as tf
import glob
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
    reconstructed_prob = np.zeros((X.shape[0],), dtype='float32')
    L = 1
    for l in range(L):
        sampled_zs = sampling([z_mean, z_log_var])
        mu_hat, log_sigma_hat = dec(sampled_zs)

        log_sigma_hat = np.float64(log_sigma_hat)
        sigma_hat = np.exp(log_sigma_hat) + 0.00001

        loss_a = np.log(2 * np.pi * sigma_hat)
        loss_m = np.square(mu_hat - X) / sigma_hat

        reconstructed_prob += -0.5 * np.sum((loss_a + loss_m), axis=(-1, -2, -3))
    reconstructed_prob /= L
    #print(reconstructed_prob)
    return reconstructed_prob


# Calculates and returns probability density of test input
def calculate_density(x_target_orig, enc, dec):
    x_target = np.clip(x_target_orig, 0, 1)
    #x_target = np.reshape(x_target_orig, (-1, 28*28))
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

    ROOT_VAE = "imagenet_vae_dola"
    VAE_TYPE = "all_classes"
    VAE = ROOT_VAE + "_" + VAE_TYPE
    VAE="imagenet_dola_train_short"

    decoder = tf.keras.models.load_model("trained/"+VAE+"/decoder", compile=False)
    encoder = tf.keras.models.load_model("trained/"+VAE+"/encoder", compile=False)

    #DATASET = "IMAGENET"
    DATASET = 'CELEBA'
    if DATASET == 'IMAGENET':
        dataset_dir = r'...'
        temp_dir = r'temp'  # a temporary directory where the data will be stored intermediately
        download_config = tfds.download.DownloadConfig(
            extract_dir=os.path.join(temp_dir, 'extracted'),
            manual_dir=dataset_dir
        )
        tfds.builder("imagenet2012").download_and_prepare(download_config=download_config)
        imagenet_data = tfds.load('imagenet2012', shuffle_files=False, as_supervised=True)
        x_test_resized, y_test = preprocess_tfds(imagenet_data['validation'], img_dim)

        test_data = x_test_resized.batch(1)

    elif DATASET == 'CELEBA':
        @tf.function
        def read_image(image_path):
            image = tf.io.read_file(image_path)
            image = tf.image.decode_image(image, channels=3, dtype=tf.float32)
            return image

        @tf.function
        def normalize(image):
            image = (image - tf.reduce_min(image)) / (tf.reduce_max(image) - tf.reduce_min(image))
            #image = (2 * image) - 1
            return image

        @tf.function
        def augment(image):
            image = tf.image.resize_with_crop_or_pad(image, 178, 178)
            image = tf.image.resize(image, (img_dim, img_dim))
            return image

        @tf.function
        def preprocess(image_path):
            image = read_image(image_path)
            image = augment(image)
            image = normalize(image)
            return image

        image_paths = glob.glob(r'...\img_align_celeba\img_align_celeba\*.jpg')[:50000]
        dataset = tf.data.Dataset.from_tensor_slices((image_paths))
        dataset = dataset.map(preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        test_data = dataset.batch(1)


    rec_losses = []
    for batch in test_data:
        rec_loss = calculate_density(batch, encoder, decoder)
        rec_losses.append(rec_loss)
    rec_loss_summary = np.vstack(rec_losses)
    np.save('losses/dola_rec_losses_' + DATASET + '_' + VAE_TYPE + '.npy', rec_loss_summary)


if __name__ == "__main__":
    main()