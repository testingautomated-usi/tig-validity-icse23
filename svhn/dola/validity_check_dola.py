import glob
import math
import csv
import ntpath

import numpy as np
import random
from tensorflow.keras import backend as K
import tensorflow as tf

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


def compute_valid(sample, encoder, decoder, tshd):
    #fp = []
    #tn = []
    #for batch in anomaly_test:
    rec_loss = calculate_density(sample, encoder, decoder)
    #print(rec_loss)
    if rec_loss < tshd or math.isnan(rec_loss):
        distr = 'ood'
        #tn.append(rec_loss)
    else:
        distr = 'id'
        #fp.append(rec_loss)

    return distr, rec_loss.item()


def main():
    csv_file = r"losses/ood_analysis_dola_all_classes.csv"
    # multiclass
    # VAE density threshold for classifying invalid inputs
    #old_vae_threshold = 5265.059520457959
    vae_threshold = 5487.52001953125
    VAE = "svhn_vae_dola_all_classes"

    # class 5 vs 4
    #old_vae_threshold = 1949.6873748748749
    #vae_threshold = 1707.9652099609375
    #VAE = "svhn_vae_dola_5"

    decoder = tf.keras.models.load_model("trained/" + VAE + "/decoder", compile=False)
    encoder = tf.keras.models.load_model("trained/" + VAE + "/encoder", compile=False)

    RESULTS_PATH = r"../../empstudy_results/results/svhn_inputs/"

    with open(csv_file, 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['TOOL', 'SAMPLE', 'ID/OOD', 'loss'])

        print("DeepJanus")
        DJ_FOLDER = RESULTS_PATH + "svhn_dj/*.npy"
        filelist = [f for f in glob.glob(DJ_FOLDER)]
        print("found samples:" + str(len(filelist)))
        # samples = [np.load(sample) for sample in filelist]
        for sample in filelist:
            s = np.load(sample)
            distr, loss = compute_valid(s, encoder, decoder, vae_threshold)
            sample_name = ntpath.split(sample)[-1]
            writer.writerow(['DJ', sample_name, distr, loss])

        print("DLFuzz")
        DLF_FOLDER = RESULTS_PATH + "svhn_dlf/*.npy"
        filelist = [f for f in glob.glob(DLF_FOLDER)]
        print("found samples:" + str(len(filelist)))
        # samples = [np.load(sample) for sample in filelist]
        for sample in filelist:
            s = np.load(sample)
            distr, loss = compute_valid(s, encoder, decoder, vae_threshold)
            sample_name = ntpath.split(sample)[-1]
            writer.writerow(['DLF', sample_name, distr, loss])

        print("DeepXplore")
        DX_FOLDER = RESULTS_PATH + "svhn_dx/*.npy"
        filelist = [f for f in glob.glob(DX_FOLDER)]
        print("found samples:" + str(len(filelist)))
        # samples = [np.load(sample) for sample in filelist]
        for sample in filelist:
            s = np.load(sample)
            distr, loss = compute_valid(s, encoder, decoder, vae_threshold)
            sample_name = ntpath.split(sample)[-1]
            writer.writerow(['DX', sample_name, distr, loss])

        print("Oxford")
        OX_FOLDER = RESULTS_PATH + "svhn_ox/*.npy"
        filelist = [f for f in glob.glob(OX_FOLDER)]
        print("found samples:" + str(len(filelist)))
        # samples = [np.load(sample) for sample in filelist]
        for sample in filelist:
            s = np.load(sample)
            distr, loss = compute_valid(s, encoder, decoder, vae_threshold)
            sample_name = ntpath.split(sample)[-1]
            writer.writerow(['OX', sample_name, distr, loss])

        print("Sinvad")
        SV_FOLDER = RESULTS_PATH + "svhn_sv/*.npy"
        filelist = [f for f in glob.glob(SV_FOLDER)]
        print("found samples:" + str(len(filelist)))
        # samples = [np.load(sample) for sample in filelist]
        for sample in filelist:
            s = np.load(sample)
            distr, loss = compute_valid(s, encoder, decoder, vae_threshold)
            sample_name = ntpath.split(sample)[-1]
            writer.writerow(['SV', sample_name, distr, loss])


if __name__ == "__main__":
    main()
