import math
import numpy as np
import matplotlib.pyplot as plt


def compute_fscore(nominal_losses, anomaly_losses, tshd):
    tp = []
    fp = []
    tn = []
    fn = []

    for rec_loss in nominal_losses:
        if rec_loss < tshd or math.isnan(rec_loss):
            fn.append(rec_loss)
        else:
            tp.append(rec_loss)

    for rec_loss in anomaly_losses:
        if rec_loss < tshd or math.isnan(rec_loss):
            tn.append(rec_loss)
        else:
            fp.append(rec_loss)

    precision = len(tp)/(len(tp)+len(fp))
    recall = len(tp)/(len(tp)+len(fn))
    f1 = (2*precision*recall)/(precision+recall+0.00001)
    return f1

# multiclass
# vae_threshold = 5487.52001953125
# f1 = 0.9549248578805114

# 5vs4
# tshd: 1707.9652099609375
# f1: 0.6689404108609841



DATASET="SVHN"

VAE_TYPE = "multiclass"


nominal = np.load('losses/dola_rec_losses_SVHN_all_classes.npy')
anomalies = np.load('losses/dola_rec_losses_CIFAR10_all_classes.npy')

losses = np.concatenate((nominal, anomalies))
losses = np.sort(losses)
losses = np.unique(losses)
thresholds = []
for idx in range(len(losses)-1):
    idx1 = len(losses)-idx-1
    idx2 = idx1-1
    temp_tshd = (losses[idx1] + losses[idx2]) / 2
    thresholds.append(temp_tshd)

scores = [compute_fscore(nominal, anomalies, t) for t in thresholds]
idx_tshd = np.flatnonzero(scores == np.max(scores))
tshd = thresholds[idx_tshd[-1]]

np.save('losses/dola_thresholds_' + DATASET + '_' + VAE_TYPE + '.npy', thresholds)
np.save('losses/dola_scores_' + DATASET + '_' + VAE_TYPE + '.npy', scores)

print(tshd)
f1 = compute_fscore(nominal, anomalies, tshd)
print(f1)