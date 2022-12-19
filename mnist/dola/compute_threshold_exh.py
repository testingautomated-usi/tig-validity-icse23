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

    #print("fp: "+str(len(fp)))
    #print("tp: "+str(len(tp)))
    #print("fn: "+str(len(fn)))
    #print("tn: "+str(len(tn)))
    #print("precision: "+str(precision))
    #print("recall: "+str(recall))

    return f1


DATASET = "MNIST"
VAE_TYPE = "multiclass"

# vae threshold: -39191.8984375
# f1: 0.667411280939761


# vae threshold: -42057.59765625
# f1: 0.611526294118256
# fp: 1131
# tp: 891
# fn: 1
# tn: 4
# precision: 0.4406528189910979
# recall: 0.9988789237668162


# vae threshold: -2881.8822026372077
# f1: 0.6106091122130706
# fp: 1124
# tp: 886
# fn: 6
# tn: 11
# precision: 0.4407960199004975
# recall: 0.9932735426008968


#VAE_TYPE = "multiclass"
# vae threshold: 547.2452392578125
# f1: 0.9933618343222792

if VAE_TYPE == "oneclass":
    nominal = np.load('losses/dola_rec_losses_MNIST_5.npy')
    #anomalies = np.load('losses/dola_rec_losses_SVHN_1.npy')
    anomalies = np.load('losses/dola_rec_losses_MNIST_1.npy')
else:
    nominal = np.load('losses/dola_rec_losses_MNIST_all_classes.npy')
    anomalies = np.load('losses/dola_rec_losses_FMNIST_all_classes.npy')

#f1 = compute_fscore(nominal, anomalies, -2881.8822026372077)
#exit()

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
#tshd = thresholds[np.argmax(scores)]

print(tshd)
print(np.max(scores))

np.save('losses/dola_thresholds_' + DATASET + '_' + VAE_TYPE + '.npy', thresholds)
np.save('losses/dola_scores_' + DATASET + '_' + VAE_TYPE + '.npy', scores)
