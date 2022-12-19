#!/usr/bin/env python
# coding: utf-8

import torch
import numpy as np
import sys

import torchvision.transforms.functional as fn

from pt_vgg16 import ImageNet_classifier

from keras_preprocessing import image
from keras.applications.vgg16 import preprocess_input


# TODO: push this file to repo, it is the working one
def preprocess_image(dec_img):

    # pt2tf
    dec_img = np.transpose(dec_img, [0, 2, 3, 1])
    dec_img = dec_img.squeeze()

    input_img_data = image.img_to_array(dec_img)
    input_img_data = np.expand_dims(input_img_data, axis=0)
    input_img_data = preprocess_input(input_img_data)  # final input shape = (1,224,224,3)

    # tf2pt
    input_img_data = np.transpose(input_img_data, [0, 3, 1, 2])
    input_img_data = input_img_data.copy()
    gen_input = torch.from_numpy(input_img_data)
    return gen_input


with torch.no_grad(): # since nothing is trained here

    batch_size = 1

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if torch.cuda.is_available():
        map_location = None
    else:
        map_location = torch.device('cpu')

    target_class_idx = 963
    # Generator
    gen_attr_dict = {'test_class': [target_class_idx], 'truncation': 1}
    supers = getattr(sys.modules[__name__], 'BigGANGenerator')
    class_to_instantiate = type('BigGANGenerator', tuple([supers]), gen_attr_dict)
    gen = class_to_instantiate()
    gen.eval()
    gen.to(device)

    classifier = ImageNet_classifier()
    classifier.load_state_dict(torch.load("imagenet_class.pt"))
    #
    classifier.eval()

    classifier.to(device)
    print("models loaded...")

    gen_num = 5
    pop_size = 5

    best_left = 20
    mut_size = 0.1
    INC_MUT_SIZE = 0.1
    MULT_MUT_SIZE = 0.7
    INIT_MUT_SIZE = 0.7
    imgs_to_samp = 20
    COUNT = 0

    all_img_lst = []
    num_iterations = 100
    ### multi-image sample loop ###
    while COUNT < imgs_to_samp:
        ### Sample image ###
        y = torch.tensor([target_class_idx])
        cond = False
        while cond == False:
            img_enc, expected_label = gen.get_input(batch_size, y)
            dec_img = gen(tuple([img_enc.view(1, 128), expected_label])).cpu().detach()
            dec_img = (dec_img + 1) / 2

            dec_img *= 255.

            dec_img = fn.resize(dec_img, size=[224])

            gen_input = preprocess_image(dec_img)

            logits = classifier(gen_input).cpu().detach()
            pred_class = torch.argmax(logits).item()


            exp_class = expected_label.item()
            if  pred_class == exp_class:
                cond = True

        gen.eval()
        # shape [1,128]

        ### Initialize optimization ###
        init_pop = [img_enc.to(device) + INIT_MUT_SIZE * torch.randn(1, 128).to(device) for _ in range(pop_size)]
        now_pop = init_pop
        prev_best = 999
        binom_sampler = torch.distributions.binomial.Binomial(probs=0.5*torch.ones(img_enc.size()))

        ### GA ###
        for g_idx in range(gen_num):
            with torch.no_grad():
                indivs = torch.cat(now_pop, dim=0)
                dec_imgs = [gen(tuple([ind.view(1,128), expected_label])).cpu().detach()
                            for ind in indivs]

                for i in dec_imgs:

                    i = (i + 1) / 2
                    i *= 255.
                dec_imgs = [fn.resize(img, size=[224]) for img in dec_imgs]
                dec_imgs = [preprocess_image(img) for img in dec_imgs]

                dec_imgs_tensor = torch.cat(dec_imgs).to(device)

                all_logits = classifier(dec_imgs_tensor).cpu().detach()


            img_enc = img_enc.to(device)
            indv_score = [999 if expected_label.item() == torch.argmax(all_logits[i_idx]).item()
                          else torch.sum(torch.abs(indivs[i_idx] - img_enc))
                          for i_idx in range(pop_size)]
            best_idxs = sorted(range(len(indv_score)), key=lambda i: indv_score[i], reverse=True)[-best_left:]
            now_best = min(indv_score)
            if now_best == prev_best:
                mut_size *= 0.7
            else:
                mut_size = 0.1
            parent_pop = [now_pop[idx] for idx in best_idxs]

            k_pop = []
            for k_idx in range(pop_size-best_left):
                mom_idx, pop_idx = np.random.choice(best_left, size=2, replace=False)
                spl_idx = np.random.choice(400, size=1)[0]
                k_gene = torch.cat([parent_pop[mom_idx][:, :spl_idx], parent_pop[pop_idx][:, spl_idx:]], dim=1) # crossover
                # mutation
                diffs = (k_gene != img_enc).float()
                k_gene += mut_size * torch.randn(k_gene.size()).to(device) * diffs # random adding noise only to diff places
                # random matching to img_enc
                interp_mask = binom_sampler.sample().to(device)
                k_gene = interp_mask * img_enc + (1 - interp_mask) * k_gene

                k_pop.append(k_gene)
            now_pop = parent_pop + k_pop
            prev_best = now_best
            if mut_size < 1e-3:
                print("early stopping criterion")
                break # that's enough and optim is slower than I expected

        mod_best = parent_pop[-1].clone()
        final_bound_img = gen(tuple([parent_pop[-1].view(1, 128), expected_label]))

        final_bound_img = final_bound_img.cpu().detach()
        final_bound_img = (final_bound_img + 1) / 2
        final_bound_img *= 255
        final_bound_img = fn.resize(final_bound_img, size=[224])
        final_bound_img = preprocess_image(final_bound_img)

        all_logits = classifier(final_bound_img).cpu().detach()
        predicted_class = torch.argmax(all_logits).item()

        if predicted_class != target_class_idx:
            COUNT += 1
            final_bound_img = final_bound_img.detach().cpu().numpy()
            all_img_lst.append(final_bound_img)
            print("found input no. "+str(COUNT))

all_imgs = np.vstack(all_img_lst)
np.save('bound_imgs_imagenet.npy', all_imgs)
print("generated inputs: "+str(len(all_img_lst)))
print("GAME OVER")