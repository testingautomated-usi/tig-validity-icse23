from abc import ABC, abstractmethod
import torch
from torch import nn
import torch.nn.functional as F
from collections import OrderedDict
import numpy as np
import utils
import folder

LR = 0.004
ADV_LOSS_W = 1.0
EPS_BOUND = 0.1
INIT_PERT_NOISE = 0.0001
CONF_TSHD = 1.0
TERM_A = 1.0
TERM_B = 0.001
CLIPPING = False

### Abstract interfaces ###

class Network(ABC, nn.Module):

    @abstractmethod
    def loss(self, operands):
        pass

    @abstractmethod
    def compute_results(self, operands):
        pass

class Generator(Network):
    @abstractmethod
    def get_input(self, batch_size, y=None):
        pass

    def compute_results(self, operands):
        operands['generated_x'] = self(operands['z'])

class Discriminator(Network):
    def compute_results(self, operands):
        operands['disc_on_x'] = self(operands['x'])
        operands['disc_on_genx'] = self(operands['generated_x'])

class Classifier(Network):
    def compute_results(self, operands):
        operands['cla_on_x'] = self(operands['x'])
        operands['cla_on_genx'] = self(operands['generated_x'])

    @abstractmethod
    def select_loss_inputs(self, operands):
        # Any classifier should also subclass a UseXData mixin class
        pass

    def loss(self, operands):
        return F.cross_entropy(*self.select_loss_inputs(operands))


class Splittable(ABC, nn.Module):
    # implemented by nn.Sequential
    @abstractmethod
    def __len__(self):
        pass
    @abstractmethod
    def __getitem__(self, idx):
        pass

# Use mixins to select classifier training data source
class UsePlainData():
    def select_loss_inputs(self, operands):
        return operands['cla_on_x'], operands['y']


def resize_image(img, size):
    from torchvision import transforms
    p = transforms.Resize(size)
    return p(img)


def make_chunky(sequential, extra_allowed=[]):
    allowed = extra_allowed + [nn.Linear, nn.modules.conv._ConvNd]
    chunk_start_idx = [0]

    for i in range(len(sequential)):
        if any([issubclass(sequential[i].__class__, al) for al in allowed]):
            chunk_start_idx.append(i+1)
    chunk_start_idx.append(len(sequential))

    chunks = [sequential[chunk_start_idx[i]:chunk_start_idx[i+1]] for i in range(len(chunk_start_idx)-1)]
    chunk_od = OrderedDict((str(j),c) for j,c in enumerate(chunks))
    return sequential.__class__(chunk_od)


def get_neuron_ranges(splittable_gen, y):
    if splittable_gen._get_name()=='BigGANGenerator':
        def get_acts(y):
            z, y = splittable_gen.get_input(y.shape[0], y=y)
            splittable_gen_gan = splittable_gen.gan.to(device=splittable_gen.device)
            embed = splittable_gen_gan.embeddings(y.to(device=splittable_gen.device))
            cond_vector = torch.cat((z.to(device=splittable_gen.device), embed), dim=1)
            activations = [z]
            for i in range(len(splittable_gen)):
                if i == 0:
                    z = splittable_gen[i](cond_vector, truncation=splittable_gen.truncation)
                    activations.append(z)
                elif i != 9 and i  != 16: # 9 is attention layer
                    z = splittable_gen[i](z,cond_vector, truncation=splittable_gen.truncation)
                    activations.append(z)
                elif i == 9 or i == 16:
                    z = splittable_gen[i](z)
                    activations.append(z)
            return activations
        with torch.no_grad():
            actss = []
            for i in range(200):
                actss.append([a.to('cpu') for a in get_acts(y)])
                if i % 10 == 0: print(i)
            acts = [torch.cat([a[i] for a in actss]) for i in range(len(actss[0]))]
            # acts is list where each element is a batch of activations
            mins = [torch.min(a, dim=0)[0] for a in acts]
            maxs = [torch.max(a, dim=0)[0] for a in acts]
            means = [torch.mean(a, dim=0) for a in acts]
            stds = [torch.std(a, dim=0) for a in acts]

        acts = get_acts(y)
        mins = [a for a in acts]
        maxs = [a for a in acts]
        for _ in range(1000):
            acts = get_acts(y)
            mins = [torch.min(m, a)[0] for m, a in zip(mins, acts)]
            maxs = [torch.max(m, a)[0] for m, a in zip(maxs, acts)]
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        z = splittable_gen.get_input(y.shape[0], y=y)
        z = z.to(device=device)
        activations = [z]
        for i in range(len(splittable_gen)):
            z = splittable_gen[i](z)
            activations.append(z)
        mins = [torch.min(a, dim=0)[0] for a in activations]
        maxs = [torch.max(a, dim=0)[0] for a in activations]
    return mins, maxs


class InternallyPerturbable(Generator):
    def __init__(self, splittable, pert_weights, which_neurons, add_neurons, y):
        super().__init__()
        if splittable._get_name() != 'BigGANGenerator' and splittable._get_name() != 'SplitTestGenerator2' and splittable._get_name() != 'SplitTestGeneratorSVHN':
            self.splittable = make_chunky(splittable)
        else:
            self.splittable = splittable

        assert(len(pert_weights) == len(self.splittable)+1)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.splittable = self.splittable.to(self.device)
        self.y = y

        if splittable._get_name() != 'BigGANGenerator':
            with torch.no_grad():
                self.n_mins, self.n_maxs = get_neuron_ranges(self.splittable, y)
            self.internal_shapes = [a.shape for a in self.n_mins]
            self.n_mins = [m.to(self.device) for m in self.n_mins]
            self.n_maxs = [m.to(self.device) for m in self.n_maxs]
        else:
            self.stds = torch.load('stds.pt')
            self.stds = [s.to(self.device) for s in self.stds]
            self.internal_shapes = [a.shape for a in self.stds]

        if which_neurons:
            self.which = [torch.zeros(self.internal_shapes[i], device=self.device) for i in range(len(pert_weights))]
            for i in range(len(pert_weights)):
                w = which_neurons
                if len(self.internal_shapes[i]) == 1:
                    self.which[i][slice(*w[i][0])] = 1
                elif len(self.internal_shapes[i]) == 3:
                    self.which[i][slice(*w[i][0]),slice(*w[i][1]), slice(*w[i][2])] = 1
                else:
                    raise ValueError('Need to add case for %d dimensional activation shape in %th layer' % (len(self.internal_shapes[i]), i))
            for (layer, mode, number, offset) in add_neurons:
                if mode == 'add filter':
                    self.which[layer][offset:offset+number,:,:] = 1
                elif mode == 'add location':
                    self.which[layer][:,offset:offset+number,offset:offset+number] = 1
                else:
                    raise ValueError(mode)
        else:
            self.which = [torch.ones(self.internal_shapes[i], device=self.device) for i in range(len(pert_weights))]
        self.which = [torch.where(self.which[i]*pert_weights[i] == 0,
                                  torch.zeros_like(self.which[i]),
                                  self.which[i])
                        for i in range(len(pert_weights))]

        self.pert_weights = [pert_weights[i] * self.which[i] for i in range(len(pert_weights))]
        self.flat_which = self.flatten_ps([w.unsqueeze(0) for w in self.which])

    def forward(self, zps): #returns semantically perturbed data
        z, ps = zps
        if self.splittable._get_name() == 'BigGANGenerator' and self.splittable.z_path:
            z = np.load(self.splittable.z_path)
            z = torch.tensor(z).to(self.device)
        # ps is a tuple of perturbations; needs to be normalised and weighted
        if self.splittable._get_name() == 'BigGANGenerator':
            ps = [self.stds[i] * self.pert_weights[i] * ps[i]
                  for i in range(len(ps))]
        else:
            ps = [(self.n_maxs[i] - self.n_mins[i]) * self.pert_weights[i] * ps[i]
                  for i in range(len(ps))]

        if self.splittable._get_name()=='BigGANGenerator':
            for i in range(len(self.splittable)+1):
                if i == 0:
                    x = z + ps[i]
                    embed = self.splittable.gan.embeddings(self.y)
                    cond_vector = torch.cat((x.float(), embed), dim=1)
                elif (i-1)  == 0 :
                    x = self.splittable[i-1](cond_vector, truncation=self.splittable.truncation) + ps[i]
                elif (i-1) != 9 and (i-1) !=16:  # 9 is attention layer
                    x = self.splittable[i-1](x.float(), cond_vector, truncation=self.splittable.truncation) + ps[i]
                elif (i-1)==9 or (i-1) ==16:
                    x = self.splittable[i-1](x.float()) + ps[i]

        else:
            # perform forwards
            for i in range(len(self.splittable)+1):
                if i == 0:
                    x = z + ps[i]
                else:
                    x = self.splittable[i-1](x) + ps[i]
        return x

    def get_z_input(self, batch_size, y=None):
        return  self.splittable.get_input(batch_size, y) # z, y

    def get_ps_input(self, batch_size, device='cpu'):
        return [torch.zeros(batch_size, *shape, device=device)
                for shape in self.internal_shapes]

    def get_input(self, batch_size, y=None):
        return (self.get_z_input(batch_size, y=y),
                self.get_ps_input(batch_size))

    def flatten_ps(self, ps):
        # flatten each perturbation, then concatenate
        batch_size = ps[0].shape[0]
        return torch.cat([p.view(batch_size, -1) for p in ps], 1)

    def unflatten_ps(self, ps):
        # unconcatenate then unflatten
        batch_size, total_plen = ps.shape
        flat_ps = []
        i = 0
        def _prod(l): return 1 if len(l)==0 else l[0]*_prod(l[1:]) # product
        for shape in self.internal_shapes:
            flat_ps.append(ps[:, i : i+_prod(shape)])
            i += _prod(shape)
        assert(i == total_plen)
        unflat_ps = [flat_ps[i].view(batch_size,*self.internal_shapes[i])
                        for i in range(len(flat_ps))]
        assert((self.flatten_ps(unflat_ps) == ps).all())
        return unflat_ps

    def loss(self, operands):
        return self.splittable.loss(operands)


# Use mixin to actually perform semantic adversarial perturbations
class SemanticPert(Generator):
    def __init__(self, splittable, attack, target_classifier, epsilons, target_label, pert_weights=None, which_neurons=None, add_neurons=[], disc=None):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if splittable._get_name() == 'BigGANGenerator':
            self.num_classes = 1000 # HARDCODED FOR NOW
            bs = len(splittable.test_class)
            y = torch.zeros(bs, self.num_classes, device=self.device)
            for i in range(bs):
                y[i, splittable.test_class[i]] = 1
        else:
            self.num_classes = 10 # HARDCODED FOR NOW
            batch_size_for_normalisations = 10  # HARDCODED FOR NOW
            y = torch.tensor([i % self.num_classes for i in
                              range(batch_size_for_normalisations)], device=self.device)

        self.perturbable_gen = InternallyPerturbable(splittable, pert_weights, which_neurons, add_neurons, y)
        self.attack = attack
        self.target_cla = target_classifier # classifier
        self.epsilons = epsilons
        self.target_label = target_label # label
        self.disc = disc
        self.splittable_name = splittable._get_name()

    def get_input(self, batch_size, y=[]):
        if hasattr(self.perturbable_gen.splittable, 'z_path'):
            try:
                z = np.load(self.perturbable_gen.splittable.z_path)
                z = torch.tensor(z).to(self.device)
                z = torch.cat([z for _ in range(batch_size)])
                zy = (z,y)
            except:
                print('Generating z at random')
                z = self.perturbable_gen.get_z_input(batch_size, y)
                self.target_cla.eval()
                self.perturbable_gen.splittable.eval()
                gen_img = self.perturbable_gen.splittable(z)
                if self.splittable_name == 'SplitTestGenerator2':
                    gen_img = resize_image(gen_img, 28)
                elif self.splittable_name == 'SplitTestGeneratorSVHN':
                    gen_img = resize_image(gen_img, 32)
                pred = self.target_cla(gen_img).argmax(dim=-1)
                count = 1
                while not pred == y and count < 100:
                    print('Prediction %d didn\'t match correct class %d, retrying (%d)' % (pred, y, count))
                    z = self.perturbable_gen.get_z_input(batch_size, y)
                    gen_img = self.perturbable_gen.splittable(z)
                    if self.splittable_name == 'SplitTestGenerator2':
                        gen_img = resize_image(gen_img, 28)
                    elif self.splittable_name == 'SplitTestGeneratorSVHN':
                        gen_img = resize_image(gen_img, 32)
                    pred = self.target_cla(gen_img).argmax(dim=-1)
                    count += 1
                    # hack for now to ensure correct class
                if pred == y:
                    print('Prediction %d does match correct class %d' % (pred, y))
                else:
                    print('Timeout')
                    return
                zy = (z, y)
        else:
            z = self.perturbable_gen.get_z_input(batch_size, y)
            z = z.to(device=self.device)
            self.target_cla.eval()
            self.perturbable_gen.splittable.eval()
            gen_x = self.perturbable_gen.splittable(z)
            if self.splittable_name == 'SplitTestGenerator2':
                gen_x = resize_image(gen_x, 28)
            elif self.splittable_name == 'SplitTestGeneratorSVHN':
                gen_x = resize_image(gen_x, 32)
            # print(gen_x.shape)
            pred = self.target_cla(gen_x).argmax(dim=-1)
            while not pred == y:
                # print('Prediction %d didn\'t match correct class %d, retrying' % (pred, y))
                z = self.perturbable_gen.get_z_input(batch_size, y)
                z = z.to(self.device)
                gen_x = self.perturbable_gen.splittable(z)
                if self.splittable_name == 'SplitTestGenerator2':
                    gen_x = resize_image(gen_x, 28)
                elif self.splittable_name == 'SplitTestGeneratorSVHN':
                    gen_x = resize_image(gen_x, 32)
                pred = self.target_cla(gen_x).argmax(dim=-1)
                # hack for now to ensure correct class
                if pred == y:
                    print('Prediction %d does match correct class %d' % (pred, y))
            zy = (z, y)
        return zy

    def forward(self, zy):
        if self.splittable_name == 'BigGANGenerator':
            z, y = zy[0]  # this zero is a hack for now, since the z seems to be z, y
        else:
            z, y = zy
        z = z.to(device=self.device)
        y = y.to(device=self.device)
        self.perturbable_gen.eval()

        class _CurriedZ(nn.Module):
            def forward(_self, ps):
                fl_ps = self.perturbable_gen.unflatten_ps(ps) # squeeze here? look into... seems no for biggan, yes for mnist?
                semantically_perturbed = self.perturbable_gen((z, fl_ps))
                return semantically_perturbed
        gen_taking_perts_only = _CurriedZ()

        initial_pert = self.perturbable_gen.get_ps_input(z.shape[0], device=z.device)
        with torch.no_grad():
            unperturbed = self.perturbable_gen((z, initial_pert))
            if self.splittable_name == 'SplitTestGenerator2':
                unperturbed = resize_image(unperturbed, 28)
            elif self.splittable_name == 'SplitTestGeneratorSVHN':
                unperturbed = resize_image(unperturbed, 32)
            unperturbed_classification = self.target_cla(unperturbed).argmax(dim=-1)
            if unperturbed_classification != y:
                print('\n\nWarning: original not correctly classified!\n')
                print('Should be %d but was %d\n' % (y, unperturbed_classification))

        flat_initial_pert = self.perturbable_gen.flatten_ps(initial_pert)

        if self.attack[0] == 'manual':
            if self.splittable_name == 'BigGANGenerator':
                hparams = {
                        'optimisation': 'c&w',
                        'lr': 0.003,
                        'loop_length': 10,
                        'timeout': 400,
                        'd_loss_type': 'none',
                        'adv_loss_weight': 1,
                        'real_loss_weight': 0,
                        'correct_class_loss_weight': 0,
                        'epsilon_bound': 1,
                        'initial_pert_noise': 0.0001,
                        'confidence_threshold': 1,
                        }
            else:
                    hparams = {
                        'optimisation': 'c&w',
                        'lr': LR,
                        'loop_length': 10,
                        'timeout': 9999,
                        'd_loss_type': 'none',
                        'adv_loss_weight': ADV_LOSS_W,
                        'real_loss_weight': 0,
                        'correct_class_loss_weight': 0,
                        'epsilon_bound': EPS_BOUND,
                        'initial_pert_noise': INIT_PERT_NOISE,
                        'confidence_threshold': CONF_TSHD,
                    }
            hparams.update(self.attack[1])

            # lr, min its, max its, eps bound, d loss type
            flat_initial_pert = flat_initial_pert + \
                torch.randn_like(flat_initial_pert)*hparams['initial_pert_noise']
            flat_pert = flat_initial_pert.detach().requires_grad_(True)

            adv_opt = torch.optim.Adam([flat_pert], lr=hparams['lr'])

            self.disc.eval()
            adv_loss = 0
            acc = 100
            epsilon_bound = hparams['epsilon_bound']
            lr = hparams['lr']
            it = 0
            while (it == 0 or (hparams['optimisation'] == 'c&w' and it % hparams['loop_length'] < (hparams['loop_length'] - 9))
                    or adv_loss > 0) and it < hparams['timeout']: # adv_loss > 0 only for now; was neq 0 before
                it += 1
                if it+1 > hparams['timeout']: raise Exception('timeout')
                if True or hparams['optimisation'] == 'new':
                    if self.splittable_name == 'BigGANGenerator':
                        epsilon_bound = epsilon_bound*1.00 + 0.1 # 2**(1/50)
                    else:
                        epsilon_bound = epsilon_bound*TERM_A + TERM_B
                    #lr =  epsilon_bound / hparams['loop_length']
                if hparams['optimisation'] == 'c&w' and it % hparams['loop_length'] == 0:
                    hparams['real_loss_weight'] *= 0.5

                gen_x = gen_taking_perts_only(flat_pert)

                if self.splittable_name == 'SplitTestGenerator2':
                    gen_x = resize_image(gen_x, 28)
                elif self.splittable_name == 'SplitTestGeneratorSVHN':
                    gen_x = resize_image(gen_x, 32)

                cla_logits = self.target_cla(gen_x)

                #if self.target_label == 0:
                if self.target_label is not None:
                    print("[models/semantpert] no target, expected label:")
                    expected_label = y.item()
                    print(expected_label)
                    print("[models/semantpert] computing loss")
                    cla_logits_except_expected = torch.cat([cla_logits[:, :expected_label],
                                                            cla_logits[:, expected_label + 2:]],
                                                           dim=1)
                    top_logits_except_expected, _ = cla_logits_except_expected.max(dim=-1)
                    print("[models/semantpert] predicted label")
                    print(cla_logits.argmax(dim=-1))
                    print("[models/semantpert] best label except expected")
                    print(cla_logits_except_expected.argmax(dim=-1))
                    top_logits, top_labs = cla_logits.max(dim=-1)
                    expected_logits = cla_logits[:, expected_label]
                    adv_losses = expected_logits - top_logits_except_expected + hparams['confidence_threshold']
                    acc = (cla_logits.argmax(dim=-1) == expected_label).float().sum().item()

                adv_loss = torch.mean(adv_losses)

                d_loss_type = hparams['d_loss_type']
                real_loss = correct_class_loss = 0
                if d_loss_type != 'none':
                    d_s, d_c = self.disc(gen_x, y)
                    if d_loss_type == 'wgan':
                        # lower value of d_s means it's being predicted to be fake
                        real_loss = -torch.mean(d_s)
                        correct_class_loss = F.cross_entropy(d_c, y)
                    elif d_loss_type == 'simple':
                        batch_size = len(d_s)
                        labels = 1 * torch.ones(batch_size, device=d_s.device)
                        real_loss = F.binary_cross_entropy(d_s, labels)
                    else: raise ValueError(d_loss_type)


                norm_reg = torch.norm(flat_pert, p=2, dim=-1).mean()
                real_loss = norm_reg

                loss = hparams['adv_loss_weight']*adv_loss + \
                        hparams['real_loss_weight']*real_loss + \
                        hparams['correct_class_loss_weight']*correct_class_loss

                adv_opt.zero_grad()
                loss.backward()


                if CLIPPING:
                    nn.utils.clip_grad_norm_([flat_pert], max_norm=5.0, norm_type=2)

                if adv_loss == 0:
                    pass
                elif hparams['optimisation'] == 'c&w':
                    print(flat_pert.data)
                    adv_opt.step()
                    print(flat_pert.data)
                elif hparams['optimisation'] in ['pgd', 'new']:
                    flat_pert.data = flat_pert - lr*flat_pert.grad.data/(1e-14+torch.norm(flat_pert.grad.data, p=2, dim=-1))
                else: raise NotImplementedError(hparams['optimisation'])


                with torch.no_grad():
                    flat_pert *= self.perturbable_gen.flat_which

                if epsilon_bound != -1:
                    with torch.no_grad():
                        norms = torch.norm(flat_pert, p=2, dim=-1, keepdim=True)
                        too_big = norms > epsilon_bound
                        projected_pert = flat_pert / norms * epsilon_bound
                        flat_pert.data = torch.where(too_big, projected_pert, flat_pert)
                else: too_big = torch.tensor(0)
                print('iteration %d  loss %.3f' % (it, loss))
            flat_adv_pert = flat_pert
            print()
        else:
            raise NotImplementedError(self.attack)

        unflat_adv_pert = self.perturbable_gen.unflatten_ps(flat_adv_pert)
        if False: # splittable may not have this attr: self.perturbable_gen.splittable.save_magnitudes:
            utils.get_layer_magnitude(unflat_adv_pert, flat_adv_pert)
        semantically_perturbed = torch.clamp(self.perturbable_gen((z, unflat_adv_pert)), -1, 1)
        if self.splittable_name == 'SplitTestGenerator2':
            semantically_perturbed = resize_image(semantically_perturbed, 28).cpu().detach()
            size = 28
        elif self.splittable_name == 'SplitTestGeneratorSVHN':
            semantically_perturbed = resize_image(semantically_perturbed, 32).cpu().detach()
            size = 32


        folder.Folder.save_img(unperturbed, semantically_perturbed, expected_label, size)
        return unperturbed, unflat_adv_pert, semantically_perturbed, adv_loss, real_loss, acc

    def loss(self, operands):
        return self.perturbable_gen.loss(operands)

    def compute_results(self, operands):
        unpert_gen_x, unflat_pert, sem_pert_x, adv, real, acc = self(operands['z'])
        if self.splittable_name == 'SplitTestGenerator2':
            unpert_gen_x = resize_image(unpert_gen_x, 28)
            sem_pert_x = resize_image(sem_pert_x, 28)
        elif self.splittable_name == 'SplitTestGeneratorSVHN':
            unpert_gen_x = resize_image(unpert_gen_x, 32)
            sem_pert_x = resize_image(sem_pert_x, 32)
        unpert_gen_x = unpert_gen_x.to(self.device)
        sem_pert_x = sem_pert_x.to(self.device)
        diffs = imagenet.utils.pix_diff(unpert_gen_x, sem_pert_x)
        operands['generated_x'] = sem_pert_x
        operands['unflat_perturbation'] = unflat_pert
        operands['unpert_generated_x'] = unpert_gen_x
        operands['semantic_pert_diffs'] = diffs

        operands['sp_adv'] = adv
        operands['sp_real'] = real
        operands['sp_acc'] = acc

