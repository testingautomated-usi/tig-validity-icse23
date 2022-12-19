import torch
from torch import nn
import torch.nn.functional as F
import sys
import numpy as np
from models import Generator, Discriminator, Classifier, Splittable
from collections import OrderedDict
from pytorch_pretrained_BigGAN.pytorch_pretrained_biggan.model import BigGAN
from pytorch_pretrained_BigGAN.pytorch_pretrained_biggan.utils import one_hot_from_names, truncated_noise_sample, save_as_images
from torchvision.models import resnext50_32x4d, resnet50, resnet18
import torch
import models


class _ReshapeBatch(nn.Module):
    def __init__(self, *new_shape):
        super().__init__()
        self.new_shape = new_shape
    def forward(self, x):
        batch_size = x.shape[0]
        return x.view(batch_size, *self.new_shape)

### Generators ###

class NullGenerator(Generator):
    '''To be used when no generator is needed'''
    def __init__(self):
        super().__init__()
        self.parameter_so_no_error = nn.Parameter(torch.ones(1))
    def loss(self, o): return torch.mean(self.parameter_so_no_error*o['x'])
    def forward(self, x): return x
    def compute_results(self, operands):
        operands['generated_x'] = self(operands['x'])
    def get_input(self, b, y=None):
        return torch.zeros(1)


class PytorchLightningExampleGenerator(Generator):
    # from https://github.com/williamFalcon/pytorch-lightning/blob/master/pl_examples/domain_templates/gan.py
    def __init__(self, latent_dim=100, img_shape=(1,28,28)):
        super(PytorchLightningExampleGenerator, self).__init__()
        self.latent_dim = latent_dim
        self.img_shape = img_shape
        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
        self.model = nn.Sequential(
            *block(latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )
        self.apply(weights_init())
    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), *self.img_shape)
        return img
    def loss(self, operands):
        # ground truth result (ie: all fake)
        # put on GPU because we created this tensor inside training_loop
        valid = torch.ones(operands['x'].size(0), 1)
        valid = valid.cuda()
        return F.binary_cross_entropy(operands['disc_on_genx'], valid)
    def get_input(self, batch_size, y=None):
        return torch.randn(batch_size, self.latent_dim, device=y.device)

def weights_init(): # for recursive model module initialisation
    def fn_to_return(module):
        if isinstance(module, nn.Linear) or isinstance(module, torch.nn.modules.conv._ConvNd):
            nn.init.kaiming_normal(module.weight)
        elif isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            nn.init.constant_(module.weight.data, 1.0)
            nn.init.constant_(module.bias.data, 0.0)
    return fn_to_return


class BigGanLayerPreProcessing(nn.Module):
    def __init__(self, gan):
        super(BigGanLayerPreProcessing, self).__init__()
        self.gan = gan

    def forward(self, cond_vector, truncation=0.4):
        z = self.gan.gen_z(cond_vector)
        z = z.view(-1, 4, 4, 16 * 128)  #128 is channel width
        z = z.permute(0, 3, 1, 2).contiguous()
        return z

class BigGanLayerPostProcessing(nn.Module):
    def __init__(self, gan, truncation=0.4):
        super(BigGanLayerPostProcessing, self).__init__()
        self.gan = gan
        self.truncation = truncation

    def forward(self, z):
        z = self.gan.bn(z, self.truncation)
        z = self.gan.relu(z)
        z = self.gan.conv_to_rgb(z)
        z = z[:, :3, ...]
        z = self.gan.tanh(z)
        return z

class BigGANGenerator(Generator, Splittable):
    def __init__(self):
        super(BigGANGenerator, self).__init__()
        model = BigGAN.from_pretrained('biggan-deep-512', cache_dir=None)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.gan = model.to(device=self.device)
        self.list_layers = [BigGanLayerPreProcessing(self.gan.generator), *self.gan.generator.layers, BigGanLayerPostProcessing(self.gan.generator, self.truncation)]

    def __len__(self):
        len_gen = len(self.list_layers)
        return len_gen

    def __getitem__(self, idx):
        if  isinstance(idx,int):
            layer = self.list_layers[idx]
        else:
            layer = nn.Sequential(*self.list_layers[idx])
        return layer

    def get_cond_vector(self, y):
        batch_size = y.shape[0]
        cond_vector = torch.zeros(batch_size, 1000, device=y.device)
        cond_vector[range(batch_size), y] = 1
        return cond_vector

    def forward(self, zy):
        z, y = zy
        cond_vector = self.get_cond_vector(y)
        cond_vector.to(device=self.device)
        return self.gan(z.to(device=self.device), cond_vector.to(device=self.device), self.truncation)

    def loss(self, operands):
        return torch.tensor([.0], requires_grad=True)

    def get_input(self, batch_size=1, y=None):
        noise_vector = truncated_noise_sample(truncation=self.truncation, batch_size=batch_size)
        noise_vector = torch.from_numpy(noise_vector)
        return ((noise_vector),y)


class SeqGenerator(nn.Sequential, Splittable, Generator):
    def __init__(self, *layers, latent_dim=100, img_shape=(1,28,28), num_classes=10):
        super().__init__(*layers)
        self.latent_dim = latent_dim
        self.img_shape = img_shape
        self.num_classes = num_classes

    def loss(self, operands):
        return torch.tensor([.0], requires_grad=True)
    #def loss(self, operands):
    #    fool_disc_loss = -operands['disc_on_genx']
    #    correct_disc_classification = F.cross_entropy(
    #            operands['disc_auxclass_on_genx'],
    #            operands['y'],
    #            reduction='none')
    #    return torch.mean(fool_disc_loss + correct_disc_classification)

    def get_input(self, batch_size, y):
        # z is latent noise
        z = torch.randn(batch_size, self.latent_dim, device=y.device)
        # c is one-hot encoding of y
        c = torch.zeros(batch_size, self.num_classes, device=y.device)
        c[range(batch_size), y] = 1
        return torch.cat((z,c), dim=1)


class OriginalGenerator(nn.Module):
    def __init__(self, ngpu, nc=1, nz=100, ngf=64):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(     nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2,     ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(    ngf,      nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output


class SplitTestGenerator2_(SeqGenerator):
    def __init__(self, layers=None, nc=1, nz=100, ngf=64):
        self.nc = 1
        self.nz = 100
        self.ngf = 64
        if layers is None:
            layers = [  # input is Z, going into a convolution
                nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
                nn.BatchNorm2d(ngf * 8),
                nn.ReLU(True),
                # state size. (ngf*8) x 4 x 4
                nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf * 4),
                nn.ReLU(True),
                # state size. (ngf*4) x 8 x 8
                nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf * 2),
                nn.ReLU(True),
                # state size. (ngf*2) x 16 x 16
                nn.ConvTranspose2d(ngf * 2,     ngf, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf),
                nn.ReLU(True),
                #nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
                nn.ConvTranspose2d(ngf, nc, kernel_size=1, stride=1, padding=2, bias=False),
                #nn.Tanh()]
                nn.Sigmoid()]
        layers = OrderedDict((str(i), l) for i, l in enumerate(layers))
        super().__init__(layers)

        # self.apply(weights_init()) TODO fix

    def get_input(self, batch_size, y):
        z = torch.randn(batch_size, self.latent_dim, 1, 1, device='cpu')

        # c is one-hot encoding of y
        c = torch.zeros(batch_size, self.num_classes, device='cpu')
        c[range(batch_size), y] = 1
        #return z,c
        return z

    #def loss(self, operands):
    #    print("[concrete_models/generator] concrete loss")
    #    return torch.tensor([.0], requires_grad=True)


class SplitTestGenerator2(SeqGenerator):
    def __init__(self, layers=None, nc=1, nz=100, ngf=64):
        self.nc = 1
        self.nz = 100
        self.ngf = 64
        if layers is None:
            layers = [  # input is Z, going into a convolution
                nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
                nn.BatchNorm2d(ngf * 8),
                nn.ReLU(True),
                # state size. (ngf*8) x 4 x 4
                nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf * 4),
                nn.ReLU(True),
                # state size. (ngf*4) x 8 x 8
                nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf * 2),
                nn.ReLU(True),
                # state size. (ngf*2) x 16 x 16
                nn.ConvTranspose2d(ngf * 2,     ngf, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf),
                nn.ReLU(True),
                nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
                #nn.ConvTranspose2d(ngf, nc, kernel_size=1, stride=1, padding=2, bias=False),
                nn.Sigmoid()]
                #nn.Sigmoid()]
        layers = OrderedDict((str(i), l) for i, l in enumerate(layers))
        super().__init__(layers)

    def get_input(self, batch_size, y):
        z = torch.randn(batch_size, self.latent_dim, 1, 1, device='cpu')

        # c is one-hot encoding of y
        c = torch.zeros(batch_size, self.num_classes, device='cpu')
        c[range(batch_size), y] = 1
        #return z,c
        return z


class SplitTestGeneratorSVHN(SeqGenerator):
    def __init__(self, layers=None, nc=3, nz=100, ngf=64):
        self.nc = 3
        self.nz = 100
        self.ngf = 64
        if layers is None:
            layers = [  # input is Z, going into a convolution
                nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
                nn.BatchNorm2d(ngf * 8),
                nn.ReLU(True),
                # state size. (ngf*8) x 4 x 4
                nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf * 4),
                nn.ReLU(True),
                # state size. (ngf*4) x 8 x 8
                nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf * 2),
                nn.ReLU(True),
                # state size. (ngf*2) x 16 x 16
                nn.ConvTranspose2d(ngf * 2,     ngf, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf),
                nn.ReLU(True),
                nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
                #nn.ConvTranspose2d(ngf, nc, kernel_size=1, stride=1, padding=2, bias=False),
                nn.Sigmoid()]
                #nn.Sigmoid()]
        layers = OrderedDict((str(i), l) for i, l in enumerate(layers))
        super().__init__(layers)

    def get_input(self, batch_size, y):
        z = torch.randn(batch_size, self.latent_dim, 1, 1, device='cpu')

        # c is one-hot encoding of y
        c = torch.zeros(batch_size, self.num_classes, device='cpu')
        c[range(batch_size), y] = 1
        #return z,c
        return z


class MyConvGenerator(SeqGenerator):
    def __init__(self, layers=None, latent_dim=100, img_shape=(1,28,28), num_classes=10):
        self.latent_dim=latent_dim
        self.img_shape=img_shape
        self.num_classes=num_classes
        if layers == None:
            layers = [
                nn.Linear(latent_dim+num_classes, 64),
                nn.ReLU(),
                _ReshapeBatch(-1, 1, 1),
                #nn.ConvTranspose2d(64, 64, 5, 2, bias=False),
                #nn.BatchNorm2d(64),
                #nn.LeakyReLU(0.2),
                #nn.Dropout(0.35),

                #nn.ConvTranspose2d(64, 64, 5, 2, bias=False),
                #nn.BatchNorm2d(64),
                #nn.LeakyReLU(0.2),
                #nn.Dropout(0.35),

                nn.ConvTranspose2d(64, 32, 5, 2, bias=False),
                nn.BatchNorm2d(32),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.35),

                nn.ConvTranspose2d(32, 8, 5, 2, bias=False),
                nn.BatchNorm2d(8),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.35),

                nn.ConvTranspose2d(8, 4, 5, 2, bias=False),
                nn.BatchNorm2d(4),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.35),
                _ReshapeBatch(-1),
                nn.Linear(3364, 28*28),
               #nn.BatchNorm1d(28*28),
               #nn.BatchNorm1d(28*28),
               #nn.LeakyReLU(0.2),
               #nn.Linear(28*28, 28*28),
               #nn.BatchNorm1d(28*28),
               #nn.LeakyReLU(0.2),
               #nn.Linear(28*28, 28*28),
               #nn.BatchNorm1d(28*28),
               #nn.LeakyReLU(0.2),
               #nn.Linear(28*28, 28*28),
                nn.Tanh(),
                _ReshapeBatch(*self.img_shape)
            ]
            layers=OrderedDict((str(i), l) for i, l in enumerate(layers))
        super().__init__(layers, latent_dim=latent_dim, img_shape=img_shape, num_classes=num_classes)
        #self.apply(weights_init()) TODO fix


### Discriminators ###
class NullDiscriminator(Discriminator):
    '''To be used when no discriminator is needed'''
    def __init__(self):
        super().__init__()
        self.parameter_so_no_error = nn.Parameter(torch.ones(1))
    def loss(self, o): return torch.mean(self.parameter_so_no_error*o['x'])
    def forward(self, o): return torch.zeros(1)


### Classifiers ###
class MnistCNN(Classifier):
    def __init__(self, out_size=10):
        super().__init__()
        self.layers = nn.Sequential(
                nn.Conv2d(1,10,kernel_size=5), nn.MaxPool2d(2), nn.LeakyReLU(0.1),
                nn.Conv2d(10,20,kernel_size=5), nn.Dropout2d(),
                    nn.MaxPool2d(2), nn.LeakyReLU(0.1),
            )
        self.lins = nn.Sequential(
                nn.Linear(320, 256), nn.LeakyReLU(0.1), nn.Dropout(),
                nn.Linear(256, 128), nn.LeakyReLU(0.1), nn.Dropout(),
                nn.Linear(128, out_size)
                )
    def forward(self, x):
        x = x.view(-1, 1, 28, 28)
        b = x.size(0)
        x = self.layers(x).view(b, -1)
        x = self.lins(x)
        return x

class LeNet5(Classifier):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1, stride=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.maxpool2 = nn.MaxPool2d(2)
        self.linear1 = nn.Linear(7 * 7 * 64, 200)
        self.relu3 = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(200, 10)

    def forward(self, x):
        out = self.maxpool1(self.relu1(self.conv1(x)))
        out = self.maxpool2(self.relu2(self.conv2(out)))
        out = out.view(out.size(0), -1)
        out = self.relu3(self.linear1(out))
        out = self.linear2(out)
        return out

class ResNet(Classifier):
    def __init__(self):
        super().__init__()
        self.model = resnext50_32x4d(pretrained=True)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.eval().to(device=self.device).float()

    def forward(self, x):
        x = x.to(device=self.device).float()
        x = self.model(x)
        return x

class ResNet50(Classifier):
    def __init__(self):
        super().__init__()
        self.model = resnet50(pretrained=True)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.eval().to(device=self.device).float()

    def forward(self, x):
        x = x.to(device=self.device).float()
        x = self.model(x)
        return x

class ResNet18(Classifier):
    def __init__(self):
        super().__init__()
        self.model = resnet18(pretrained=True)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.eval().to(device=self.device).float()

    def forward(self, x):
        x = x.to(device=self.device).float()
        x = self.model(x)
        return x


def load_model(classnames, ckpt_path, attr_dict={}):
    if len(classnames) == 1 and classnames[0] != 'BigGANGenerator':
        assert(attr_dict == {}) # attr_dict is for multi classes only
        class_to_instantiate = getattr(sys.modules[__name__], classnames[0])
    else:
        fullname = ''.join(classnames)
        # for supers, look in models.py for all but last, which is here
        supers = [getattr(models, cl) for cl in classnames[:-1]]
        supers.append(getattr(sys.modules[__name__], classnames[-1]))
        class_to_instantiate = type(fullname, tuple(supers), attr_dict)
    model = class_to_instantiate()

    if ckpt_path is not None:
        checkpoint = torch.load(ckpt_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        if 'model_state_dict' in checkpoint:
            pretrained_state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            pretrained_state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint:
            pretrained_state_dict = checkpoint['model']
        else:
            pretrained_state_dict = checkpoint

        if True: # ignore naming of parameters, only order matters
            pretrained_weights_items = list(pretrained_state_dict.items())
            new_sdict = model.state_dict()
            count = 0
            for k, v in new_sdict.items():
                layername, weights = pretrained_weights_items[count]
                new_sdict[k] = weights
                count += 1
            model.load_state_dict(new_sdict, strict=True)
        else:
            model.load_state_dict(pretrained_state_dict, strict=True)
    return model
