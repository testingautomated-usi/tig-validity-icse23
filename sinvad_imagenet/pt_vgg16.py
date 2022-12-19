from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from keras.applications.vgg16 import VGG16


class Conv2dSame(torch.nn.Conv2d):

    def calc_same_pad(self, i: int, k: int, s: int, d: int) -> int:
        return max((math.ceil(i / s) - 1) * s + (k - 1) * d + 1 - i, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ih, iw = x.size()[-2:]

        pad_h = self.calc_same_pad(i=ih, k=self.kernel_size[0], s=self.stride[0], d=self.dilation[0])
        pad_w = self.calc_same_pad(i=iw, k=self.kernel_size[1], s=self.stride[1], d=self.dilation[1])

        if pad_h > 0 or pad_w > 0:
            x = F.pad(
                x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2]
            )
        return F.conv2d(
            x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


class ImageNet_classifier(nn.Module):
    def __init__(self):
        super(ImageNet_classifier, self).__init__()

        # # Block 1
        # x = layers.Conv2D(
        #     64, (3, 3), activation='relu', padding='same', name='block1_conv1')(
        #     img_input)
        self.block1_conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3),
                               padding='same', stride=1)
        # (3, 3, 3, 64)
        self.block1_relu1 = nn.ReLU()
        # (64,)
        # x = layers.Conv2D(
        #     64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
        self.block1_conv2 = nn.Conv2d(64, 64, kernel_size=(3, 3),
                                      padding='same', stride=1)
        # (3, 3, 64, 64)
        self.block1_relu2 = nn.ReLU()
        # (64,)

        # x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)
        self.block1_pool = nn.MaxPool2d(2, stride=(2, 2))

        # # Block 2
        # x = layers.Conv2D(
        #     128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
        self.block2_conv1 = nn.Conv2d(64, 128, kernel_size=(3, 3),
                                      padding='same', stride=1)
        # (3, 3, 64, 128)
        self.block2_relu1 = nn.ReLU()
        # x = layers.Conv2D(
        #     128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
        self.block2_conv2 = nn.Conv2d(128, 128, kernel_size=(3, 3),
                                      padding='same', stride=1)
        self.block2_relu2 = nn.ReLU()
        # x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)
        self.block2_pool = nn.MaxPool2d(2, stride=(2, 2))

        # # Block 3
        # x = layers.Conv2D(
        #     256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
        self.block3_conv1 = nn.Conv2d(128, 256, kernel_size=(3, 3),
                                      padding='same', stride=1)
        self.block3_relu1 = nn.ReLU()

        # x = layers.Conv2D(
        #     256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
        self.block3_conv2 = nn.Conv2d(256, 256, kernel_size=(3, 3),
                                      padding='same', stride=1)
        self.block3_relu2 = nn.ReLU()

        # x = layers.Conv2D(
        #     256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
        self.block3_conv3 = nn.Conv2d(256, 256, kernel_size=(3, 3),
                                      padding='same', stride=1)
        self.block3_relu3 = nn.ReLU()

        # x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)
        self.block3_pool = nn.MaxPool2d(2, stride=(2, 2))

        # # Block 4
        # x = layers.Conv2D(
        #     512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
        self.block4_conv1 = nn.Conv2d(256, 512, kernel_size=(3, 3),
                                      padding='same', stride=1)
        self.block4_relu1 = nn.ReLU()

        # x = layers.Conv2D(
        #     512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
        self.block4_conv2 = nn.Conv2d(512, 512, kernel_size=(3, 3),
                                      padding='same', stride=1)
        self.block4_relu2 = nn.ReLU()

        # x = layers.Conv2D(
        #     512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
        self.block4_conv3 = nn.Conv2d(512, 512, kernel_size=(3, 3),
                                      padding='same', stride=1)
        self.block4_relu3 = nn.ReLU()

        # x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)
        self.block4_pool = nn.MaxPool2d(2, stride=(2, 2))


        # # Block 5
        # x = layers.Conv2D(
        #     512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
        self.block5_conv1 = nn.Conv2d(512, 512, kernel_size=(3, 3),
                                      padding='same', stride=1)
        self.block5_relu1 = nn.ReLU()

        # x = layers.Conv2D(
        #     512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
        self.block5_conv2 = nn.Conv2d(512, 512, kernel_size=(3, 3),
                                      padding='same', stride=1)
        self.block5_relu2 = nn.ReLU()

        # x = layers.Conv2D(
        #     512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
        self.block5_conv3 = nn.Conv2d(512, 512, kernel_size=(3, 3),
                                      padding='same', stride=1)
        self.block5_relu3 = nn.ReLU()

        # x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)
        self.block5_pool = nn.MaxPool2d(2, stride=(2, 2))
        self.fc1 = nn.Linear(25088, 4096)
        self.out1_relu = nn.ReLU()
        self.fc2 = nn.Linear(4096, 4096)
        self.out2_relu = nn.ReLU()
        self.predictions = nn.Linear(4096, 1000)



    def forward(self, x):
        out = self.block1_relu1(self.block1_conv1(x))
        out = self.block1_pool(self.block1_relu2(self.block1_conv2(out)))

        out = self.block2_relu1(self.block2_conv1(out))
        out = self.block2_pool(self.block2_relu2(self.block2_conv2(out)))
        # expected (16,16,96)

        out = self.block3_pool(
            self.block3_relu3(self.block3_conv3(
                self.block3_relu2(self.block3_conv2(
                    self.block3_relu1(self.block3_conv1(
                        out
                    ))
                ))
            ))
        )

        # expected (16,16,192)
        out = self.block4_pool(
            self.block4_relu3(self.block4_conv3(
                self.block4_relu2(self.block4_conv2(
                    self.block4_relu1(self.block4_conv1(
                        out
                    ))
                ))
            ))
        )


        # expected (8,8,192)
        out = self.block5_pool(
            self.block5_relu3(self.block5_conv3(
                self.block5_relu2(self.block5_conv2(
                    self.block5_relu1(self.block5_conv1(
                        out
                    ))
                ))
            ))
        )


        # expected (8,8,192)
        out = np.transpose(out, (0, 2, 3, 1))
        # print(out.shape)
        out = torch.flatten(out, start_dim=1)
        # print(out.shape)


        out = self.out1_relu(self.fc1(out))
        out = self.out2_relu(self.fc2(out))

        #out = torch.flatten(out, start_dim=1)
        #out = self.out3_relu(self.predictions(out))
        out = self.predictions(out)
        out = F.softmax(out, dim=1)

        return out



from keras.layers import Input
import numpy as np
# input image dimensions
img_rows, img_cols = 224, 224
img_chn = 3
input_shape = (img_rows, img_cols, img_chn)
# define input tensor as a placeholder
input_tensor = Input(shape=input_shape)
# load multiple models sharing same input tensor
model1 = VGG16(input_tensor=input_tensor)
print(model1.summary())

tf_weights = model1.get_weights()

net = ImageNet_classifier()
sd = net.state_dict()
#print(sd)


conv1_w = tf_weights[0]
conv1_w = np.transpose(conv1_w, (3, 2, 0, 1))
conv1_w = torch.from_numpy(conv1_w)
sd['block1_conv1.weight'] = conv1_w

conv1_b = tf_weights[1]
#print("original shape conv1 bias")
#print(conv1_b.shape)
conv1_b = torch.from_numpy(conv1_b)
#print(conv1_b)
#print("final shape conv1 bias")
#print(conv1_b.shape)
sd['block1_conv1.bias'] = conv1_b
#print("AFTER SAVING")
#print(sd['conv1.bias'])
#exit()

conv2_w = tf_weights[2]
#print(conv2_w.shape)
conv2_w = np.transpose(conv2_w, (3, 2, 0, 1))
conv2_w = torch.from_numpy(conv2_w)
#sd['conv2.conv.weight'] = conv2_w
sd['block1_conv2.weight'] = conv2_w
#print(sd['conv2.conv.weight'].shape)
#exit()

conv2_b = tf_weights[3]
#print(conv2_b.shape)
conv2_b = torch.from_numpy(conv2_b)
#print(conv2_b)
#sd['conv2.conv.bias'] = conv2_b
sd['block1_conv2.bias'] = conv2_b

#print("AFTER SAVING")
#print(sd['conv2.conv.bias'])

conv3_w = tf_weights[4]
#print(conv3_w.shape)
conv3_w = np.transpose(conv3_w, (3, 2, 0, 1))
conv3_w = torch.from_numpy(conv3_w)
sd['block2_conv1.weight'] = conv3_w

conv3_b = tf_weights[5]
#print(conv3_b.shape)
conv3_b = torch.from_numpy(conv3_b)
sd['block2_conv1.bias'] = conv3_b

conv4_w = tf_weights[6]
#print(conv4_w.shape)
conv4_w = np.transpose(conv4_w, (3, 2, 0, 1))
conv4_w = torch.from_numpy(conv4_w)
#sd['conv4.conv.weight'] = conv4_w
sd['block2_conv2.weight'] = conv4_w


conv4_b = tf_weights[7]
#print(conv4_b.shape)
conv4_b = torch.from_numpy(conv4_b)
#sd['conv4.conv.bias'] = conv4_b
sd['block2_conv2.bias'] = conv4_b


conv5_w = tf_weights[8]
#print(conv3_w.shape)
conv5_w = np.transpose(conv5_w, (3, 2, 0, 1))
conv5_w = torch.from_numpy(conv5_w)
sd['block3_conv1.weight'] = conv5_w

conv5_b = tf_weights[9]
#print(conv3_b.shape)
conv5_b = torch.from_numpy(conv5_b)
sd['block3_conv1.bias'] = conv5_b

conv6_w = tf_weights[10]
conv6_w = np.transpose(conv6_w, (3, 2, 0, 1))
conv6_w = torch.from_numpy(conv6_w)
sd['block3_conv2.weight'] = conv6_w

conv6_b = tf_weights[11]
conv6_b = torch.from_numpy(conv6_b)
sd['block3_conv2.bias'] = conv6_b

conv7_w = tf_weights[12]
conv7_w = np.transpose(conv7_w, (3, 2, 0, 1))
conv7_w = torch.from_numpy(conv7_w)
sd['block3_conv3.weight'] = conv7_w

conv7_b = tf_weights[13]
conv7_b = torch.from_numpy(conv7_b)
sd['block3_conv3.bias'] = conv7_b

conv8_w = tf_weights[14]
conv8_w = np.transpose(conv8_w, (3, 2, 0, 1))
conv8_w = torch.from_numpy(conv8_w)
sd['block4_conv1.weight'] = conv8_w

conv8_b = tf_weights[15]
conv8_b = torch.from_numpy(conv8_b)
sd['block4_conv1.bias'] = conv8_b

conv9_w = tf_weights[16]
conv9_w = np.transpose(conv9_w, (3, 2, 0, 1))
conv9_w = torch.from_numpy(conv9_w)
sd['block4_conv2.weight'] = conv9_w

conv9_b = tf_weights[17]
conv9_b = torch.from_numpy(conv9_b)
sd['block4_conv2.bias'] = conv9_b

conv10_w = tf_weights[18]
conv10_w = np.transpose(conv10_w, (3, 2, 0, 1))
conv10_w = torch.from_numpy(conv10_w)
sd['block4_conv3.weight'] = conv10_w

conv10_b = tf_weights[19]
conv10_b = torch.from_numpy(conv10_b)
sd['block4_conv3.bias'] = conv10_b

conv11_w = tf_weights[20]
conv11_w = np.transpose(conv11_w, (3, 2, 0, 1))
conv11_w = torch.from_numpy(conv11_w)
sd['block5_conv1.weight'] = conv11_w

conv11_b = tf_weights[21]
conv11_b = torch.from_numpy(conv11_b)
sd['block5_conv1.bias'] = conv11_b

conv12_w = tf_weights[22]
conv12_w = np.transpose(conv12_w, (3, 2, 0, 1))
conv12_w = torch.from_numpy(conv12_w)
sd['block5_conv2.weight'] = conv12_w

conv12_b = tf_weights[23]
conv12_b = torch.from_numpy(conv12_b)
sd['block5_conv2.bias'] = conv12_b

conv13_w = tf_weights[24]
conv13_w = np.transpose(conv13_w, (3, 2, 0, 1))
conv13_w = torch.from_numpy(conv13_w)
sd['block5_conv3.weight'] = conv13_w

conv14_b = tf_weights[25]
conv14_b = torch.from_numpy(conv14_b)
sd['block5_conv3.bias'] = conv14_b

out1_w = tf_weights[26]
out1_w = np.transpose(out1_w)
out1_w = torch.from_numpy(out1_w)
sd['fc1.weight'] = out1_w

out1_b = tf_weights[27]
out1_b = torch.from_numpy(out1_b)
sd['fc1.bias'] = out1_b

out2_w = tf_weights[28]
out2_w = np.transpose(out2_w)
out2_w = torch.from_numpy(out2_w)
sd['fc2.weight'] = out2_w

out2_b = tf_weights[29]
out2_b = torch.from_numpy(out2_b)
sd['fc2.bias'] = out2_b

pred_w = tf_weights[30]
pred_w = np.transpose(pred_w)
pred_w = torch.from_numpy(pred_w)
sd['predictions.weight'] = pred_w

pred_b = tf_weights[31]
pred_b = torch.from_numpy(pred_b)
sd['predictions.bias'] = pred_b

torch.save(sd, "imagenet_class.pt")




