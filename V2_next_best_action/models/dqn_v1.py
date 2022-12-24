'''DQN 
Input: Height map + RGB color map (top view orthographic)
Output: Q value for each of the 16 possible push directions + Q value for grasp
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
from collections import OrderedDict
from torch.autograd import Variable

class pushDQN(nn.Module):
    def __init__(self, use_cuda) -> None:
        super(pushDQN, self).__init__()
        self.use_cuda = use_cuda

        # Initialize push network trunk with DenseNet pre-trained on ImageNet
        self.push_color_trunk = torchvision.models.densenet.densenet121(pretrained=True) # (weights='DenseNet121_Weights.DEFAULT') # (pretrained=True) # 7*7*1024 # These pre-trained models will also be trained
        self.push_height_trunk = torchvision.models.densenet.densenet121(pretrained=True) #(weights='DenseNet121_Weights.DEFAULT') # (pretrained=True) # 7*7*1024 (given input==3*224*224)These pre-trained models will also be trained


        # Additional Layers for the model
        self.pushnet = nn.Sequential(OrderedDict([
            ('push-norm0', nn.BatchNorm2d(2048)), # here 2048 is the number of channels for the image
            ('push-relu0', nn.ReLU(inplace=True)),
            # ('push-pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=0)),
            ('push-conv1', nn.Conv2d(2048, 64, kernel_size=5, stride=1, padding='same')),
            ('push-norm1', nn.BatchNorm2d(64)), # here 2048 is the number of channels for the image
            ('push-relu1', nn.ReLU(inplace=True)),
            # ('push-pool1', nn.MaxPool2d(kernel_size=3, stride=2, padding=0)),
            ('push-conv2', nn.Conv2d(64, 32, kernel_size=5, stride=1, padding='same')),
            ('push-norm2', nn.BatchNorm2d(32)), # here 2048 is the number of channels for the image
            ('push-relu2', nn.ReLU(inplace=True)),
            # ('push-pool2', nn.MaxPool2d(kernel_size=3, stride=2, padding=0)) 
        ])) # Output 7*7*32
        
        self.linear_layers = nn.Sequential(OrderedDict([
            ('push-linear0', nn.Linear(7*7*32, 1024)),
            ('push-relu3', nn.ReLU(inplace=True)),
            ('push-linear1', nn.Linear(1024, 16)),
            # ('push-tanh', nn.Tanh()) # ('push-relu4', nn.ReLU(inplace=True))
        ]))

        # Weights initialization for the newly added layers

        for m in self.named_modules():
            if 'push-' in m[0]:
                if isinstance(m[1], nn.Conv2d):
                    nn.init.kaiming_normal_(m[1].weight.data)
                elif isinstance(m[1], nn.BatchNorm2d):
                    m[1].weight.data.fill_(1)
                    m[1].bias.data.zero_()
                elif isinstance(m[1], nn.Linear):
                    nn.init.xavier_uniform_(m[1].weight.data)
                    m[1].bias.data.fill_(0.01)
        
        # Initialize output variable (for backprop)
        self.interim_feat = []
        self.output_prob = []
        
    def forward(self, input_color_data, input_height_data, is_volatile=False):
        # if is_volatile:
        #     with torch.no_grad():
        #         # Fill up the code later
        #         pass
        # else:
        self.interim_feat = []
        self.output_prob = []

        interim_color_feat = self.push_color_trunk.features(input_color_data)
        interim_height_feat = self.push_height_trunk.features(input_height_data)
        interim_push_feat = torch.cat((interim_color_feat, interim_height_feat), dim=1)

        appended_activated_feat = self.pushnet(interim_push_feat)
        appended_activated_feat = appended_activated_feat.view(appended_activated_feat.size(0), -1)

        output_probs = self.linear_layers(appended_activated_feat)
    
        return output_probs
