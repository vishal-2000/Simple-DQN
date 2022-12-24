'''DQN
Consumes the current state, and the next state as input, and spits out Q values for each of the 16 standard actions
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
from collections import OrderedDict
from torch.autograd import Variable

class pushDQN2(nn.Module):
    '''Input Output Description
    Input Space: (x1, y1, theta1, x2, y2, theta2)

    Output: 0-15 (Q values for push actions (pushing in the 16 standard directions))
    '''
    def __init__(self, n_observations, n_actions, use_cuda) -> None:
        super(pushDQN2, self).__init__()
        self.use_cuda = use_cuda

        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)
        
        # self.linear_layers = nn.Sequential(OrderedDict([

        #     ('push-linear0', nn.Linear(7*7*32, 1024)),
        #     ('push-relu3', nn.ReLU(inplace=True)),
        #     ('push-linear1', nn.Linear(1024, 16)),
        #     # ('push-tanh', nn.Tanh()) # ('push-relu4', nn.ReLU(inplace=True))
        # ]))

        # # Weights initialization for the newly added layers

        # for m in self.named_modules():
        #     if 'push-' in m[0]:
        #         if isinstance(m[1], nn.Conv2d):
        #             nn.init.kaiming_normal_(m[1].weight.data)
        #         elif isinstance(m[1], nn.BatchNorm2d):
        #             m[1].weight.data.fill_(1)
        #             m[1].bias.data.zero_()
        #         elif isinstance(m[1], nn.Linear):
        #             nn.init.xavier_uniform_(m[1].weight.data)
        #             m[1].bias.data.fill_(0.01)
        
        # Initialize output variable (for backprop)
        # self.interim_feat = []
        self.output_prob = []
        
    def forward(self, x, is_volatile=False):
        # if is_volatile:
        #     with torch.no_grad():
        #         # Fill up the code later
        #         pass
        # else:
        # self.interim_feat = []
        self.output_prob = []

        # interim_color_feat = self.push_color_trunk.features(input_color_data)
        # interim_height_feat = self.push_height_trunk.features(input_height_data)
        # interim_push_feat = torch.cat((interim_color_feat, interim_height_feat), dim=1)

        # appended_activated_feat = self.pushnet(interim_push_feat)
        # appended_activated_feat = appended_activated_feat.view(appended_activated_feat.size(0), -1)

        # output_probs = self.linear_layers(appended_activated_feat)

        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        output_probs = self.layer3(x)
    
        return output_probs
