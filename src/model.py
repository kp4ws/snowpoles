'''
Original author: Catherine Breen (July 1, 2024)
Updated by: Kent Pawson (2026) Adapted for multi-pole keypoint configuration and custom dataset pipelines.

adapted from: 
https://debuggercafe.com/advanced-facial-keypoint-detection-with-pytorch/

'''

import torch.nn as nn
import torch.nn.functional as F
import pretrainedmodels

class snowPoleResNet50(nn.Module):
    def __init__(self, pretrained, requires_grad, num_keypoints = 12):
    #def __init__(self, pretrained, requires_grad, input_size, hidden_size, num_layers, num_classes):
        super(snowPoleResNet50, self).__init__()
        if pretrained == True:
            self.model = pretrainedmodels.__dict__['resnet50'](pretrained='imagenet')
        else:
            self.model = pretrainedmodels.__dict__['resnet50'](pretrained=None)
        if requires_grad == True:
            for param in self.model.parameters():
                param.requires_grad = True
            print('Training intermediate layer parameters...')
        elif requires_grad == False:
            for param in self.model.parameters():
                param.requires_grad = False
            print('Freezing intermediate layer parameters...')
        # change the final layer
        
        #Final layer of neural network
        self.l0 = nn.Linear(2048, num_keypoints)  #### the second value is the number of points you want to predict

    def forward(self, x):
        # get the batch size only, ignore (c, h, w)
        batch, _, _, _ = x.shape
        x = self.model.features(x)
        x = F.adaptive_avg_pool2d(x, 1).reshape(batch, -1)
        l0 = self.l0(x)
        return l0
