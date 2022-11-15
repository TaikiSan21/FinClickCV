# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from torch import nn
from torchvision.models import resnet18, ResNet18_Weights

class FCNet(nn.Module):
    
    def __init__(self, cfg):
        '''
        new model constructor
        '''
        super(FCNet).__init__()
        self.feature_extractor = resnet18(weights=ResNet18_Weights.DEFAULT)
        last_layer = self.feature_extractor.fc
        in_features = last_layer.in_features
        self.feature_extractor.fc = nn.Identity()
        self.classifier = nn.Linear(in_features, cfg['num_classes']) # should be 2
        
    def forward(self, x):
        features = self.feature_extractor(x)
        prediction = self.classifier(features)
        
        return prediction
        