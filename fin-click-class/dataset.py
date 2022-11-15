# -*- coding: utf-8 -*-
"""
Created on Fri Nov 11 16:50:11 2022

@author: tnsak
"""

from torch.utils.data import Dataset
from torchvision.transforms import Compose, ToPILImage, Resize, ToTensor
import pandas as pd
import os
import numpy as np

class FCDataset(Dataset):
    
    def __init__(self, cfg, label_csv, transform=None):
        df = pd.read_csv(label_csv)
        
        if transform is None:
            transform = Compose([ToPILImage(), Resize([224, 224]), ToTensor()])
        self.transform = transform
        self.file = [os.path.join(cfg['data_dir'], x) for x in df.file.tolist()]
        self.label = df.label.tolist()
        
    def __len__(self):
        return len(self.file)
    
    def __getitem__(self, ix):
        image = np.load(self.file[ix])
        image = np.repeat(image[..., np.newaxis], 3, -1)
        if self.transform:
            image = self.transform(image)
        label = self.label[ix]
        
        return image, label
        
        