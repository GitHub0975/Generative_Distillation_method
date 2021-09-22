
from __future__ import print_function, division

import torch
import numpy as np
import torchvision
from torchvision import datasets, models, transforms

import sys
import shutil
import pdb

from PIL import Image
import os
import os.path
import errno
import pickle as pk

import codecs
import numpy as np
from torch.utils.data import Dataset
from CIFAR_split import *
from torch.utils.data import DataLoader as loader
from torch.distributions.multivariate_normal import MultivariateNormal

TASK1_CLASS = 50
TASKN_CLASS = 10
prototype = 2048
device = torch.device("cuda")
cpu = torch.device("cpu")


class Prototype_generate(datasets.CIFAR100):    # 自定義資料集
    """`MNIST <http://yann.lecun.com/exdb/mnist/>`_ Dataset.

    Args:
        root (string): Root directory of dataset where ``processed/training.pt``
            and  ``processed/test.pt`` exist.
        train (bool, optional): If True, creates dataset from ``training.pt``,
            otherwise from ``test.pt``.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """
  
    

    def __init__(self, total_sample, class_num, root='', train=True, transform=None, target_transform=None, 
                    download=False,digits=[1,2], real_label = True, model=None, New_data = False, file_size = 1):
        #super(Prototype_generate, self).__init__(total_sample, class_num, root, train, transform, target_transform, download)
        super(Prototype_generate, self).__init__(root, train, transform, target_transform, download)
        """
            total_sample: (class per task list) [50, 10, 10]
            class_num: (samples per task list) [5000, 500, 500]
        """
        
        Data = torch.load('old_data.pth.tar')
        self.all_data = Data['input']
        #self.all_soft_label = Data['Soft_label']
        self.target_label = Data['T_label']    # True label
        #self.Task_labels = Data['Task_label']
        
    
    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
       
        #img, target = self.digit_data[index], self.digit_labels[index]
        #prototype, target = self.pca_transform_data[index], self.all_soft_label[index]
        #prototype, Soft_labels, True_label = self.all_data[index], self.all_soft_label[index], self.target_label[index]
        prototype, True_label = self.all_data[index], self.target_label[index]


        #if self.target_transform is not None:
        #    target = self.target_transform(target)
        
        #img=img.view(-1, 3, 32, 32) # 轉成一維tensor(直接丟入全連層)
        
       
        #return prototype, Soft_labels, True_label
        return prototype, True_label

    def __len__(self):
        return(self.target_label.size()[0])  # 總data數量

  