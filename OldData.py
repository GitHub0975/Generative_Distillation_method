
from __future__ import print_function, division

import torch
import numpy as np
import torchvision
from torchvision import models

import sys
import shutil
import pdb

from PIL import Image
import os
import os.path
import errno
import pickle as pk

import codecs
from torch.distributions.multivariate_normal import MultivariateNormal

#TASK1_CLASS = 50
#TASKN_CLASS = 10
prototype = 2048
device = torch.device("cuda")
cpu = torch.device("cpu")


class test():    # 自定義資料集

    def __init__(self, total_sample, class_num, root='', model=None, New_data = False, file_size = 1):
        
        
        mean_file_name = []
        cov_file_name = []
        decoder_name = []
        
        print(model)
        
        for i in range(1, file_size):
            mean_file_name.append('./General_utils/information/feature_mean' + str(i) + '.npy')
            cov_file_name.append('./General_utils/information/feature_cov' + str(i) + '.npy')
            #decoder_name.append('./AuxModel/Decoder' + str(i) + '.pth.tar')
        
        
        # 對每一個類別做sample
        self.all_data = None
        self.all_labels = None
        self.target_label = None
        print('class_num:{}'.format(class_num))
        print(mean_file_name)
        acc_class = 0
        
        with torch.no_grad():
            ## 有幾個task的舊資料要產生
            for t in range(len(class_num)): # 幾個任務的資料要產生

                s = 0
                e = class_num[t]
                # 每個class有幾個sample
                samples = int(total_sample[t] / class_num[t])# + 1
            #print(samples)
            #print(mean_file_name[t])
                all_mean = np.load(mean_file_name[t])
                all_std = np.load(cov_file_name[t])
                print(mean_file_name[t])
                # 針對不同任務製作不同大小的decoder1
                #class_end = sum(class_num[:t + 1]) # size of decoder
                
                #decoder1 = torch.nn.Linear(class_end, prototype)
                
                #decoder1.weight.data = model.decoder1[0].weight.data[:, 0:class_end]
                #decoder1.bias.data = model.decoder1[0].bias.data

                
                #print(decoder1)

            
                for i in range(s, e): # 一個task有多個class
                
                    mean = torch.from_numpy(all_mean[i])
                    std = torch.from_numpy(all_std[i])
                
                    np.set_printoptions(threshold=np.inf)
                    #print(mean)
                    #print(std)
                    print(i)
                    n = MultivariateNormal(mean, std)
                
                    sampling = n.sample((samples, ))
                    sampling = sampling.to(device).float()
                    
                    sampling = model.decoder1(sampling)
                    sampling = model.decoder2(sampling)
                    sampling = model.decoder3(sampling).to(cpu) 
                    instance = sampling.clone().detach()

                
                    if self.all_data is None:
                        self.all_data = instance
                    else:
                        self.all_data = torch.cat((self.all_data, instance), 0)
                    
                    New_target_label = torch.zeros((samples, )) + i + acc_class
                    if self.target_label is None:
                        self.target_label = New_target_label
                    else:
                        self.target_label = torch.cat((self.target_label, New_target_label), 0)
                    #print(New_target_label)
                    #print(New_target_label.size())
            
                    
                acc_class += e
            
                torch.set_printoptions(threshold=np.inf)
        
 
        print('Virtual data: {}'.format(self.all_data.size()))   # 總共產生幾筆虛擬資料    
        torch.set_printoptions(threshold=np.inf)

        # 是否加入新data
        '''if New_data:
            New_data = torch.load('Data.pth.tar')
            New_input = New_data['input']#.cuda()
            #New_label = New_data['label']
            
            New_label = torch.zeros(New_input.size()[0]) + (file_size - 1)
            New_target_label = New_data['N_label']
            self.target_label = torch.cat((self.target_label, New_target_label), 0)
            #New_label = torch.ones((New_data['label'].size()[0], ))#New_data['label']#.cuda()
        
            self.all_data = torch.cat((self.all_data, New_input), 0)
            #self.all_labels = torch.cat((self.all_labels, New_label ), 0)'''
        
        #self.all_data = self.all_data.detach()    
        #print(self.all_labels)
        
        print('Total old data : {}'.format(self.all_data.size()))
        torch.save({
            'input': self.all_data,
            #'label': self.all_labels,    # Task label(沒用到!!!)
            'T_label': self.target_label,    # True label
            }, 'old_data.pth.tar')
        