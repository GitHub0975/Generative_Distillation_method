from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as pltD
import time
import copy
import os
import shutil
import sys
sys.path.append('General_utils')
sys.path.append('MAS_utils')

from MAS_based_Training import *

from Data_generate import *
from Merge_generate import *
import OldData
import pdb

import pandas as pd
from torch.utils.data import DataLoader as loader

from NCM_calculate_1 import NCM_result
from evaluate_1 import accuracy_result

prototype = 2048
feature_size = 8192
old_samples = 500

def exp_lr_scheduler(optimizer, epoch, init_lr=0.0008, lr_decay_epoch=200):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    lr = init_lr * (0.1**(epoch // lr_decay_epoch))
    print('lr is '+str(lr))
    if epoch % lr_decay_epoch == 0:
        print('LR is set to {}'.format(lr))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return optimizer
    
def kaiming_normal_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
    elif isinstance(m, nn.Linear):
        #nn.init.kaiming_normal_(m.weight, nonlinearity='sigmoid')
        nn.init.xavier_uniform_(m.weight)


def Big_Separate(dataset_path, previous_pathes, previous_task_model_path, exp_dir, data_dirs, reg_sets, 
                reg_lambda=1, norm='L2', num_epochs=100, lr=0.0008, batch_size=200, weight_decay=1e-5, 
                b1=False, after_freeze=1, class_num = 1, task = 1, digits = [1,2], base_class_size = 50, inc_class_size = 5):
    """Call MAS on mainly a sequence of tasks with a head for each where between at each step it sees samples from all the previous tasks to approximate the importance weights 
    dataset_path=new dataset path
    exp_dir=where to save the new model
    previous_task_model_path: previous task in the sequence model path to start from 
    pevious_pathes:pathes of previous methods to use the previous heads in the importance weights computation. We assume that each task head is not changed in classification setup of different tasks
    reg_sets,data_dirs: sets of examples used to compute omega. Here the default is the training set of the last task
    b1=to mimic online importance weight computation, batch size=1
    reg_lambda= regulizer hyper parameter. In object recognition it was set to 1.
    norm=the norm used to compute the gradient of the learned function
    """

    print('Dataset Path: {}'.format(dataset_path))
    dsets = torch.load(dataset_path)    # dictionary dataset(train and test data), new data
    
    # New training data
    dset_loaders = {x: torch.utils.data.DataLoader(dsets[x], batch_size=batch_size,
                                           shuffle=True, num_workers=0)
                for x in ['train_true', 'val_true']}
    dset_sizes = {x: len(dsets[x]) for x in ['train_true', 'val_true']}
    dset_classes = dsets['train_true'].classes

    use_gpu = torch.cuda.is_available()
    checkpoint = torch.load(previous_task_model_path)
    model_ft = checkpoint['model']
    
    # 製作新label(雙標籤)、新input
    '''train_encode = None
    soft_label = None
    N_label = None
    with torch.no_grad():
        for step, data in enumerate(dset_loaders['train_true']):
            input, label = data
            input = input.view(-1, 3, 32, 32).cuda()
        
            encoded_array, _, _, features = model_ft(input)
        
            if train_encode is None:
                train_encode = features.clone().detach()
                N_label = label
            else:
                train_encode = torch.cat((train_encode, features), 0)
                N_label = torch.cat((N_label, label), 0)
            
    save_checkpoint({
            'input': train_encode.to(cpu),
            'T_label': N_label,
            }, 'Data.pth.tar')'''
            
    
    new_size = task

    
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
    resume=os.path.join(exp_dir,'epoch.pth.tar')
    
    # 準備舊資料(training data)
    train_total_sample = [base_class_size * old_samples] + [inc_class_size*old_samples] * (new_size - 2)  
    train_class_num = [base_class_size] + [inc_class_size] * (new_size - 2)

    root = ''
    
    print('-------old class num: {}---------'.format(train_class_num))
    print('-------old training samples: {}---------'.format(train_total_sample))
    
    # save old data
    a = OldData.test(train_total_sample, class_num = train_class_num, 
        root = root, model=model_ft, New_data = True, file_size = new_size)
    
    train_dset = Prototype_generate(train_total_sample, class_num = train_class_num, 
            root = root, digits = digits, model=model_ft, New_data = True, file_size = new_size)
            
    train_old_size = len(train_dset)
    
    oldData_loader = loader(train_dset, batch_size = batch_size, shuffle = True)
                       

    for name, param in model_ft.encoder1.named_parameters():
        param.requires_grad = True
    for name, param in model_ft.encoder2.named_parameters():
        param.requires_grad = True
    for name, param in model_ft.decoder2.named_parameters():
        param.requires_grad = False
    for name, param in model_ft.decoder3.named_parameters():
        param.requires_grad = False
    for name, param in model_ft.encoder3.named_parameters():
        param.requires_grad = True
    for name, param in model_ft.decoder1.named_parameters():
        param.requires_grad = False
    #print(model_ft)

    
    # produce new neurons for encoder3 and decoder1 (load old weights)
    # encoder
    
    last_weight = model_ft.encoder3[1].weight.data#.state_dict()[layer_names[-2]]   #最後一層
    last_bias = model_ft.encoder3[1].bias.data#.state_dict()[layer_names[-1]]    # 倒數第二層(bias層)
    
    f = nn.Linear(prototype, class_num)
    #kaiming_normal_init(f.weight)
    f.weight.data[0:class_num - inc_class_size] = last_weight
    f.bias.data[0:class_num - inc_class_size] = last_bias
    
    
    model_ft.encoder3 = nn.Sequential(
        nn.ELU(inplace=False),
        f)
    
    # decoder    
    #layer_names = []
    #for param_tensor in model_ft.decoder1.state_dict():
    #    layer_names.append(param_tensor)
    
    last_weight1 = model_ft.decoder1[0].weight.data#.state_dict()[layer_names[-2]]   #最後一層
    last_bias1 = model_ft.decoder1[0].bias.data#.state_dict()[layer_names[-1]]    # 倒數第二層(bias層)
    
    f1 = nn.Linear(class_num, prototype)
    
    #print(last_weight1.size())
    f1.weight.data[:, 0:(class_num - inc_class_size)] = last_weight1
    f1.bias.data = last_bias1
        
    model_ft.decoder1 = nn.Sequential(
       f1)
    
    if use_gpu:
        model_ft = model_ft.cuda()
    
    # MAS_based_Training的train_model
    model_ft = train_Big_Separate(model_ft, previous_task_model_path, exp_lr_scheduler, lr, dset_loaders, oldData_loader, 
            dset_sizes, use_gpu, num_epochs, exp_dir, resume, class_num, task = task, reg_sets = reg_sets, 
            train_dset_size = train_old_size, new_size = new_size, base_class_size = base_class_size, inc_class_size = inc_class_size)

    
    #-----------------Test accuracy(包含前面的task)---------------------------
    # 紀錄每個任務的準確率
    all_task_acc=[]
    total_corects = 0
    total_data = 0
    for t in range(len(reg_sets)):
        print(reg_sets[t])
        dset = torch.load(reg_sets[t])
        dset_loader= torch.utils.data.DataLoader(dset['val_true'], batch_size=150,
                                               shuffle=False, num_workers=0)
             
        #print(dset_loader)
        dset_sizes = len(dset['val_true'])
        dset_classes = dset['val_true'].classes
        #print("dset_classes--------------------------------------------")
        #print(dset_classes)
        print('------------task {:} acc ---------------'.format(t + 1))
        
        epoch_acc, corrects, data_num = accuracy_result(model_ft, dset_loader, dset_sizes, len(dset_classes), t, base_class_size=base_class_size, inc_class_size=inc_class_size)
        total_corects += corrects
        total_data += data_num
        #epoch_acc = float(running_corrects) / len(dset['val_true'])

        print('{} {:} {} Acc: {:.4f}'.format('Task', t + 1, 'val', epoch_acc))
        all_task_acc.append(epoch_acc)
    
    
    
    print("------------task current acc ---------------")
    dset = torch.load(dataset_path)
    print('data/Pytorch_ImageNet_dataset_normalize//split' + str(len(reg_sets) + 1) + '_dataset.pth.tar')
    dset_loaders= torch.utils.data.DataLoader(dset['val_true'], batch_size=150,
                                               shuffle=False, num_workers=0)
                                               
    dset_sizes = len(dsets['val_true'])
    dset_classes = dsets['val_true'].classes
    

    epoch_acc, corrects, data_num = accuracy_result(model_ft, dset_loaders, dset_sizes, len(dset_classes),len(reg_sets), base_class_size = base_class_size, inc_class_size = inc_class_size)
    total_corects += corrects
    total_data += data_num
    
    print('{} Acc: {:.4f}'.format('val', epoch_acc))
    all_task_acc.append(epoch_acc)
    mean_acc = total_corects / total_data
    #print(model_ft.classifier)
    #exit()
    
    return model_ft, all_task_acc, mean_acc
    
            
def save_checkpoint(state, filename='checkpoint.pth.tar'):
    #best_model = copy.deepcopy(model)
    torch.save(state, filename)