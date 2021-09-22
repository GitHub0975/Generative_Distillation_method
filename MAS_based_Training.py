from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from MNIST_Net import *
import matplotlib.pyplot as plt
import torch.optim as optim
import time
import copy
import os
import pdb
import math
import shutil
from torch.utils.data import DataLoader
from NCM_calculate_1 import NCM_result
from NCM_calculate_current import NCM_result_current
from evaluate_1 import accuracy_result
from itertools import cycle
from Data_generate import *
from torch import autograd
from torch.utils.data import DataLoader as loader
from torch.nn import functional as F
#end of imports
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

# the weights of losses
prototype = 2048
feature_size = 8192
weight_decay_phase2 = 0.0
mse_epoch = 100
weight_decay_phase1=1e-5

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def exp_lr_scheduler(optimizer, epoch, decay_rate=0.0008, lr_decay_epoch=200):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    if epoch % lr_decay_epoch ==0 and epoch !=0 and epoch >99:
        for param_group in optimizer.param_groups:
        
            param_group['lr'] = param_group['lr'] * 0.1
            print('LR is set to {}'.format(param_group['lr']))

    return optimizer


def _compute_loss( output, target, old_target, class_weight):
    #output=self.model(imgs)
    #target = get_one_hot(target, self.numclass)
    #output, target = output.to(device), target.to(device)
    
    #old_target=torch.sigmoid(self.old_model(imgs))
    old_task_size = old_target.shape[1]
    # new task loss
    loss1 = class_weight*F.binary_cross_entropy_with_logits(output, target)
    # old task loss
    loss2 = F.binary_cross_entropy_with_logits(output[:,:old_task_size], old_target)
    #target[:, :old_task_size] = old_target

    return loss1+loss2#F.binary_cross_entropy_with_logits(output, target)
    
    
def train_Big_Separate(model, previous_task_model_path,lr_scheduler,lr,dset_loaders, oldData_loader, 
        dset_sizes,use_gpu, num_epochs,exp_dir='./',resume='', class_num=2, 
        task = 1, reg_sets=[], train_dset_size = 1, new_size = 1, base_class_size = 50, inc_class_size = 5):
    """Train a given model using MAS optimizer. The only unique thing is that it passes the importnace params to the optimizer"""

    print('dictoinary length'+str(len(dset_loaders)))
    since = time.time()
    
    #optimizer_ft = optim.SGD(model.parameters(), lr, momentum=0.9)
    #optimizer_ft = optim.Adam(model.parameters(), lr, amsgrad = True)
    #optimizer_ft = optim.Adam(model.parameters(), lr)


    best_model = model
    best_acc = 0.0
    if os.path.isfile(resume):
        print("=> loading checkpoint '{}'".format(resume))
        checkpoint = torch.load(resume)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        #print('load')
        optimizer_ft.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(resume, checkpoint['epoch']))
    else:
            start_epoch=0
            print("=> no checkpoint found at '{}'".format(resume))
    #start_epoch=0
    print('Start_epoch: {}'.format(str(start_epoch)))

    cos_history = []
    mse_history = []
    l1_history = []
    total_loss = []
    acc_history = []
    regulizer = []
    
    
    optimizer_ft = optim.Adam(model.parameters(), lr, amsgrad=True,weight_decay=weight_decay_phase1)
                
    # training new_class encoder            
    # joint training
    
    # 載入未加權重的model for distillation loss
    old_model = torch.load(previous_task_model_path)['model']
    old_model.eval().cuda()
    
    for epoch in range(start_epoch, num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        
        optimizer_ft = exp_lr_scheduler(optimizer_ft, epoch, decay_rate = 0.1, lr_decay_epoch=50)
        
        # Each epoch has a training and validation phase
        model.train(True)  # Set model to training mode

        running_loss = 0.0
        running_corrects = 0
            
        # 因為是兩個Dataloader一起使用，怕數量會不一樣，因此都重新計算
        total_samples = 0
      
        for step, data in enumerate(zip(oldData_loader, cycle(dset_loaders['train_true']))):
                
            # get the inputs
            old_input = data[0][0].view(-1, feature_size)
            old_label = data[0][1]
            
            new_input = data[1][0].view(-1, 3, 32, 32)    # data_loader1的image   
            label2 = data[1][1]    # 舊模型輸出
                
            if use_gpu:
                model = model.cuda()
                new_input = new_input.cuda()
                old_input = old_input.cuda()
                old_label = old_label.to(device="cuda", dtype=torch.int64)
                label2 = label2.to(device="cuda", dtype=torch.int64)
                

            # zero the parameter gradients
            optimizer_ft.zero_grad()    #不然gradient會每個batch加總


            # New task forward
            encoded, _, _, features = model(new_input)
            #all_preds = model.encoder3(model.encoder2(model.encoder1(new_input)))
            # old model forward
            #soft_target = old_model.encoder3(old_model.encoder2(old_model.encoder1(new_input)))
            soft_target,_,_,_ = old_model(new_input)
            soft_target = torch.sigmoid(soft_target[2])
            
            one_hot=torch.zeros(label2.shape[0],class_num).cuda()#.to(device)
            one_hot=one_hot.scatter(dim=1,index=label2.long().view(-1,1),value=1.).cuda()
            class_weight = 0.1
            #if encoded[2].size()[1] > 50:
            #    class_weight = 0.01
            
            loss_new = _compute_loss(encoded[2], one_hot, soft_target, class_weight)
            
            # old task forward
            old_encoded = model.encoder3(model.encoder2(model.encoder1(old_input)))
            old_soft_target = old_model.encoder3(old_model.encoder2(old_model.encoder1(old_input)))
            old_soft_target = torch.sigmoid(old_soft_target)
            
            old_one_hot=torch.zeros(old_label.shape[0],class_num).cuda()#.to(device)
            old_one_hot=old_one_hot.scatter(dim=1,index=old_label.long().view(-1,1),value=1.).cuda()
            
            loss_old = _compute_loss(old_encoded, old_one_hot, old_soft_target, class_weight)
            
            
            _, preds = torch.max(encoded[2], 1)
            _, old_preds = torch.max(old_encoded, 1)
            
            #print('=======================================')
            #print(preds)
            #print('---------------------------------------')
            #print(label2)
            
            loss = loss_old + loss_new
            loss.backward()
            optimizer_ft.step()


            # statistics
            running_loss += loss.data
            running_corrects += torch.sum(preds == label2.data) + torch.sum(old_preds == old_label.data)
            total_samples = total_samples + encoded[2].size()[0] + old_encoded.size()[0]
            
        epoch_loss = running_loss / total_samples
        epoch_acc = running_corrects.float() / total_samples
        
        #cls_loss = new_loss / label2.size()[0]
        #dis_loss = dist_loss / label2.size()[0]
            #儲存各loss值
        #total_loss.append(loss)
        #acc_history.append(epoch_acc)
            
        print('Training Phase1 Loss: {:} Total Acc: {:.4f}'.format(
                epoch_loss, epoch_acc))
        print('[Last batch] Old task loss: {:} New task loss: {:}'.format(
                loss_old, loss_new))
                
    #exit()            
    # Fix the layer of encoder
    for name, param in model.encoder1.named_parameters():
        param.requires_grad = False
    for name, param in model.encoder2.named_parameters():
        param.requires_grad = False
    for name, param in model.encoder3.named_parameters():
        param.requires_grad = False
    for name, param in model.decoder1.named_parameters():
        param.requires_grad = True
    for name, param in model.decoder2.named_parameters():
        param.requires_grad = True
    for name, param in model.decoder3.named_parameters():
        param.requires_grad = True
        
        
    
    optimizer = optim.Adam(model.parameters(), 1e-4, amsgrad=True, weight_decay = weight_decay_phase2)
    
    # training with MSE Loss           
    for epoch in range(mse_epoch):
        print('Epoch {}/{}'.format(epoch, mse_epoch - 1))
        print('-' * 10)
        
        # Each epoch has a training and validation phase
        model.train(True)  # Set model to training mode

        running_loss = 0.0
        main_running_loss = 0.0
        running_corrects = 0
        
            
        # 因為是兩個Dataloader一起使用，怕數量會不一樣，因此都重新計算
        total_samples = 0
      
        for step, data in enumerate(zip(oldData_loader, cycle(dset_loaders['train_true']))):
                
            # get the inputs
            old_input = data[0][0].view(-1, feature_size)
            old_label = data[0][1]
            
            new_input = data[1][0].view(-1, 3, 32, 32)    # data_loader1的image   
            new_label = data[1][1]
                
            if use_gpu:
                old_input = old_input.cuda()
                new_input = new_input.cuda()
                old_labels = old_label.to(device="cuda", dtype=torch.int64)
                labels = new_label.to(device="cuda", dtype=torch.int64)

            # zero the parameter gradients
            optimizer.zero_grad(set_to_none=True)    #不然gradient會每個batch加總

            # New task forward
            encoded, _, _, features = model(new_input)
            
            #encoded = model.encoder1(new_input)
            #encoded1 = model.encoder2(encoded)
            #all_preds = model.encoder3(encoded1)
            
            decoded = model.decoder1(encoded[2])
            decoded1 = model.decoder2(decoded)
            decoded2 = model.decoder3(decoded1)
            
            # old task forward
            old_encoded = model.encoder1(old_input)
            old_encoded1 = model.encoder2(old_encoded)
            old_all_preds = model.encoder3(old_encoded1)
            
            old_decoded = model.decoder1(old_all_preds)
            old_decoded1 = model.decoder2(old_decoded)
            old_decoded2 = model.decoder3(old_decoded1)
            
            loss_mse = nn.MSELoss()
            
            main_loss = loss_mse(features, decoded2)
            old_main_loss = loss_mse(old_input, old_decoded2)
            
            loss_new = loss_mse(encoded[1], decoded) + loss_mse(features, decoded2) + loss_mse(encoded[0], decoded1)
            loss_old = loss_mse(old_encoded1, old_decoded) + loss_mse(old_input, old_decoded2) + loss_mse(old_encoded, old_decoded1)
                
            _, preds = torch.max(encoded[2], 1)
            _, old_preds = torch.max(old_all_preds, 1)
            
            loss = loss_old + loss_new
            loss.backward()
            #optimizer_ft3.step()
            #optimizer_ft2.step()
            optimizer.step()
            

            # statistics
            main_running_loss += main_loss.data + old_main_loss.data
            running_loss += loss.data
            running_corrects += torch.sum(preds == labels.data) + torch.sum(old_preds == old_labels.data)
            total_samples = total_samples + encoded[2].size()[0] + old_all_preds.size()[0]
        
        epoch_main_loss = main_running_loss / total_samples
        epoch_loss = running_loss / total_samples
        epoch_acc = running_corrects.float() / total_samples

            
        print('Training Phase2 Loss: {:} Acc: {:.4f} main loss: {:}'.format(
                epoch_loss, epoch_acc, epoch_main_loss))          

    
          
    # 儲存decoder (防止bias跑掉)
    #torch.save(model.decoder1, './AuxModel/Decoder' + str(new_size) + '.pth.tar')
    # old data loader(新+舊NCM都需要被更新)
    # old data calculation
    NCM_result(model, oldData_loader, train_dset_size, class_num, task, base_class_size, inc_class_size)
    # New task data calculation
    NCM_result_current(model, dset_loaders['val_true'], dset_sizes['val_true'], class_num, task, inc_class_size)
    #exit()
   
    epoch_file_name=exp_dir+'/'+'epoch'+'.pth.tar'
    save_checkpoint({
            'epoch': epoch + 1,
            'model': model,
            'state_dict': model.state_dict(),
            'optimizer' : optimizer_ft.state_dict(),
                },epoch_file_name)
    
    #exit()
    return model
    



def save_checkpoint(state, filename='checkpoint.pth.tar'):
    #best_model = copy.deepcopy(model)
    torch.save(state, filename)

def random_shuffle(encoded_vec, label):

    tensor_size = encoded_vec.size()[0]
    order = torch.randperm(tensor_size)    # 順序表
    shuffle_encode = torch.empty(tensor_size, prototype)
    shuffle_label = torch.empty(tensor_size)
    
    # 按順序表填入相對應的tensor以及標籤
    for i in range(tensor_size):
        shuffle_encode[i] = encoded_vec[order[i]]
        shuffle_label[i] = label[order[i]]
        
    return shuffle_encode, shuffle_label
   
