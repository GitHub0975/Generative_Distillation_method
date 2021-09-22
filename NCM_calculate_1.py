import os
import numpy as np
 
import torch
import torch.nn as nn
import torchvision.models as models
from torch.autograd import Variable 
import torch.cuda
import torchvision.transforms as transforms
import torchvision
import torch.utils.data as Data
import datetime
import pickle as pk
from torchvision.datasets import ImageFolder
from scipy.stats import multinomial
from numpy.random import multivariate_normal
from torch.distributions.multivariate_normal import MultivariateNormal
from sklearn.decomposition import PCA

from PIL import Image


feature_size = 8192

def sort_data(img, label):    #因為train_data是沒有順序的
    argsort_label = torch.argsort(label)
    sorted_img = torch.empty(img.size())
    sorted_label = torch.empty(label.size())
    for i in range(len(argsort_label)):
        sorted_img[i] = img[argsort_label[i]]
        sorted_label[i] = label[argsort_label[i]]
    
    return sorted_img.numpy(), sorted_label.numpy()

def NCM_result(autoencoder, train_loaders, TRAIN_SAMPLE, TRAIN_CLASS, t, base_class_size, inc_class_size):
    
    #autoencoder = torch.load(model_name)
    autoencoder = autoencoder.eval()
    device = torch.device("cuda")
    autoencoder.to(device)
    prototype = TRAIN_CLASS
    
    #model = make_model()


    
    with torch.no_grad():
        
        all_features = torch.empty(TRAIN_SAMPLE, prototype)
        all_labels = torch.empty(TRAIN_SAMPLE, )
        accu_data = 0
        data_per_class = np.zeros(TRAIN_CLASS)
        
        for step, (x, b_label) in enumerate(train_loaders):
            
            #features = extract_feature(model, x).view((BATCH_SIZE, 8192 )).to(device)
            #print(features.size())
            
            b_label = b_label.int()

            torch.cuda.empty_cache()
            x = x.view(-1, feature_size).to(device)
            train_encode = autoencoder.encoder3(autoencoder.encoder2(autoencoder.encoder1(x)))
            #train_encode = encoded_array[2]
            
            for j in range(train_encode.size()[0]):
                #print(train_encode[j])
                all_features[accu_data + j] = train_encode[j]
                all_labels[accu_data + j] = b_label[j]
                
                # 累加各類別的張數分別有多少張
                data_per_class[b_label[j]] += 1
            # 目前有多少筆data
            accu_data += train_encode.size()[0]
        
        torch.set_printoptions(threshold=np.inf)

    # 把訓練資料按照類別排好
    sort_feature, sort_label = sort_data(all_features, all_labels)
    
    #mean_feature = np.zeros([TRAIN_CLASS, prototype])
    #cov_feature = np.zeros([TRAIN_CLASS, prototype, prototype])
    
    init = 0
    for i in range(t):
        class_start = base_class_size * int(i>0) + int(i>1) * inc_class_size * (i-1)
        class_end = class_start + inc_class_size
        
        mean_feature = np.zeros([(class_end - class_start),prototype])
        cov_feature = np.zeros([(class_end - class_start),prototype, prototype])
    
    
        for j in range(class_start, class_end):
            # 計算目標索引值
            des = int(init + data_per_class[j])
            
            mean_feature[(j-class_start)] = np.mean(sort_feature[init:des], axis = 0)
            cov_feature[(j-class_start)] = np.cov(sort_feature[init:des].T)

            # 更新初始索引值
            init += int(data_per_class[i])
            
        np.set_printoptions(threshold=np.inf)
        print(mean_feature.shape)

        np.save('./General_utils/information/feature_mean' + str(i + 1), mean_feature)
        np.save('./General_utils/information/feature_cov' + str(i + 1), cov_feature)
    
    np.set_printoptions(threshold=np.inf)


    
if __name__ == '__main__':
    NCM_result('autoencoder.pkl')