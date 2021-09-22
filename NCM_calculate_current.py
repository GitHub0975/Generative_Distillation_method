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

EPOCH = 1
LR = 1e-4      # learning rate
DOWNLOAD_DATA = False
#prototype = 50

def sort_data(img, label):    #因為train_data是沒有順序的
    argsort_label = torch.argsort(label)
    sorted_img = torch.empty(img.size())
    sorted_label = torch.empty(label.size())
    for i in range(len(argsort_label)):
        sorted_img[i] = img[argsort_label[i]]
        sorted_label[i] = label[argsort_label[i]]
    
    return sorted_img.numpy(), sorted_label.numpy()

def NCM_result_current(autoencoder, train_loaders, TRAIN_SAMPLE, TRAIN_CLASS, task, inc_class_size):
    
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
            
            torch.cuda.empty_cache()
            x = x.view(-1, 3, 32, 32).to(device)
            encoded_array, _,_,_ = autoencoder(x)
            train_encode = encoded_array[2]
            
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
    
    mean_feature = np.zeros([TRAIN_CLASS, prototype])
    cov_feature = np.zeros([TRAIN_CLASS, prototype, prototype])
    
    init = 0
    for i in range(TRAIN_CLASS - inc_class_size, TRAIN_CLASS):
        # 計算目標索引值
        des = int(init + data_per_class[i])
        
        mean_feature[i] = np.mean(sort_feature[init:des], axis = 0)
        cov_feature[i] = np.cov(sort_feature[init:des].T)

        # 更新初始索引值
        init += int(data_per_class[i])
    
    np.set_printoptions(threshold=np.inf)

    np.save('./General_utils/information/feature_mean' + str(task), mean_feature[ -inc_class_size:])
    np.save('./General_utils/information/feature_cov' + str(task), cov_feature[-inc_class_size:])


    
    
if __name__ == '__main__':
    NCM_result('autoencoder.pkl')