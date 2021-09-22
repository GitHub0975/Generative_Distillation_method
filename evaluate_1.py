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
import pickle as pk
from torch.distributions.multivariate_normal import MultivariateNormal
from torchvision.datasets import ImageFolder
from PIL import Image
from sklearn.metrics import confusion_matrix, accuracy_score

def accuracy_result(autoencoder, test_loaders, TEST_SAMPLE, TEST_CLASS, task, base_class_size = 50, inc_class_size = 5):
    
    #計算autoencoder的size
    layer_names = []
    for param_tensor in autoencoder.encoder3.state_dict():
        layer_names.append(param_tensor)
    
    model_size = autoencoder.encoder3.state_dict()[layer_names[-1]].size(0) - base_class_size
    print('Model size : {}'.format(model_size))
    
    # 要比較的center檔案數量
    file_size = 1
    while model_size > 0:
        file_size +=1
        model_size -=inc_class_size
    
    
    file_name = task + 1
    #file_name = ((task + 1) // 2 + 1)
    
    autoencoder = autoencoder.eval()
    device = torch.device("cuda")
    cpu = torch.device("cpu")
    autoencoder.to(device)
    
    # 判斷該任務使用哪組mean和cov來分類
    print('./General_utils/information/feature_mean' + str(file_name) + '.npy')
    mean_feature = np.load('./General_utils/information/feature_mean' + str(file_name) + '.npy')
    mean_feature = torch.from_numpy(mean_feature).to(device)
    dis_size = mean_feature.shape[1]
    #print(dis_size)
    
    cov_feature = np.load('./General_utils/information/feature_cov' + str(file_name) + '.npy')
    cov_feature = torch.from_numpy(cov_feature).to(device)
    
    # 每個file的起始class
    class_start = [0]
    for i in range(1, file_size):
        if i == 1:
            class_start.append(base_class_size)
        else:
         class_start.append(class_start[i-1] + inc_class_size)
         
    # 載入所有mean和cov
    all_mean = []
    all_cov = []
    for i in range(file_size):
        mean = np.load('./General_utils/information/feature_mean' + str(i+1) + '.npy')
        mean = torch.from_numpy(mean).to(device)
        all_mean.append(mean)
        
        cov = np.load('./General_utils/information/feature_cov' + str(i+1) + '.npy')
        cov = torch.from_numpy(cov).to(device)
        all_cov.append(cov)
    
    # 現在的任務是在判別哪幾個class
    if task == 0:
        start = 0
        end = base_class_size
    else:
        start = base_class_size+(file_name-2)*inc_class_size
        end = base_class_size+(file_name-2)*inc_class_size + inc_class_size
        
    #print(start)
    #print(end)
    
    with torch.no_grad():
        
        predict_class = torch.zeros(TEST_SAMPLE)
        predict_class_classify = torch.zeros(TEST_SAMPLE)
        all_labels = torch.zeros(TEST_SAMPLE, )
        acc_data = 0
        numbers = 0
        
        
        for step, (x, b_label) in enumerate(test_loaders):
            #print(b_label)
            #print("------------------------")
            x = x.view(-1, 3, 32, 32).to(device)   # batch x, shape (batch, 28*28)
            
            # 階層一分類(New task or Old task)
            _, _, classify, features = autoencoder(x)
           
            
            # 判斷是哪個分類器
            _, preds = torch.max(classify, 1)
            #print(preds)
            
          

            data_size = classify.size()[0]    # 該batch的數量
            
            all_labels[acc_data:acc_data+data_size] = b_label
            
            # 用分類器判斷類別
            encoded = autoencoder.encoder3(autoencoder.encoder2(autoencoder.encoder1(features)))
            
            classClf = encoded
            #print(classClf.size())
            
            _, Clf_preds = torch.max(classClf, 1)
            
            predict_class_classify[acc_data:acc_data + data_size] = Clf_preds
            
            # 用NCM判斷類別 (用正確的file_name當center)
            mean_feature = np.load('./General_utils/information/feature_mean' + str(file_name) + '.npy')
            mean_feature = torch.from_numpy(mean_feature).to(device)
            dis_size = mean_feature.shape[1]
            #print(dis_size)
            
            cov_feature = np.load('./General_utils/information/feature_cov' + str(file_name) + '.npy')
            cov_feature = torch.from_numpy(cov_feature).to(device)
            
            clf_size = end - start
            distribution = torch.empty((clf_size, data_size))
            #for i in range(clf_size):
            #    m = MultivariateNormal(mean_feature[i], cov_feature[i])
            #    distribution[i] = m.log_prob(encoded[:, :dis_size])
                
            Clf_preds_NCM = torch.argmax(distribution, dim=0).to(device)  # 在正確任務中屬於哪一個類別
            #Clf_preds_NCM = torch.where(correct_preds > 0, Clf_preds_NCM, neg)    
            #predict_class[acc_data:acc_data + data_size] = Clf_preds_NCM.to(cpu)
            
            
            # 判斷最大機率值是否在正確的task內
            maxV_dis = torch.empty((file_size, data_size))
            
            for i in range(len(class_start)):   # 任務數量
                
                # 待計算機率密度函數的class數量
                if i != len(class_start) - 1:
                    clf_size = class_start[i+1] - class_start[i]
                else:
                    clf_size = inc_class_size
                    
                    #clf_size = end - class_start[i]
                #print('clf_size:{}'.format(clf_size))    
                # 載入mean和cov
                mean_feature = all_mean[i]
                cov_feature = all_cov[i]
                dis_size = mean_feature.size()[1]    # 橫向維度
                #dstart = class_start[i]
                #print(dstart)
                #print(dis_size)
                distribution = torch.empty((clf_size, data_size))    # 機率分佈(class_num, data_num)
                
                #for j in range(clf_size):
                #    m = MultivariateNormal(mean_feature[j], cov_feature[j])
                #    distribution[j] = m.log_prob(encoded[:, :dis_size])    # data_size
                max_value, _ = torch.max(distribution, 0) # data_size, class中的最大值是多少
                maxV_dis[i] = max_value
            #print(maxV_dis.size())
            #print(maxV_dis)
            #print("--"*100)
            _, Task_label = torch.max(maxV_dis, 0)    # 最大值屬於哪個task
            Task_label = Task_label.to(device)
            #print(Task_label)
            #print("="*100)
            
            #pos = torch.ones(classify.size(0)).to(device)
            neg = (torch.ones(classify.size(0)) * -1).to(device)
            neg = neg.type_as(Clf_preds_NCM)
            
            Clf_preds_NCM = class_start[(file_name - 1)] + Clf_preds_NCM
            correct_preds = torch.where(Task_label == (file_name - 1), Clf_preds_NCM, neg)
            #print(correct_preds)
            predict_class[acc_data:acc_data + data_size] = correct_preds.to(cpu)
            
            
            acc_data += data_size
            
        #print(predict_class)
        torch.set_printoptions(precision=None, threshold=np.inf, edgeitems=None, linewidth=None, profile=None)
      
        running_corrects_NCM = torch.sum(predict_class == all_labels.data)
        #print('判斷正確的張數: {}'.format(running_corrects_NCM))
        #print('總張數:{}'.format(acc_data))
        #print(running_corrects_NCM / acc_data)
        classifier_acc = accuracy_score(all_labels.numpy(), predict_class_classify.numpy())

        NCM_acc = accuracy_score(all_labels.numpy(), predict_class.numpy())
        Final_acc = classifier_acc
        #Final_acc = 0
        #if classifier_acc > NCM_acc:
        #    Final_acc = classifier_acc
        #else:
        #    Final_acc = NCM_acc
        
        print('classifier acc: {}'.format(classifier_acc))
        print('NCM acc: {}'.format(accuracy_score(all_labels.numpy(), predict_class.numpy())))
        #print('################################ test result ####################################')

        return Final_acc, running_corrects_NCM, acc_data

    
if __name__ == '__main__':
    accuracy_result('autoencoder.pkl')