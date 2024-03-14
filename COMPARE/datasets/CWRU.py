import os
from xml import dom
from scipy.io import loadmat
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from datasets.SequenceDatasets import dataset
from datasets.sequence_aug import *
from tqdm import tqdm

#Digital data was collected at 12,000 samples per second
signal_size = 1024
dataname= {0:["0_A.mat","1_B.mat","2_C.mat","3_D.mat"],  
           1:["0_A_T.mat","0_A_Test.mat","1_B_Test.mat","2_C_Test.mat","3_D_Test.mat"]} 

datasetname = ["source domain", "target domain","test domain"]
axis = ["data"]#["_DE_time", "_FE_time", "_BA_time"]

label = [i for i in range(0, 4)]

def get_files(root, N, domain):
    '''
    This function is used to generate the final training set and test set.
    root:The location of the data set
    '''
    data = []
    lab =[]
    if domain=="source":
        for k in range(len(N)):
            for n in tqdm(range(len(dataname[N[k]]))):
                print(dataname[N[k]][n])
                path1 =os.path.join(root,datasetname[N[k]], dataname[N[k]][n])
                data1, lab1 = data_load(path1,dataname[N[k]][n],label=label[n])
                data += data1
                lab +=lab1
    elif domain=="target_train":
        for k in range(len(N)):
                print(dataname[N[k]][0])
                path1 =os.path.join(root,datasetname[N[k]], dataname[N[k]][0])
                data1, lab1 = data_load(path1,dataname[N[k]][0],label=label[0])
                data += data1
                lab +=lab1
    elif domain=="target_test":
        for k in range(len(N)):
            for n in tqdm(range(len(dataname[N[k]]))):
                if n==0:
                    continue
                else:
                    print(dataname[N[k]][n])
                    path1 =os.path.join(root,datasetname[N[k]+1], dataname[N[k]][n])
                data1, lab1 = data_load(path1,dataname[N[k]][n],label=label[n-1])
                data += data1
                lab +=lab1

    return [data, lab]




# def get_files(root, N):
#     '''
#     This function is used to generate the final training set and test set.
#     root:The location of the data set
#     '''
#     data = []
#     lab =[]
#     for k in range(len(N)):
#         for n in tqdm(range(len(dataname[N[k]]))):
#             if n==0:
#                path1 =os.path.join(root,datasetname[3], dataname[N[k]][n])
#             else:
#                 path1 = os.path.join(root,datasetname[0], dataname[N[k]][n])
#             data1, lab1 = data_load(path1,dataname[N[k]][n],label=label[n])
#             data += data1
#             lab +=lab1

#     return [data, lab]


def data_load(filename, axisname, label):
    '''
    This function is mainly used to generate test data and training data.
    filename:Data location
    axisname:Select which channel's data,---->"_DE_time","_FE_time","_BA_time"
    '''
    datanumber = axisname.split(".")
    # if eval(datanumber[0]) < 100:
    #     realaxis = "X0" + datanumber[0] + axis[0]
    # else:
    #     realaxis = "X" + datanumber[0] + axis[0]
    fl = loadmat(filename)[axis[0]]
    data = []
    lab = []
    start, end = 0, signal_size
    while end <= fl.shape[0]:
        x = fl[start:end]
        x = np.fft.fft(x) #傅里叶变换
        x = np.abs(x) / len(x) #计算绝对值
        x = x[range(int(x.shape[0] / 2))]
        x = x.reshape(-1,1)
        data.append(x)
        lab.append(label)
        start += signal_size
        end += signal_size

    return data, lab

#--------------------------------------------------------------------------------------------------------------------
class CWRU(object):
    num_classes = 4
    inputchannel = 1
    def __init__(self, data_dir, transfer_task, normlizetype="0-1"):
        self.data_dir = data_dir
        self.source_N = transfer_task[0]
        self.target_N = transfer_task[1]
        self.normlizetype = normlizetype
        self.data_transforms = {
            'train': Compose([
                Reshape(),
                Normalize(self.normlizetype),
                # RandomAddGaussian(),
                # RandomScale(),
                # RandomStretch(),
                # RandomCrop(),
                Retype(),
                # Scale(1)
            ]),
            'val': Compose([
                Reshape(),
                Normalize(self.normlizetype),
                Retype(),
                # Scale(1)
            ])
        }

    def data_split(self, transfer_learning=True):
        if transfer_learning:
            # get source train and val
            list_data = get_files(self.data_dir, self.source_N,"source")
            data_pd = pd.DataFrame({"data": list_data[0], "label": list_data[1]})
            train_pd, val_pd = train_test_split(data_pd, test_size=0.2, random_state=40, stratify=data_pd["label"])
            source_train = dataset(list_data=train_pd, transform=self.data_transforms['train'])
            source_val = dataset(list_data=val_pd, transform=self.data_transforms['val'])

            # get target train and val
            list_data_target_train = get_files(self.data_dir, self.target_N,"target_train")
            list_data_target_test = get_files(self.data_dir, self.target_N,"target_test")

            data_pd_target_train = pd.DataFrame({"data": list_data_target_train[0], "label": list_data_target_train[1]})
            data_pd_target_test = pd.DataFrame({"data": list_data_target_test[0], "label": list_data_target_test[1]})

            target_train = dataset(list_data=data_pd_target_train, transform=self.data_transforms['train'])
            target_val = dataset(list_data=data_pd_target_test, transform=self.data_transforms['val'])

            return source_train, source_val, target_train, target_val
        else:
            #get source train and val
            list_data = get_files(self.data_dir, self.source_N,"source")
            data_pd = pd.DataFrame({"data": list_data[0], "label": list_data[1]})
            train_pd, val_pd = train_test_split(data_pd, test_size=0.2, random_state=40, stratify=data_pd["label"])
            source_train = dataset(list_data=train_pd, transform=self.data_transforms['train'])
            source_val = dataset(list_data=val_pd, transform=self.data_transforms['val'])

            # get target train and val
            list_data_target_train = get_files(self.data_dir, self.target_N,"target_train")
            list_data_target_test = get_files(self.data_dir, self.target_N,"target_test")

            data_pd_target_train = pd.DataFrame({"data": list_data_target_train[0], "label": list_data_target_train[1]})
            data_pd_target_test = pd.DataFrame({"data": list_data_target_test[0], "label": list_data_target_test[1]})
            
            val_pd_target_val = pd.concat([data_pd_target_train, data_pd_target_test])
            target_val = dataset(list_data=val_pd_target_val, transform=self.data_transforms['val'])

            return source_train, source_val, target_val


"""
    def data_split(self):

"""
