import random
import numpy as np
import scipy.io as scio
from sklearn import preprocessing
from models.CNN_1 import CNN as CNN_1d

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score,accuracy_score
from sklearn.manifold import TSNE
import matplotlib.colors as col
import torch.utils.data
from torch.utils.data import DataLoader, TensorDataset
import torchvision
# from torchvision import datasets, transforms

import datasets

import models

def TSNE1(model,dataloaders,filename):
    model.eval()
    source_feature = []
    source_label = []
    target_feature = []
    target_label = []
    with torch.no_grad():
        for data, target in dataloaders["source_train"]:
            feature = model(data)
            source_feature.append(feature)
            source_label.append(target)
        for data, target in dataloaders["source_val"]:
            feature = model(data)
            source_feature.append(feature)
            source_label.append(target)
        for data, target in dataloaders["target_val"]:
            feature = model(data)
            target_label.append(target)
            target_feature.append(feature)
    source_feature=torch.cat(source_feature, dim=0)
    target_feature=torch.cat(target_feature, dim=0)
    source_label=torch.cat(source_label, dim=0)
    target_label=torch.cat(target_label, dim=0)
    visualize(source_feature,source_label, target_feature,target_label,filename)
        

def test_Model(model, test_loader,filename):
    model.eval()
    # test_loss = 0
    correct = 0
    
    list_test_in = []
    list_test_out = []
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)  # logits
            # print(output)
            # loss_fn = nn.MSELoss(reduce=True, reduction='sum')
            # test_loss += loss_fn(output.float(), target.float()).item()  # sum up batch loss
            pred = output.argmax(dim=1)  # get the index of the max log-probability
            # print((pred==4).sum())
            # target = target.argmax(dim=1, keepdim=True)
            list_test_in += target.tolist()
            list_test_out += pred.tolist()
            correct += torch.eq(pred, target).float().sum().item()
    draw(list_test_in,list_test_out,filename)
    # test_loss /= len(test_loader.dataset)

    print('\nTest: accuracy: {}/{} ({:.2f}%)'.format(
         correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    return correct / len(test_loader.dataset)


def draw(actual_labels,predicted_labels,filename):
        # actual_labels = self.temp['true']
        # predicted_labels = self.temp['pred']
        
        cm = confusion_matrix(actual_labels, predicted_labels)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        # 绘制混淆矩阵
        plt.imshow(cm_normalized, interpolation='nearest', cmap=plt.cm.Blues, vmin=0, vmax=1)
        plt.colorbar()

        # 添加行标签
        classes = np.unique(actual_labels)
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes)
        plt.yticks(tick_marks, classes)

        # 添加各行和各列的样本数量标注
        thresh = cm_normalized.max() / 2.0
        for i in range(cm_normalized.shape[0]):
            for j in range(cm.shape[1]):
                # plt.text(j, i, str('%d%%'%float((cm_normalized[i, j])*100)), horizontalalignment="center",
                #          color="white" if cm_normalized[i, j] > thresh else "black")
                plt.text(j, i, str('%0.3f' % float((cm_normalized[i, j]))), horizontalalignment="center",
                         color="white" if cm_normalized[i, j] > thresh else "black")
                # plt.text(j, i, str(cm[i, j]), horizontalalignment="center",
                #          color="white" if cm[i, j] > thresh else "black")

        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.savefig(filename)

        # 计算预测精确度
        precision = precision_score(actual_labels, predicted_labels, average='weighted')

        # 计算召回率
        recall = recall_score(actual_labels, predicted_labels, average='weighted')

        # 计算F1分数
        f1 = f1_score(actual_labels, predicted_labels, average='weighted')

        # acc
        acc = accuracy_score(actual_labels, predicted_labels)
        # 输出结果
        print('Precision:', precision)
        print('Recall:', recall)
        print('F1 Score:', f1)
        print('acc Score:', acc)
        
def visualize(source_feature: torch.Tensor, source_label: torch.Tensor, target_feature: torch.Tensor,target_label: torch.Tensor,
              filename: str, source_color='r', target_color='b'):
    """
    Visualize features from different domains using t-SNE.

    Args:
        source_feature (tensor): features from source domain in shape :math:`(minibatch, F)`
        target_feature (tensor): features from target domain in shape :math:`(minibatch, F)`
        filename (str): the file name to save t-SNE
        source_color (str): the color of the source features. Default: 'r'
        target_color (str): the color of the target features. Default: 'b'

    """
    source_feature = source_feature.numpy()
    target_feature = target_feature.numpy()
    source_label = source_label.numpy()
    target_label = target_label.numpy()
    features = np.concatenate([source_feature, target_feature], axis=0)

    # map features to 2-d using TSNE
    X_tsne = TSNE(n_components=2, random_state=33).fit_transform(features)

    # domain labels, 1 represents source while 0 represents target
    domains = np.concatenate((source_label, target_label))

    # visualize using matplotlib
    fig, ax = plt.subplots(figsize=(10, 10))
    label_0=[]
    label_1=[]
    label_2=[]
    label_3=[]
    for i in range(0,len(domains)):
        if domains[i]==0:
            label_0.append(i)
        elif domains[i]==1:
            label_1.append(i)
        elif domains[i]==2:
            label_2.append(i)
        elif domains[i]==3:
            label_3.append(i)

    plt.scatter(X_tsne[label_0, 0], X_tsne[label_0, 1], cmap=col.ListedColormap(['r']), s=20,label='0')
    plt.scatter(X_tsne[label_1, 0], X_tsne[label_1, 1], cmap=col.ListedColormap(['b']), s=20,label='1')
    plt.scatter(X_tsne[label_2, 0], X_tsne[label_2, 1], cmap=col.ListedColormap(['g']), s=20,label='2')
    plt.scatter(X_tsne[label_3, 0], X_tsne[label_3, 1], cmap=col.ListedColormap(['y']), s=20,label='3')
    plt.legend()
    plt.savefig(filename)

def visualize1(source_feature: torch.Tensor, source_label: torch.Tensor, target_feature: torch.Tensor,target_label: torch.Tensor,
              filename: str, source_color='r', target_color='b'):
    """
    Visualize features from different domains using t-SNE.

    Args:
        source_feature (tensor): features from source domain in shape :math:`(minibatch, F)`
        target_feature (tensor): features from target domain in shape :math:`(minibatch, F)`
        filename (str): the file name to save t-SNE
        source_color (str): the color of the source features. Default: 'r'
        target_color (str): the color of the target features. Default: 'b'

    """
    source_feature = source_feature.numpy()
    target_feature = target_feature.numpy()
    source_label = source_label.numpy()
    target_label = target_label.numpy()
    features = np.concatenate([source_feature, target_feature], axis=0)

    # map features to 2-d using TSNE
    X_tsne = TSNE(n_components=2, random_state=33).fit_transform(features)

    # domain labels, 1 represents source while 0 represents target
    domains = np.concatenate((source_label, target_label))

    # visualize using matplotlib
    fig, ax = plt.subplots(figsize=(10, 10))
    # ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False)
    # ax.spines['bottom'].set_visible(False)
    # ax.spines['left'].set_visible(False)
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=domains, cmap=col.ListedColormap(['r', 'b','g','y']), s=20)
    plt.savefig(filename)

def main():
    args_batch_size=100
    args_num_workers=0
    args_last_batch=True
    args_data_name="CWRU"
    args_data_dir="./data/CWRU"
    args_bottleneck=True
    args_model_name="CNN_1d"
    args_pretrained=False
    args_bottleneck_num=256
    args_transfer_task= ['[', '0', ']', ',', '[', '1', ']']
    device = torch.device("cpu")
    args_normlizetype="mean-std"
    Dataset = getattr(datasets, args_data_name)
    datasets1 = {}
    if isinstance(args_transfer_task[0], str):
        #print(args.transfer_task)
        args_transfer_task = eval("".join(args_transfer_task))


    datasets1['source_train'], datasets1['source_val'], datasets1['target_val'] = Dataset(args_data_dir, args_transfer_task, args_normlizetype).data_split(transfer_learning=False)


    dataloaders = {x: torch.utils.data.DataLoader(datasets1[x], batch_size=args_batch_size,
                                                        shuffle=(True if x.split('_')[1] == 'train' else False),
                                                        num_workers=args_num_workers,
                                                        pin_memory=(True if device == 'cuda' else False),
                                                        drop_last=(True if args_last_batch and x.split('_')[1] == 'train' else False))
                        for x in ['source_train', 'source_val', 'target_val']}

    model = getattr(models, args_model_name)(args_pretrained)
    model.fc = torch.nn.Linear(model.fc.in_features, Dataset.num_classes)


    # basic
    # model.load_state_dict(torch.load(r'./checkpoint_adabn/CNN_1d_0730-104205/1-0.8940-best_model.pth'))
    # acc = test_Model(model, dataloaders['target_val'],'./confusion_matrix_basis')
    # TSNE1(model,dataloaders,'./TSNE_basis')

    # adabn
    model.load_state_dict(torch.load(r'./checkpoint_adabn/CNN_1d_0730-110845/5-0.8024-best_model.pth'))
    acc = test_Model(model, dataloaders['target_val'], './confusion_matrix_adabn')
    TSNE1(model, dataloaders, './TSNE_adabn')


main()