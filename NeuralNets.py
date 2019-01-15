# -*- coding: utf-8 -*-
"""
Created on Sat Dec  8 13:59:22 2018

@author: Mathieu
"""

import torch
import numpy as np
import torch.utils.data as utils
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import pandas as pd
from sklearn.utils import shuffle
from sklearn import datasets
from skorch import NeuralNetClassifier

from torch.utils.data.sampler import SubsetRandomSampler


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.dropout=nn.Dropout(0.5)
        self.fc1 = nn.Linear(30, 15)
        self.fc2 = nn.Linear(15, 1)
        self.fc3 = nn.Sigmoid()

    def forward(self, x):
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def train_test_dataloaders(df_features, df_targets, train_portion=0.8, rand=23,
                           batch_size=1):
    np_features = df_features.values
    np_targets = df_targets.values
    x = torch.from_numpy(np_features).float()
    y = torch.from_numpy(np_targets).float()
    
    full_dataset = utils.TensorDataset(x,y) 
    train_size = int(train_portion * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split( \
            full_dataset, \
            [train_size, test_size])
    test_dataloader = utils.DataLoader(test_dataset,
                        shuffle=True, num_workers=0,batch_size=batch_size)
    train_dataloader = utils.DataLoader(train_dataset,
                        shuffle=True, num_workers=0,batch_size=batch_size)
    
    return train_dataloader, test_dataloader

def nn_test(dataloader, model):
    for i, data in enumerate(dataloader, 0):
        predict = net()

def nn_train (dataloader, net=Net(), epochs=10,
              criterion_o = torch.nn.MSELoss(),
              optimizer_o = optim.SGD):
    '''
    Alert : Directly alters the given model
    '''
    net=net
    model.train()
    criterion = criterion_o
    #criterion = nn.CrossEntropyLoss()
    optimizer = optimizer_o(net.parameters(), lr=0.1, momentum=0.9)
    print(net.parameters())
    for epoch in range(epochs):  # loop over the dataset multiple times
        weights1 = net.fc1.weight
        running_loss = 0.0
        total_loss = 0.0
        for i, data in enumerate(dataloader, 0):
            # get the inputs
            inputs, labels = data
            labels = labels.unsqueeze(0)
            # zero the parameter gradients
            optimizer.zero_grad()
    
            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    
            # print statistics
            running_loss += loss.item()
            total_loss += loss.item()
            if i % 30 == 29:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
        print('Epoch : ',epoch, ' - Loss :', total_loss)
#        assert not(torch.all(torch.eq(weights1, net.fc1.weight)))
    print('Finished Training')
    return net

model= Net()
epoch= 10
rand = 23
batch = 64
iris = datasets.load_iris()
bc = datasets.load_breast_cancer()

df2 = pd.DataFrame(iris.data)
df2['target'] = iris.target
df2 = shuffle(df2, random_state =rand)

df3=pd.DataFrame.from_csv('data.csv')
df3=df3.drop('Unnamed: 32', axis=1)

df3c2=df3.copy()
df3c2['diagnosis']=df3c2['diagnosis'].apply(lambda x: 1 if x=='M' else 0)
df3c2 = shuffle(df3c2, random_state =rand)

df = df3c2.copy()
target = 'diagnosis'
net = Net()

torch.set_default_tensor_type('torch.FloatTensor')

df_features = df.drop(target,axis=1)
df_features_n=(df_features-df_features.mean())/df_features.std()
df_targets = df[target]
#df_targets_n = (df_targets-df_targets.mean())/df_targets.std()

train_data, test_data = train_test_dataloaders(df_features_n,df_targets)



