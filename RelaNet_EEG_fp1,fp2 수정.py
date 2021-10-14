#!/usr/bin/env python
# coding: utf-8

# In[50]:


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
import torchvision.datasets as dset   #토치비전은 영상처리용 데이터셋, 모델, 이미지 변환기가 들어있는 패키지 dset: 데이터 읽어오는 역할
import torchvision.transforms as transforms  #불러온 이미지를 필요에 따라 변환해주는 역할
from torch.utils.data import DataLoader  #데이터를 배치 사이즈로 묶어서 전달하거나 정렬 또는 섞는 등의 방법으로 데이터 모델에 전달해줌
from sklearn.preprocessing import MaxAbsScaler, StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
from random import random, shuffle
from torch.autograd import Variable
from pylab import *
from sklearn.model_selection import train_test_split


"""
Motor data load
"""
total_sub=109    #총 sub 수: 109
data_size = 160  #1초=160
sample_class = 10  #만들 모델 수

#data : (109, 9600, 2)
data = scipy.io.loadmat('../JAEWON_RelationNet_study/DATA/motor_dataset/Motor_Imagery.mat')['data']


#스케일링 위해 2D로 변환
#data_2D : (2092800, 1)
data_2D = data.reshape(-1, 1)


#스케일링
SDscaler = StandardScaler()
SDscaler.fit(data_2D)
scaled_data = SDscaler.transform(data_2D)
scaled_data = scaled_data.reshape(total_sub, 9600, 2) #합쳤다가 풀어준거니 데이터 안섞임


#fp1, fp2 쉽게 나누기 위해 transpose해줌
#person: (109, )
person=[]
for i in range(total_sub):
    person.insert(i, scaled_data[i].T)


person = torch.Tensor(person)  #person: (109, 2, 9600)

fp1 = person[:,0,:]  #fp1: (109, 9600)  fp1만 빼옴
fp2 = person[:,1,:]  #fp2: (109, 9600)  fp2만 빼옴

reshape_fp1 = fp1.reshape(total_sub,1,9600)
reshape_fp2 = fp2.reshape(total_sub,1, 9600)

reshape_data = (reshape_fp1, reshape_fp2)
reshape_data = torch.cat(reshape_data, dim=1)  #reshape_data : (109, 2, 9600)


# In[51]:


"""
Embedding function
"""

class embedding_function(nn.Module):
    def __init__(self):
        super(embedding_function, self).__init__()
        
        self.layer1 = nn.Sequential(nn.Conv1d(2, 16, kernel_size=3, stride=1, padding=0),
                                    nn.BatchNorm1d(16, momentum=1, affine=True),
                                    nn.ReLU())
        self.layer2 = nn.Sequential(nn.Conv1d(16, 32, kernel_size=3, padding=0),
                                    nn.BatchNorm1d(32, momentum=1, affine=True),
                                    nn.ReLU())
        self.layer3 = nn.Sequential(nn.Conv1d(32, 64, kernel_size=3, padding=0),
                                    nn.BatchNorm1d(64, momentum=1, affine=True),
                                    nn.ReLU())
        
    
    def forward(self,x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        return out
  

    
"""
Relation function
"""

class relation_function(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(relation_function, self).__init__()
         
        self.layer1 = nn.Sequential(nn.Conv1d(128, 16, kernel_size=3, padding=1),
                                    nn.BatchNorm1d(16, momentum=1, affine=True),
                                    nn.ReLU())
        self.layer2 = nn.Sequential(nn.Conv1d(16, 32, kernel_size=3, padding=1),
                                    nn.BatchNorm1d(32, momentum=1, affine=True),
                                    nn.ReLU())
        self.layer3 = nn.Sequential(nn.Conv1d(32, 64, kernel_size=3, padding=1),
                                    nn.BatchNorm1d(64, momentum=1, affine=True),
                                    nn.ReLU())
#         self.layer4 = nn.Sequential(nn.Conv1d(64, 64, kernel_size=3, padding=1),
#                                     nn.BatchNorm1d(64, momentum=1, affine=True),
#                                     nn.ReLU())

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)

        
    def forward(self,x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.view(out.size(0), -1)     #(batch_size, -1)
        out = F.relu(self.fc1(out))
        out = F.sigmoid(self.fc2(out))
        
        return out
    
    
    
"""
가중치 초기화
"""

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        n = m.kernel_size[0] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm') != -1:
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        n = m.weight.size(1)
        m.weight.data.normal_(0, 0.01)
        m.bias.data = torch.ones(m.bias.data.size())
        
        
feature_encoder=[]
relation_network=[]
feature_encoder_optim=[]
relation_network_optim=[]

for i in range(sample_class):
    #각 sub 모델 생성
    feature_encoder.insert(i, embedding_function())
    relation_network.insert(i, relation_function(64*2394, 8))
    
    
    #모델 초기화
    feature_encoder[i].apply(weights_init)
    relation_network[i].apply(weights_init)
    
    
    #모델마다 최적화(학습률:0.0001)
    feature_encoder_optim.insert(i, torch.optim.Adam(feature_encoder[i].parameters(), lr=0.0001))
    relation_network_optim.insert(i, torch.optim.Adam(relation_network[i].parameters(), lr=0.0001)) 


# In[52]:


"""
Support, Query, Test 나누기
"""

Support=[]
Query=[]
Test=[]

sub = 0  #정답 class
reshape_data = reshape_data.numpy()  #tensor였던 것을 numpy로 바꿔줌(나중에 텐서로 다 바꾸줄기 위해)

#Support : (2, 160*15) * 109
#Query : (2, 160*15) * 109
#Test : (2, 160*30) * 109
for i in range(total_sub):
    Support.append(reshape_data[i][:,0:160*15])
    Query.append(reshape_data[i][:, 160*15:160*30])
    Test.append(reshape_data[i][:, 160*30:160*60])
  

#데이터 형태 tensor로 바꿔줌
Support = np.array(Support)
Support = torch.Tensor(Support)  #Support: (109, 2, 2400)

Query = np.array(Query)
Query = torch.Tensor(Query)     #Query: (109, 2, 4800)

Test = np.array(Test)
Test = torch.Tensor(Test)      #Support: (109, 2, 4800)


for model in range(sample_class):
    for episode in range (1000):
                
        sample = Support[:sample_class]  #sample: (sample_class, 2, 2400)

        query = Query[model, :,:]  #query: (2, 2400)
        query = query.reshape(1, 2, 160*15)  #query: (1, 2, 2400)
        
        
        sample_feature = feature_encoder[model](sample) #(sample_class, 64, 2394)
        query_feature = feature_encoder[model](query)  #(1, 64, 2394)


        query_feature_ext = query_feature.repeat(sample_class, 1, 1) #(sample_class, 64, 2394)

        relation_pair = torch.cat((sample_feature, query_feature_ext), dim=1) #(sample_class, 64*2, 2394)
        relations = relation_network[model](relation_pair) #(sample_class, 1)
    
        mse = nn.MSELoss()
        one_hot_labels = torch.zeros(sample_class, 1)
        one_hot_labels[model, :] = 1
    
        loss = mse(relations, one_hot_labels)
    
        feature_encoder[model].zero_grad()
        relation_network[model].zero_grad()
    
        loss.backward()
    
        feature_encoder_optim[model].step()
        relation_network_optim[model].step()
    
        if (episode)%1000 == 0:
            print("model:", model)
        
        if (episode+1)%100 == 0:
            #print("episode:", episode+1, "relations:", relations, "loss", loss.data)
            print("episode:", episode+1, ", loss:", loss.data)
            
    #torch.save(feature_encoder[model],str("./models/1/"+str(model)+"feature_encoder_"+str(model_cnt)+"way_"+str(3)+"shot.pkl"))
    #torch.save(relation_network[model],str("./models/1/"+str(model)+"relation_network_"+ str(model_cnt) +"way_" + str(3) +"shot.pkl"))
       


# In[81]:


for model in range(sample_class):
    for request in range(sample_class):
        print("model",model)
        print("request",request)
        
        test1 = Query[request, :, 0:160*15]  #test: (2, 2400)
        test1 = test1.reshape(1, 2, 160*15)

        test2 = Test[request, :, 160*15:160*30]  #test: (2, 2400)
        test2 = test2.reshape(1, 2, 160*15)

        sample_feature = feature_encoder[model](sample) #(sample_class, 64, 2394)
        test_feature = feature_encoder[model](test1)  #(1, 64, 2394)


        test_feature_ext = test_feature.repeat(sample_class, 1, 1) #(sample_class, 64, 1594)

        relation_pair = torch.cat((sample_feature, test_feature_ext), dim=1) #(sample_class, 64*2, 2394)

        relations = relation_network[model](relation_pair) #(sample_class, 1)

        me_class = 10000
        prob_other = 0
            
        for i in range(sample_class):
            if (relations[i]>0.3):
                me_class = i
                prob_me = relations[i]
            else:
                prob_other = prob_other + relations[i]

        if(me_class == 10000):
            me_class = "other"
        
        print("결과: ", me_class, "\n")
        


# In[87]:


model = 1
request = 1

test1 = Test[request, :, 0:160*15]  #test: (2, 2400)
test1 = test1.reshape(1, 2, 160*15)

test2 = Test[request, :, 160*15:160*30]  #test: (2, 2400)
test2 = test2.reshape(1, 2, 160*15)

sample_feature = feature_encoder[model](sample) #(sample_class, 64, 2394)
test_feature = feature_encoder[model](test1)  #(1, 64, 2394)


test_feature_ext = test_feature.repeat(sample_class, 1, 1) #(sample_class, 64, 1594)

relation_pair = torch.cat((sample_feature, test_feature_ext), dim=1) #(sample_class, 64*2, 2394)

relations = relation_network[model](relation_pair) #(sample_class, 1)
 
print(relations)


# for i in range(sample_class):
#     if (relations[i]>0.7):
#         relations[i]=1
#     else:
#         relations[i]=0
        
# print(relations)
 
result = 0
for i in range(sample_class):
    if (relations[i]>result):
        label=i
        result = relations[i]

print("label: ",label)

