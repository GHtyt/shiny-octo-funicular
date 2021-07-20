import torch
import numpy as np

import random

class binset:

    def __init__(self, x, y, label_len, data=None):
        self.x = x
        self.y = y
        self.ll = label_len
        self.dl = x-label_len
        self.unique = False
        if data!=None:
            self.data = torch.tensor(data).view(y, x)
        else:
            self.data = torch.zeros(y, x)
        #print(x,y,self.data)
        self.initavg()
    
    def initavg(self):
        self.avg = self.mean()
        self.mse = self.calculate_mse(self.avg)
        if self.mse == 0:
            self.unique=True
        
        #print(self.avg)

    def mean(self):
        #print(self.ll)
        #print(self.data[:, int(self.ll):])
        avg = torch.sum(self.data[:, int(self.ll):], dim=0)/self.y
        #print("avg  ", avg)

        return torch.round(avg)

    def distance(self, i):
        #assert i.shape == (1, self.x-self.ll), i.shape
        #i = i.repeat(self.y, 1)
        #print(self.data[:, int(self.ll):])
        #print("why", i)
        return torch.sum(self.data[:, int(self.ll):]-i, dim=1)

    
    def calculate_mse(self, i):
        #assert i.shape == (1, self.x-self.ll), i.shape

        dis = self.distance(i)
        #print(dis)
        f = lambda x, y: x*x
        ms = dis.map_(dis, f)

        return torch.sum(ms)

    def split(self, i):
        index0 = (self.data[:, i] == 0)
        index1 = (self.data[:, i] == 1)
        
        data0 = self.data[index0].to(torch.int)
        data1 = self.data[index1].to(torch.int)
        #print(data0.shape)
        bset0 = binset(data0.shape[1], data0.shape[0], self.ll, data0)
        bset1 = binset(data1.shape[1], data1.shape[0], self.ll, data1)
        return bset0, bset1

    def train(self):
        return self.data[:, :self.ll]

    def label(self):
        return self.data[:, self.ll:]

    def index(self, idx):
        d = self.data[idx]
        return binset(d.shape[1], d.shape[0], self.ll, d)


def tobinary(a, n):
    a = bin(a)[2:]
    a = a.rjust(n, '0')
    res = []
    for i in range(n):
        res.append(int(a[i]))
    return res

def build():
    random.seed(0)
    x=[]
    y=[]
    for i in range(2**2):
        for j in range(2**1):
            x.append(tobinary(i,2) + tobinary(j,1)+[round(random.random()), round(random.random())])
            y.append(int(int((i*j)/16)%2))
            #print(i,j,x[-1],y[-1])

    #print(x)

    x=binset(5, 8, 3, x)

    print("here\n", x.data)
    print(x.split(2)[0].unique)
    return x

#build()