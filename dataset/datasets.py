import torch
import torch.utils.data as Data

import struct

import numpy as np


def tobinary(a, n):
    a = bin(a)[2:]
    a = a.rjust(n, '0')
    res = []
    for i in range(n):
        res.append(int(a[i]))
    return res


class MyDataLoader:
    def __init__(self, batch_size, shuffle=True, used = 1.0):
        self.BATCH_SIZE = batch_size
        x = []
        y = []

        '''
        for i in range(2**4):
            s = bin(i)[2:]
            s = s.rjust(4, '0')
            x.append([int(s[3]), int(s[2]), int(s[1]), int(s[0])])
            y.append( ((int(s[1])*int(s[3])) + (int(s[0])*int(s[2]))) % 2 )

        '''
        
        for i in range(2**4):
            for j in range(2**4):
                x.append(tobinary(i,4) + tobinary(j,4))
                y.append([int(int((i*j)/16)%2), int(int((i*j)/8)%2), int(int((i*j)/4)%2)])
                #print(i,j,x[-1],y[-1])
        '''
        for i in range(2**8):
            for j in range(2**8):
                x.append(tobinary(i,8) + tobinary(j,8))
                y.append(int(int((i+j)/128)%2))'''
                #print(i,j,x[-1],y[-1])
        #x = torch.linspace(1,10,10)
        #y = torch.linspace(10,1,10)
        
        #print(x,y)
        '''
        indices = range(len(x)) # indices = the number of images in the source data set
        np.random.shuffle(indices)
        '''
        x = torch.tensor(x).to(torch.float)
        y = torch.tensor(y).to(torch.float)
        #print(x.shape, y.shape)
        torch_dataset= Data.TensorDataset(x,y)

        self.loader = Data.DataLoader(
            dataset=torch_dataset,
            batch_size=self.BATCH_SIZE,
            shuffle=shuffle,
            num_workers=0
        )

if __name__ == "__main__":
    train_loader = MyDataLoader(1).loader
    for step, (b_x, b_y) in enumerate(train_loader):
        print(step, b_x, b_y)
        if step > 5:
            break