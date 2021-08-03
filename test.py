import torch
import torch.utils.data as Data
from sklearn.decomposition import PCA
import struct
import matplotlib.pyplot as plt


def tobinary(a, n):
    a = bin(a)[2:]
    a = a.rjust(n, '0')
    res = []
    for i in range(n):
        res.append(int(a[i]))
    return res

x = []
y = []

'''
for i in range(2**4):
    s = bin(i)[2:]
    s = s.rjust(4, '0')
    res =  ((int(s[1])*int(s[3])) + (int(s[0])*int(s[2]))) % 2
    if res == 1:        
        x.append([int(s[3]), int(s[2]), int(s[1]), int(s[0])])
    #y.append( ((int(s[1])*int(s[3])) + (int(s[0])*int(s[2]))) % 2 )

print(x)'''
'''
for i in range(2**8):
    for j in range(2**8):
        if int(int((i*j)/256)%2) == 0:
            x.append(tobinary(i,8) + tobinary(j,8))
        #y.append(int(int((i*j)/128)%2))
        #print(i,j,x[-1],y[-1])

print(len(x))



pca = PCA(8)
pca.fit(x)

print(pca)
print(pca.explained_variance_ratio_)
print(pca.explained_variance_)

X_new = pca.transform(x)
plt.scatter(X_new[:, 0], X_new[:, 1],marker='o')
plt.show()



#from scipy.cluster.vq import kmeans
'''
'''
import torch
#def f(x,*y): return 4*x+10
f = lambda x, y: 4*x+10
x = torch.randint(0,3,(2,3))
print('原来的x是\n{}'.format(x))
x.map_(x,f)
print(x)

'''
'''
from customlayer import *

a= torch.tensor([[1,0,1]])
nl = MyNand3Layer(3)
b = nl(a)
print(b)
print(b.shape)

'''
from dataset.binset import *

x = build()
y = x.data.sum(dim=0)/256
print(y)