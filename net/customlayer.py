import torch
import torch.nn as nn
from torch.autograd import Function


class Quantize(Function):
    @staticmethod
    def forward(self, input, bit):
        quan_input = input.clone()
        if bit == 0:
            return quan_input
        else:
            quan_input.mul_(bit)
            quan_input = torch.round(quan_input)
            quan_input.div_(bit)
        return quan_input

    @staticmethod
    def backward(self, grad_output):
        return grad_output, None


class FlattenLayer(nn.Module):
    def __init__(self):
        super(FlattenLayer, self).__init__()

    def forward(self, x):
        return x.view(x.shape[0], -1)


class MyNotLayer(nn.Module):
    def __init__(self):
        super(MyNotLayer, self).__init__()

    def forward(self, x):  # x shape: (batch, *, *, ...)
        x = torch.stack((x, 1-x), dim=1)
        x = x.transpose(1,2).contiguous().view(x.size()[0],-1)
        return x

class MyRelu(nn.Module):
    def __init__(self):
        super(MyRelu, self).__init__()

    def forward(self, x):  # x shape: (batch, *, *, ...)
        #x = x.clamp(0,1)
        a = 0.1
        x = torch.where(x>1, (x-1)*a+1, x)
        x = torch.where(x<0, x*a, x)
        return x

class MyNandLayer(nn.Module):  # 自己定义层Flattenlayer
    def __init__(self, n):
        super(MyNandLayer, self).__init__()
        self.n = n
        self.mask = torch.zeros(n, n)
        for i in range(n):
            for j in range(n):
                if i > j:
                    self.mask[i, j] = 1

                    
    def forward(self, x):
        a = x.view(x.shape[0], self.n, 1)
        b = x.view(x.shape[0], 1, self.n)
        #print(a,b)
        x = a*b
        #print(self.mask)
        x = torch.where(self.mask>0, 1-x, x)
        return x

class MyNand2Layer(nn.Module):  # 自己定义层Flattenlayer
    def __init__(self, n, cuda=False):
        super(MyNand2Layer, self).__init__()
        self.n = n
        self.mask = torch.zeros(n, n)
        for i in range(n):
            for j in range(n):
                if i > j:
                    self.mask[i, j] = 1
        if cuda:
            self.mask = self.mask.to("cuda")


                    
    def forward(self, x):
        a = x.view(x.shape[0], self.n, 1)
        b = x.view(x.shape[0], 1, self.n)
        #print(a,b)
        x0 = a*b
        #x1 = (1-a)*(1-b)
        x1 = a+b-2*a*b
        #print(self.mask)
        #print(x0.device)
        #print(x1.device)
        #print(self.mask.device)
        #print(x.shape)
        x = torch.where(self.mask>0, x1, x0)
        #print(x.shape)
        x = torch.unsqueeze(x, axis=1)
        #print(x.shape)
        return x


class MyNand3Layer(nn.Module):  # 自己定义层Flattenlayer
    def __init__(self, n, cuda=False):
        super(MyNand3Layer, self).__init__()
        self.n = n
        self.mask = torch.zeros(n, n)
        self.mask2 = torch.zeros(n, n)
        for i in range(n):
            for j in range(n):
                if i > j:
                    self.mask[ i, j] = 1
                    self.mask2[j, i] = 1
        if cuda:
            self.mask  = self.mask.to("cuda")
            self.mask2 = self.mask2.to("cuda")


                    
    def forward(self, x):
        a = x.view(x.shape[0], self.n, 1)
        b = x.view(x.shape[0], 1, self.n)
        #print(a,b)
        x0 = a*b
        x1 = (1-a) * (1-b)
        x2 = a * (1-b)
        x3 = (1-a) * b
        #print(x0, x1, x2, x3)

        #y0 = torch.diag_embed(b).view(self.n, self.n)
        #y1 = torch.diag_embed((1-b)).view(self.n, self.n)

        #print(y0.shape, y1.shape)

        #y0 = torch.where(self.mask>0, x1, y0)
        #y0 = torch.where(self.mask2>0, x0, y0)
        #y1 = torch.where(self.mask>0, x3, y1)
        #y1 = torch.where(self.mask2>0, x2, y1)

        #print(y0.shape, y1.shape)

        #x = torch.where(self.mask>0, x1, x0)
        #x = torch.unsqueeze(x, axis=1)
        y2 = torch.stack((x0, x1, x2, x3)).view(x.shape[0], 4, self.n, self.n)
        return y2


class MyDense(nn.Module):
    def __init__(self, cin, cout, bit=0):
        super(MyDense, self).__init__()
        #self.weight=nn.Parameter(torch.zeros(cin, cout)+0.2)
        self.weight = nn.Parameter(torch.rand(cin, cout))
        self.bias = nn.Parameter(torch.zeros(cout))
        self.bit = bit


    def forward(self, x):
        a = 0.1
        #w = self.weight
        #b = self.bias
        #with torch.no_grad():
        self.quantize = Quantize.apply
        w = self.quantize(self.weight, self.bit)
        b = self.quantize(self.bias, self.bit)
        #print(self.bit, w)

        #w = torch.where(w>1, (w-1)*a+1, w)
        #w = torch.where(w<0, w*a, w)
        #w = self.weight.clamp(-0.5,1.5)
        #print(x.device)
        #print(w.device)
        #print(b.device)
        x = torch.mm(x, w) + b
        return x
    
    def setbit(self, b):
        self.bit = b
        print("set bit %d"%(b))
        return


    '''
    def quantize(self, t):
            #with torch.no_grad():
        quanz_t = t.clone()
        quan_input = t.clone()
        #print(quan_input.requires_grad, quan_input.grad)
        #with torch.no_grad():
        if 1:
            if self.bit == 0:
                return quanz_t
            else:
                quan_input.mul_(self.bit)
                quan_input = torch.round(quan_input)
                quan_input.div_(self.bit)

                #quanz_t = torch.round(t*self.bit).to(torch.float))/self.bit
                #return (torch.round(t*self.bit).to(torch.float))/self.bit
        #print(quan_input.requires_grad, quan_input.grad)
        return quan_input
    '''
'''
a = torch.tensor([0.1,0.4,0.6,0.9])
q = Quantize.apply
print(q(a,2))
print(q(a,1))
'''