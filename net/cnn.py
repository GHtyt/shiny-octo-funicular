import torch

from dataset.datasets import *
from net.customlayer import *
from torch.nn.parameter import Parameter

USE_CUDA = 1

class CNN(nn.Module):
    def __init__(self, n, k):
        super(CNN, self).__init__()
        self.nand1 = MyNand2Layer(n, True)
        self.conv1 = nn.Conv2d(1, 4, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(4, 8, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv3 = nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu1 = MyRelu()
        self.flat1 = FlattenLayer()
        '''
        self.fc = []
        for i in range(k):
            fci = nn.Sequential(MyDense(8*n*n, 2*n), MyDense(2*n, 2))
            if USE_CUDA:
                fci.cuda()
            self.fc.append(fci)
        '''
            #fc1 = MyDense(8*n*n, 2*n)
            #fc2 = MyDense(2*n, 2)
            #fc1.cuda()
            #fc2.cuda()
            #self.fc.append([fc1, fc2])


        self.fc1 = nn.Sequential(MyDense(8*n*n, 2*n*k), MyDense(2*n*k, 2*k))
        #self.fc2 = nn.Sequential(MyDense(8*n*n, 2*n*k), MyDense(2*n*k, 2*k))
        #self.fc.to("cuda")
        #self.fc1 = MyDense(8*n*n, 2*n)
        #self.fc2 = MyDense(2*n, 2)
        self.relu2 = MyRelu()


    def forward(self, x, k):
        #print(x.size())
        x = self.nand1(x)
        #print(x.size())
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        #print(x.size())
        x = self.relu1(x)
        x = self.flat1(x)

        #x = self.fc[k](x)
        #x = self.fc[k][0](x)
        #x = self.fc[k][1](x)

        x = self.fc1(x)[:, 2*k:2*k+2]

        '''
        if k == 0:
            x = self.fc1(x)
        else:
            x = self.fc2(x)
        '''
        
        #x = self.fc2(x)
        x = self.relu2(x)
       
        '''
        x0 = x
        x1 = x

        x0 = self.fc1(x0)
        x0 = self.fc2(x0)
        x0 = self.relu2(x0)

        x1 = self.fc1(x1)
        x1 = self.fc2(x1)
        x1 = self.relu2(x1)'''

        #print(x.shape)
        output = torch.max(x,dim=1)
        #print(output.shape)
        #output1 = torch.max(x1,dim=1)
        return output  #, output1

INBIT = 8
OUTBIT = 3

cnn = CNN(INBIT, OUTBIT)
if USE_CUDA:
    cnn = cnn.cuda()  #.to("cuda")



lr = 0.001
optimizer = torch.optim.Adam(cnn.parameters(), lr=lr)
criterion = nn.MSELoss(reduction='mean')#nn.CrossEntropyLoss()

train_loader = MyDataLoader(32).loader
test_loader = MyDataLoader(256, False).loader


def train(epoch, net, loader):
    net.train()
    for step, (b_x, b_y) in enumerate(loader):
        if USE_CUDA:
            b_x = b_x.to("cuda")
            b_y = b_y.to("cuda")
        for i in range(OUTBIT):
            bb_y = b_y[:, i]
            output = cnn(b_x, i)[0]
            loss = criterion(output, bb_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()





def test(bit, epoch, net, loader):
    net.eval()
    test_all = 0
    test_right = 0
    for test_step, (t_x, t_y) in enumerate(loader):
        if USE_CUDA:
            t_x = t_x.to("cuda")
            t_y = t_y.to("cuda")
        #print(t_x.shape, t_y.shape)
        #print(torch.sum(net.fc[0][1].weight))
        #print(net.fc[1][1].weight)
        #print(torch.sum(net.fc1[1].weight))
        for i in range(OUTBIT):
            tt_y = t_y[:, i]
            output = net(t_x, i)[0]
            test_right += (torch.sum(torch.abs(torch.round(output)-tt_y)))
            #print(i, test_right)
    if test_right == 0 or (epoch+1) %10 == 0:
        print("\ntesting: %d"%(bit))
        print("epoch:", epoch, "accuracy: ", (1-test_right/t_y.shape[0]).item())
        print("test err: ", test_right.item(), "\n")
    if test_right == 0:
        save_path = "cnn_%d.pkl"%(bit)
        print("saving models to", save_path)
        #torch.save(net.state_dict(), save_path)
        return True
    return False

def quantize(t, bit):
    with torch.no_grad():
        quanz_t = t.clone()
        if bit == 0:
            return (t)
        else:
            return Parameter((torch.round(t*bit).to(torch.float))/bit)




EPOCH=3000

epsilon = 0.1

test(0, 0, cnn, test_loader)

def quanz_all(net, bit):
    print("before quant: \n", cnn.nand1[2].weight)
    net.nand1[2].weight = quantize(cnn.nand1[2].weight, bit)
    net.nand1[2].bias = quantize(cnn.nand1[2].bias, bit)
    net.nand2[2].weight = quantize(cnn.nand2[2].weight, bit)
    net.nand2[2].bias = quantize(cnn.nand2[2].bias, bit)

#cnn.load_state_dict(torch.load("cnn_2.pkl"))

#quanz_all(cnn, 8)
#before = cnn.nand1[2].weight.clone()
for bit in [0]: #, 128, 64, 32,16,8,4,2,1]:

    #quanz_all(cnn)
    #cnn.nand1[2].setbit(bit)
    #cnn.nand2[2].setbit(bit)
    for epoch in range(EPOCH):
        #quanz_all(cnn)
        #if torch.sum(before != cnn.nand1[2].weight) == 0:
        #    print("no updating")
        #before = cnn.nand1[2].weight.clone()
        #print(cnn.nand1[2].weight)
        train(epoch, cnn, train_loader)
        '''
        with torch.no_grad():
            
            #print(cnn.nand1[2].weight)
            cnn.nand1[2].weight = quantize(cnn.nand1[2].weight, bit)
            #cnn.nand1[2].bias = quantize(cnn.nand1[2].bias, bit)
            cnn.nand2[2].weight = quantize(cnn.nand2[2].weight, bit)
        '''
        

        if test(bit, epoch, cnn, test_loader):
            #print(cnn.nand1[2].weight)
            print(cnn.conv1.weight)
            #print(tf.round(cnn.nand1[2].weight))
            break
'''
cnn2 = CNN(4)
cnn2.load_state_dict(torch.load("cnn_1.pkl"))

cnn2.nand1[2].setbit(1)
cnn2.nand2[2].setbit(1)
print(cnn2.nand1[2].weight)
print(cnn2.nand1[2].bit)
print(cnn2.nand2[2].weight)
print("final test:", test(0, 0, cnn2, test_loader))
'''
