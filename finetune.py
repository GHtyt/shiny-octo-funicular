import torch

from datasets import *
from customlayer import *
from torch.nn.parameter import Parameter



class CNN(nn.Module):
    def __init__(self, n):
        super(CNN, self).__init__()
        self.not1 = MyNotLayer()
        self.nand1 = nn.Sequential(
            MyNandLayer(2*n),
            FlattenLayer(),
            MyDense(4*n*n, n),
            #nn.Linear(4*n*n, n),
            #nn.ReLU()
            MyRelu()
        )
        self.nand2 = nn.Sequential(
            MyNandLayer(n),
            FlattenLayer(),
            MyDense(n*n, 1),
            #nn.Linear(n*n, 1),
            #nn.ReLU()
            MyRelu()
        )
    def forward(self, x):
        x = self.not1(x)
        x = self.nand1(x)
        x = self.nand2(x)
        #x = self.flat1(x)
        #x = self.linear1(x)
        #x = self.nand2(x)
        #x = self.nand3(x)
        #x = self.nand4(x)
        #output = self.pool1(x)
        output = torch.max(x,dim=1)
        return output

cnn = CNN(4)


def train(epoch, net, loader):
    net.train()
    for step, (b_x, b_y) in enumerate(loader):
        output = cnn(b_x)[0]
        #print(b_x, output, b_y)
        loss = criterion(output, b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()





def test(bit, epoch, net, loader):
    net.eval()
    test_all = 0
    test_right = 0
    for test_step, (t_x, t_y) in enumerate(loader):
        output = net(t_x)[0]
        test_right += (torch.sum(torch.abs(torch.round(output)-t_y)))

    if test_right == 0 or (epoch+1) %100 == 0:
        print("\ntesting: %d"%(bit))
        print("epoch:", epoch, "accuracy: ", (1-test_right/t_y.shape[0]).item())
        print("test err: ", test_right.item(), "\n")
    if test_right == 0:
        save_path = "cnn_%d.pkl"%(bit)
        print("saving models to", save_path)
        torch.save(net.state_dict(), save_path)
        return True
    return False

def quantize(t, bit):
    with torch.no_grad():
        quanz_t = t.clone()
        if bit == 0:
            return (t)
        else:
            return Parameter((torch.round(t*bit).to(torch.float))/bit)



lr = 0.1
optimizer = torch.optim.Adam(cnn.parameters(), lr=lr)
criterion = nn.MSELoss(reduction='mean')#nn.CrossEntropyLoss()

train_loader = MyDataLoader(16).loader
test_loader = MyDataLoader(16, False).loader

EPOCH=300

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
for bit in [0, 128, 64, 32,16,8,4,2,1]:

    #quanz_all(cnn)
    cnn.nand1[2].setbit(bit)
    cnn.nand2[2].setbit(bit)
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
            print(cnn.nand1[2].weight)
            #print(tf.round(cnn.nand1[2].weight))
            break

cnn2 = CNN(4)
cnn2.load_state_dict(torch.load("cnn_1.pkl"))

cnn2.nand1[2].setbit(1)
cnn2.nand2[2].setbit(1)
print(cnn2.nand1[2].weight)
print(cnn2.nand1[2].bit)
print(cnn2.nand2[2].weight)
print("final test:", test(0, 0, cnn2, test_loader))
   
