import torch
from color import *
from datasets import *
from customlayer import *
from torch.nn.parameter import Parameter

#USE_CUDA = 1
#INBIT = 8
#OUTBIT = 2

class CNN(nn.Module):
    def __init__(self, INPUT_BIT, OUTPUT_BIT):
        super(CNN, self).__init__()
        n = INPUT_BIT
        k = OUTPUT_BIT
        self.nand1 = MyNand2Layer(n, True)
        self.conv1 = nn.Conv2d(1, 4, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu1 = MyRelu()
        self.conv2 = nn.Conv2d(4, 8, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu2 = MyRelu()
        self.conv3 = nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu3 = MyRelu()
        self.flat1 = FlattenLayer()

        self.fc1 = nn.Sequential(MyDense(8*n*n, 2*n*k), MyDense(2*n*k, 2*k))
        self.relu4 = MyRelu()


    def forward(self, x, k):
        x = self.nand1(x)
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.flat1(x)
        x = self.fc1(x)[:, 2*k:2*k+2]
        x = self.relu4(x)
        output = torch.max(x,dim=1)
        return output 


class cnnnode:
    def __init__(self, INPUT_BIT, OUTPUT_BIT, train_data, label_data, learning_rate=0.001, USE_CUDA=True):
        self.ib = INPUT_BIT
        self.ob = OUTPUT_BIT
        self.cnn = CNN(INPUT_BIT, OUTPUT_BIT)
        self.USE_CUDA = USE_CUDA
        if USE_CUDA:
            self.cnn = self.cnn.cuda() 
        self.lr = learning_rate

        self.correctness = 0

        self.EPOCH = 3000

        self.train_batch_size = 32
        self.test_batch_size = 256

        
        self.optimizer = torch.optim.Adam(self.cnn.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss(reduction='mean')#nn.CrossEntropyLoss()

        if train_data != None:

            torch_dataset= Data.TensorDataset(train_data, label_data)

            self.train_loader = Data.DataLoader(
                dataset=torch_dataset,
                batch_size=self.train_batch_size,
                shuffle=True,
                num_workers=0
            )

            self.test_loader = Data.DataLoader(
                dataset=torch_dataset,
                batch_size=self.test_batch_size,
                shuffle=False,
                num_workers=0
            )

        else:
                
            self.train_loader = MyDataLoader(32).loader
            self.test_loader = MyDataLoader(256, False).loader
            
    def train(self, epoch):
        self.cnn.train()
        for step, (b_x, b_y) in enumerate(self.train_loader):
            if self.USE_CUDA:
                b_x = b_x.to("cuda")
                b_y = b_y.to("cuda")
            for i in range(self.ob):
                bb_y = b_y[:, i]
                output = self.cnn(b_x, i)[0]
                loss = self.criterion(output, bb_y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            
    def test(self, epoch):
        self.cnn.eval()
        test_all = 0
        test_right = 0
        for test_step, (t_x, t_y) in enumerate(self.test_loader):
            if self.USE_CUDA:
                t_x = t_x.to("cuda")
                t_y = t_y.to("cuda")
            for i in range(self.ob):
                tt_y = t_y[:, i]
                output = self.cnn(t_x, i)[0]
                test_right += (torch.sum(torch.abs(torch.round(output)-tt_y)))
                #print(i, test_right)
        '''
        if test_right == 0 or (epoch+1) %10 == 0:
            print("\ntesting: %d"%(self.ib))
            print("epoch:", epoch, "accuracy: ", (1-test_right/t_y.shape[0]).item())
            print("test err: ", test_right.item(), "\n")
            self.correctness = test_right.item()
        '''
        if test_right == 0:
            save_path = "cnn_%d.pkl"%(self.ib)
            print("saving models to", save_path)
            #torch.save(net.state_dict(), save_path)
            return test_right
        return test_right
        return False

    def fit(self, Epoch=0):
        if Epoch == 0:
            Epoch = self.Epoch
        for bit in [0]: 
            for epoch in range(Epoch):
                self.train(epoch)

                acc = self.test(epoch)

                
                if (epoch+1) %10 == 0:                       
                    INFO("    epoch %d   testing  test err : %f"%(epoch, acc))

                if acc == 0:
                    #print(self.cnn.conv1.weight)
                    break
        return acc
    
    def predict(self, x):
        '''
        loader = torch.utils.data.DataLoader(
            dataset=torch_dataset,
            batch_size=self.BATCH_SIZE,
            shuffle=shuffle,
            num_workers=0
        )'''
        #DEBUG(x)
        x = torch.tensor(x, dtype=torch.float).view(1, self.ib).cuda()
        output = []
        for i in range(self.ob):
            output.append(self.cnn(x, i)[0].cpu())
        output = torch.cat(output)
        res = torch.round(output)
        #print(res)
        #print(res.detach())
        return res.detach().numpy()





if __name__ == "__main__":
    cnode = cnnnode(8, 3, None, None)
    cnode.fit()
    #cnode.predict()

    
        
