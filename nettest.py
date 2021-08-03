from net.net import *
import torch
import pickle


if __name__ == "__main__":
    cnode = cnnnode(8, 3, None, None)
    cnode.fit()
    #cnode.predict()
    
    #cnode2 = cnnnode(8, 3, None, None)
    #cnode2.cnn.load_state_dict(torch.load("pkl/cnn_8_3.pkl"))
    #print(cnode2.cnn.conv1.weight)
    #print(cnode2.cnn.conv1.bias)
    #f = open("pkl/cnn_8_3.pkl", "rb")
    #p = pickle.load(f)
    #print(p)
