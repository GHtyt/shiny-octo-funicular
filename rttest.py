import tree.regressiontree as regressiontree
import dataset.binset as binset
import torch
from util.color import SHOW


def main():

    data = binset.build()
    
    x = torch.tensor([
        [0, 0, 0, 0, 1, 0],
        [0, 0, 1, 1, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 1, 1, 0, 0, 0],
        [1, 0, 0, 1, 1, 0],
        [1, 0, 1, 0, 1, 0],
        [1, 1, 0, 1, 1, 0],
        [1, 1, 1, 0, 0, 1]])
    data = binset.binset(6, 8, 3, x)
    
    

    # Train model
    reg = regressiontree.RegressionTree(bit=data.ll)
    reg.fit(data_all=data, max_depth=5)
    # Show rules
    #print(reg)
    pred = reg.predict(data)
    print(pred)
    print(data.label())
    #print(torch.sum(torch.abs(data.label()-pred)))
    SHOW("final error rate : " + str(torch.sum(torch.abs(data.label() - pred)).item()))
    #print(reg)
    # Model evaluation
    #get_r2(reg, data_test, label_test)


if __name__ == "__main__":
    main()
