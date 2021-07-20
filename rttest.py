import regressiontree
import binset
import torch


def main():

    data = binset.build()
    x = torch.tensor([[0, 0, 0, 0, 1],
        [0, 0, 1, 1, 0],
        [0, 1, 0, 0, 0],
        [0, 1, 1, 0, 0],
        [1, 0, 0, 1, 1],
        [1, 0, 1, 0, 1],
        [1, 1, 0, 1, 1],
        [1, 1, 1, 0, 0]])
    data = binset.binset(5, 8, 3, x)
    

    # Train model
    reg = regressiontree.RegressionTree(bit=data.ll)
    reg.fit(data_all=data, max_depth=5)
    # Show rules
    #print(reg)
    print(reg.predict(data.train()))
    print(data.data)
    #print(reg)
    # Model evaluation
    #get_r2(reg, data_test, label_test)


if __name__ == "__main__":
    main()
