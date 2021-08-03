import tree.regressiontree as regressiontree
import dataset.binset as binset
import tree.gdbt as gdbt
from util.color import SHOW
import torch

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
    
    gb = gdbt.GradientBoostingRegressor()
    gb.fit(data=data, n_estimators=5,
            learning_rate=0.5, max_depth=5, min_samples_split=2, max_nodes=100)

    #print(gb.predict(data.train()))
    #print(data.data)

    
    pred = gb.predict(data.train())
    print(pred)
    SHOW("final error rate : " + str(torch.sum(torch.abs(data.label() - pred)).item()))


if __name__ == "__main__":
    main()
