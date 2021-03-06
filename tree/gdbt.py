from tree.regressiontree import *
from dataset.binset import *
import numpy as np


class GradientBoostingRegressor:
    """GBDT base class.
    http://statweb.stanford.edu/~jhf/ftp/stobst.pdf

    Attributes:
        trees {list}: A list of RegressionTree objects.
        lr {float}: Learning rate.
        init_val {float}: Initial value to predict.
    """

    def __init__(self):
        self.trees = None
        self.learning_rate = None
        self.init_val = None

    def _get_init_val(self, data):
        return data.avg

    def fit(self, data, n_estimators, learning_rate, max_depth, min_samples_split, max_nodes, subsample=None):

        # Calculate the initial prediction of y.
        self.init_val = self._get_init_val(data)
        # Initialize prediction.
        n_rows = data.y
        prediction = self.init_val  #.repeat(data.y, 1)
        #print(prediction)
        # Initialize the residuals.
        residuals = torch.abs(data.label() - prediction)

        print("prediction: ", prediction)
        print("residual: ", residuals)

        # Train Regression Trees
        self.trees = []
        self.learning_rate = learning_rate
        for T in range(n_estimators):
            SHOW("----BUILDING TREE: "+str(T))
            # Sampling with replacement
            idx = range(n_rows)
            if subsample:
                k = int(subsample * n_rows)
                idx = np.random.choice(idx, k, replace=True)
            data_sub = data.index(idx)
            residuals_sub = residuals[idx]
            #prediction_sub = prediction[idx]

            data_sub.data[:, data.ll:] = residuals_sub

            data_sub.initavg()

            # Train a Regression Tree by sub-sample of X, residuals
            tree = RegressionTree(bit = data.ll)
            #print("data_sub:\n ",data_sub.data)
            tree.fit(data_sub, max_depth, min_samples_split, max_nodes=max_nodes)

            #self._update_score(tree, data_sub, prediction_sub, residuals_sub)

            self.trees.append(tree)

            INFO("----CALCULATING PREDICTION")
            prediction = self.predict(data.data) #(prediction + tree.predict(data)) % 2
            INFO("----UPDATING RESITDUAL")
            residuals = torch.abs(data.label() - prediction)
            
            #print(tree.predict(data))
            print("pre: ", prediction)
            print("res: ", residuals)


            if torch.sum(residuals) == 0:
                break

    def predict_one(self, row):
        #print("one:")
        #print([tree.predict_one(row).numpy() for tree in self.trees])
        residual = np.sum([tree.predict_one(row) for tree in self.trees], axis=0) % 2
        #print(self.init_val, residual)
        #print((self.init_val.numpy() + residual) % 2)
        return (self.init_val.numpy() + residual) % 2

    
    def predict(self, data):
        data = data.numpy()
        #print(self.predict_one(data[1]))
        return np.apply_along_axis(self.predict_one, 1, data)

    
    def printf(self):
        for t in self.tree:
            print("tree:")
            for i in range(len(t.nodes)):
                print(t.nodes[i].pos, t.nodes[i].son, t.nodes[i].avg, t.nodes[i].split, t.nodes[i].mse, t.nodes[i].gain, t.nodes[i].remain)
            print()