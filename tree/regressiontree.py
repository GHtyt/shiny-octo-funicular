from dataset.binset import *
from net.net import *
from util.heap import *
from util.color import *
import torch

class Node:
    
    attr_names = ("pos", "avg", "son", "split", "mse")

    LEAF=0
    TREE=1
    NET =2
    FUNC=3

    def __init__(self, pos=0, avg=-1, left=0, right=0, split=-1, mse=None, gain=1, remain = None):
        self.pos = pos
        self.avg = avg
        self.son = [left, right]
        self.split = split
        self.mse = mse
        self.remain = [0]
        self.gain = gain

        self.type = self.LEAF
        self.func = None
        '''
        if not remain:
            self.remain = [0]*bit
        else:
            self.remain = remain
        '''

    def predict(self, row):
        if self.type == Node.LEAF:
            return self.avg

        elif self.type == Node.NET:
            return self.func.predict(row)

    def copy(self, node):

        for attr_name in self.attr_names:
            attr = getattr(node, attr_name)
            setattr(self, attr_name, attr)

    def functype(self):
        if self.type == Node.LEAF:
            return "LEAF"
        else:
            return "NET"

    def __str__(self):
        s = ""
        if self.avg != None:
            #print("val: %d, spllit: %d, mse: %d\n\n"%(self.avg, self.split, self.mse))
            s += str(self.avg)+str(self.split)+str(self.mse)+"\n"
        else:
            s += str("None")+"\n"
        s += self.son[0]
        s += self.son[1]
        return s
    
    def update(self, k):
        self.remain[k] = 1
    
    def printf(self):
        #print("node:", self.avg, self.split, self.mse)
        if self.avg != None:
            #print("val: %d, spllit: %d, mse: %d\n\n"%(self.avg, self.split, self.mse))
            print("node:", self.avg, self.split, self.mse)
        else:
            print("None")

        print("left")
        if self.son[0]:
            self.son[0].printf()
        
        print("right")
        if self.son[1]:
            self.son[1].printf()


class RegressionTree:
    def __init__(self, bit=0, max_nodes=8):
        self.nodes = [Node()]
        self.bit = bit
        self.next = 0
        self.root = self.newnodes(0)
        self.nodes[self.root].remain = [0]*self.bit
        self.depth = 1
        self._rules = None

        self.max_nodes = max_nodes


    def newnodes(self, i):
        self.next += 1
        self.nodes.append(Node(pos=self.next))
        #print(self.nodes[i].remain[:])
        #print(self.nodes)
        self.nodes[self.next].remain = self.nodes[i].remain[:]
        return self.next


    @staticmethod
    def _get_split_mse(data_all, split_pos):

        data_left, data_right = data_all.split(split_pos)

        mse = (data_left.mse + data_right.mse) / data_all.y


        return mse  #, node

    def _calculate_mse_gain(self, node, data_all):
        n = self.nodes[node]
        n.mse = data_all.mse
        n.avg = data_all.avg
        k = -1
        min_mse = 1e6
        for i in range(self.bit):
            #print(i)
            if n.remain[i]==0:
                m = self._get_split_mse(data_all, i)
                #print(i, m)
                if m < min_mse:
                    k = i
                    min_mse = m
        if k == -1:
            return -1
        else:
            n.split=k
            dleft, dright = data_all.split(n.split)
            n.gain = dleft.mse + dright.mse - n.mse
            
            #print(node, ": ", k, dleft.data, "\n", dright.data)
        return k

    def _choose_feature(self, node, data_all):
        
        n = node
        if n.gain == 1 or n.split == -1:
            return [0, 0, 0]
        else:
            n.son = [self.newnodes(node.pos), self.newnodes(node.pos)]
            n1, n2 = n.son
            self.nodes[n1].update(n.split)
            self.nodes[n2].update(n.split)
            dleft, dright = data_all.split(n.split)
            #print("choosing", data_all.data, dleft.data, dright.data)
            self._calculate_mse_gain(n1, dleft)
            self._calculate_mse_gain(n2, dright)
            return [1, dleft, dright]
            


    def fit(self, data_all, max_depth=7, min_samples_split=1):

        self._calculate_mse_gain(self.root, data_all)
        que = [(self.depth + 1, self.nodes[self.root], data_all)]

        # Breadth-First Search.
        j = 0
        while que and self.next <= self.max_nodes:
            #print(j)
            j += 1
            #print(que)
            #depth, node, dall = que.pop(0)
            depth, node, dall = heappop(que, lambda x:x[1].gain)

            if depth > max_depth:
                depth -= 1
                break

            if dall.y < min_samples_split or dall.unique: # or sum(remain) == self.bit: # or all(_label == label[0]):
                #node.avg = dall.avg
                continue

            INFO("learning node %d"%(node.pos))
            train_data = dall.train().to(torch.float)
            label_data = dall.label().to(torch.float)
            #print(train_data.data, label_data.data)
            cnode = cnnnode(dall.ll, dall.dl, train_data, label_data)
            acc = cnode.fit(100) / dall.x            
            SHOW("    node %d, acc : %f"%(node.pos, 1-(acc/dall.x)))

            if acc < 0.1:
                SHOW("    method pass %d"%(node.pos))
                node.type = Node.NET
                node.func = cnode
            
            else:
                ERR("    spliting node")



                err, dleft, dright = self._choose_feature(node, dall)
                if err == 0:
                    continue
                else:
                    #que.append((depth + 1, self.nodes[node.son[0]], dleft))
                    #que.append((depth + 1, self.nodes[node.son[1]], dright))
                    heappush(que, (depth + 1, self.nodes[node.son[0]], dleft),  key=lambda x:x[1].gain)
                    heappush(que, (depth + 1, self.nodes[node.son[1]], dright), key=lambda x:x[1].gain)

            self.depth = depth
        #self.get_rules()
        
        #self.root.printf()

        
    def predict_one(self, row):

        node = self.nodes[self.root]
        while node.split != -1 and node.son[row[node.split]] != 0:
            #print(node.pos, row, node.son[row[node.split]] )
            node = self.nodes[node.son[row[node.split]]]
            #print(node.pos, row, node.son[row[node.split]] )
        #print(node.pos)

        return node.predict(row)

    def predict(self, data):
        data = data.data.numpy()
        self.printf()
        #print(self.predict_one(data[1]))
        return np.apply_along_axis(self.predict_one, 1, data)

    #def __str__(self):
    #    self.root.printf()

    def printf(self):
        for i in range(len(self.nodes)):
            print(self.nodes[i].pos, self.nodes[i].functype(), self.nodes[i].son, self.nodes[i].avg, self.nodes[i].split, self.nodes[i].mse, self.nodes[i].gain, self.nodes[i].remain)
