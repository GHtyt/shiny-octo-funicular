import torch

def tobinary(a, n):
    a = bin(a)[2:]
    if n > len(a):
        a = a.rjust(n, '0')
    else:
        a = a[-n:]
    res = [int(i) for i in a]
    return res

def toint(a, left=-1, right=-1):
    if left == -1:
        b = a
    else:
        b = a[left:right]
    #print(b)
    b = "".join([str(i) for i in b])
    return int(b, 2)

def sampling8(n, mask, val):
    a = torch.round(torch.rand([n, 8]))
    print(a)
    mask0 = [mask[i] for i in range(len(val)) if val[i]==0]
    mask1 = [mask[i] for i in range(len(val)) if val[i]==1]
    a[:, mask0] = 0
    a[:, mask1] = 1

    a = a.numpy()
    print(a)
    for i in a:
        print(i)
    
    
    '''
    print(mask)
    m = torch.tensor([0,1,0,2,0,0,0,0])
    print(m)
    m2 = torch.zeros(n, len(mask)) + torch.tensor(val)
    print(m2)
    b = torch.zeros(n, 8).scatter(1, m, m2)
    print(b)
    '''
    x = torch.rand(n, len(mask))
    print(x)
    print(torch.LongTensor([mask]*n))
    print(torch.zeros(n, 8).scatter_(1, torch.LongTensor([mask]*n), x))

    
    res = torch.tensor([[2,3], [1,2]], dtype=torch.float)
    lm = [0,1]
    #print(res.detach())
    if len(lm) > 0:
        #print("res", res)
        res = torch.zeros(2, 3).scatter_(1, torch.LongTensor([lm]*res.shape[0]), res)
        print(res)
    #for i in range(16):


if __name__ == "__main__":
    #print(toint([1,1,1,1], 2, 3))
    #print(tobinary(15, 3))
    sampling8(4, [0,3,5], [1,0,1])