
import numpy as np
from Processing import list_to_bin,bin_to_list
from functools import reduce


def winning_move(piles, binary=False):
    if binary:
        piles = bin_to_list(piles)
    # print(piles)
    znot = list(map(lambda x: x-(reduce(lambda a, b: a ^ b, piles) ^ x), piles))
    idx = znot.index(max(znot))
    val = max(max(znot), 0)
    if val:
        loc = np.zeros([1, 3], dtype=np.float64)
        loc[0, idx] = 1
        results = list_to_bin([piles[idx] - val])
        results = np.concatenate((results, loc), axis=1)

        return results
    else:
        return np.zeros([1, 9], dtype=np.float64)

def get_perm(bound):
    for i in range(bound):
        for j in range(bound):
            for k in range(bound):
                yield i,j,k


def rebuild():
    bound = 63
    piles = list(get_perm(bound))
    n = len(piles)
    # data = np.zeros([int(64**3),27], dtype=np.ndarray)
    x,y=[],[]

    [x.append(list_to_bin(p)[0]) for p in piles]
    [y.append(winning_move(p)[0]) for p in piles]
    x=np.stack(x)
    y=np.stack(y)
    np.save("X",x)
    np.save("Y",y)
    return x,y

def labeled_data():
    try:
        x = np.load("X" + ".npy")
        y = np.load("Y" + ".npy")
    except:
        x,y = rebuild()
    # np.random.seed(1000)
    # np.random.shuffle(kl)
    # return kl[0:200000,:],kl[200001::,:]
    return x,y

if __name__ == '__main__':
    x,y = labeled_data()