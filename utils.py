import os
import numpy as np
import scipy.sparse as sp
import torch
import copy
import torch.nn.functional as F
import random

def road_network_sim(path='../data/Road_Network_JN_2016Q1_SIM.txt'):
    with open(path, 'r') as f:
        nodenum = f.readline()
        for i in range(nodenum):
            node = f.readline()



def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def evaluate(x, re_x, drop_rate=0.1, mask_value=0):

    if type(x) != torch.Tensor:
        x = torch.from_numpy(x)
    if type(re_x) != torch.Tensor:
        re_x = torch.from_numpy(re_x)

    def random_cover(x, drop_rate, mask_value):
        # random cover some (dicided by drop_rate) filled data to evaluate the model
        noise_x = copy.deepcopy(x)
        filled_index = torch.where(noise_x != 0)
        noise_index = random.sample(range(0, len(filled_index[0])), int(drop_rate * len(filled_index[0])))

        noise_index = (filled_index[0][np.array(noise_index)], filled_index[1][np.array(noise_index)])
        noise_x[noise_index] = mask_value

        return noise_x, noise_index

    noise_x, noise_index = random_cover(x, drop_rate, mask_value)

    loss = F.mse_loss(re_x[noise_index], x[noise_index])
    return loss.sum().item()


def random_cover(x, drop_rate, mask_value):
    # random cover some (dicided by drop_rate) filled data to evaluate the model
    x = torch.Tensor(x)
    noise_x = copy.deepcopy(x)
    filled_index = torch.where(noise_x != 0)
    noise_index = random.sample(range(0,len(filled_index[0])), int(drop_rate*len(filled_index[0])))

    index = []
    for i in range(len(filled_index)):
        index.append(filled_index[i][np.array(noise_index)])
    # index = tuple(index)
    # print(index, noise_x.shape)
    noise_x[index] = mask_value

    return noise_x, index


def setup_random_seed(seed=623):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
