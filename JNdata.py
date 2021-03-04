from torch.utils import data
import torch.utils.data
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import os
import torch
import datetime
import json
import networkx as nx
import scipy.sparse as sp
import copy
import random
import scipy.stats

import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import math
import matplotlib.mlab as mlab
from scipy.stats import norm
from model_configs import MODEL_CONFIGS

from utils import random_cover

# total dates
# traning_op, training_ed = '20170903', '20170930'

# training date
# testing_op, testing_ed = '20170901', '20170902'

mapped_edge_filepath = f'{MODEL_CONFIGS["PROJ_ROOT"]}/data/road_network/JN_mapped_edges.json'
edge_info_filepath = f'{MODEL_CONFIGS["PROJ_ROOT"]}/data/road_network/JN_edges.json'
node_neighbor_edge_filepath = f'{MODEL_CONFIGS["PROJ_ROOT"]}/data/road_network/JN_node_neighbor_edges.json'
datadir = f'{MODEL_CONFIGS["PROJ_ROOT"]}/data/'

edge_num = 608 #2348 #608 #397

day_time_interval_num = int((60/5)*24)



seqlen = 12

class JN(Dataset):
    def __init__(self, args, mode, drop_rate=0):
        if mode == 'train':
            date_op, date_ed = args.training_op, args.training_ed
        else:
            date_op, date_ed = args.testing_op, args.testing_ed

        # the traffic data here is the average speed of each road in the network
        self.data = load_data('speed', seqlen, date_op, date_ed)

        # the support feature is the number of trajectory used to calculate the speed
        self.support = load_data('support', seqlen, date_op, date_ed, normalize=False)

        if drop_rate != 0 and mode == 'train':
            self.data, noise_index = random_cover(self.data, drop_rate, mask_value=0)
            self.support[noise_index] = 0

        # the historical data is the average speed over pass a month
        self.historical = torch.mean(self.data, 0)
        self.historical = torch.stack([self.historical for i in range(self.data.shape[0])])


    def __getitem__(self, index):
        data = self.data[index]

        support = self.support[index]

        historical = self.historical[index]

        return data, support, historical

    def __len__(self):
        return len(self.data)

def create_graph(data:np.ndarray, norm=True):
    zeros = np.where(data==0)
    if norm:
        norm_data = np.tril(data, -1)
        norm_data = norm_data[np.where(norm_data!=0)]
        min_ = np.min(norm_data)
        max_ = np.max(norm_data)
        data = (data - min_)/(max_ - min_)
    data[zeros] = 0
    h, w = data.shape
    assert h == w
    G = nx.Graph() 
    G.add_nodes_from([str(i) for i in range(h)])
    G.add_weighted_edges_from([(str(i), str(i), 1.) for i in range(h)])
    where = np.where(data != 0)
    G.add_weighted_edges_from(zip(*[item.astype('str') for item in where], data[where]))
    return G

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def load_adj():

    with open(edge_info_filepath, 'r') as f:
        edge_info = json.load(f)
    with open(mapped_edge_filepath, 'r') as f:
        mapped_edges = json.load(f)
    with open(node_neighbor_edge_filepath, 'r') as f:
        node_neighbors = json.load(f)

    nodeNum = len(mapped_edges)
    adj = np.eye(nodeNum)
    for edge in edge_info:
        edge_id = mapped_edges[edge]
        sn = str(edge_info[edge]['sn'])
        en = str(edge_info[edge]['en'])
        neighbors = []
        neighbors.extend(node_neighbors['in'][sn])
        neighbors.extend(node_neighbors['in'][en])
        neighbors.extend(node_neighbors['out'][sn])
        neighbors.extend(node_neighbors['out'][en]) 
        for neighbor in neighbors:
            adj[edge_id][mapped_edges[str(neighbor)]] = 1
    sparse_adj = sp.csr_matrix(adj)
    sparse_adj = normalize(sparse_adj)
    sparse_adj = sparse_mx_to_torch_sparse_tensor(sparse_adj)

    adj = torch.FloatTensor(normalize(adj))
    print('adj shape:', sparse_adj.shape)
    return adj



def load_data(data_source, seqlen, date_op, date_ed, normalize=True):
    # data shape: batch * seqlen * feature
    with open(mapped_edge_filepath, 'r') as f:
        mapped_edges = json.load(f)
    totle_time_interval_num = day_time_interval_num * (int(date_ed) - int(date_op) + 1)
    data = np.zeros((edge_num, totle_time_interval_num))
    speedfiles = os.listdir(datadir + data_source + '/')
    for speedfile in speedfiles:
        if speedfile.startswith('.') or speedfile.split('.')[0] > date_ed or speedfile.split('.')[0] < date_op:
            continue
        day = int(speedfile.split('.')[0][-2:]) - int(date_op[-2:])
        with open(datadir + data_source + '/' + speedfile, 'r') as f:
            speed = json.load(f)
            for edge in speed:
                # current edge not in the selected region
                if edge not in mapped_edges:
                    continue
                for time_interval in speed[edge]:
                    edge_id = mapped_edges[edge]
                    data[edge_id, day * day_time_interval_num + int(time_interval)] = speed[edge][time_interval]
    
    ub_index = np.where(data == 0)
    if normalize:
        data = (data-data.min()) / data.max()
    data[ub_index] = 0
    
    output = []
    for i in range(totle_time_interval_num - seqlen):
        output.append(data[:,i:i+seqlen])
    output = torch.FloatTensor(output).permute(0,2,1)

    filled_rate = len(np.where(output.numpy() != 0)[0]) / (output.shape[0]*output.shape[1]*output.shape[2])
    
    print(len(np.where(output.numpy()[:72,:,:] != 0)[0]) / output[:72,:,:].numel())
    print(len(np.where(output.numpy()[72:120,:,:] != 0)[0]) / output[72:120,:,:].numel())
    print(len(np.where(output.numpy()[120:204,:,:] != 0)[0]) / output[120:204,:,:].numel())
    print(len(np.where(output.numpy()[204:240,:,:] != 0)[0]) / output[204:240,:,:].numel())
    print(len(np.where(output.numpy()[240:288,:,:] != 0)[0]) / output[240:288,:,:].numel())

    print(data_source, 'data shape:', output.shape, 'filled rate:', filled_rate)
    return output


if __name__ == "__main__":
    
    class Args: pass
    args = Args()
    args.training_op, args.training_ed, args.testing_op, args.testing_ed = '20170903', '20170930','20170901', '20170902'
    test_dataset = JN(args,mode='test')
    test_support = test_dataset.support.numpy()
    test_speed = test_dataset.data.numpy()
#     np.save(f'{MODEL_CONFIGS["PROJ_ROOT"]}/data/road_network/test_support_intro.npy', test_support)

#     train_dataset = JN(args,mode='train')
#     train_support = train_dataset.support.numpy()
#     train_speed = train_dataset.data.numpy()

#     re_mask = np.load(f'{os.getcwd()}/data/model/re_mask.npy')
#     re_x = np.load(f'{os.getcwd()}/data/model/re_x.npy')

    # adj = load_adj()


