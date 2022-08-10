# -*- coding:utf-8 -*-
# Author : Leehom
# Data : 2021/1/13 10:00

import torch
from scipy import sparse
from torch.utils.data import Dataset
from enum import Enum
import random
from datetime import datetime
import networkx as nx
import time
import numpy as np
import itertools
import scipy.sparse as sp
from sklearn.metrics import roc_auc_score, average_precision_score,explained_variance_score,mean_absolute_error,mean_squared_error
import torch.nn as nn
import math
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error

class GowallaLoader():
    def __init__(self,file,min_checkins,interval_hour,max_users=-1):

        self.user2id = {}   # 重编码的user集合 {原编码: 重编码;原编码: 重编码... }
        self.max_users = max_users
        self.min_checkins = min_checkins
        self.file=file
        self.poi2id = {}    # 重编码的poi集合 {原编码: 重编码;原编码: 重编码... }
        self.interval_hour = interval_hour
        # self.consecutively_Visit_dics = {}
        self.users = []
        self.times = []
        self.coords = []
        self.locs = []


    def load_users(self):
        f = open(self.file, 'r')
        lines = f.readlines()

        prev_user = int(lines[0].split('\t')[0])  # 先提取第一行数据的第一列（即第一个值）
        visit_cnt = 0  # 计数
        for i, line in enumerate(lines):
            # print("第",i,"行记录")
            tokens = line.strip().split('\t')
            user = int(tokens[0])
            if user == prev_user:
                visit_cnt += 1
            else:
                if visit_cnt >= self.min_checkins:
                    self.user2id[prev_user] = len(self.user2id)
                prev_user = user
                visit_cnt = 1
                if self.max_users > 0 and len(self.user2id) >= self.max_users:
                    break
        if visit_cnt >= self.min_checkins:  # 当user不等于prev_user后，查看计数是否超过最小签到数量（101）
            self.user2id[prev_user] = len(self.user2id)
        return self.user2id

    def POINumber(self):
        f = open(self.file, 'r')
        lines = f.readlines()
        # store location ids
        all_poi = []
        for i, line in enumerate(lines):
            tokens = line.strip().split('\t')
            poi = int(tokens[0])
            all_poi.append(poi)
        PoiNumber = len(np.unique(all_poi))
        self.load_users()
        # collect checkins for all collected users:
        users,times,coords,locs = self.load_pois()
        PoiNumber02=len(np.unique(list(itertools.chain.from_iterable(locs)))) # 获得poi数量

        return PoiNumber,PoiNumber02

    def load_pois(self):
        f = open(self.file, 'r')
        lines = f.readlines()

        user_time = []
        user_coord = []
        user_loc = []

        prev_user = int(lines[0].split('\t')[0])
        prev_user = self.user2id.get(prev_user)
        for i, line in enumerate(lines):
            tokens = line.strip().split('\t')
            user = int(tokens[0])
            if self.user2id.get(user) is None:
                continue
            user = self.user2id.get(user)

            timeArray = time.strptime(tokens[1], "%Y-%m-%dT%H:%M:%SZ")
            timeSeq = int(time.mktime(timeArray))
            lat = float(tokens[3])  # 纬度
            long = float(tokens[2])  # 经度
            coord = (lat, long)

            location = int(tokens[4])  # location nr
            if self.poi2id.get(location) is None:  # get-or-set locations
                self.poi2id[location] = len(self.poi2id)  ##location重新编号
            location = self.poi2id.get(location)

            if user == prev_user:
                user_time.insert(0, timeSeq)
                user_coord.insert(0, coord)
                user_loc.insert(0, location)
            else:
                self.users.append(prev_user)
                self.times.append(user_time)
                self.coords.append(user_coord)
                self.locs.append(user_loc)

                prev_user = user
                user_time = [timeSeq]
                user_coord = [coord]
                user_loc = [location]

        self.users.append(prev_user)
        self.times.append(user_time)
        self.coords.append(user_coord)
        self.locs.append(user_loc)

        return self.users,self.times,self.coords,self.locs

    def load_pois_V(self):

        f = open(self.file, 'r')
        lines = f.readlines()

        user_coord = []
        user_loc = []
        for i, line in enumerate(lines):
            tokens = line.strip().split(' ')
            lat = float(tokens[2])
            long = float(tokens[1])
            coord = (lat, long)

            location = int(tokens[0])
            user_coord.insert(0, coord)
            user_loc.insert(0, location)

        self.coords.append(user_coord)
        self.locs.append(user_loc)

        return self.coords,self.locs


    def with_h_hour(self,interval,h):

        if interval<=h*60*60:
            return True
        else:
            return False

    def dicInster(self,d, endnode, startnode):  # 修改后start和end的顺序

        try:
            d[startnode][endnode]['weight'] += 1
        except:
            d = d.setdefault(startnode, {})
            d = d.setdefault(endnode, {})
            d = d.setdefault("weight", 1)
        return d

    def consecutively_Visit(self,user_times,locs):

        consecutively_Visit_dics = {}
        print("all", len(user_times))
        for i in range(len(user_times)):
            i_user_times = user_times[i]
            for j in range(len(i_user_times)-1):
                interval = i_user_times[j+1]-i_user_times[j]
                if self.with_h_hour(interval,self.interval_hour):
                    self.dicInster(consecutively_Visit_dics,int(locs[i][j+1]),int(locs[i][j])) # cons...dics,end,start
        return consecutively_Visit_dics

    def load_data(self):
        self.load_users()
        users,times,coords,locs = self.load_pois()
        PoiNumber=len(np.unique(list(itertools.chain.from_iterable(locs)))) # 获得poi数量
        print("PoiNumber",PoiNumber)
        dod=self.consecutively_Visit(times,locs)
        DG = nx.DiGraph(dod)
        squares = []
        for x in range(PoiNumber):
            squares.append(x)
        adj = sparse.csr_matrix(nx.to_numpy_matrix(DG,nodelist=squares))

        features_init = nn.Embedding(adj.shape[0], 64)
        features_init = torch.FloatTensor(features_init.weight)

        return adj, features_init

    def load_data_onlyfeature(self,nodenumber):
        features_init = nn.Embedding(nodenumber, 16)
        features_init = torch.FloatTensor(features_init.weight)

        return features_init
    def load_data_Bipartitefeature(self, nodenumber):

        features_init_Output = nn.Embedding(nodenumber, 16)
        features_init_Output = torch.FloatTensor(features_init_Output.weight)

        features_init_Input = nn.Embedding(nodenumber, 16)
        features_init_Input = torch.FloatTensor(features_init_Input.weight)

        return features_init_Output,features_init_Input

    def load_adj(self):
        self.load_users()
        users,times,coords,locs=self.load_pois()
        print("locs长度",len(locs))
        PoiNumber=len(np.unique(list(itertools.chain.from_iterable(locs)))) # 获得poi数量
        print("PoiNumber",PoiNumber)

        dod=self.consecutively_Visit(times,locs)
        DG = nx.DiGraph(dod)
        squares = []
        for x in range(PoiNumber):
            squares.append(x)
        # 生成访问次数的adj邻接矩阵
        adj = sparse.csr_matrix(nx.to_numpy_matrix(DG,nodelist=squares))
        return users,times,coords,locs,adj

    def load_adj_V(self):
        # self.load_users()
        coords,locs=self.load_pois_V()
        print("locs长度",len(locs))
        PoiNumber=len(np.unique(list(itertools.chain.from_iterable(locs)))) # 获得poi数量
        print("PoiNumber",PoiNumber)
        return coords,locs

    def load_BipartiteAdj(self):

        self.load_users()
        users,times,coords,locs=self.load_pois()
        print("locs长度",len(locs))
        PoiNumber=len(np.unique(list(itertools.chain.from_iterable(locs)))) # 获得poi数量
        print("PoiNumber",PoiNumber)

        dod=self.consecutively_Visit(times,locs)
        DG = nx.DiGraph(dod)

        squares = []
        for x in range(PoiNumber):
            squares.append(x)
        OutputAdj = sparse.csr_matrix(nx.to_numpy_matrix(DG,nodelist=squares))
        InputAdj = OutputAdj.T

        return OutputAdj,InputAdj


def mask_test_edges_Bipartite(adj):

    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)

    assert np.diag(adj.todense()).sum() == 0

    adj_tuple = sparse_to_tuple(adj)
    edges = adj_tuple[0]
    edges_all = sparse_to_tuple(adj)[0]
    print("所有连边",edges_all)
    num_test = int(np.floor(edges.shape[0] / 10.))
    num_val = int(np.floor(edges.shape[0] / 10.))

    all_edge_idx = list(range(edges.shape[0]))
    np.random.shuffle(all_edge_idx)
    val_edge_idx = all_edge_idx[:num_val]
    test_edge_idx = all_edge_idx[num_val:(num_val + num_test)]
    test_edges = edges[test_edge_idx]
    val_edges = edges[val_edge_idx]
    train_edges = np.delete(edges, np.hstack([test_edge_idx, val_edge_idx]), axis=0)

    def ismember(a, b, tol=2):

        rows_close = np.all(np.round(a - b[:, None], tol) == 0, axis=-1)
        return np.any(rows_close)

    test_edges_false = []
    while len(test_edges_false) < len(test_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], edges_all):
            continue
        if test_edges_false:
            if ismember([idx_i, idx_j], np.array(test_edges_false)):
                continue
        test_edges_false.append([idx_i, idx_j])
    val_edges_false = []
    while len(val_edges_false) < len(val_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], train_edges):
            continue
        if ismember([idx_i, idx_j], val_edges):
            continue
        if ismember([idx_i, idx_j], edges_all):
            continue
        if val_edges_false:
            if ismember([idx_i, idx_j], np.array(val_edges_false)):
                continue
        val_edges_false.append([idx_i, idx_j])

    assert ~ismember(test_edges_false, edges_all)
    assert ~ismember(val_edges_false, edges_all)
    assert ~ismember(val_edges, train_edges)
    assert ~ismember(test_edges, train_edges)
    assert ~ismember(val_edges, test_edges)


    data = getEdgeValue(np.sort(all_edge_idx[(num_val + num_test):]),adj_tuple[1])

    adj_train_Value = sp.csr_matrix((data, (train_edges[:, 0], train_edges[:, 1])), shape=adj.shape)

    return adj_train_Value, train_edges, val_edges, val_edges_false, test_edges, test_edges_false

def getEdgeValue(edges,valueList):

    return valueList[edges]

def mask_test_edges(adj):

    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)

    assert np.diag(adj.todense()).sum() == 0

    adj_triu = sp.triu(adj)
    adj_tuple = sparse_to_tuple(adj_triu)
    edges = adj_tuple[0]
    edges_all = sparse_to_tuple(adj)[0]
    print(edges_all)
    num_test = int(np.floor(edges.shape[0] / 10.))
    num_val = int(np.floor(edges.shape[0] / 20.))

    all_edge_idx = list(range(edges.shape[0]))
    np.random.shuffle(all_edge_idx)
    val_edge_idx = all_edge_idx[:num_val]
    test_edge_idx = all_edge_idx[num_val:(num_val + num_test)]
    test_edges = edges[test_edge_idx]
    val_edges = edges[val_edge_idx]
    train_edges = np.delete(edges, np.hstack([test_edge_idx, val_edge_idx]), axis=0)

    def ismember(a, b, tol=2):

        rows_close = np.all(np.round(a - b[:, None], tol) == 0, axis=-1)
        return np.any(rows_close)

    test_edges_false = []
    while len(test_edges_false) < len(test_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], edges_all):
            continue
        if test_edges_false:
            if ismember([idx_j, idx_i], np.array(test_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(test_edges_false)):
                continue
        test_edges_false.append([idx_i, idx_j])
    val_edges_false = []
    while len(val_edges_false) < len(val_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], train_edges):
            continue
        if ismember([idx_j, idx_i], train_edges):
            continue
        if ismember([idx_i, idx_j], val_edges):
            continue
        if ismember([idx_j, idx_i], val_edges):
            continue
        if ismember([idx_i, idx_j], edges_all): # 是否随机的[idx_i, idx_j] 存在于edges_all
            continue
        if val_edges_false:
            if ismember([idx_j, idx_i], np.array(val_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(val_edges_false)):
                continue
        val_edges_false.append([idx_i, idx_j])

    assert ~ismember(test_edges_false, edges_all)
    assert ~ismember(val_edges_false, edges_all)
    assert ~ismember(val_edges, train_edges)
    assert ~ismember(test_edges, train_edges)
    assert ~ismember(val_edges, test_edges)

    data = np.ones(train_edges.shape[0])

    adj_train = sp.csr_matrix((data, (train_edges[:, 0], train_edges[:, 1])), shape=adj.shape)
    adj_train = adj_train + adj_train.T
    return adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false
def sparse_to_tuple(sparse_mx):

    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape

def preprocess_graph_csr(adj):
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    return adj_normalized

def preprocess_graph(adj):
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    return sparse_mx_to_torch_sparse_tensor(adj_normalized.T)
def sparse_mx_to_torch_sparse_tensor(sparse_mx):

    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def preprocess_graph_Bipartite(adj):
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    # return sparse_to_tuple(adj_normalized)
    # return sparse_mx_to_torch_sparse_tensor_Bipartite(adj_normalized)
    return adj_normalized

def sparse_mx_to_torch_sparse_tensor_Bipartite(sparse_mx):

    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)



def get_roc_score(emb, adj_orig, edges_pos, edges_neg):


    def sigmoid(x):
        if x >=0:
            return 1 / (1 + np.exp(-x))
        else:
            return np.exp(x)/(1+np.exp(x))
    adj_rec = np.dot(emb, emb.T)
    preds = []
    pos = []
    for e in edges_pos:
        preds.append(sigmoid(adj_rec[e[0], e[1]]))
        pos.append(adj_orig[e[0], e[1]])

    preds_neg = []
    neg = []
    for e in edges_neg:
        preds_neg.append(sigmoid(adj_rec[e[0], e[1]]))
        neg.append(adj_orig[e[0], e[1]])

    preds_all = np.stack([preds,preds_neg])
    labels_all = np.stack([np.ones(len(preds)),np.zeros(len(preds_neg))])

    roc_score = roc_auc_score(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)
    cost = mean_squared_error(np.stack([np.ones(len(preds))]),np.stack([preds]))

    return roc_score, ap_score,cost

def get_MSE(emb, adj_orig, edges_pos, edges_neg):

    def sigmoid(x):
        if x >=0:
            return 1 / (1 + np.exp(-x))
        else:
            return np.exp(x)/(1+np.exp(x))
    adj_rec = np.dot(emb, emb.T)
    preds = []
    pos = []
    for e in edges_pos:
        preds.append(adj_rec[e[0], e[1]])
        pos.append(adj_orig[e[0], e[1]])

    preds_neg = []
    neg = []
    for e in edges_neg:
        preds_neg.append(sigmoid(adj_rec[e[0], e[1]]))
        neg.append(adj_orig[e[0], e[1]])

    preds_all = np.stack([preds])
    labels_all = np.stack([np.array(pos)]) # 仅正样本

    cost = mean_squared_error(preds_all,labels_all)

    return cost


def get_MSE_Bipartite(emb,mu_output, mu_input, adj_orig, edges_pos, edges_neg):


    def sigmoid(x):
        if x >=0:
            return 1 / (1 + np.exp(-x))
        else:
            return np.exp(x)/(1+np.exp(x))
    adj_rec = np.dot(mu_output, mu_input.T)

    preds = []
    pos = []
    for e in edges_pos:
        preds.append(adj_rec[e[0], e[1]])  #【？】不理解sigmod的用处
        pos.append(adj_orig[e[0], e[1]])


    preds_all = np.stack([preds])

    labels_all = np.stack([np.array(pos)]) # 仅正样本
    roc_score=1
    cost = mean_squared_error(labels_all,preds_all)

    return cost


