# -*- coding:utf-8 -*-
# Author : Leehom
# Data : 2021/1/11 9:08

import os
# from torch.utils.tensorboard import SummaryWriter
import argparse
import time
from utils.utilsModel import GowallaLoader
from model.models import VGAE_P,GAE_P,VGAE_P_Distance
from utils.utilsModel import preprocess_graph,preprocess_graph_Bipartite,get_MSE,mask_test_edges,sparse_mx_to_torch_sparse_tensor,mask_test_edges_Bipartite
from optiminzers.optimizer import loss_function
import numpy as np
import scipy.sparse as sp
import torch
from torch import optim
import pandas as pd


parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='gcn_vae', help="models used")
parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')

# parser.add_argument('--gpu', type=str, default='2', help='specify gpu device')
parser.add_argument('--min-checkins', default=100, type=int, help='amount of checkins required')
parser.add_argument('--interval-hour', default=100, type=int, help='amount of checkins required')
parser.add_argument('--scope', default=0, type=int, help='the thresholds to control the scope for two POIs')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=80, help='Number of epochs to train.')
parser.add_argument('--hidden1', type=int, default=64, help='Number of units in hidden layer 1.')
parser.add_argument('--hidden2', type=int, default=40, help='Number of units in hidden layer 2, ')
parser.add_argument('--cluster', type=int, default=250, help='簇的数量')
parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate.')
parser.add_argument('--dropout', type=float, default=0., help='Dropout rate (1 - keep probability).')
args = parser.parse_args()

def train(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    dataname='NYC_recode'+str(args.cluster)+'_cluster'
    emb_size=args.hidden2
    data_path='./data/'+dataname+'.csv'
    adj_path='./data/PreData/adj_D_'+dataname+'.npz'
    adj_D_path='./data/PreData/adj_D_'+dataname+'.npz'
    z_path='./data/embeddingData/z_'+'_'+str(emb_size)+'.npy'

    adj = sp.load_npz(adj_path)
    print(adj.shape[0])
    features_init=GowallaLoader(data_path,args.min_checkins,args.interval_hour).load_data_onlyfeature(adj.shape[0])

    n_nodes, feat_dim = features_init.shape

    adj_orig, val_edges, val_edges_false, test_edges, test_edges_false \
        , adj_norm, norm, pos_weight, adj_label=Adjprepro(adj)


    model = VGAE_P(feat_dim, args.hidden1, args.hidden2,args.dropout)  # feat_dim=1433;hidden1=32;hidden2=16;dropout=0 【？？为什么是0维】

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    features_init = features_init.to(device)
    adj_norm = adj_norm.to(device)
    adj_label = adj_label.to(device)
    pos_weight = pos_weight.to(device)
    adj_d = sp.load_npz(adj_D_path)


    adj_d=adj_d.todense()
    adj_d = np.array(adj_d)
    print(type(adj_d),adj_d.shape)

    adj_distance = (torch.tensor(adj_d)).to(device)
    print(type(adj_distance),adj_distance.shape)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    hidden_emb = None

    loss1=[];ap=[]
    model.train()
    for epoch in range(args.epochs):
        t = time.time()
        optimizer.zero_grad()
        z, recovered, mu, logvar = model(features_init, adj_norm,t_dis,adj_distance)
        loss = loss_function(preds=recovered, labels=adj_label,
                             mu=mu, logvar=logvar, n_nodes=n_nodes,
                             norm=norm, pos_weight=pos_weight)
        loss.backward()
        cur_loss = loss.item()
        optimizer.step()

        hidden_emb = mu.data.cpu().numpy()
        cost = get_MSE(hidden_emb, adj_norm, val_edges, val_edges_false)
        #
        loss1.append(cur_loss);ap.append(cost)
        print("Epoch:",'%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(cur_loss),
              "val_cost=", "{:.5f}".format(cost),
              "time=","{:.5f}".format(time.time() - t))

    z=z.cpu().detach().numpy()
    np.save(z_path, z)
    print("Optimization Finished!")

    # ------------------------- 测试
    cost = get_MSE(hidden_emb, adj_norm, test_edges, test_edges_false)
    print('Test COST score:' + str(cost))

def Adjprepro(adj):
    adj_orig = adj
    adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
    adj_train_Value, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges_Bipartite(adj)
    adj_norm = preprocess_graph(adj)
    adj_label = preprocess_graph_Bipartite(adj_train_Value.toarray())
    adj_label = torch.FloatTensor(adj_label.T.toarray())
    pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
    pos_weight=torch.tensor(pos_weight, dtype=torch.float)
    norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)

    return adj_orig, val_edges, val_edges_false, test_edges, test_edges_false\
        ,adj_norm,norm,pos_weight,adj_label


if __name__ == '__main__':
    train(args)










