# -*- coding:utf-8 -*-
# Author : Leehom
# Data : 2021/1/29 17:02

import torch
import torch.nn as nn
import torch.nn.functional as F
from model.layers import GraphConvolution, GraphConvSparse

class VGAE_P(nn.Module):
    def __init__(self,input_feat_dim, hidden_dim1,hidden_dim2,dropout):
        super(VGAE_P, self).__init__()
        self.gc1 = GraphConvolution(input_feat_dim,hidden_dim1,dropout,act=F.relu)
        self.gc2 = GraphConvolution(hidden_dim1,hidden_dim2,dropout,act=lambda x: x)
        self.gc3 = GraphConvolution(hidden_dim1,hidden_dim2,dropout,act=lambda x: x)
        self.dc = InnerProductDecoder(dropout, act=lambda x: x)

    def encode(self, x, adj):
        hidden1 = self.gc1(x, adj)
        return self.gc2(hidden1,adj), self.gc3(hidden1, adj)


    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, x, adj,t,adj_distance):
        mu, logvar = self.encode(x,adj) # 【？？】
        z = self.reparameterize(mu, logvar)
        return z,self.dc(z), mu, logvar

class InnerProductDecoder(nn.Module):
    def __init__(self, dropout, act=torch.sigmoid):
        super(InnerProductDecoder, self).__init__()
        self.dropout = dropout
        self.act = act

    def forward(self, z):
        z = F.dropout(z, self.dropout, training=self.training)
        adj = self.act(torch.mm(z,z.t()))
        return adj


class VGAE_P_Distance(nn.Module):
    def __init__(self,input_feat_dim, hidden_dim1,hidden_dim2,dropout):
        super(VGAE_P_Distance, self).__init__()
        self.gc1 = GraphConvolution(input_feat_dim,hidden_dim1,dropout,act=F.relu)
        self.gc2 = GraphConvolution(hidden_dim1,hidden_dim2,dropout,act=lambda x: x)
        self.gc3 = GraphConvolution(hidden_dim1,hidden_dim2,dropout,act=lambda x: x)
        self.dc = InnerProductDecoder_Distance(dropout, act=lambda x: x)

    def encode(self, x, adj):
        hidden1 = self.gc1(x, adj)
        return self.gc2(hidden1,adj), self.gc3(hidden1, adj)


    def reparameterize(self, mu, logvar):

        if self.training:
            std = torch.exp(logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, x, adj,t,adj_distance):
        mu, logvar = self.encode(x,adj) # 【？？】
        z = self.reparameterize(mu, logvar)
        return z,self.dc(z,t,adj_distance), mu, logvar

class InnerProductDecoder_Distance(nn.Module):
    def __init__(self, dropout, act=torch.sigmoid):
        super(InnerProductDecoder_Distance, self).__init__()
        self.dropout = dropout
        self.act = act

    def forward(self, z,t,adj_distance):

        z = F.dropout(z, self.dropout, training=self.training)
        adj = self.act(torch.mul(torch.mm(z,z.t()),adj_distance))
        return adj

class VGAE_P_Bipartite(nn.Module):
    def __init__(self,input_feat_dim, hidden_dim1,hidden_dim2,dropout):
        super(VGAE_P_Bipartite, self).__init__()
        self.gc1 = GraphConvolution(input_feat_dim,hidden_dim1,dropout,act=F.relu)
        self.gc2 = GraphConvolution(hidden_dim1,hidden_dim2,dropout,act=lambda x: x)
        self.gc3 = GraphConvolution(hidden_dim1,hidden_dim2,dropout,act=lambda x: x)
        self.dc = InnerProductDecoder_Bipartite(dropout, act=lambda x: x)

    def encode(self, x, adj):
        hidden1 = self.gc1(x, adj)
        return self.gc2(hidden1,adj), self.gc3(hidden1, adj)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, x_Output, x_Input, Output_adj_norm, Input_adj_norm):

        mu_output, logvar_oupt = self.encode(x_Output,Output_adj_norm) # 【？？】
        z_output = self.reparameterize(mu_output, logvar_oupt)
        mu_input, logvar_input = self.encode(x_Input,Input_adj_norm) # 【？？】
        z_input = self.reparameterize(mu_input, logvar_input)
        return z_output,z_input,self.dc(z_output, z_input),mu_output, mu_input, logvar_oupt, logvar_input

class InnerProductDecoder_Bipartite(nn.Module):

    def __init__(self, dropout, act=torch.sigmoid):
        super(InnerProductDecoder_Bipartite, self).__init__()
        self.dropout = dropout
        self.act = act

    def forward(self, z_output, z_input):

        z_output = F.dropout(z_output, self.dropout, training=self.training)
        z_input = F.dropout(z_input, self.dropout, training=self.training)

        adj = self.act(torch.mm(z_output,z_input.t()))
        return adj




class GAE_P(nn.Module):
    def __init__(self, input_dim, hidden1_dim, hidden2_dim):
        super(GAE_P, self).__init__()
        self.base_gcn = GraphConvSparse(input_dim, hidden1_dim)
        self.gcn_mean = GraphConvSparse(hidden1_dim, hidden2_dim, activation=lambda x: x)

    def encode(self, X):
        hidden = self.base_gcn(X)
        z = self.mean = self.gcn_mean(hidden)
        return z

    def forward(self, X,adj,t,adj_distance):
        Z = self.encode(X)
        A_pred = dot_product_decode(Z)
        return A_pred,

def dot_product_decode(Z):
    A_pred = torch.sigmoid(torch.matmul(Z,Z.t()))
    return A_pred