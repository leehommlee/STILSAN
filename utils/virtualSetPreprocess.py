# -*- coding:utf-8 -*-
# Author : Leehom
# Data : 2021/3/22 8:44
import sys
import pickle as pkl
import numpy as np
from numpy import mat
import scipy
from scipy import sparse
from scipy.sparse import csr_matrix
import scipy.sparse as sp
from utils.PowerLaw import dist
from scipy.sparse import dia_matrix
from utils.utilsModel import GowallaLoader
import itertools

class datapreprocess():

    def __init__(self,datafilepath,savefileRootpath,dataname,min_checkins, interval_hour,scope,max_users):
        self.datafilepath=datafilepath
        self.saveAdjfilepath=savefileRootpath+'adj'+dataname+'.npz'  # 邻接矩阵保存的路径
        self.saveOutAdjfilepath=savefileRootpath+'adj'+dataname+'output'+'.npz'
        self.saveInAdjfilepath=savefileRootpath+'adj'+dataname+'input'+'.npz'
        self.saveAdj_D_filepath=savefileRootpath+'adj_D_'+dataname+'.npz'  # 邻接矩阵保存的路径
        self.dataname=dataname
        self.min_checkins=min_checkins
        self.interval_hour=interval_hour
        self.scope=scope
        self.max_users=max_users
        self.gowalla=GowallaLoader(self.datafilepath, self.min_checkins, self.interval_hour, self.max_users)

    def create_V(self):
        self.load_adj_distance_V()

    def load_adj_distance_V(self):

        coords, locs= self.gowalla.load_adj_V()
        print("原始locs",len(locs))
        coords_locs_list=[]
        coords_f=[]
        locs_f=[]
        for i in range(len(locs)):
            for j in range(len(locs[i])):
                coords_tuple = coords[i][j]
                locs_value = locs[i][j]
                if locs_value in locs_f:
                    continue
                else:
                    coords_f.append(coords_tuple)
                    locs_f.append(locs_value)

        print(coords_f,'\n',locs_f)
        print(len(coords_f),'\n',len(locs_f))

        adj_shape=len(coords_f)
        adj_d= np.ones((adj_shape,adj_shape))
        for i in range(adj_shape):
            print("i",i)
            for j in range(adj_shape):

                distance = int(dist(coords_f[i],coords_f[j]))
                row = locs_f[i]
                column = locs_f[j]
                if distance < 20 and distance > 0:
                    print(distance)
                    adj_d[row,column] = 1/distance
                else:
                    adj_d[row, column] = 0
        adj_d=sp.csr_matrix(adj_d)
        sp.save_npz(self.saveAdj_D_filepath, adj_d)
        return adj_d



    def createAdjnpz(self):
        adj, features_init = self.gowalla.load_data()
        sp.save_npz(self.saveAdjfilepath, adj)

    def createAdjnpzBipartite(self):

        OutputAdj, InputAdj = self.gowalla.load_BipartiteAdj()
        OutputAdj = OutputAdj - sp.dia_matrix((OutputAdj.diagonal()[np.newaxis, :], [0]), shape=OutputAdj.shape)
        InputAdj = InputAdj - sp.dia_matrix((InputAdj.diagonal()[np.newaxis, :], [0]), shape=OutputAdj.shape)
        sp.save_npz(self.saveOutAdjfilepath, OutputAdj)
        sp.save_npz(self.saveInAdjfilepath, InputAdj)



    def load_adj_distance(self):

        users, times, coords, locs, adj = self.gowalla.load_adj()

        print("原始locs",len(locs))

        coords_locs_list=[]
        coords_f=[]
        locs_f=[]
        for i in range(len(locs)):
            for j in range(len(locs[i])):
                coords_tuple = coords[i][j]
                locs_value = locs[i][j]
                if locs_value in locs_f:
                    continue
                else:
                    coords_f.append(coords_tuple)
                    locs_f.append(locs_value)

        print(coords_f,'\n',locs_f)
        print(len(coords_f),'\n',len(locs_f))

        adj_shape=adj.shape[0]
        adj_d= np.ones((adj_shape,adj_shape))
        for i in range(adj_shape):
            print("i",i)
            for j in range(adj_shape):

                distance = int(dist(coords_f[i],coords_f[j]))
                row = locs_f[i]
                column = locs_f[j]
                if distance > self.scope:
                    adj_d[row,column] = 1/distance
                else:
                    adj_d[row, column] = 0
        adj_d=sp.csr_matrix(adj_d)
        sp.save_npz(self.saveAdj_D_filepath, adj_d)
        return adj_d










