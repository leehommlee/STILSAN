# -*- coding:utf-8 -*-
# Author : Leehom
# Data : 2020/12/24 16:50

import math
import time
import numpy as np
from collections import defaultdict
from utils.utilsModel import GowallaLoader
from tkinter import _flatten
import scipy.sparse as sp
from numpy.random import uniform

# 距离计算
def dist(loc1, loc2):
    lat1, long1 = loc1[0], loc1[1]
    lat2, long2 = loc2[0], loc2[1]
    if abs(lat1 - lat2) < 1e-6 and abs(long1 - long2) < 1e-6:
        return 0.0
    degrees_to_radians = math.pi/180.0
    phi1 = (90.0 - lat1)*degrees_to_radians
    phi2 = (90.0 - lat2)*degrees_to_radians
    theta1 = long1*degrees_to_radians
    theta2 = long2*degrees_to_radians
    cos = (math.sin(phi1)*math.sin(phi2)*math.cos(theta1 - theta2) +
           math.cos(phi1)*math.cos(phi2))
    arc = math.acos( cos )
    earth_radius = 6371
    return arc * earth_radius

# 指数距离函数计算
rang = 0.5
a = uniform(-rang, rang)
b = uniform(-rang, rang)
def f_d(d):
    return a * (d ** b)# * exp(self.c * d)

def load_adj_distance_connect(users,times,coords,locs,adj,scope):
    calculated_list=[]
    for i in range(len(locs)):
        for j in range(len(locs[i])):
            end_loc=j, start_loc=j+1
            if ((start_loc,end_loc)) in calculated_list:
                continue
            else:
                distance=dist(coords[i][end_loc],coords[i][start_loc])
                row = locs[i][start_loc]
                column = locs[i][end_loc]
                if distance>scope:
                    adj[row,column]=distance
                else:
                    adj[row,column]=distance

                calculated_list.append((start_loc,end_loc))

            pass


def load_adj_distance(data_path, min_checkins, interval_hour,scope,max_users):

    users, times, coords, locs, adj = GowallaLoader(data_path, min_checkins, interval_hour,max_users).load_adj()

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
            if distance > scope:
                adj_d[row,column] = 1/distance
            else:
                adj_d[row, column] = 0
    adj_d=sp.csr_matrix(adj_d)
    sp.save_npz('../data/adj_type_200_re.npz', adj_d)
    return adj_d










