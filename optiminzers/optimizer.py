# -*- coding:utf-8 -*-
# Author : Leehom
# Data : 2020/12/24 16:39

import torch
import torch.nn.modules.loss
import torch.nn.functional as F


def loss_function(preds, labels, mu, logvar, n_nodes, norm, pos_weight):

    cost = norm * F.mse_loss(preds, labels)
    KLD = -0.5 / n_nodes*torch.mean(torch.sum(
        1+2*logvar - mu.pow(2) - logvar.exp().pow(2), 1))

    return cost + KLD


def loss_function_Bipartite(preds, labels, mu_output, mu_input, logvar_ouput, logvar_input, n_nodes, norm, pos_weight):

    cost = norm * F.mse_loss(preds, labels)

    KLD_output = -0.5 / n_nodes * torch.mean(torch.sum(
        1 + 2 * logvar_ouput - mu_output.pow(2) - logvar_ouput.exp().pow(2), 1))

    KLD_input = -0.5 / n_nodes * torch.mean(torch.sum(
        1 + 2 * logvar_input - mu_input.pow(2) - logvar_input.exp().pow(2), 1))

    return cost + KLD_output + KLD_input

