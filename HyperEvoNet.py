#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import copy
import math
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor,optim
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from time import time
from scipy.sparse import csr_matrix, diags
from torch.utils.data import DataLoader, TensorDataset
from loader import *
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = '1'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print(device)

### <!-- Dual Hypergraph Convolutional Network for learning hypergraphs --!> ###

class HyperEvoNet(nn.Module):
	def __init__(self, in_ch, n_hid, dty_nets, inter, intra, dropout=0.5):
		super(HyperEvoNet, self).__init__()
		self.dropout = dropout
		self.dty_nets = dty_nets
		self.dim_emb = n_hid[-1]
		self.inter = inter
		self.intra = intra
		self.HyperConv_1 = MultiHyperConv(in_ch, n_hid[0], self.dty_nets, self.inter, self.intra, self.dropout)
		self.HyperConv_2 = MultiHyperConv(n_hid[0], n_hid[1], self.dty_nets, self.inter, self.intra, self.dropout)
		self.Linear_u = nn.Linear(n_hid[-1]*len(dty_nets), n_hid[-1])
		# print(dty_nets)

	def dropout_layer(self, X):
		out=dict()
		for dty in self.dty_nets:
			out[dty] = F.dropout(X[dty], self.dropout)
		return out

	def forward(self, X, G, H,nodes,subnodes_info,id_to_index):
		X_1= self.HyperConv_1(X, G, H)
		X_1 = self.dropout_layer(X_1)
		X_2= self.HyperConv_2(X_1, G, H)
		dty_nets = self.dty_nets-['base']

		all_x = X_2['base']
		for dty in dty_nets:
			add_x_1 = X_2[dty]
			node_ids_list = extract_node_ids(subnodes_info[int(dty)])
			add_x = map_and_fill_add_x(nodes, node_ids_list, add_x_1,id_to_index)
			all_x = torch.cat((all_x, add_x.to(device)), 1)
		opt = self.Linear_u(all_x)
		return opt

class Embed_layer(nn.Module):
	def __init__(self, in_ft, out_ft, dty_nets):
		super(Embed_layer, self).__init__()
		self.weight = Parameter(torch.Tensor(in_ft, out_ft)).to(device)
		self.reset_parameters()
		self.dty_nets = dty_nets

	def reset_parameters(self):
		stdv = 1. / math.sqrt(self.weight.size(1))
		self.weight.data.uniform_(-stdv, stdv)

	def forward(self, X:torch.Tensor):
		X_ = dict()
		for dty in self.dty_nets:
			X_[dty] = X[dty].matmul(self.weight)
		return X_

class HyperConv(nn.Module):
	def __init__(self, in_ft, out_ft, inter=False, intra=True, bias=True):
		super(HyperConv, self).__init__()
		self.weight_u = Parameter(torch.Tensor(in_ft, out_ft)).to(device)
		self.bias = Parameter(torch.Tensor(out_ft)).to(device) if bias else None
		# if bias:
		# 	self.bias = Parameter(torch.Tensor(out_ft)).to(device)
		# else:
		# 	self.register_parameter(torch.Tensor(out_ft)).to(device)
		self.WB = Parameter(torch.Tensor(out_ft, out_ft)).to(device)
		self.reset_parameters()
		self.inter = inter
		self.intra = intra

	def reset_parameters(self):
		stdv = 1. / math.sqrt(self.weight_u.size(1))
		self.weight_u.data.uniform_(-stdv, stdv)
		if self.bias is not None:
			self.bias.data.uniform_(-stdv, stdv)
		self.WB.data.uniform_(-stdv, stdv)



	def forward(self,Xu:torch.Tensor,Gu:torch.Tensor,Hu:torch.Tensor,B:torch.Tensor,_intra:torch.bool):
		Xu = Xu.matmul(self.weight_u)
		X = Gu.matmul(Xu)

		# 确保Gu是稀疏矩阵
		# # X = torch.sparse.mm(Gu, Xu)
		# Xu = Xu.to(torch.float32)  # 转换为float32
		# Gu = Gu.to_dense().to(torch.float32)  # 如果Gu是稀疏矩阵，先转为稠密矩阵，再转换
		# X = Gu.matmul(Xu)

		if self.intra and _intra:
			X = X + B.matmul(self.WB)
		if self.bias is not None:
			X = X + self.bias
		X = F.relu(X)
		return X

class MultiHyperConv(nn.Module):
	def __init__(self, in_ft, out_ft, dty_nets, inter, intra, dropout, bias=True):
		super(MultiHyperConv, self).__init__()
		self.dty_nets = dty_nets
		self.dropout = dropout
		self.HyperConv = dict()
		for dty in self.dty_nets:
			self.HyperConv[dty] = HyperConv(in_ft, out_ft, inter=inter, intra=intra)

	def forward(self,X:torch.Tensor,G:torch.Tensor,H:torch.Tensor):
		self._dty_nets = self.dty_nets-['base']
		### HyperConv
		out_x = dict()
		base_x = self.HyperConv['base'](X['base'], G['base'], H['base'],X,False)
		out_x['base'] = base_x
		for dty in self._dty_nets:
			if dty not in self.HyperConv:
				print(f"Warning: Key '{dty}' not found in HyperConv. Available keys: {self.HyperConv.keys()}")
				continue
			add_x = self.HyperConv[dty](X[dty], G[dty], H[dty], base_x, False)
			out_x[dty] = add_x
			# print("dty:", dty)
			# print("Available keys:", self.HyperConv.keys())
			# add_x = self.HyperConv[dty](X[dty],G[dty],H[dty],base_x,True)
			# out_x[dty] = add_x
		return out_x

# 将大矩阵划分成块，block_size是块的大小
def split_into_blocks(matrix, block_size):
    rows, cols = matrix.shape
    row_blocks = rows // block_size
    col_blocks = cols // block_size
    blocks = []

    for i in range(row_blocks):
        for j in range(col_blocks):
            row_start, row_end = i * block_size, (i + 1) * block_size
            col_start, col_end = j * block_size, (j + 1) * block_size
            block = matrix[row_start:row_end, col_start:col_end]
            blocks.append(block)

    return blocks

def generate_G_from_H(args, H):
	H = np.array(H)
	n_edge = H.shape[1]
	W = np.ones(n_edge)
	DV = np.sum(H * W, axis=1)
	DE = np.sum(H, axis=0)
	DV = DV + 1e-12
	DE = DE + 1e-12
	invDE = np.mat(np.diag(np.power(DE, -1)))
	W = np.mat(np.diag(W))
	H = np.mat(H)
	HT = H.T
	if args.conv == "sym":
		DV2 = np.mat(np.diag(np.power(DV, -0.5)))
		G = DV2 * H * W * invDE * HT * DV2   #sym
	elif args.conv == "asym":
		DV1 = np.mat(np.diag(np.power(DV, -1)))
		G = DV1 * H * W * invDE * HT  # asym
	return G

def generate_Gs_from_Hs(args, Hs):
	Gs = dict()
	for key,val in Hs.items():
		Gs[key] = generate_G_from_H(args, val)
	return Gs


def generate_G_from_H_bit(args, H):
	# 将 H 转换为稀疏矩阵
	H = csr_matrix(H,dtype=np.float64)
	n_edge = H.shape[1]

	# 边的权重
	W = np.ones(n_edge,dtype=np.float64)

	# 节点度和边度
	DV = np.array(H.dot(W)).flatten().astype(np.float64)  # 节点度
	DE = np.array(H.sum(axis=0)).flatten().astype(np.float64)  # 边度

	# 防止除以零
	DV += 1e-12
	DE += 1e-12

	# 构建稀疏对角矩阵
	invDE = diags(1.0 / DE.astype(np.float64))
	W_diag = diags(W)

	# 转置超图矩阵
	HT = H.transpose()

	# 选择对称或非对称卷积
	if args.conv == "sym":
		DV2 = diags(np.power(DV, -0.5),dtype=np.float64)  # 稀疏对角矩阵
		G = DV2 @ H @ W_diag @ invDE @ HT @ DV2  # 对称卷积
	elif args.conv == "asym":
		DV1 = diags(1.0 / DV,dtype=np.float64)  # 稀疏对角矩阵
		G = DV1 @ H @ W_diag @ invDE @ HT  # 非对称卷积
	else:
		raise ValueError("Invalid convolution type. Use 'sym' or 'asym'.")

	return G


def generate_Gs_from_Hs_bit(args, Hs):
	Gs = {}
	for key, val in Hs.items():
		Gs[key] = generate_G_from_H_bit(args, val)
	return Gs

def embedding_loss(embeddings, positive_links, negtive_links, lamb):


	left_p = embeddings[positive_links[:, 0]]
	right_p = embeddings[positive_links[:, 1]]
	dots_p = torch.sum(torch.mul(left_p, right_p), dim=1)
	positive_loss = torch.mean(-1.0 * F.logsigmoid(dots_p))
	left_n = embeddings[negtive_links[:, 0]]
	right_n = embeddings[negtive_links[:, 1]]
	dots_n = torch.sum(torch.mul(left_n, right_n), dim=1)
	negtive_loss = torch.mean(-1.0 * torch.log(1.01 - torch.sigmoid(dots_n)))
	loss =  lamb*positive_loss + (1-lamb)*negtive_loss
	return loss


def train(args, model, X, mapped_target1 , mapped_target2, G, H,nodes,subnodes_info,id_to_index):
	lr = args.lr
	weight_decay = args.weight_decay

	if args.optimizer == "Adam":
		optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
	elif args.optimizer == "SGD":
		optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
	n_epoch = args.epoch
	feats, target1, target2 = X, mapped_target1 , mapped_target2

	best_loss = float('inf')
	patience_counter = 0
	scaler = GradScaler()
	scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

	for epoch in tqdm(range(n_epoch), desc="Training Progress"):
		model.train()
		optimizer.zero_grad()
		embeds = model.forward(feats, G, H, nodes,subnodes_info,id_to_index)
		loss = embedding_loss(embeds, target1, target2, args.lamb)
		loss.backward()
		optimizer.step()

		# 更新学习率调度器
		scheduler.step(loss)

		# 提前停止逻辑
		if loss < best_loss:
			best_loss = loss
			patience_counter = 0
		# 保存模型或其他需要的操作
		else:
			patience_counter += 1

		if patience_counter >= 10:  # 设置提前停止的阈值
			print("Early stopping triggered.")
			break

		# if (epoch + 1) % 10 == 0 or epoch == 0:
		print('\n The loss of %d-th epoch: %0.4f' % (epoch + 1, loss.item()))
	# 	model.train()
	# 	optimizer.zero_grad()
	# 	with autocast():
	# 		embeds = model.forward(feats, G, H, nodes, subnodes_info, id_to_index)
	# 		loss = embedding_loss(embeds, target1, target2, args.lamb)
	#
	# 	scaler.scale(loss).backward()
	# 	scaler.step(optimizer)
	# 	scaler.update()
	# 	# embeds = model.forward(feats, G, H, nodes,subnodes_info,id_to_index)
	# 	# loss = embedding_loss(embeds, target1, target2, args.lamb)
	# 	# loss.backward()
	# 	# optimizer.step()
	# 	if (epoch+1) % 100 == 0 or epoch == 0:
	# 		print('The loss of %d-th epoch: %0.4f' % (epoch+1, loss))
	# # 提前停止逻辑
	# 	if loss < best_loss:
	# 		best_loss = loss
	# 		counter = 0  # 重置计数器
	# 	else:
	# 		counter += 1
	#
	# 	if counter >= patience:
	# 		print(f"Early stopping at epoch {epoch + 1}. Best loss: {best_loss:.4f}")
	# 		break

	model.eval()
	outputs = model.forward(feats, G, H,nodes, subnodes_info, id_to_index)
	return outputs
def scipy_to_torch_sparse(scipy_sparse_matrix):
    # 转换为 COO 格式以便于稀疏表示
    coo = scipy_sparse_matrix.tocoo()
    values = coo.data
    indices = np.vstack((coo.row, coo.col))

    # 将 indices 和 values 转换为 PyTorch 张量
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = coo.shape

    # 创建 PyTorch 稀疏张量
    return torch.sparse_coo_tensor(i, v, size=shape, dtype=torch.float32)

def train_HyperEvoNet(args, X, Hs, mapped_target1 , mapped_target2,nodes,subnodes_info,id_to_index):
	Xs = dict()
	# in_ft= {}
	for key,val in X.items():
		# Xs[key] = Tensor(val).to(device)
		X_ = X[key]
		Xs[key] = Tensor(X_).to(device)
		# in_ft[key]=X[key].shape[1]
	# n_sample = X.shape[0]
	in_ft = X['base'].shape[1]
	G = generate_Gs_from_Hs_bit(args, Hs)
	# G = generate_Gs_from_Hs(args, Hs)
	Gs_u = dict()
	Hs_u = dict()
	for key,val in G.items():
		# Gs_u[key] = scipy_to_torch_sparse(val).to(device, dtype=torch.float32)
		Gs_u[key] = scipy_to_torch_sparse(G[key]).to(device, dtype=torch.float32)
		# Gs_u[key] = Tensor(G[key]).to(device, dtype=torch.float32)
		torch.cuda.empty_cache()
		Hs_u[key] = Tensor(Hs[key]).to(device, dtype=torch.float32)
		torch.cuda.empty_cache()
	model = HyperEvoNet(in_ch=in_ft,n_hid=args.dim,dty_nets=Hs.keys(),inter=args.inter,intra=args.intra,dropout=args.dropout)
	model = model.to(device)
	emb = train(args, model, Xs, mapped_target1 , mapped_target2, Gs_u, Hs_u,nodes,subnodes_info,id_to_index)
	return emb.detach().cpu().numpy()

