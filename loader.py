#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import networkx as nx
from scipy.sparse import csr_matrix
from data_helper import train_tiedAE
from sklearn import preprocessing
import community  # 导入社区检测库
import pickle
from karateclub import DeepWalk
import torch
from tqdm import tqdm



def load_pickle(fname):
	with open(fname, 'rb') as f:
		return pickle.load(f)

def write_pickle(G, fname):
	with open(fname, 'wb') as f:
		pickle.dump(G, f)
	print("存储成功！")
	return True

def load_attributes(btypes,feats):
	# print(feats.shape)
	init_feats = dict()
	for btype in btypes:
		init_feats[btype] = preprocessing.normalize(feats)
	return init_feats


def normalize_features(feats):
	"""
    对节点特征进行归一化。

    参数：
    feats (dict): 节点特征字典，每个节点包含一个特征字典。

    返回：
    dict: 归一化后的节点特征字典。
    """
	# 提取所有特征的名称
	feature_keys = list(next(iter(feats.values())).keys())

	# 将特征转换为数组
	feature_matrix = []
	for node_id, feature_dict in feats.items():
		# 保证特征顺序一致
		feature_vector = [feature_dict[key] for key in feature_keys]
		feature_matrix.append(feature_vector)

	# 转换为二维数组并归一化
	feature_matrix = np.array(feature_matrix)
	normalized_matrix = preprocessing.normalize(feature_matrix)
	normalized_features_list = normalized_matrix.tolist()
	# # 将归一化后的特征重新存回字典
	# normalized_feats = {}
	# for i, node_id in enumerate(node_ids):
	# 	normalized_feats[node_id] = {key: normalized_matrix[i][j] for j, key in enumerate(feature_keys)}

	return normalized_features_list


def convert_dataframe_to_dict(features_df):
	"""
    将 DataFrame 转换为节点特征字典。

    参数：
    features_df (DataFrame): 包含节点特征的 DataFrame。

    返回：
    dict: 每个节点的特征字典。
    """
	# 将 DataFrame 转换为字典
	features_dict = {}

	for index, row in features_df.iterrows():
		node_id = row['Node']  # 假设 'Node' 列是节点的唯一标识
		features_dict[node_id] = row.drop('Node').to_dict()  # 去掉 'Node' 列并转换为字典

	return features_dict

def load_attributes_bit(btypes,feats,subgraph_info):
	init_feats = dict()
	for btype in btypes:
		if btype == 'base':
			init_feats[btype] = np.array(normalize_features(feats))
		else:
			features_df = subgraph_info[int(btype)]['features']
			features_dict = convert_dataframe_to_dict(features_df)
			init_feats[btype] = np.array(normalize_features(features_dict))
			# init_feats[btype] = np.array(normalize_features(subgraph_info[int(btype)]['features'].to_dict(orient='list')))

	return init_feats

def _construct_hypergraph(graph,nodes):
	Hygraph = dict()
	n_neigs = 0
	for u in nodes:
		if u in graph.nodes():
			neighbors = graph.edges(u)
			neigs = [i for u, i in neighbors]
			Hygraph[u] = neigs
			n_neigs += len(neigs)
		else:
			Hygraph[u] = [],
	print('### The number of hyper-edges: %d' % (len(nodes)))
	print('### The average nodes in each hyper-edge: %0.2f (%0.2f for nodes)'
		  % ((n_neigs) / (len(nodes)), n_neigs / len(nodes)))
	return Hygraph

def extract_node_ids(subgraph_info):
    """
    提取单个子图的节点 ID 并存储在列表中。

    :param subgraph_info: 包含单个子图信息的字典
    :return: 子图的节点 ID 列表
    """
    node_ids = subgraph_info['nodes']  # 假设这里是包含节点信息的 DataFrame

    # 提取节点 ID（假设节点 ID 存在 "Node" 列中）
    node_ids_list = node_ids['Node'].tolist()  # 获取节点 ID 列表

    return node_ids_list



def map_and_fill_add_x(original_nodes, subgraph_node_ids, subgraph_features,id_to_index):
    """
    将子图的节点特征映射到原图的特征矩阵中。

    :param original_nodes: 原图的节点 ID 列表
    :param subgraph_node_ids: 子图的节点 ID 列表
    :param subgraph_features: 子图节点特征的张量
    :return: 填充后的特征矩阵
    """
    # 创建原图节点 ID 到索引的映射


    # 初始化 add_x，用于填充特征
    add_x = torch.zeros(len(original_nodes), subgraph_features.size(1))  # 假设特征维度一致

    # 填充特征
    for subgraph_id in subgraph_node_ids:
        if subgraph_id in id_to_index:  # 检查子图节点是否在原图中
            original_index = id_to_index[subgraph_id]  # 获取原图中节点的索引
            subgraph_index = subgraph_node_ids.index(subgraph_id)  # 获取子图节点的索引
            add_x[original_index] = subgraph_features[subgraph_index]  # 填充特征

    return add_x

def create_Hypergraph_with_community(graph,nodes):
	undirected_G = graph.to_undirected()
	# 使用Louvain社区检测算法对无向图进行聚类
	partition = community.best_partition(undirected_G)
	Hygraph = dict()
	n_comm = 0
	for u in nodes:
		if u in graph.nodes():
			community_u = [comm for i, comm in enumerate(partition)]
			Hygraph[u] = community_u
			n_comm += len(community_u)
		else:
			Hygraph[u] = []
	print('### The number of hyper-edges: %d' % (max(list(partition.values()))+1))
	print('### The average nodes in each hyper-edge: %0.2f (%0.2f for nodes)'
		  % ((n_comm) / (len(nodes)), n_comm / len(nodes)))
	return Hygraph

def construct_hierarchical_hypergraph(G,nodes):
	hygraphs = dict()
	Gs = load_pickle('./Ethereum/Sub_MultiGraph_num_Win_10000_2.pkl')
	# Gs = Gs[0:2]
	print(len(Gs))
	hygraphs['base'] = _construct_hypergraph(G,nodes)
	# hygraphs['base'] = create_Hypergraph_with_community(G,nodes)
	for i, graph in enumerate(Gs):
		# hygraphs[str(i)] = create_Hypergraph_with_community(graph, nodes)
		hygraphs[str(i)] = _construct_hypergraph(graph,nodes)
	return hygraphs

def generate_adj(data,nodes,btype,Gs):
	N = len(nodes)
	adj = np.zeros((N, N), dtype=int)
	if btype == 'base':
		for line in data:
			adj[int(line[0]),int(line[1])] = 1
	else:
		for node, neighbors in Gs[btype].items():
			for neighbor in neighbors:
				adj[node][neighbor] = 1
	return csr_matrix(adj).astype('float32')

def initialize_features(args, data, nodes,  Gs):
	# Encoder Based Approach
	print('### Generating initial features by Encoder-Based-Approach...')
	initial_feats = dict()
	btypes = Gs.keys()
	for btype in btypes:
		A = generate_adj(data, nodes, btype, Gs).todense()
		initial_feat = train_tiedAE(A,dim=args.dim_f,lr=args.lr_eba,weight_decay=args.weight_decay_eba,n_epochs=args.epoch_eba, batch_size=args.batch_size)
		initial_feats[btype] = preprocessing.normalize(initial_feat)
		# print(btype)
	return initial_feats

def generate_incidence_matrix_multiple(hygraphs):
	Hs = dict()
	btypes = hygraphs.keys()
	n_smp = len(hygraphs['base'])
	for btype in btypes:
		H = generate_incidence_matrix_o(hygraphs[btype],n_smp)
		# H = generate_incidence_matrix(hygraphs[btype])
		Hs[btype] = H
	return Hs

def generate_incidence_matrix_o(hyedges, n_smp):
	H = np.zeros((n_smp,n_smp))
	for key,val in hyedges.items():
		for v in val:
			H[v,key] = 1
	return H

def generate_incidence_matrix_bit(hygraphs):
	Hs = dict()
	btypes = hygraphs.keys()
	for btype in btypes:
		H = construct_hypergraph_matrix_bit(hygraphs[btype])
		Hs[btype] = H
	return Hs


def construct_hypergraph_matrix_bit(hyperedge_dict):
	"""
    根据超图字典构建超图矩阵。

    参数：
    hyperedge_dict (dict): 超边字典，每个超边包含节点的集合。

    返回：
    np.ndarray: 超图矩阵，行表示节点，列表示超边。
    """
	# 提取所有节点和超边
	all_nodes = set()
	for hyperedge in hyperedge_dict.values():
		all_nodes.update(hyperedge)

	node_to_index = {node: i for i, node in enumerate(sorted(all_nodes))}
	num_nodes = len(all_nodes)
	num_hyperedges = len(hyperedge_dict)

	# 初始化超图矩阵
	hypergraph_matrix = np.zeros((num_nodes, num_hyperedges), dtype=int)

	# 填充矩阵
	for j, (hyperedge_key, hyperedge) in enumerate(hyperedge_dict.items()):
		for node in hyperedge:
			i = node_to_index[node]
			hypergraph_matrix[i, j] = 1  # 表示该节点属于该超边

	return hypergraph_matrix

def generate_incidence_matrix(hygraph):
	G = nx.MultiDiGraph()
	for node, neighbors in hygraph.items():
		G.add_node(node)
		for neighbor in neighbors:
			G.add_edge(node, neighbor)

	undirected_G = G.to_undirected()

	# 使用Louvain社区检测算法对无向图进行聚类
	partition = community.best_partition(undirected_G)

	# 获取所有节点
	all_nodes = list(G.nodes)

	# 获取社区标签
	community_labels = list(partition.values())
	num_clusters = max(community_labels) + 1

	# 初始化稀疏关联矩阵
	num_nodes = len(all_nodes)
	num_edges = num_clusters
	H = np.zeros((num_nodes, num_edges), dtype=int)
	# H = np.zeros((n_smp,n_smp), dtype=int)

	# 将每个节点分配给其对应的社区
	for i, label in enumerate(community_labels):
		H[i][label] = 1

	# 输出关联矩阵
	# print("关联矩阵：")
	# print(H.shape)
	# print(H)
	return H

def generate_negative_samples(G, num_neg_samples, nodes):
	neg_samples = []

	degrees = dict(G.degree())
	total_degree = sum(degrees.values())
	node_probabilities = [degree / total_degree for degree in degrees.values()]
	i = 0
	for node in G.nodes():
		neighbors = list(G.neighbors(node))
		non_neighbors = list(set(nodes) - set(neighbors) - {node})
		# neg_nodes = np.random.choice(list(non_neighbors), num_neg_samples, replace=False)
		node_pro = []
		for neighbor in non_neighbors:
			node_pro.append(node_probabilities[neighbor])
		node_pro = np.array(node_pro)
		# node_pro = [node_probabilities[neighbor] for neighbor in non_neighbors]
		total_pro = sum(node_pro)
		node_probabilities_normalized = [prob / total_pro for prob in node_pro]
		neg_nodes = np.random.choice(non_neighbors, num_neg_samples, replace=False,p=node_probabilities_normalized)
		i+=1
		print(i)

		for neg_node in neg_nodes:
			neg_samples.append((node, neg_node))
			print((node,neg_node))
	np.random.shuffle(neg_samples)
	neg_samples = np.array(neg_samples)
	neg_samples = neg_samples.astype(int)
	return neg_samples


def generate_negative_samples_bitcoin(G, num_neg_samples, num_edges):
	# 获取所有节点和已有的边
	nodes = list(G.nodes())
	existing_edges = set(G.edges())

	negative_edges = []
	total_needed = num_neg_samples * num_edges

	# 使用 tqdm 显示进度条
	with tqdm(total=total_needed, desc="Generating Negative Samples") as pbar:
		while len(negative_edges) < total_needed:
			# 随机选择两个节点
			u, v = np.random.choice(nodes, 2, replace=False)

			# 确保选择的边不在图中（即为负边）
			if (u, v) not in existing_edges and (v, u) not in existing_edges:
				negative_edges.append((u, v))
				pbar.update(1)  # 更新进度条

	# 打乱负样本的顺序
	np.random.shuffle(negative_edges)

	# 转换为 numpy 数组并确保类型为 int
	neg_samples = np.array(negative_edges, dtype=int)

	return neg_samples

