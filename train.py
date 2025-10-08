#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import torch
from HyperEvoNet import train_HyperEvoNet
from evaluation import node_classification
from Ethereum_subgraph import *
from bitcoin_subgraph import *
from tqdm import tqdm
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'



import networkx as nx
def parse_args():
	parser = argparse.ArgumentParser(description="Run HyperEvoNet")
	parser.add_argument('--data', type=str, default='Bitcoin')
	parser.add_argument('--task', type=str, default='NC')
	parser.add_argument('--ratio', type=float, default=0.7)
	parser.add_argument('--n_neg', type=int, default=1)
	parser.add_argument('--conv', type=str, default='asym')
	parser.add_argument('--attr', type=bool, default=True)
	# Parameters of initializing features (TiedAutoEncoder)
	parser.add_argument('--dim_f', type=int, default=32)
	parser.add_argument('--lr_eba', type=float, default=0.001)
	parser.add_argument('--epoch_eba', type=int, default=100)
	parser.add_argument('--weight_decay_eba', type=float, default=5e-4)
	parser.add_argument('--batch_size', type=int, default=16)
	# Parameters of HyperEvoNet
	parser.add_argument('--dim', type=list, default=[16,16])
	parser.add_argument('--lr', type=float, default=0.01)
	parser.add_argument('--epoch', type=int, default=100)
	parser.add_argument('--batch_size_Dual', type=int, default=8)
	parser.add_argument('--optimizer', type=str, default='Adam')
	parser.add_argument('--weight_decay', type=float, default=5e-4)
	parser.add_argument('--dropout', type=float, default=0.5)
	parser.add_argument('--inter', type=str, default=True)
	parser.add_argument('--intra', type=str, default=False)
	parser.add_argument('--lamb', type=float, default=0.5)

	args = parser.parse_args()
	return args

if __name__ == '__main__':
	# 检查是否有可用的 CUDA 设备
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	print(device)
	args = parse_args()
	print(args.data, args.task, args.n_neg, args.conv)
	print(args.dim, args.lr, args.epoch, args.optimizer, args.dropout,args.inter, args.intra, args.lamb)
	#Etherum
	# 加载节点特征
	# G = load_pickle('./Ethereum/Sub_MultiGraph_num_10000_2.pkl') #37897node 130 phishing node
	# print(nx.info(G))
	# feats, label = get_feature(G)
	# nodes, data = get_edges(G)
	#
	# num = len(nodes)
	# hygraphs = construct_hierarchical_hypergraph(G,nodes)  # ['base','0','1','2','3','4']
	# hygraph = hygraphs['base']
	# print(hygraphs.keys())

    #Bitcoin
	# 加载图
	G = load_graph('.DataSet/Bitcoin/G_bit.pkl') #由bitcoin文件夹下csv生成graph

	# 获取图的基本信息
	nodes, data, feats, label = get_graph_info(G)

	num = len(nodes)

	subgraph_info , hygraphs = construct_hierarchical_hypergraph_bit(G)  # ['base','0','1'...]
	hygraph = hygraphs['base']
	data = np.array(data)
	pos_samples = data[:, :2]
	print("获取正样本")
	pos_samples = pos_samples.astype(int)
	neg_samples = generate_negative_samples_bitcoin(G, args.n_neg,len(data))
	print("### Positive samples: %d, Negative samples: %d" % (pos_samples.shape[0], neg_samples.shape[0]))
	samples = dict()
	samples['pos_samples'] = pos_samples
	samples['neg_samples'] = neg_samples
	id_to_index = {node_id: index for index, node_id in enumerate(nodes)}
	if args.attr == False:
		init_embs = initialize_features(args,data,nodes,hygraphs)
	elif args.attr == True:
		init_embs = load_attributes_bit(hygraphs.keys(), feats, subgraph_info)
	print("### Hs is constructing!!!")
	Hs = generate_incidence_matrix_bit(hygraphs)
	print("### Start Train!!!")
	embs = train_HyperEvoNet(args, init_embs, Hs, mapped_target1 , mapped_target2, nodes,subgraph_info,id_to_index)

	print("### Node Classification task:")
	acc,auc,f1,precision,recall = node_classification(embs, label,id_to_index)
	print('Acc: {:.4f}'.format(acc))
	print('AUC: {:.4f}'.format(auc))
	print('F1: {:.4f}, Precision: {:.4f}, Recall: {:.4f}'.format(f1, precision, recall))

print()