import pickle
import random
import networkx as nx
import community  # 导入社区检测库
import csv
import numpy as np
from train_test_split import *

def load_pickle(fname):
    with open(fname, 'rb') as f:
        return pickle.load(f)

def write_pickle(G, fname):
    with open(fname, 'wb') as f:
        pickle.dump(G, f)
    print("存储成功！")
    return True

def _statistic_feature(G):
    attribution_dict = {}
    for node in G.nodes():
        in_degree = G.in_degree(node) #入度
        out_degree = G.out_degree(node) #出度
        count = in_degree + out_degree #交易总数
        #计算交易频率
        edge_time_in = G.in_edges(node,data = True)
        if edge_time_in:
            # 使用sorted函数按时间属性排序边
            sorted_edges = sorted(edge_time_in, key=lambda edge: edge[2]['timestamp'])
            # 获取最小时间和最大时间
            min_time_in = sorted_edges[0][2]['timestamp']
            max_time_in = sorted_edges[-1][2]['timestamp']
        edge_time_out = G.out_edges(node, data=True)
        if edge_time_out:
            # 使用sorted函数按时间属性排序边
            sorted_edges = sorted(edge_time_out, key=lambda edge: edge[2]['timestamp'])
            # 获取最小时间和最大时间
            min_time_out = sorted_edges[0][2]['timestamp']
            max_time_out = sorted_edges[-1][2]['timestamp']
        if min_time_in < min_time_out:
            min_time = min_time_in
        else:
            min_time = min_time_out
        if max_time_in > max_time_out:
            max_time = max_time_in
        else:
            max_time = max_time_out
        time = max_time-min_time
        freq = time / count
        # if time == 0 :
        #     freq = count / (time+1)
        # else:
        #     freq = count / time #交易频率

        #计算转入/转出金额，差值，总额
        in_amount = 0
        out_amount = 0
        # 遍历与节点关联的每条边，获取金额属性并相加
        for source,_,attributes in G.in_edges(node, data=True):
            in_amount += attributes['amount']
        for source, _, attributes in G.out_edges(node, data=True):
            out_amount += attributes['amount']
        all_amount = in_amount + out_amount #总额
        diff_amount = in_amount - out_amount #差值
        #属性存入字典
        key = node
        # 创建一个属性字典，将属性和值配对
        attributes = {'in_degree':in_degree,'out_degree':out_degree,'count_transaction':count,'freqent':freq, 'all_amount':all_amount,'in_amount':in_amount,'out_amount':out_amount,'diff_amount':diff_amount}
        # 将键值对存储到字典
        attribution_dict[key] = attributes
    nx.set_node_attributes(G, attribution_dict)
    return G

def create_subgraphs_sliding_window(mulditigraph, window_size,step_size):
    subgraphs = []
    edge_data = list(mulditigraph.edges(data=True, keys=True))
    edge_data.sort(key=lambda edge: edge[3]['timestamp'])  # 按时间嵌入属性排序
    # start = edge_data[0][3]['timestamp']
    # end = edge_data[-1][3]['timestamp']
    # count = (end - start) / 60 / 60 / 24
    # print(count)
    start_time = edge_data[0][3]['timestamp']
    end_time = start_time + window_size
    current_subgraph = nx.MultiDiGraph()

    for u, v, k, data in edge_data:
        if data['timestamp'] <= end_time:
            current_subgraph.add_edge(u, v, key=k, amount = data['amount'], timestamp = data['timestamp'])
        else:
            subgraphs.append(current_subgraph.copy())
            current_subgraph = nx.MultiDiGraph()
            current_subgraph.add_edge(u, v, key=k, amount = data['amount'], timestamp=data['timestamp'])
            start_time = end_time - step_size
            end_time = start_time + window_size

    # 复制节点嵌入到子图
    all_current_nodes = current_subgraph.nodes()
    attribution_dict = {}
    for node in all_current_nodes:
        attributes = mulditigraph.nodes[node]
        attribution_dict[node] = attributes
    nx.set_node_attributes(current_subgraph, attribution_dict)
    subgraphs.append(current_subgraph)  # 添加最后一个子图
    return subgraphs

def subgraph_F(G,subgraphs):
    subgraphs_f = []
    for current_subgraph in subgraphs:
        # 复制节点嵌入到子图
        all_current_nodes = current_subgraph.nodes()
        attribution_dict = {}
        for node in all_current_nodes:
            attributes = G.nodes[node]
            attribution_dict[node] = attributes
        nx.set_node_attributes(current_subgraph, attribution_dict)
        subgraphs_f.append(current_subgraph)
    return subgraphs_f

def _subgraph_F(G,current_subgraph):

    all_current_nodes = current_subgraph.nodes()
    attribution_dict = {}
    for node in all_current_nodes:
        attributes = G.nodes[node]
        attribution_dict[node] = attributes
    nx.set_node_attributes(current_subgraph, attribution_dict)
    return current_subgraph

def random_node_split(graph, test_ratio=0.3, seed=None):
    if seed is not None:
        random.seed(seed)

    nodes = list(graph.nodes())
    num_test_nodes = int(len(nodes) * test_ratio)

    test_nodes = random.sample(nodes, num_test_nodes)
    train_nodes = [node for node in nodes if node not in test_nodes]

    return train_nodes, test_nodes

def create_H_with_dfs(G, max_depth=4):
    num_nodes = len(G.nodes)
    num_edges = 0

    all_nodes = list(G.nodes)

    # 使用NumPy创建一个全零关联矩阵
    adjacency_matrix = [[0] * num_edges for _ in range(num_nodes)]

    # 定义一个函数来执行深度优先搜索
    def dfs(node, depth, path):
        if depth == max_depth:
            return
        for neighbor in G.successors(node):
            if neighbor not in path:
                path.append(neighbor)
                for i in range(num_nodes):
                    if i in (int(all_nodes.index(node)),int(all_nodes.index(neighbor))):
                        adjacency_matrix[i].append(1)
                    else:
                        adjacency_matrix[i].append(0)
                dfs(neighbor, depth + 1, path)
                path.pop()
        return

    # 遍历所有节点并执行深度优先搜索
    for node in all_nodes:
        path = [node]
        dfs(node, 0, path)
        num_edges += 1

    # 输出关联矩阵
    print("关联矩阵：")
    print(adjacency_matrix)

    return G, adjacency_matrix

def create_H_with_community(G):
    # 将有向图转换为无向图
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
    adjacency_matrix = np.zeros((num_nodes, num_edges), dtype=int)

    # 将每个节点分配给其对应的社区
    for i, label in enumerate(community_labels):
        adjacency_matrix[i][label] = 1

    # 输出关联矩阵
    print("关联矩阵：")
    print(adjacency_matrix.shape)
    print(adjacency_matrix)


    return adjacency_matrix

def create_dataset_subgraphs(mulditigraph, window_size):
    subgraphs = []
    edge_data = list(mulditigraph.edges(data=True, keys=True))
    edge_data.sort(key=lambda edge: edge[3]['timestamp'])  # 按时间嵌入属性排序
    # start = edge_data[0][3]['timestamp']
    # end = edge_data[-1][3]['timestamp']
    # count = (end - start) / 60 / 60 / 24
    # print(count)
    start_time = edge_data[0][3]['timestamp'] + 67.5 * 24 * 60 * 60 * 3
    end_time = start_time + window_size
    current_subgraph = nx.MultiDiGraph()

    for u, v, k, data in edge_data:
        if data['timestamp'] <= end_time:
            current_subgraph.add_edge(u, v, key=k, amount = data['amount'], timestamp = data['timestamp'])

    # 复制节点嵌入到子图
    all_current_nodes = current_subgraph.nodes()
    attribution_dict = {}
    for node in all_current_nodes:
        attributes = mulditigraph.nodes[node]
        attribution_dict[node] = attributes
    nx.set_node_attributes(current_subgraph, attribution_dict)
    return current_subgraph

if __name__ == '__main__':

    G = load_pickle('./Ethereum/MulDiGraph_F.pkl')
    print(nx.info(G))
    for idx, nd in enumerate(nx.nodes(G)):
        print(nd)
        print(G.nodes[nd])
        break
    # #子图划分
    window_size = 67.5 * 24 * 60 * 60 * 4  # 67.5 days in seconds
    subgraph = create_dataset_subgraphs(G,window_size)
    window_size = 67.5 * 24 * 60 * 60   # 67.5 days in seconds
    step_size = 30 * 24 * 60 * 60   # 30 days in seconds
    subgraphs = create_subgraphs_sliding_window(G, window_size, step_size)
    print(nx.info(subgraphs[0]))
    subgraphs = subgraph_F(G,subgraphs)
    print(len(subgraphs))
    print(nx.info(subgraphs[0]))
    








