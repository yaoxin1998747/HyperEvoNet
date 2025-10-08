import pickle
import random
import networkx as nx
import torch
from loader import *
import scipy.sparse as sp

def train_test_split(data):
    # 随机划分节点为测试集和验证集
    num_nodes = len(data.edge_index[0].unique())
    indices = list(range(num_nodes))
    random.shuffle(indices)
    split_idx = int(0.8 * num_nodes)  # 80% 节点用于训练，20% 用于测试
    train_idx, test_idx = indices[:split_idx], indices[split_idx:]

    return train_idx, test_idx

def get_feature(G):
    X=[]
    label = []
    for node in G.nodes():
        data = list(G.nodes[node].values())
        label.append(data[0])
        X.append(data[1:])
    return X,label
def get_edges(G):
    # name_to_number = {name: i for i, name in enumerate(G.nodes())}
    # 创建一个空列表来存储所有边
    edges_lists = []
    nodes = list(G.nodes())
    # nodes = list(name_to_number.values())
    # 遍历图的边，将节点名称转换为数字编号并提取权重，添加到列表中
    for u, v, data in G.edges(data=True):
        u_number = u
        v_number = v
        amount = data['amount']
        timestamp = data['timestamp']
        edge_list = [u_number, v_number, amount,timestamp]
        edges_lists.append(edge_list)

    return nodes, edges_lists
def subgraph_F(G,current_subgraph):

    all_current_nodes = current_subgraph.nodes()
    attribution_dict = {}
    for node in all_current_nodes:
        attributes = G.nodes[node]
        attribution_dict[node] = attributes
    nx.set_node_attributes(current_subgraph, attribution_dict)
    return current_subgraph

def is_phishing_node(node):
    # 假设节点特征是一个列表，其中第一维表示是否为钓鱼节点
    return G.nodes[node]['isp'] == 1  # 假设第一维为1表示钓鱼节点

def extract_random_subgraph(G, num_nodes=5000):
    subgraph = nx.MultiGraph()
    all_nodes = list(G.nodes())
    phishing_nodes = [node for node in all_nodes if is_phishing_node(node)]
    select_phishing_nodes = random.sample(list(set(phishing_nodes)), 500)
    select_remaining_nodes = random.sample(list(set(all_nodes) - set(phishing_nodes)), num_nodes-500)
    select_nodes = select_phishing_nodes + select_remaining_nodes
    i=0
    for edge in G.edges(keys=True,data=True):
        u, v, key,data = edge
        # 检查边的两个节点是否符合要求
        if (u not in select_nodes) or (v not in select_nodes):
            continue
        else:
            subgraph.add_edge(u, v, key=key,data=data)  # 添加边到多重图子图
        i = i+1
        print(i)

    # 遍历普通节点并将其添加到多重图子图
    for node in select_remaining_nodes:
        subgraph.add_node(node)

    # 遍历钓鱼节点并将其添加到多重图子图
    for node in phishing_nodes:
        subgraph.add_node(node)
    return subgraph



if __name__ == '__main__':
    # 检查是否有可用的 CUDA 设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    G = load_pickle('./Ethereum/MulDiGraph_F.pkl')
    print(nx.info(G))
    # 设置起始节点和步数
    subgraph = extract_random_subgraph(G,5000)
    write_pickle(subgraph,'./Ethereum/Dataset.pkl')
    print("子图节点数量: ", len(subgraph.nodes()))
    print("子图边数量: ", len(subgraph.edges()))

