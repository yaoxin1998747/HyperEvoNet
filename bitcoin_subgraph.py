
import networkx as nx
import pandas as pd
from networkx.algorithms import community
from community import community_louvain  # Louvain 算法更适合大图
import torch
def load_graph(filepath):
    """加载图文件并返回NetworkX图对象"""
    return nx.read_gpickle(filepath)


def get_graph_info(G):
    """获取图的基本信息，包括节点列表、边列表、特征和标签"""
    nodes = list(G.nodes())
    edges = list(G.edges())
    features = {node: G.nodes[node].get('features', {}) for node in G.nodes()}
    labels = {node: G.nodes[node].get('label') for node in G.nodes()}
    # d = dict(nx.degree(G))


    print("Graph Information:")
    print(f"Number of nodes: {len(nodes)}")
    print(f"Number of edges: {len(edges)}")
    # print(d)
    # print("平均度为：", sum(d.values()) / len(G.nodes))
    return nodes, edges, features, labels


def create_subgraph(G, timestamp):
    """基于指定的时间戳创建子图，并返回子图的节点、边、特征和标签信息"""
    nodes_in_slice = [
        node for node, data in G.nodes(data=True)
        if data.get('timestamp') == timestamp
    ]
    subgraph = G.subgraph(nodes_in_slice).copy()

    subgraph_nodes = list(subgraph.nodes())
    subgraph_edges = list(subgraph.edges())
    subgraph_features = {node: subgraph.nodes[node].get('features', {}) for node in subgraph_nodes}
    subgraph_labels = {node: subgraph.nodes[node].get('label') for node in subgraph_nodes}

    return subgraph, subgraph_nodes, subgraph_edges, subgraph_features, subgraph_labels


def subgraphs_to_dataframes(subgraph_nodes, subgraph_edges, subgraph_features, subgraph_labels):
    """将子图信息转换为DataFrame格式，便于查看"""
    nodes_df = pd.DataFrame(subgraph_nodes, columns=['Node'])
    edges_df = pd.DataFrame(subgraph_edges, columns=['Source', 'Target'])
    features_df = pd.DataFrame.from_dict(subgraph_features, orient='index').reset_index().rename(
        columns={'index': 'Node'})
    labels_df = pd.DataFrame.from_dict(subgraph_labels, orient='index', columns=['Label']).reset_index().rename(
        columns={'index': 'Node'})

    return nodes_df, edges_df, features_df, labels_df


def generate_all_subgraphs(G, start_timestamp=1, end_timestamp=49):
    """生成所有指定时间范围的子图信息"""
    subgraphs_info = []

    for timestamp in range(start_timestamp, end_timestamp + 1):
        subgraph, subgraph_nodes, subgraph_edges, subgraph_features, subgraph_labels = create_subgraph(G, timestamp)
        nodes_df, edges_df, features_df, labels_df = subgraphs_to_dataframes(
            subgraph_nodes, subgraph_edges, subgraph_features, subgraph_labels
        )

        subgraphs_info.append({
            "timestamp": timestamp,
            "graph": subgraph,
            "nodes": nodes_df,
            "edges": edges_df,
            "features": features_df,
            "labels": labels_df
        })

    return subgraphs_info


def display_subgraph_info(subgraphs_info):
    """输出所有子图的基本信息"""
    for info in subgraphs_info:
        print(f"\nTimestamp: {info['timestamp']}")
        print(f"Number of nodes: {info['nodes'].shape[0]}")
        print(f"Number of edges: {info['edges'].shape[0]}")
        print("Nodes DataFrame:")
        print(info['nodes'].head())
        print("Edges DataFrame:")
        print(info['edges'].head())
        print("Features DataFrame:")
        print(info['features'].head())
        print("Labels DataFrame:")
        print(info['labels'].head())


def cluster_graph(G):
    """使用Girvan-Newman算法对图进行聚类，并返回每个聚类簇作为列表"""
    Hygraph = dict()
    if nx.is_directed(G):
        G = G.to_undirected()
    partition = community_louvain.best_partition(G)  # 获取每个节点的社区
    clusters = {}
    for node, community_id in partition.items():
        if community_id not in clusters:
            clusters[community_id] = []
        clusters[community_id].append(node)
    clusters= list(clusters.values())
    n_neigs = 0

    for i, cluster in enumerate(clusters):
        hyperedge_id = f"cluster_{i}"  # 超边ID
        Hygraph[hyperedge_id] = cluster  # 每个簇作为超边存储
        n_neigs += len(cluster)  # 计算所有超边中的节点总数

    # 输出统计信息
    print('### The number of hyper-edges (clusters): %d' % len(clusters))
    print('### The average nodes in each hyper-edge: %0.2f' % (n_neigs / len(clusters)))

    return Hygraph


def construct_hierarchical_hypergraph_bit(G):
    hygraphs = dict()
    Gs = generate_all_subgraphs(G)
    print(len(Gs))
    hygraphs['base'] = cluster_graph(G)
    for i, graph in enumerate(Gs):
        subgraph = graph["graph"]
        hygraphs[str(i)] = cluster_graph(subgraph)
    return Gs, hygraphs


def display_hypergraph_info(H):
    """输出超图的基本信息"""
    print("Hypergraph Information:")
    num_hyperedges = sum(1 for n, data in H.nodes(data=True) if data.get('type') == 'hyperedge')
    num_nodes = H.number_of_nodes() - num_hyperedges
    print(f"Number of hyperedges (clusters): {num_hyperedges}")
    print(f"Number of nodes in original graph: {num_nodes}")
    print(f"Total nodes in hypergraph (including hyperedges): {H.number_of_nodes()}")
    print(f"Total edges in hypergraph: {H.number_of_edges()}")

    # 打印每个超边节点及其连接的实际节点
    for n, data in H.nodes(data=True):
        if data.get('type') == 'hyperedge':
            connected_nodes = list(H.neighbors(n))
            print(f"{n} connects to nodes: {connected_nodes}")
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



def map_and_fill_add_x(original_nodes, subgraph_node_ids, subgraph_features):
    """
    将子图的节点特征映射到原图的特征矩阵中。

    :param original_nodes: 原图的节点 ID 列表
    :param subgraph_node_ids: 子图的节点 ID 列表
    :param subgraph_features: 子图节点特征的张量
    :return: 填充后的特征矩阵
    """
    # 创建原图节点 ID 到索引的映射
    id_to_index = {node_id: index for index, node_id in enumerate(original_nodes)}

    # 初始化 add_x，用于填充特征
    add_x = torch.zeros(len(original_nodes), subgraph_features.size(1))  # 假设特征维度一致

    # 填充特征
    for subgraph_id in subgraph_node_ids:
        if subgraph_id in id_to_index:  # 检查子图节点是否在原图中
            original_index = id_to_index[subgraph_id]  # 获取原图中节点的索引
            subgraph_index = subgraph_node_ids.index(subgraph_id)  # 获取子图节点的索引
            add_x[original_index] = subgraph_features[subgraph_index]  # 填充特征

    return add_x

# 示例运行流程
if __name__ == "__main__":
    # 加载图
    G = load_graph('./Bitcoin/G_bit.pkl')

    # 获取图的基本信息
    nodes, edges, features, labels = get_graph_info(G)
    print(nodes[0:2])

    # 生成所有时间片的子图信息
    subgraphs_info = generate_all_subgraphs(G)
    node_ids_dictionary = extract_node_ids(subgraphs_info[0])
    node_ids_list = node_ids_dictionary[0:2]
    subgraph_features = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    add_x = map_and_fill_add_x(nodes, node_ids_list, subgraph_features)
    print(add_x.size())

    # 输出每个子图的基本信息
    # display_subgraph_info(subgraphs_info)
    #
    # hygraphs = construct_hierarchical_hypergraph_bit(G)  # ['base','0','1','2','3','4']
    #
    # hygraph = hygraphs['base']
    #
    # print(hygraphs.keys())
