import networkx as nx
from networkx.algorithms.community import louvain_communities, girvan_newman
import numpy as np


def compute_degree_centrality(G):
    return nx.degree_centrality(G)


def compute_betweenness_centrality(G):
    return nx.betweenness_centrality(G)


def compute_closeness_centrality(G):
    return nx.closeness_centrality(G)


def compute_pagerank(G):
    return nx.pagerank(G)


def compute_eigenvector_centrality(G):
    try:
        return nx.eigenvector_centrality(G, max_iter=1000)
    except:
        return {node: 0.0 for node in G.nodes()}


def compute_katz_centrality(G):
    try:
        return nx.katz_centrality(G, alpha=0.1, beta=1.0, max_iter=1000)
    except:
        return {node: 0.0 for node in G.nodes()}


def compute_harmonic_centrality(G):
    return nx.harmonic_centrality(G)


def detect_communities(G):
    communities = louvain_communities(G, seed=42)
    node_to_community = {}
    for idx, community in enumerate(communities):
        for node in community:
            node_to_community[node] = idx
    return node_to_community, communities


def detect_communities_girvan_newman(G, num_communities=2):
    comp = girvan_newman(G)
    for _ in range(num_communities - 1):
        try:
            communities = next(comp)
        except StopIteration:
            break
    node_to_community = {}
    for idx, community in enumerate(communities):
        for node in community:
            node_to_community[node] = idx
    return node_to_community, list(communities)


def get_top_nodes(centrality_dict, n=5):
    sorted_nodes = sorted(centrality_dict.items(), key=lambda x: x[1], reverse=True)
    return sorted_nodes[:n]


def compute_network_metrics(G):
    # Use undirected version for metrics that require it
    G_undirected = G.to_undirected() if G.is_directed() else G
    
    # Check connectivity on undirected version
    is_connected = nx.is_connected(G_undirected)
    
    metrics = {
        'nodes': G.number_of_nodes(),
        'edges': G.number_of_edges(),
        'density': nx.density(G),
        'average_clustering': nx.average_clustering(G_undirected),
        'transitivity': nx.transitivity(G_undirected),
        'average_degree': sum(dict(G.degree()).values()) / G.number_of_nodes() if G.number_of_nodes() > 0 else 0,
        'diameter': nx.diameter(G_undirected) if is_connected else 'N/A (disconnected)',
        'average_shortest_path': nx.average_shortest_path_length(G_undirected) if is_connected else 'N/A',
        'assortativity': nx.degree_assortativity_coefficient(G),
        'num_connected_components': nx.number_connected_components(G_undirected)
    }
    return metrics


def compute_link_prediction_scores(G):
    # Link prediction algorithms work on undirected graphs
    G_undirected = G.to_undirected() if G.is_directed() else G
    non_edges = list(nx.non_edges(G_undirected))
    
    # Limit to avoid memory issues with large graphs
    if len(non_edges) > 1000:
        import random
        random.seed(42)
        non_edges = random.sample(non_edges, 1000)
    
    jaccard = list(nx.jaccard_coefficient(G_undirected, non_edges))
    adamic_adar = list(nx.adamic_adar_index(G_undirected, non_edges))
    preferential = list(nx.preferential_attachment(G_undirected, non_edges))
    
    predictions = []
    for i, (u, v) in enumerate(non_edges):
        predictions.append({
            'node1': u,
            'node2': v,
            'jaccard': jaccard[i][2],
            'adamic_adar': adamic_adar[i][2],
            'preferential_attachment': preferential[i][2]
        })
    return sorted(predictions, key=lambda x: x['adamic_adar'], reverse=True)


def identify_bridge_nodes(G):
    # Convert to undirected for bridge detection (bridges only defined for undirected graphs)
    G_undirected = G.to_undirected() if G.is_directed() else G
    bridges = list(nx.bridges(G_undirected))
    bridge_nodes = set()
    for u, v in bridges:
        bridge_nodes.add(u)
        bridge_nodes.add(v)
    return list(bridge_nodes), bridges


def compute_node_vulnerability(G):
    # Use undirected version for connected components
    G_undirected = G.to_undirected() if G.is_directed() else G
    original_components = nx.number_connected_components(G_undirected)
    vulnerability = {}
    for node in G.nodes():
        H = G_undirected.copy()
        H.remove_node(node)
        new_components = nx.number_connected_components(H)
        vulnerability[node] = new_components - original_components
    return vulnerability


def compute_k_core_decomposition(G):
    # K-core decomposition only works on undirected graphs
    G_undirected = G.to_undirected() if G.is_directed() else G
    return nx.core_number(G_undirected)
