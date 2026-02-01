
import networkx as nx
from networkx.algorithms.community import louvain_communities, girvan_newman
import numpy as np
from typing import Any, Dict, List, Tuple


def compute_degree_centrality(G: nx.Graph) -> Dict[Any, float]:
    """
    Compute the degree centrality for each node in the graph.
    Args:
        G (nx.Graph): The input graph.
    Returns:
        Dict[Any, float]: Mapping of node to its degree centrality.
    """
    return nx.degree_centrality(G)


def compute_betweenness_centrality(G: nx.Graph) -> Dict[Any, float]:
    """
    Compute the betweenness centrality for each node in the graph.
    Args:
        G (nx.Graph): The input graph.
    Returns:
        Dict[Any, float]: Mapping of node to its betweenness centrality.
    """
    return nx.betweenness_centrality(G)


def compute_closeness_centrality(G: nx.Graph) -> Dict[Any, float]:
    """
    Compute the closeness centrality for each node in the graph.
    Args:
        G (nx.Graph): The input graph.
    Returns:
        Dict[Any, float]: Mapping of node to its closeness centrality.
    """
    return nx.closeness_centrality(G)


def compute_pagerank(G: nx.Graph) -> Dict[Any, float]:
    """
    Compute the PageRank for each node in the graph.
    Args:
        G (nx.Graph): The input graph.
    Returns:
        Dict[Any, float]: Mapping of node to its PageRank score.
    """
    return nx.pagerank(G)


def compute_eigenvector_centrality(G: nx.Graph) -> Dict[Any, float]:
    """
    Compute the eigenvector centrality for each node in the graph.
    Args:
        G (nx.Graph): The input graph.
    Returns:
        Dict[Any, float]: Mapping of node to its eigenvector centrality.
    """
    try:
        return nx.eigenvector_centrality(G, max_iter=1000)
    except:
        return {node: 0.0 for node in G.nodes()}


def compute_katz_centrality(G: nx.Graph) -> Dict[Any, float]:
    """
    Compute the Katz centrality for each node in the graph.
    Args:
        G (nx.Graph): The input graph.
    Returns:
        Dict[Any, float]: Mapping of node to its Katz centrality.
    """
    try:
        return nx.katz_centrality(G, alpha=0.1, beta=1.0, max_iter=1000)
    except:
        return {node: 0.0 for node in G.nodes()}


def compute_harmonic_centrality(G: nx.Graph) -> Dict[Any, float]:
    """
    Compute the harmonic centrality for each node in the graph.
    Args:
        G (nx.Graph): The input graph.
    Returns:
        Dict[Any, float]: Mapping of node to its harmonic centrality.
    """
    return nx.harmonic_centrality(G)


def detect_communities(G: nx.Graph) -> Tuple[Dict[Any, int], List[List[Any]]]:
    """
    Detect communities in the graph using the Louvain method.
    Args:
        G (nx.Graph): The input graph.
    Returns:
        Tuple[Dict[Any, int], List[List[Any]]]: Mapping of node to community and list of communities.
    """
    communities = louvain_communities(G, seed=42)
    node_to_community: Dict[Any, int] = {}
    for idx, community in enumerate(communities):
        for node in community:
            node_to_community[node] = idx
    return node_to_community, communities


def detect_communities_girvan_newman(G: nx.Graph, num_communities: int = 2) -> Tuple[Dict[Any, int], List[List[Any]]]:
    """
    Detect communities in the graph using the Girvan-Newman algorithm.
    Args:
        G (nx.Graph): The input graph.
        num_communities (int): Number of communities to find.
    Returns:
        Tuple[Dict[Any, int], List[List[Any]]]: Mapping of node to community and list of communities.
    """
    comp = girvan_newman(G)
    for _ in range(num_communities - 1):
        try:
            communities = next(comp)
        except StopIteration:
            break
    node_to_community: Dict[Any, int] = {}
    for idx, community in enumerate(communities):
        for node in community:
            node_to_community[node] = idx
    return node_to_community, list(communities)


def get_top_nodes(centrality_dict: Dict[Any, float], n: int = 5) -> List[Tuple[Any, float]]:
    sorted_nodes = sorted(centrality_dict.items(), key=lambda x: x[1], reverse=True)
    return sorted_nodes[:n]


def compute_network_metrics(G: nx.Graph) -> Dict[str, Any]:
    metrics = {
        'nodes': G.number_of_nodes(),
        'edges': G.number_of_edges(),
        'density': nx.density(G),
        'average_clustering': nx.average_clustering(G),
        'transitivity': nx.transitivity(G),
        'average_degree': sum(dict(G.degree()).values()) / G.number_of_nodes(),
        'diameter': nx.diameter(G) if nx.is_connected(G) else 'N/A (disconnected)',
        'average_shortest_path': nx.average_shortest_path_length(G) if nx.is_connected(G) else 'N/A',
        'assortativity': nx.degree_assortativity_coefficient(G),
        'num_connected_components': nx.number_connected_components(G)
    }
    return metrics


def compute_link_prediction_scores(G: nx.Graph) -> List[Dict[str, Any]]:
    non_edges = list(nx.non_edges(G))
    jaccard = list(nx.jaccard_coefficient(G, non_edges))
    adamic_adar = list(nx.adamic_adar_index(G, non_edges))
    preferential = list(nx.preferential_attachment(G, non_edges))
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


def identify_bridge_nodes(G: nx.Graph) -> Tuple[List[Any], List[Tuple[Any, Any]]]:
    bridges = list(nx.bridges(G))
    bridge_nodes = set()
    for u, v in bridges:
        bridge_nodes.add(u)
        bridge_nodes.add(v)
    return list(bridge_nodes), bridges


def compute_node_vulnerability(G: nx.Graph) -> Dict[Any, int]:
    original_components = nx.number_connected_components(G)
    vulnerability: Dict[Any, int] = {}
    for node in G.nodes():
        H = G.copy()
        H.remove_node(node)
        new_components = nx.number_connected_components(H)
        vulnerability[node] = new_components - original_components
    return vulnerability


def compute_k_core_decomposition(G: nx.Graph) -> Dict[Any, int]:
    return nx.core_number(G)
