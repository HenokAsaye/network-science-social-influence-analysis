# This module provides visualization utilities for social network analysis.
# It supports community visualization, information diffusion visualization,
# and comparison of different centrality metrics using NetworkX and Matplotlib.

import matplotlib.pyplot as plt
import networkx as nx


def visualize_network_with_communities(G, node_to_community, title="Social Network with Communities"):
    """
    Visualizes a social network graph with nodes colored by their community.

    Args:
        G (networkx.Graph): The social network graph.
        node_to_community (dict): Mapping of node -> community ID.
        title (str): Title of the plot.
    """
    # Create a figure with a fixed size for better readability  
    plt.figure(figsize=(14, 10))
    # Compute node positions using spring layout for clear visualization
    pos = nx.spring_layout(G, seed=42, k=2)
    # Determine the number of unique communities
    num_communities = len(set(node_to_community.values()))
    # Generate distinct colors for each community
    colors = plt.cm.tab10(range(num_communities))
    # Assign a color to each node based on its community
    node_colors = [colors[node_to_community[node]] for node in G.nodes()]
    # Draw edges with low opacity to reduce visual clutter
    nx.draw_networkx_edges(G, pos, alpha=0.3, edge_color='gray')
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=500, alpha=0.9)
    # Draw node labels
    nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold')
    
    plt.title(title, fontsize=16, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    plt.show()


def visualize_diffusion(G, node_to_community, activated_nodes, seed_nodes, title="Information Diffusion"):
    """
    Visualizes the spread of information in a network.

    Seed nodes, activated nodes, and non-activated nodes are shown
    using different colors.

    Args:
        G (networkx.Graph): The social network graph.
        node_to_community (dict): Mapping of node -> community ID (optional context).
        activated_nodes (set): Nodes activated during diffusion.
        seed_nodes (set): Initial seed nodes.
        title (str): Title of the plot.
    """
    plt.figure(figsize=(14, 10))
    
    pos = nx.spring_layout(G, seed=42, k=2)
    
    node_colors = []
    for node in G.nodes():
        if node in seed_nodes:
            node_colors.append('red')
        elif node in activated_nodes:
            node_colors.append('orange')
        else:
            node_colors.append('lightgray')
    
    nx.draw_networkx_edges(G, pos, alpha=0.3, edge_color='gray')
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=500, alpha=0.9)
    nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold')
    
    legend_elements = [
        plt.scatter([], [], c='red', s=100, label='Seed Nodes'),
        plt.scatter([], [], c='orange', s=100, label='Activated Nodes'),
        plt.scatter([], [], c='lightgray', s=100, label='Not Activated')
    ]
    plt.legend(handles=legend_elements, loc='upper left', fontsize=10)
    
    plt.title(title, fontsize=16, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    plt.show()


def visualize_centrality_comparison(centrality_metrics, top_n=10):
    """
    Compares different centrality measures by plotting the top-N nodes
    for each metric.

    Args:
        centrality_metrics (list of dict): List containing centrality dictionaries
                                           (Degree, Betweenness, Closeness, PageRank).
        top_n (int): Number of top nodes to display for each metric.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    metric_names = ['Degree', 'Betweenness', 'Closeness', 'PageRank']
    
    for ax, (metric_name, centrality_dict) in zip(axes.flatten(), zip(metric_names, centrality_metrics)):
        sorted_items = sorted(centrality_dict.items(), key=lambda x: x[1], reverse=True)[:top_n]
        nodes = [item[0] for item in sorted_items]
        values = [item[1] for item in sorted_items]
        
        bars = ax.barh(nodes, values, color='steelblue')
        ax.set_xlabel('Centrality Score')
        ax.set_title(f'{metric_name} Centrality (Top {top_n})')
        ax.invert_yaxis()
    
    plt.tight_layout()
    plt.show()
