import matplotlib.pyplot as plt
import networkx as nx


def visualize_network_with_communities(G, node_to_community, title="Social Network with Communities"):
    plt.figure(figsize=(14, 10))
    
    pos = nx.spring_layout(G, seed=42, k=2)
    
    num_communities = len(set(node_to_community.values()))
    colors = plt.cm.tab10(range(num_communities))
    
    node_colors = [colors[node_to_community[node]] for node in G.nodes()]
    
    nx.draw_networkx_edges(G, pos, alpha=0.3, edge_color='gray')
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=500, alpha=0.9)
    nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold')
    
    plt.title(title, fontsize=16, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    plt.show()


def visualize_diffusion(G, node_to_community, activated_nodes, seed_nodes, title="Information Diffusion"):
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
