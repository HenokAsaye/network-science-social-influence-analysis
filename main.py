
import logging
from src.load_data import load_network_data
from src.build_graph import build_graph
from src.analysis import (
    compute_degree_centrality,
    compute_betweenness_centrality,
    compute_closeness_centrality,
    compute_pagerank,
    detect_communities,
    get_top_nodes
)
from src.diffusion import run_diffusion_simulation
from src.visualize import (
    visualize_network_with_communities,
    visualize_diffusion,
    visualize_centrality_comparison
)


def print_separator():
    print("=" * 60)


def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )
    print_separator()
    logging.info("SOCIAL NETWORK ANALYSIS - INFLUENCE & DIFFUSION STUDY")
    print_separator()

    logging.info("[1] Loading network data...")
    df = load_network_data("data/sample_network.csv")
    logging.info(f"Loaded {len(df)} edges from dataset")

    logging.info("[2] Building graph...")
    G = build_graph(df)
    logging.info(f"Nodes: {G.number_of_nodes()}")
    logging.info(f"Edges: {G.number_of_edges()}")
    logging.info(f"Density: {round(2 * G.number_of_edges() / (G.number_of_nodes() * (G.number_of_nodes() - 1)), 4)}")

    logging.info("[3] Computing centrality metrics...")

    degree_cent = compute_degree_centrality(G)
    betweenness_cent = compute_betweenness_centrality(G)
    closeness_cent = compute_closeness_centrality(G)
    pagerank_cent = compute_pagerank(G)
    
    print("\n    >> Degree Centrality (Top 5):")
    for node, score in get_top_nodes(degree_cent, 5):
        print(f"       {node}: {round(score, 4)}")
    
    print("\n    >> Betweenness Centrality (Top 5):")
    for node, score in get_top_nodes(betweenness_cent, 5):
        print(f"       {node}: {round(score, 4)}")
    
    print("\n    >> Closeness Centrality (Top 5):")
    for node, score in get_top_nodes(closeness_cent, 5):
        print(f"       {node}: {round(score, 4)}")
    
    print("\n    >> PageRank (Top 5):")
    for node, score in get_top_nodes(pagerank_cent, 5):
        print(f"       {node}: {round(score, 4)}")
    
    print("\n[4] Detecting communities (Louvain algorithm)...")
    node_to_community, communities = detect_communities(G)
    print(f"    Found {len(communities)} communities")
    for idx, community in enumerate(communities):
        print(f"    Community {idx + 1}: {sorted(community)}")
    
    print("\n[5] Running Independent Cascade diffusion simulation...")
    top_influencers = [node for node, _ in get_top_nodes(pagerank_cent, 3)]
    print(f"    Seed nodes (top 3 by PageRank): {top_influencers}")
    
    diffusion_result = run_diffusion_simulation(G, top_influencers, probability=0.15)
    
    print(f"    Propagation probability: 0.15")
    print(f"    Iterations until convergence: {diffusion_result['iterations']}")
    print(f"    Total nodes activated: {diffusion_result['total_activated']} / {G.number_of_nodes()}")
    print(f"    Activation rate: {round(diffusion_result['total_activated'] / G.number_of_nodes() * 100, 2)}%")
    
    print("\n    >> Spread History:")
    for i, wave in enumerate(diffusion_result['spread_history']):
        if i == 0:
            print(f"       Initial seeds: {wave}")
        else:
            print(f"       Wave {i}: {wave}")
    
    print_separator()
    print("ANALYSIS COMPLETE - GENERATING VISUALIZATIONS")
    print_separator()
    
    print("\n[6] Displaying centrality comparison chart...")
    visualize_centrality_comparison([degree_cent, betweenness_cent, closeness_cent, pagerank_cent])
    
    print("\n[7] Displaying network with community coloring...")
    visualize_network_with_communities(G, node_to_community)
    
    print("\n[8] Displaying diffusion visualization...")
    visualize_diffusion(G, node_to_community, diffusion_result['final_activated'], top_influencers)
    
    print_separator()
    print("ALL TASKS COMPLETED SUCCESSFULLY")
    print_separator()


if __name__ == "__main__":
    main()
