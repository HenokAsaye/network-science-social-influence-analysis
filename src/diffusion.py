# This module implements diffusion, epidemic spreading, network robustness,
# and immunization strategies on graphs using NetworkX.
# It includes Independent Cascade, Linear Threshold, SIR models,
# and network attack/immunization simulations.
import random
import numpy as np
import networkx as nx


def independent_cascade(G, seeds, probability=0.1, max_iterations=100):
    """
    Simulates information diffusion using the Independent Cascade (IC) model.

    Each activated node has one chance to activate its neighbors
    with a fixed probability.

    Args:
        G (networkx.Graph): The input network.
        seeds (list or set): Initial seed nodes.
        probability (float): Activation probability for each edge.
        max_iterations (int): Maximum number of diffusion steps.

    Returns:
        activated (set): All activated nodes.
        history (list): Nodes activated at each iteration.
    """
    random.seed(42)
    activated = set(seeds)
    newly_activated = set(seeds)
    history = [list(seeds)]
    
    for _ in range(max_iterations):
        if not newly_activated:
            break
        
        candidates = set()
        for node in newly_activated:
            neighbors = set(G.neighbors(node)) - activated
            for neighbor in neighbors:
                if random.random() < probability:
                    candidates.add(neighbor)
        
        newly_activated = candidates
        activated.update(newly_activated)
        
        if newly_activated:
            history.append(list(newly_activated))
    
    return activated, history


def run_diffusion_simulation(G, seed_nodes, probability=0.1):
    """
    Runs an Independent Cascade diffusion simulation and summarizes results.

    Args:
        G (networkx.Graph): The network graph.
        seed_nodes (list): Initial seed nodes.
        probability (float): Activation probability.

    Returns:
        dict: Summary statistics of the diffusion process.
    """
    final_activated, spread_history = independent_cascade(G, seed_nodes, probability)
    return {
        'seed_nodes': seed_nodes,
        'final_activated': final_activated,
        'total_activated': len(final_activated),
        'spread_history': spread_history,
        'iterations': len(spread_history)
    }


def sir_model(G, initial_infected, beta=0.3, gamma=0.1, max_steps=100):
    """
    Simulates epidemic spreading using the SIR (Susceptible–Infected–Recovered) model.

    Args:
        G (networkx.Graph): The contact network.
        initial_infected (list): Initially infected nodes.
        beta (float): Infection probability.
        gamma (float): Recovery probability.
        max_steps (int): Maximum simulation steps.

    Returns:
        dict: Time-series data of susceptible, infected, and recovered populations.
    """
    susceptible = set(G.nodes()) - set(initial_infected)
    infected = set(initial_infected)
    recovered = set()
    
    history = {
        'susceptible': [len(susceptible)],
        'infected': [len(infected)],
        'recovered': [len(recovered)],
        'infected_nodes': [list(infected)],
        'recovered_nodes': [list(recovered)]
    }
    
    random.seed(42)
    
    for step in range(max_steps):
        if not infected:
            break
        
        new_infected = set()
        new_recovered = set()
        
        for node in list(infected):
            if random.random() < gamma:
                new_recovered.add(node)
        
        for node in infected - new_recovered:
            for neighbor in G.neighbors(node):
                if neighbor in susceptible and random.random() < beta:
                    new_infected.add(neighbor)
        
        susceptible -= new_infected
        infected = (infected - new_recovered) | new_infected
        recovered |= new_recovered
        
        history['susceptible'].append(len(susceptible))
        history['infected'].append(len(infected))
        history['recovered'].append(len(recovered))
        history['infected_nodes'].append(list(infected))
        history['recovered_nodes'].append(list(recovered))
    
    return history


def linear_threshold(G, seeds, thresholds=None, max_iterations=100):
    """
    Simulates diffusion using the Linear Threshold (LT) model.

    A node becomes active when the fraction of its active neighbors
    exceeds its threshold.

    Args:
        G (networkx.Graph): The network.
        seeds (list): Initial active nodes.
        thresholds (dict): Node-specific activation thresholds.
        max_iterations (int): Maximum diffusion steps.

    Returns:
        activated (set): All activated nodes.
        history (list): Activation history per iteration.
    """
    random.seed(42)
    
    if thresholds is None:
        thresholds = {node: random.uniform(0.1, 0.5) for node in G.nodes()}
    
    activated = set(seeds)
    newly_activated = set(seeds)
    history = [list(seeds)]
    
    for _ in range(max_iterations):
        if not newly_activated:
            break
        
        candidates = set()
        for node in G.nodes():
            if node in activated:
                continue
            neighbors = set(G.neighbors(node))
            activated_neighbors = neighbors & activated
            if len(neighbors) > 0:
                influence = len(activated_neighbors) / len(neighbors)
                if influence >= thresholds[node]:
                    candidates.add(node)
        
        newly_activated = candidates
        activated.update(newly_activated)
        
        if newly_activated:
            history.append(list(newly_activated))
    
    return activated, history


def simulate_network_attack(G, attack_type='random', percentage=0.1):
    """
    Simulates network robustness under node removal (attack).

    Args:
        G (networkx.Graph): Original network.
        attack_type (str): 'random', 'targeted_degree', or 'targeted_betweenness'.
        percentage (float): Fraction of nodes to remove.

    Returns:
        dict: Network fragmentation statistics after attack.
    """
    H = G.copy()
    num_to_remove = max(1, int(G.number_of_nodes() * percentage))
    
    if attack_type == 'random':
        nodes_to_remove = random.sample(list(H.nodes()), num_to_remove)
    elif attack_type == 'targeted_degree':
        degree_sorted = sorted(H.degree(), key=lambda x: x[1], reverse=True)
        nodes_to_remove = [n for n, d in degree_sorted[:num_to_remove]]
    elif attack_type == 'targeted_betweenness':
        betweenness = nx.betweenness_centrality(H)
        sorted_nodes = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)
        nodes_to_remove = [n for n, b in sorted_nodes[:num_to_remove]]
    else:
        nodes_to_remove = []
    
    H.remove_nodes_from(nodes_to_remove)
    
    # Use undirected version for connected components (only defined for undirected graphs)
    G_undirected = G.to_undirected() if G.is_directed() else G
    H_undirected = H.to_undirected() if H.is_directed() else H
    
    original_components = nx.number_connected_components(G_undirected)
    new_components = nx.number_connected_components(H_undirected)
    
    if len(H.nodes()) > 0:
        largest_cc = max(nx.connected_components(H_undirected), key=len)
        largest_cc_size = len(largest_cc)
    else:
        largest_cc_size = 0
    
    return {
        'removed_nodes': nodes_to_remove,
        'original_nodes': G.number_of_nodes(),
        'remaining_nodes': H.number_of_nodes(),
        'original_components': original_components,
        'new_components': new_components,
        'largest_cc_size': largest_cc_size,
        'fragmentation': 1 - (largest_cc_size / G.number_of_nodes()) if G.number_of_nodes() > 0 else 0
    }


def compare_immunization_strategies(G, infected_seeds, beta=0.3, gamma=0.1, immunization_percentage=0.1):
    """
    Compares epidemic outcomes under different immunization strategies.

    Strategies:
        - No immunization
        - Random immunization
        - Targeted immunization (high-degree nodes)

    Args:
        G (networkx.Graph): The network.
        infected_seeds (list): Initially infected nodes.
        beta (float): Infection rate.
        gamma (float): Recovery rate.
        immunization_percentage (float): Fraction of nodes to immunize.

    Returns:
        dict: Maximum number of infected nodes for each strategy.
    """
    results = {}
    
    history_no_immunization = sir_model(G, infected_seeds, beta, gamma)
    results['no_immunization'] = max(history_no_immunization['infected'])
    
    num_to_immunize = max(1, int(G.number_of_nodes() * immunization_percentage))
    
    random.seed(42)
    random_immunized = set(random.sample(list(set(G.nodes()) - set(infected_seeds)), 
                                          min(num_to_immunize, len(G.nodes()) - len(infected_seeds))))
    H_random = G.copy()
    H_random.remove_nodes_from(random_immunized)
    history_random = sir_model(H_random, [s for s in infected_seeds if s in H_random.nodes()], beta, gamma)
    results['random_immunization'] = max(history_random['infected'])
    
    degree_sorted = sorted(G.degree(), key=lambda x: x[1], reverse=True)
    hub_immunized = set()
    for node, degree in degree_sorted:
        if node not in infected_seeds:
            hub_immunized.add(node)
            if len(hub_immunized) >= num_to_immunize:
                break
    H_targeted = G.copy()
    H_targeted.remove_nodes_from(hub_immunized)
    history_targeted = sir_model(H_targeted, [s for s in infected_seeds if s in H_targeted.nodes()], beta, gamma)
    results['targeted_immunization'] = max(history_targeted['infected'])
    
    return results
