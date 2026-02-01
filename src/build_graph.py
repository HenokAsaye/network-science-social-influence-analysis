import networkx as nx
import pandas as pd


def build_graph(df, weight_column=None):
    """
    Build a NetworkX graph from a pandas DataFrame.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with 'source' and 'target' columns, and optionally a weight column
    weight_column : str, optional
        Name of the column to use as edge weights. If None, creates unweighted graph.
    
    Returns:
    --------
    networkx.Graph
        The constructed graph (weighted if weight_column is provided)
    """
    if weight_column and weight_column in df.columns:
        G = nx.from_pandas_edgelist(
            df, 
            source='source', 
            target='target', 
            edge_attr=weight_column
        )
        
        for u, v, data in G.edges(data=True):
            if weight_column in data:
                data['weight'] = data.pop(weight_column)
    else:
        G = nx.from_pandas_edgelist(df, source='source', target='target')
    
    
    self_loops = list(nx.selfloop_edges(G))
    if self_loops:
        G.remove_edges_from(self_loops)
    
    return G
