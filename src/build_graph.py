import networkx as nx
import pandas as pd


def build_graph(df):
    G = nx.from_pandas_edgelist(df, source='source', target='target')
    return G
