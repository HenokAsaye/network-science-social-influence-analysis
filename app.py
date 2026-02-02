import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
from pyvis.network import Network
import tempfile
import os

from src.load_data import load_network_data
from src.build_graph import build_graph
from src.analysis import (
    compute_degree_centrality,
    compute_betweenness_centrality,
    compute_closeness_centrality,
    compute_pagerank,
    compute_eigenvector_centrality,
    compute_katz_centrality,
    compute_harmonic_centrality,
    detect_communities,
    detect_communities_girvan_newman,
    get_top_nodes,
    compute_network_metrics,
    compute_link_prediction_scores,
    identify_bridge_nodes,
    compute_node_vulnerability,
    compute_k_core_decomposition
)
from src.diffusion import (
    run_diffusion_simulation,
    sir_model,
    linear_threshold,
    simulate_network_attack,
    compare_immunization_strategies
)

st.set_page_config(
    page_title="Social Network Analysis Dashboard",
    page_icon="🕸️",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.session_state['project_info'] = {
    "version": "1.0.0",
    "description": "Dashboard for social network analysis and diffusion modeling",
    "last_updated": "2026-02-01",             
    "license": "MIT",                          
    "data_source": "NYC Taxi Network + Sample Data",  
}

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data(filepath):
    return load_network_data(filepath)


@st.cache_data
def create_graph(_df, weight_column=None):
    return build_graph(_df, weight_column)


def create_interactive_network(G, node_to_community, title="Network Visualization"):
    net = Network(height="600px", width="100%", bgcolor="#ffffff", font_color="black")
    net.barnes_hut(gravity=-3000, central_gravity=0.3, spring_length=200)
    
    colors = px.colors.qualitative.Set3
    
    for node in G.nodes():
        community = node_to_community.get(node, 0)
        color = colors[community % len(colors)]
        net.add_node(node, label=str(node), color=color, size=20, title=f"Node: {node}\nCommunity: {community}")
    
    for edge in G.edges():
        net.add_edge(edge[0], edge[1], color="#888888")
    
    with tempfile.NamedTemporaryFile(delete=False, suffix='.html', mode='w', encoding='utf-8') as f:
        net.save_graph(f.name)
        return f.name


def create_plotly_network(G, node_to_community, node_sizes=None, title="Network Graph"):
    pos = nx.spring_layout(G, seed=42, k=2)
    
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines'
    )
    
    node_x = []
    node_y = []
    node_text = []
    node_color = []
    sizes = []
    
    colors = px.colors.qualitative.Set3
    
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(f"{node}<br>Community: {node_to_community.get(node, 0)}")
        node_color.append(node_to_community.get(node, 0))
        if node_sizes:
            sizes.append(node_sizes.get(node, 20) * 100 + 10)
        else:
            sizes.append(25)
    
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        text=[str(n) for n in G.nodes()],
        textposition="top center",
        hovertext=node_text,
        marker=dict(
            showscale=True,
            colorscale='Rainbow',
            color=node_color,
            size=sizes,
            colorbar=dict(
                thickness=15,
                title=dict(text='Community', side='right'),
                xanchor='left'
            ),
            line_width=2
        )
    )
    
    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title=title,
                        showlegend=False,
                        hovermode='closest',
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        height=600
                    ))
    return fig


def create_centrality_chart(centrality_dict, title, top_n=10):
    sorted_items = sorted(centrality_dict.items(), key=lambda x: x[1], reverse=True)[:top_n]
    nodes = [item[0] for item in sorted_items]
    values = [item[1] for item in sorted_items]
    
    fig = go.Figure(go.Bar(
        x=values,
        y=nodes,
        orientation='h',
        marker_color='steelblue'
    ))
    fig.update_layout(
        title=title,
        xaxis_title="Centrality Score",
        yaxis_title="Node",
        yaxis=dict(autorange="reversed"),
        height=400
    )
    return fig


def main():
    st.markdown('<h1 class="main-header">🕸️ Social Network Analysis Dashboard</h1>', unsafe_allow_html=True)
    st.markdown("### Network Science-Driven Analysis of Information Diffusion and Influence")
    
    with st.sidebar:
        st.header("📁 Data Configuration")
        
        uploaded_file = st.file_uploader("Upload CSV (source, target columns)", type=['csv'])
        
        weight_column = None
        
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ File uploaded: {uploaded_file.name}")
            st.success(f"📊 Loaded {len(df)} edges")
            st.write("**Preview (first 5 rows):**")
            st.dataframe(df.head(), use_container_width=True)
            st.write(f"**Columns detected:** {', '.join(df.columns.tolist())}")
            if 'source' not in df.columns or 'target' not in df.columns:
                st.error("⚠️ CSV must have 'source' and 'target' columns!")
                st.stop()
            
            # Weight column selection
            other_columns = [col for col in df.columns if col not in ['source', 'target']]
            if other_columns:
                weight_options = ["None (unweighted)"] + other_columns
                weight_selection = st.selectbox(
                    "🔢 Weight Column (optional)",
                    weight_options,
                    help="Select a column to use as edge weights for weighted graph analysis"
                )
                if weight_selection != "None (unweighted)":
                    weight_column = weight_selection
                    st.info(f"📏 Using '{weight_column}' as edge weights")
        else:
            df = load_data("data/nyc_taxi_network_dashboard.csv")
            st.info("📂 Using sample dataset: data/nyc_taxi_network_dashboard.csv")
            st.write(f"**Edges:** {len(df)}")
            
            # Check if sample data has weight column
            other_columns = [col for col in df.columns if col not in ['source', 'target']]
            if other_columns:
                weight_options = ["None (unweighted)"] + other_columns
                weight_selection = st.selectbox(
                    "🔢 Weight Column (optional)",
                    weight_options,
                    help="Select a column to use as edge weights for weighted graph analysis"
                )
                if weight_selection != "None (unweighted)":
                    weight_column = weight_selection
        
        st.markdown("---")
        st.header("⚙️ Settings")
        
        community_algorithm = st.selectbox(
            "Community Detection Algorithm",
            ["Louvain", "Girvan-Newman"]
        )
        
        if community_algorithm == "Girvan-Newman":
            num_communities = st.slider("Number of Communities", 2, 10, 3)
    
    G = create_graph(df, weight_column)
    
    if community_algorithm == "Louvain":
        node_to_community, communities = detect_communities(G)
    else:
        node_to_community, communities = detect_communities_girvan_newman(G, num_communities)
    
    tabs = st.tabs([
        "📊 Overview",
        "🎯 Centrality Analysis",
        "🏘️ Communities",
        "🔄 Diffusion Models",
        "🦠 SIR Epidemic",
        "🔗 Link Prediction",
        "💥 Network Resilience"
    ])
    
    with tabs[0]:
        st.header("Network Overview")
        
        # Show if graph is weighted
        is_weighted = weight_column is not None
        if is_weighted:
            st.success(f"📊 **Weighted Graph** - Using '{weight_column}' as edge weights")
        else:
            st.info("📊 **Unweighted Graph** - No weight column selected")
        
        metrics = compute_network_metrics(G)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Nodes", metrics['nodes'])
            st.metric("Edges", metrics['edges'])
        with col2:
            st.metric("Density", f"{metrics['density']:.4f}")
            st.metric("Avg Clustering", f"{metrics['average_clustering']:.4f}")
        with col3:
            st.metric("Transitivity", f"{metrics['transitivity']:.4f}")
            st.metric("Avg Degree", f"{metrics['average_degree']:.2f}")
        with col4:
            st.metric("Components", metrics['num_connected_components'])
            st.metric("Assortativity", f"{metrics['assortativity']:.4f}")
        
        st.subheader("Interactive Network Visualization")
        
        viz_type = st.radio("Visualization Type", ["Plotly (Static)", "PyVis (Interactive)"], horizontal=True)
        
        if viz_type == "Plotly (Static)":
            fig = create_plotly_network(G, node_to_community, title="Network with Community Coloring")
            st.plotly_chart(fig, use_container_width=True)
        else:
            html_file = create_interactive_network(G, node_to_community)
            with open(html_file, 'r', encoding='utf-8') as f:
                html_content = f.read()
            st.components.v1.html(html_content, height=620, scrolling=True)
            os.unlink(html_file)
        
        st.subheader("Degree Distribution")
        degrees = [d for n, d in G.degree()]
        fig_degree = px.histogram(x=degrees, nbins=20, labels={'x': 'Degree', 'y': 'Count'},
                                   title="Degree Distribution")
        st.plotly_chart(fig_degree, use_container_width=True)
    
    with tabs[1]:
        st.header("Centrality Analysis")
        
        centrality_metrics = {
            'Degree': compute_degree_centrality(G),
            'Betweenness': compute_betweenness_centrality(G),
            'Closeness': compute_closeness_centrality(G),
            'PageRank': compute_pagerank(G),
            'Eigenvector': compute_eigenvector_centrality(G),
            'Katz': compute_katz_centrality(G),
            'Harmonic': compute_harmonic_centrality(G)
        }
        
        selected_metrics = st.multiselect(
            "Select Centrality Metrics to Display",
            list(centrality_metrics.keys()),
            default=['Degree', 'Betweenness', 'PageRank']
        )
        
        top_n = st.slider("Number of Top Nodes", 5, 15, 10)
        
        cols = st.columns(min(len(selected_metrics), 3))
        for i, metric in enumerate(selected_metrics):
            with cols[i % 3]:
                fig = create_centrality_chart(centrality_metrics[metric], f"{metric} Centrality", top_n)
                st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Centrality Comparison Table")
        comparison_data = []
        for node in G.nodes():
            row = {'Node': node}
            for metric in selected_metrics:
                row[metric] = round(centrality_metrics[metric].get(node, 0), 4)
            comparison_data.append(row)
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values(by=selected_metrics[0] if selected_metrics else 'Node', ascending=False)
        st.dataframe(comparison_df, use_container_width=True)
        
        st.subheader("Centrality Correlation Heatmap")
        corr_data = pd.DataFrame({m: list(centrality_metrics[m].values()) for m in selected_metrics})
        corr_matrix = corr_data.corr()
        fig_corr = px.imshow(corr_matrix, text_auto=True, title="Centrality Metrics Correlation")
        st.plotly_chart(fig_corr, use_container_width=True)
    
    with tabs[2]:
        st.header("Community Detection")
        
        num_communities = len(communities)
        st.metric("Number of Communities Detected", num_communities)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            pagerank = compute_pagerank(G)
            fig = create_plotly_network(G, node_to_community, pagerank, "Network with Communities (Node size = PageRank)")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Community Details")
            for i, comm in enumerate(communities):
                with st.expander(f"Community {i + 1} ({len(comm)} nodes)"):
                    st.write(", ".join(sorted([str(n) for n in comm])))
        
        st.subheader("Community Size Distribution")
        comm_sizes = [len(c) for c in communities]
        fig_comm = px.pie(values=comm_sizes, names=[f"Community {i+1}" for i in range(len(communities))],
                          title="Community Size Distribution")
        st.plotly_chart(fig_comm, use_container_width=True)
        
        st.subheader("Community Statistics")
        comm_stats = []
        for i, comm in enumerate(communities):
            subgraph = G.subgraph(comm)
            comm_stats.append({
                'Community': i + 1,
                'Nodes': len(comm),
                'Internal Edges': subgraph.number_of_edges(),
                'Density': nx.density(subgraph) if len(comm) > 1 else 0,
                'Avg Clustering': nx.average_clustering(subgraph)
            })
        st.dataframe(pd.DataFrame(comm_stats), use_container_width=True)
    
    with tabs[3]:
        st.header("Information Diffusion Simulation")
        
        model_type = st.selectbox("Diffusion Model", ["Independent Cascade", "Linear Threshold"])
        
        col1, col2 = st.columns(2)
        
        with col1:
            seed_selection = st.selectbox(
                "Seed Node Selection Strategy",
                ["Top PageRank", "Top Degree", "Top Betweenness", "Custom"]
            )
            
            num_seeds = st.slider("Number of Seed Nodes", 1, min(10, G.number_of_nodes()), 3)
        
        with col2:
            if model_type == "Independent Cascade":
                probability = st.slider("Propagation Probability", 0.01, 1.0, 0.15)
            else:
                threshold_type = st.selectbox("Threshold Type", ["Random", "Uniform"])
        
        if seed_selection == "Top PageRank":
            pagerank = compute_pagerank(G)
            seed_nodes = [n for n, _ in get_top_nodes(pagerank, num_seeds)]
        elif seed_selection == "Top Degree":
            degree_cent = compute_degree_centrality(G)
            seed_nodes = [n for n, _ in get_top_nodes(degree_cent, num_seeds)]
        elif seed_selection == "Top Betweenness":
            betweenness = compute_betweenness_centrality(G)
            seed_nodes = [n for n, _ in get_top_nodes(betweenness, num_seeds)]
        else:
            seed_nodes = st.multiselect("Select Seed Nodes", list(G.nodes()), default=list(G.nodes())[:num_seeds])
        
        st.info(f"Selected Seed Nodes: {', '.join([str(n) for n in seed_nodes])}")
        
        if st.button("Run Diffusion Simulation", type="primary"):
            if model_type == "Independent Cascade":
                result = run_diffusion_simulation(G, seed_nodes, probability)
            else:
                thresholds = None
                if threshold_type == "Uniform":
                    uniform_threshold = st.slider("Uniform Threshold", 0.1, 0.9, 0.3)
                    thresholds = {node: uniform_threshold for node in G.nodes()}
                activated, history = linear_threshold(G, seed_nodes, thresholds)
                result = {
                    'seed_nodes': seed_nodes,
                    'final_activated': activated,
                    'total_activated': len(activated),
                    'spread_history': history,
                    'iterations': len(history)
                }
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Activated", f"{result['total_activated']} / {G.number_of_nodes()}")
            with col2:
                st.metric("Activation Rate", f"{result['total_activated'] / G.number_of_nodes() * 100:.1f}%")
            with col3:
                st.metric("Iterations", result['iterations'])
            
            st.subheader("Diffusion Spread Over Time")
            spread_data = []
            cumulative = 0
            for i, wave in enumerate(result['spread_history']):
                cumulative += len(wave)
                spread_data.append({'Wave': i, 'New Activations': len(wave), 'Cumulative': cumulative})
            
            spread_df = pd.DataFrame(spread_data)
            fig_spread = px.line(spread_df, x='Wave', y='Cumulative', markers=True,
                                 title="Cumulative Activations Over Time")
            st.plotly_chart(fig_spread, use_container_width=True)
            
            st.subheader("Diffusion Visualization")
            activated_colors = {}
            for node in G.nodes():
                if node in seed_nodes:
                    activated_colors[node] = -1
                elif node in result['final_activated']:
                    activated_colors[node] = 1
                else:
                    activated_colors[node] = 0
            
            fig_diff = create_plotly_network(G, activated_colors, title="Diffusion Result (Red=Seed, Yellow=Activated, Gray=Inactive)")
            st.plotly_chart(fig_diff, use_container_width=True)
    
    with tabs[4]:
        st.header("SIR Epidemic Model Simulation")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            beta = st.slider("Infection Rate (β)", 0.01, 1.0, 0.3)
        with col2:
            gamma = st.slider("Recovery Rate (γ)", 0.01, 1.0, 0.1)
        with col3:
            initial_infected = st.multiselect("Initial Infected Nodes", list(G.nodes()), 
                                               default=[list(G.nodes())[0]])
        
        if st.button("Run SIR Simulation", type="primary"):
            history = sir_model(G, initial_infected, beta, gamma)
            
            sir_df = pd.DataFrame({
                'Step': range(len(history['susceptible'])),
                'Susceptible': history['susceptible'],
                'Infected': history['infected'],
                'Recovered': history['recovered']
            })
            
            fig_sir = px.line(sir_df, x='Step', y=['Susceptible', 'Infected', 'Recovered'],
                              title="SIR Model Dynamics",
                              labels={'value': 'Population', 'variable': 'State'})
            st.plotly_chart(fig_sir, use_container_width=True)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Peak Infected", max(history['infected']))
            with col2:
                st.metric("Total Recovered", history['recovered'][-1])
            with col3:
                st.metric("Final Susceptible", history['susceptible'][-1])
        
        st.subheader("Immunization Strategy Comparison")
        immunization_pct = st.slider("Immunization Percentage", 0.05, 0.5, 0.1)
        
        if st.button("Compare Immunization Strategies"):
            if len(initial_infected) > 0:
                comparison = compare_immunization_strategies(G, initial_infected, beta, gamma, immunization_pct)
                
                comp_df = pd.DataFrame({
                    'Strategy': list(comparison.keys()),
                    'Peak Infected': list(comparison.values())
                })
                
                fig_comp = px.bar(comp_df, x='Strategy', y='Peak Infected', color='Strategy',
                                  title="Peak Infection by Immunization Strategy")
                st.plotly_chart(fig_comp, use_container_width=True)
            else:
                st.warning("Please select at least one initial infected node.")
    
    with tabs[5]:
        st.header("Link Prediction")
        
        predictions = compute_link_prediction_scores(G)
        
        st.subheader("Top Predicted Links")
        top_n_links = st.slider("Number of predictions to show", 5, 30, 15)
        
        pred_df = pd.DataFrame(predictions[:top_n_links])
        pred_df = pred_df.round(4)
        st.dataframe(pred_df, use_container_width=True)
        
        st.subheader("Prediction Score Distributions")
        all_pred_df = pd.DataFrame(predictions)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            fig_jac = px.histogram(all_pred_df, x='jaccard', nbins=20, title="Jaccard Coefficient")
            st.plotly_chart(fig_jac, use_container_width=True)
        with col2:
            fig_aa = px.histogram(all_pred_df, x='adamic_adar', nbins=20, title="Adamic-Adar Index")
            st.plotly_chart(fig_aa, use_container_width=True)
        with col3:
            fig_pa = px.histogram(all_pred_df, x='preferential_attachment', nbins=20, title="Preferential Attachment")
            st.plotly_chart(fig_pa, use_container_width=True)
    
    with tabs[6]:
        st.header("Network Resilience Analysis")
        
        st.subheader("Bridge Nodes & Vulnerabilities")
        
        bridge_nodes, bridges = identify_bridge_nodes(G)
        vulnerability = compute_node_vulnerability(G)
        k_cores = compute_k_core_decomposition(G)
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Bridge Nodes:**", ", ".join([str(n) for n in bridge_nodes]) if bridge_nodes else "None detected")
            st.write("**Number of Bridges:**", len(bridges))
        
        with col2:
            vuln_df = pd.DataFrame([
                {'Node': k, 'Vulnerability': v, 'K-Core': k_cores[k]}
                for k, v in sorted(vulnerability.items(), key=lambda x: x[1], reverse=True)
            ])
            st.dataframe(vuln_df, use_container_width=True)
        
        st.subheader("Network Attack Simulation")
        
        col1, col2 = st.columns(2)
        with col1:
            attack_type = st.selectbox("Attack Type", ["random", "targeted_degree", "targeted_betweenness"])
        with col2:
            attack_pct = st.slider("Percentage of Nodes to Remove", 0.05, 0.5, 0.1)
        
        if st.button("Simulate Attack", type="primary"):
            attack_result = simulate_network_attack(G, attack_type, attack_pct)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Nodes Removed", len(attack_result['removed_nodes']))
            with col2:
                st.metric("Remaining Nodes", attack_result['remaining_nodes'])
            with col3:
                st.metric("New Components", attack_result['new_components'])
            with col4:
                st.metric("Fragmentation", f"{attack_result['fragmentation']:.2%}")
            
            st.write("**Removed Nodes:**", ", ".join([str(n) for n in attack_result['removed_nodes']]))
        
        st.subheader("K-Core Decomposition")
        k_core_df = pd.DataFrame([
            {'Node': k, 'K-Core Number': v}
            for k, v in sorted(k_cores.items(), key=lambda x: x[1], reverse=True)
        ])
        
        col1, col2 = st.columns(2)
        with col1:
            st.dataframe(k_core_df, use_container_width=True)
        with col2:
            fig_kcore = px.histogram(k_core_df, x='K-Core Number', title="K-Core Distribution")
            st.plotly_chart(fig_kcore, use_container_width=True)


if __name__ == "__main__":
    main()
