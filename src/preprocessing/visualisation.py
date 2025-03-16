import os
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')  # or 'Agg' if running in a headless environment
import numpy as np

def visualize_garimella_graph(graph_directory):
    """
    Visualizes the graphs stored as .gml files in the given directory, optimized for large graphs.

    Parameters:
    graph_directory (str): Path to the directory containing .gml graph files.
    """
    if not os.path.exists(graph_directory):
        print(f"Directory '{graph_directory}' does not exist.")
        return

    graph_files = [f for f in os.listdir(graph_directory) if f.endswith('.gml')]

    if not graph_files:
        print("No .gml files found in the directory.")
        return

    for graph_file in graph_files:
        graph_path = os.path.join(graph_directory, graph_file)
        G = nx.read_gml(graph_path)

        plt.figure(figsize=(10, 8))
        plt.title(f"Visualization of {graph_file}")

        if len(G.nodes) > 1000:
            print(
                f"Graph {graph_file} is too large ({len(G.nodes)} nodes). Using a spring layout with reduced node visibility.")
            pos = nx.spring_layout(G, k=5 / np.sqrt(len(G.nodes)), seed=42)
            nx.draw(G, pos, node_size=10, edge_color='gray', alpha=0.5, with_labels=False)
        else:
            pos = nx.spring_layout(G)
            nx.draw(G, pos, with_labels=True, node_size=50, font_size=8, edge_color='gray')

        plt.show()


def visualize_large_garimella_graph(graph_directory):
    """
    Efficiently visualizes large graphs by sampling nodes and using a faster layout.
    """
    if not os.path.exists(graph_directory):
        print(f"Directory '{graph_directory}' does not exist.")
        return

    graph_files = [f for f in os.listdir(graph_directory) if f.endswith('.gml')]

    if not graph_files:
        print("No .gml files found in the directory.")
        return

    for graph_file in graph_files:
        graph_path = os.path.join(graph_directory, graph_file)
        G = nx.read_gml(graph_path)

        plt.figure(figsize=(12, 8))
        plt.title(f"Visualization of {graph_file} (Sampled)")

        if len(G.nodes) > 10000:
            print(f"Graph {graph_file} is too large ({len(G.nodes)} nodes). Sampling 10,000 nodes.")
            sampled_nodes = np.random.choice(G.nodes, 10000, replace=False)
            G = G.subgraph(sampled_nodes)

        pos = nx.spring_layout(G, k=1 / np.sqrt(len(G.nodes)), seed=42)

        nx.draw_networkx_edges(G, pos, alpha=0.1)
        nx.draw_networkx_nodes(G, pos, node_size=5, alpha=0.7)

        plt.savefig(f"{graph_file}.png")
        plt.close()

