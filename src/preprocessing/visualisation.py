import os
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
matplotlib.use('TkAgg')  # or 'Agg' if running in a headless environment
import numpy as np
import graphviz
from tqdm import tqdm
import ast
from PIL import Image

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


def visualize_vaccination_graph(gml_file):
    if not os.path.exists(gml_file):
        print(f"Error: File {gml_file} not found.")
        return

    # Load the graph
    G = nx.read_gml(gml_file)
    G = nx.convert_node_labels_to_integers(G)

    # Convert to AGraph for Graphviz visualization
    A = nx.nx_agraph.to_agraph(G)

    # Use the sfdp layout for visualization
    A.layout(prog='sfdp')

    # Output image file
    output_file = gml_file.replace('.gml', '.png')
    A.draw(output_file)

    # Show the result
    img = plt.imread(output_file)
    plt.figure(figsize=(10, 10))
    plt.imshow(img)
    plt.axis('off')
    plt.show()

    print(f"Graph visualization saved as {output_file}")


