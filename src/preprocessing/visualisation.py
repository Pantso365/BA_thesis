import os
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
from scipy.sparse import coo_matrix
from tqdm import tqdm
matplotlib.use('TkAgg')  # or 'Agg' if running in a headless environment
import numpy as np
import graphviz
from fa2_modified import ForceAtlas2

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
            nx.draw(G, pos, node_size=0.1, edge_color='gray', alpha=0.5, with_labels=False)
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

'''
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
'''

'''
def visualize_vaccination_graph(gml_file):
    if not os.path.exists(gml_file):
        print(f"Error: File {gml_file} not found.")
        return

    # Load the graph
    G = nx.read_gml(gml_file)
    G = nx.convert_node_labels_to_integers(G)

    # Extract edge weights
    edge_weights = nx.get_edge_attributes(G, 'weight')

    # Normalize edge weights for edge length scaling
    if edge_weights:
        max_weight = max(edge_weights.values())
        min_weight = min(edge_weights.values())
        edge_lengths = {
            (u, v): 1 + (max_weight - w) / (max_weight - min_weight + 1)
            for (u, v), w in edge_weights.items()
        }
    else:
        edge_lengths = {e: 1 for e in G.edges()}  # Default if no weights exist

    # Convert to Graphviz AGraph
    A = nx.nx_agraph.to_agraph(G)

    # Graph-level attributes (improves spacing)
    A.graph_attr["K"] = "30000"      # Controls repulsion (higher = more spacing)
    A.graph_attr["sep"] = "+600"   # Prevents node overlap

    # Apply node styling (small dots, no labels)
    for node in A.nodes():
        node.attr["width"] = "0.005"     # Makes nodes small
        node.attr["height"] = "0.005"    # Makes nodes small
        node.attr["fontsize"] = "8"     # Prevents invisible text from adding size
        node.attr["label"] = ""         # Removes node labels (only dots remain)

    # Apply edge length attributes for better community separation
    for edge in A.edges():
        u, v = edge
        edge.attr["len"] = str(edge_lengths.get((int(u), int(v)), 1) * 3)  # Scale distances

    # Use 'sfdp' for force-directed layout
    A.layout(prog='sfdp')

    # Save and display the graph
    output_file = gml_file.replace('.gml', '.png')
    A.draw(output_file)

    img = plt.imread(output_file)
    plt.figure(figsize=(20, 20))
    plt.imshow(img)
    plt.axis('off')
    plt.show()

    print(f"Graph visualization saved as {output_file}")
'''


def visualize_vaccination_graph(gml_file):
    if not os.path.exists(gml_file):
        print(f"Error: File {gml_file} not found.")
        return

    # Load the graph
    G = nx.read_gml(gml_file)
    G = nx.convert_node_labels_to_integers(G)

    # Apply ForceAtlas2 layout
    forceatlas2 = ForceAtlas2(
        outboundAttractionDistribution=True,
        linLogMode=False,
        adjustSizes=False,
        edgeWeightInfluence=1.0,
        jitterTolerance=1.0,
        barnesHutOptimize=True,
        barnesHutTheta=1.2,
        scalingRatio=2.0,
        strongGravityMode=False,
        gravity=1.0
    )
    positions = forceatlas2.forceatlas2_networkx_layout(G, iterations=1000)

    # Draw the graph
    plt.figure(figsize=(10, 10))
    nx.draw(G, pos=positions, with_labels=True, node_size=50, edge_color="gray")

    # Save the output image
    output_file = gml_file.replace('.gml', '_forceatlas2.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.show()

    print(f"Graph visualization saved as {output_file}")


'''
def force_atlas2_layout(graph, atlas_properties):
    print("Start creating Force Atlas Layout")
    iterations = atlas_properties.get("iterations", 1000)
    linlog = atlas_properties.get("linlog", False)
    pos = atlas_properties.get("pos", None)
    nohubs = atlas_properties.get("nohubs", False)
    k = atlas_properties.get("k", None)
    dim = atlas_properties.get("dim", 2)

    A = nx.to_scipy_sparse_matrix(graph, dtype='f')
    nnodes, _ = A.shape

    try:
        A = A.tolil()
    except Exception as e:
        A = (coo_matrix(A)).tolil()
    if pos is None:
        pos = np.asarray(np.random.random((nnodes, dim)), dtype=A.dtype)
    else:
        pos = pos.astype(A.dtype)
    if k is None:
        k = np.sqrt(1.0 / nnodes)
    t = 0.1

    dt = t / float(iterations + 1)
    displacement = np.zeros((dim, nnodes))
    for _ in tqdm(range(iterations)):
        displacement *= 0
        for i in range(A.shape[0]):
            delta = (pos[i] - pos).T
            distance = np.sqrt((delta ** 2).sum(axis=0))
            distance = np.where(distance < 0.01, 0.01, distance)
            Ai = np.asarray(A.getrowview(i).toarray())
            dist = k * k / distance ** 2
            if nohubs:
                dist = dist / float(Ai.sum(axis=1) + 1)
            if linlog:
                dist = np.log(dist + 1)
            displacement[:, i] += \
                (delta * (dist - Ai * distance / k)).sum(axis=1)
        length = np.sqrt((displacement ** 2).sum(axis=0))
        length = np.where(length < 0.01, 0.01, length)
        pos += (displacement * t / length).T
        t -= dt

    print("Force Atlas done")
    return dict(zip(graph, pos))
'''

def visualize_covid_graph(graph_directory):
    """
    Visualizes the COVID-19 related graphs stored as .gml files in the given directory, optimized for large graphs.

    Parameters:
    graph_directory (str): Path to the directory containing .gml graph files related to COVID-19.
    """
    if not os.path.exists(graph_directory):
        print(f"Directory '{graph_directory}' does not exist.")
        return

    # Get all .gml files in the specified directory
    graph_files = [f for f in os.listdir(graph_directory) if f.endswith('.gml')]

    if not graph_files:
        print("No .gml files found in the directory.")
        return

    for graph_file in graph_files:
        graph_path = os.path.join(graph_directory, graph_file)
        G = nx.read_gml(graph_path)

        plt.figure(figsize=(10, 8))
        plt.title(f"Visualization of {graph_file}")

        # Check the graph size and adjust layout accordingly
        if len(G.nodes) > 1000:
            print(f"Graph {graph_file} is too large ({len(G.nodes)} nodes). Using spring layout with reduced node visibility.")
            pos = nx.spring_layout(G, k=5 / np.sqrt(len(G.nodes)), seed=42, weight='weight')
            nx.draw(G, pos, node_size=0.5, edge_color='gray', alpha=0.5, with_labels=False)
        else:
            pos = nx.spring_layout(G)
            nx.draw(G, pos, with_labels=True, node_size=50, font_size=8, edge_color='gray')

        plt.show()