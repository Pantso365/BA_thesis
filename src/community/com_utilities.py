import networkx as nx
from networkx.algorithms.community import partition_quality, louvain_communities
from tqdm import tqdm
import time
from math import ceil
from random import random, randint
from collections import Counter
import os



from src.community.log_writer import log_write_com_result, log_write_graph_info, print_difference


def create_multi_graph(G):
    G_multi = nx.MultiGraph()
    for edge in G.edges(data = True):
        weight = ceil(edge[2]['weightWithSentiment'])
        for _ in range(weight):
            G_multi.add_edge(edge[0], edge[1])

    return G_multi



def get_community_dict_and_set(list_com):
    '''
    From communities partitions returns a list of set
    and dictionary of these
    '''
    com = 0
    commun_dict = dict()
    commun_list = list()
    single_com = set()

    for communities in list_com:
        for member in communities:
            commun_dict[member] = com
            single_com.add(member)
        com += 1
        commun_list.append(single_com)
        single_com = set()
    return commun_dict, commun_list


def get_communities(G, alg, typology, seed=0):
    if typology == 'sentiment':
        weight = 'weightWithSentiment'
    elif typology == 'topic':
        weight = 'weightWithTopic'
    elif typology == 'hybrid':
        weight = 'Hibrid'
    else:
        weight = 'weight'

    if alg == 'Louvain':
        start = time.time()
        communities = nx.community.louvain_communities(G, weight=weight, seed=seed, resolution=1.0)
        end = time.time()

        # Map nodes to their community
        list_com = {node: i for i, com in enumerate(communities) for node in com}
        set_com = set(list_com.values())  # Unique community labels

        print(f'Number of communities detected: {len(communities)}')

        info = [len(com) for com in communities]  # List of community sizes
    elif alg == 'Kernighan-Lin':
        start = time.time()

        # Apply Kernighan-Lin bisection to split into exactly 2 communities
        partition_A, partition_B = nx.community.kernighan_lin_bisection(G, weight=weight)

        end = time.time()

        # Combine the two partitions into a list of communities
        partitions = [set(partition_A), set(partition_B)]

        # Map nodes to their selected community
        list_com = {node: i for i, com in enumerate(partitions) for node in com}
        set_com = set(list_com.values())  # Unique community labels
        info = [len(com) for com in partitions]  # Community sizes
    else:
        print('Wrong algorithm name')
        set_com = -1
        list_com = -1
        info = -1
    return list_com, set_com, info, end - start


def label_node_communities(compactGraph, communities, type_com, name, opt):

    if opt == 1:
        directGraph = nx.read_gml(f'Final_DiGraph_{name}.gml')

    for member in communities:
        compactGraph.nodes[member][f'{type_com}Comm'] = communities[member]
        if opt == 1:
            directGraph.nodes[member][f'{type_com}Comm'] = communities[member]
        
    if opt == 1:
        nx.write_gml(directGraph, f'./Final_DiGraph_{name}.gml')
        nx.write_gml(compactGraph, f'./Final_Graph_{name}.gml')
    else:
        nx.write_gml(compactGraph, f'./{name}.gml')

def extract_community(G, id_com):
    community_node = list()
    
    for node in G.nodes():
        if  G.nodes[node]['community'] != id_com:
            community_node.append(node)
    subgraph = G.subgraph(community_node)
    return subgraph

'''
def community_detection(name, opt, typology):
    #### READING GRAPH
    ## OPT == 0 --> Garimella ELSE VAX/COVID
    if opt == 0:
        graph = nx.read_gml(f'{name}.gml')
        multi = nx.read_gml(f'Multi_{name}.gml')
    else:
        graph = nx.read_gml(f'Final_Graph_{name}.gml')
        multi = nx.read_gml(f'Final_MultiGraph_{name}.gml')
    if typology == 'weight':
        log_write_graph_info(name, nx.info(graph), nx.info(multi))
    #### METIS
    list_com_metis, set_com_metis, info, exe_time = get_communities(graph, 'Metis', typology)
    mod_m = modularity(list_com_metis, graph, weight='weight')
    cov_m = coverage(multi, set_com_metis)
    log_write_com_result('Metis', info, mod_m, cov_m, exe_time, opt, typology, name)
    
    #### FLUID
    # seed = 1
    # if opt == 1:
    #    seed = 76
    # if sent:
    #    multi_fluid = create_multi_graph(graph)
    #    if name == 'Covid':
    #        seed = 76
    #    else:
    #        seed = 38
    #else:
    #    multi_fluid = multi
    # print('BEFORE')
    # print(nx.info(multi_fluid))
    # print()
    # list_com_fluid, set_com_fluid, info, exe_time = get_communities(multi_fluid, 'Fluid', typology, 2, seed)

    # print()
    # print(Counter(list_com_fluid.values()))
    # print()
    # mod_f = modularity(list_com_fluid, graph, weight='weight')
    # cov_f = coverage(multi, set_com_fluid)
    # log_write_com_result('Fluid', info, mod_f, cov_f, exe_time, opt, typology)
    # return [list_com_metis, mod_m, cov_m], [list_com_fluid, mod_f, cov_f]

    label_node_communities(graph, list_com_metis, typology, name, opt)
    return [list_com_metis, mod_m, cov_m], [0, 0, 0]
'''

def get_graph_info(graph):
    """Replicates nx.info() by returning node and edge count as a string."""
    return f"Graph with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges"


def community_detection(name, opt, typology):
    #### READING GRAPH
    ## OPT == 0 --> Garimella ELSE VAX/COVID
    if opt == 0:
        graph = nx.read_gml(f'{name}.gml')
        multi = nx.read_gml(f'Multi_{name}.gml')
    else:
        graph = nx.read_gml(f'Final_Graph_{name}.gml')
        multi = nx.read_gml(f'Final_MultiGraph_{name}.gml')

    if typology == 'weight':
        log_write_graph_info(name, get_graph_info(graph), get_graph_info(multi))

    #### METIS (Replaced with Louvain)
    list_com_metis, set_com_metis, info, exe_time = get_communities(graph, 'Kernighan-Lin', typology)


    partition = []
    for community_id in set_com_metis:
        partition.append({node for node, com in list_com_metis.items() if com == community_id})
    mod_m = nx.community.modularity(graph, partition)
    cov_m = partition_quality(graph, partition)[0] #extract only coverage [0] leave out performance [1]

    log_write_com_result('Kernighan-Lin', info, mod_m, cov_m, exe_time, opt, typology, name)

    label_node_communities(graph, list_com_metis, typology, name, opt)
    return [list_com_metis, mod_m, cov_m], [0, 0, 0]


def note_difference(info_no_sent, info_sent, alg, type_diff):

    same = 0
    notsame = 0
    no_sent = info_no_sent[0]
    sent = info_sent[0]
    for user in no_sent:
        if no_sent[user] == sent[user]:
            same += 1
        else:
            notsame += 1

    mod_difference = info_sent[1] - info_no_sent[1]
    cov_difference = info_sent[2] - info_no_sent[2]

    print_difference(alg, same, notsame, mod_difference, cov_difference, type_diff)
    
