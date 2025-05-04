from preprocessing.ops_on_raw_data import ops_on_corona, ops_on_vac, refine_data
from preprocessing.ops_build_graph import graph_ops, garimella_graph, vax_graph

# from community.community_detection import start_community_detection
# from preprocessing.topic_modelling import add_topic
# from controversy_detection.start_controversy_detection import start_detection
# from link_prediction.start_link_prediction import start_link_opt
from preprocessing.visualisation import visualize_large_garimella_graph, visualize_vaccination_graph, visualize_covid_graph

import sys

from src.community.community_detection import start_community_detection
from src.preprocessing.ops_sentiment_vader import add_sentiment


def preprocessing_operation():
    # Note: No Covid dataset, so we can skip this for now...
    ops_on_corona()
    # ops_on_vac()
    graph_ops()
    # add_sentiment()
    print("PREPROCESSING DONE")
    print("")

def community_detection():
   start_community_detection()

#def controversy_detection():
  #  start_detection()

#def link_prediction():
    #start_link_opt()

if __name__ == '__main__':
    # Note: This function is needed to be run only for the first time
    preprocessing_operation()
    community_detection()
    # visualize_covid_graph("C:/Users/panos/GitHub/echochambers/src/data/corona_virus/Graph")
    # visualize_large_garimella_graph("C:/Users/panos/GitHub/echochambers/src/data/garimella_data/Graph")
    #visualize_vaccination_graph("C:/Users/panos/GitHub/echochambers/src/data/vaccination/Graph/Final_Graph_Vax.gml")


