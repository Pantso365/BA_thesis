import os
import ssl

import nltk
from afinn import Afinn
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.tokenize import TweetTokenizer
from collections import Counter
import itertools
from nltk.corpus import stopwords
import string
from nltk import wordpunct_tokenize
from nltk.stem.lancaster import LancasterStemmer
import networkx as nx
from tqdm import tqdm
import re 
import pickle
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from src.preprocessing.utilities import delete_url
import math

def delete_outliers(dataset, field):
    dataset_value = dataset[field]
    removed_outliers = dataset_value.between(dataset_value.quantile(.25), dataset_value.quantile(0.75))
    index_names = dataset[~removed_outliers].index
    dataset = dataset.drop(index_names)
    return dataset

def add_sentiment():
    # Note: For some reason we need this bit of dark magic code here, in order
    # to download the punkt and stopwords...
    # ref: https://stackoverflow.com/questions/38916452/nltk-download-ssl-certificate-verify-failed
    # try:
    #     _create_unverified_https_context = ssl._create_unverified_context
    # except AttributeError:
    #    pass
    # else:
    #   ssl._create_default_https_context = _create_unverified_https_context

    # nltk.download('punkt')
    # nltk.download('stopwords')
    # Note: No covid dataset so skipping...
    covid()
    # vax()

def get_stopword():
    with open("./preprocessing/stopwords_vader.txt", "rb") as fp:
        stop_words = pickle.load(fp)
    return stop_words


def add_sent_weight(DiGraph, CompGraph, name):
    # Load the final dataset that includes the vader_compound column
    os.chdir(os.path.join(os.getcwd(), '..'))
    path = os.path.join(os.getcwd(),  'final_data', 'Final_data.csv')
    print("Constructed path:", path)

    df = pd.read_csv(path)

    # Create a dictionary mapping users to their vader_compound sentiment scores
    user_sentiment = df[['original_author', 'vader_compound']].drop_duplicates()
    sentiment_dict = dict(zip(user_sentiment['original_author'], user_sentiment['vader_compound']))

    # Update the edges of both graphs with the new 'weightWithSentiment'
    for graph in [DiGraph, CompGraph]:
        for u, v, data in graph.edges(data=True):
            # Retrieve the sentiment scores for the two users involved in the edge
            sentiment_u = sentiment_dict.get(u, 0)  # Default to 0 if no sentiment data found
            sentiment_v = sentiment_dict.get(v, 0)

            # Calculate the new weightWithSentiment based on the formula
            sentiment_diff = abs(sentiment_u - sentiment_v)
            sentiment_weight = (2 - sentiment_diff)

            # Compute the weightWithSentiment
            topological_weight = data['weight']
            weight_with_sentiment = topological_weight * sentiment_weight

            # Add the 'weightWithSentiment' to the edge attributes
            data['weightWithSentiment'] = weight_with_sentiment

    # Save the updated graphs
    output_path = os.path.join(os.getcwd(), 'Graph')
    nx.write_gml(CompGraph, os.path.join(output_path, f'Final_Graph_{name}.gml'))
    nx.write_gml(DiGraph, os.path.join(output_path, f'Final_DiGraph_{name}.gml'))



def covid():
    starting_path = os.getcwd()
    path = os.path.join(starting_path, 'data/corona_virus/Graph')
    os.chdir(os.path.join(path))

    CompGraph = nx.read_gml('Final_Graph_Covid.gml')
    # if not 'weightWithSentiment' in list(CompGraph.edges(data=True))[0][2]:
    print(f"Graph Info:\n{CompGraph}")
    print()
    DiGraph = nx.read_gml('Final_DiGraph_Covid.gml')
    print(f"Graph Info:\n{DiGraph}")
    print()
    add_sent_weight(DiGraph, CompGraph, 'Covid')

    os.chdir(starting_path)

def vax():
    starting_path = os.getcwd()
    stop_words = get_stopword()
    path = os.path.join(starting_path, 'data/vax_no_vax/Graph')
    os.chdir(os.path.join(path))

    CompGraph = nx.read_gml('Final_Graph_Vax.gml')
    # Note: Again, this runs only if there is this 'weightWithSentiment' missing...
    if not 'weightWithSentiment' in list(CompGraph.edges(data=True))[0][2]:
        print(nx.info(CompGraph))
        print()
        DiGraph = nx.read_gml('Final_DiGraph_Vax.gml')
        print(nx.info(DiGraph))
        print()
        add_sent_weight(DiGraph, CompGraph, stop_words, 'Vax')

    os.chdir(starting_path)