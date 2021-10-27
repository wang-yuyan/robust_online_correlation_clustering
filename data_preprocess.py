import glob
import pandas as pd
# need this for printing pandas data framework in a more detailed way
pd.set_option("display.max_rows", None, "display.max_columns", None)
import numpy as np
import math
import os

# re-order the nodes and give them IDs as 0, 1, ..., n - 1
# and change the representation of edges accordingly
def reorder(nodes, edges):
    n = len(nodes)
    nodes.sort()
    node_dict = {nodes[i]: i for i in range(n)}
    new_edges = [[node_dict[p], node_dict[q]] for [p, q] in edges]
    return new_edges

# used to load non-temporal data
# used for the following two datasets
# https://snap.stanford.edu/data/ego-Facebook.html, extracted from facebook.tar.gz
# https://snap.stanford.edu/data/ego-Gplus.html, extracted from gplus.tar.gz
# these two datasets have similar structures
def load_data(filename):
    f = open(filename, 'r')
    edges = []
    nodes = set()
    for line in f.readlines():
        edge = [int(x) for x in line.strip().split()]
        edges.append(edge)
        for x in edge:
            nodes.add(x)
    print("network id: {}  nodes: {}  edges: {}".format(filename.split('\\')[-1].split('.')[0], len(nodes), len(edges)))
    f.close()
    nodes = list(nodes)
    new_edges = reorder(nodes, edges)
    return len(nodes), new_edges

# used to preprocess the original data files for the following two datasets:
# https://snap.stanford.edu/data/soc-RedditHyperlinks.html (soc-redditHyperlinks-body.tsv, renamed as reddit.tsv)
# https://snap.stanford.edu/data/soc-sign-bitcoin-otc.html (soc-sign-bitcoinotc.csv, renamed as bitcoin.csv)
def preprocess_temporal_data(filename, subsample_direc, num_edges=None, num_samples=None):

    if not os.path.exists(subsample_direc):
        os.makedirs(subsample_direc)

    if filename == './reddit.tsv':
        data = pd.read_csv(filename, sep='\t')
        data = data.sort_values(by=['TIMESTAMP'])

        # print(data.columns)
        # print(data.head(5))

        # select only the positive edges
        data = data[data['LINK_SENTIMENT'] > 0]

        # put the edges into equal-length edge groups according to time
        # each group has num_edges edges
        record_groups = data.groupby(np.arange(len(data)) // num_edges)
        for i in range(num_samples):
            subsamples = record_groups.get_group(i)[['SOURCE_SUBREDDIT', 'TARGET_SUBREDDIT']]
            nodes = []
            for index, row in subsamples.iterrows():
                if row['SOURCE_SUBREDDIT'] not in nodes:
                    nodes.append(row['SOURCE_SUBREDDIT'])
                if row['TARGET_SUBREDDIT'] not in nodes:
                    nodes.append(row['TARGET_SUBREDDIT'])
            print(len(nodes))

            # order the nodes and give them ID in range(n) and re-represent the edges
            # write everything to output files
            node_dict = {nodes[i]: i for i in range(len(nodes))}
            edges = []
            for index, row in subsamples.iterrows():
                edges.append([node_dict[row['SOURCE_SUBREDDIT']], node_dict[row['TARGET_SUBREDDIT']]])
            output_name = '{}/{}edges-{}.txt'.format(subsample_direc, str(num_edges), str(i))
            np.savetxt(output_name, edges, fmt='%s')

    elif filename == './data/bitcoin.csv':
        data = pd.read_csv(filename, header=None)
        # print(data.shape)

        # select only the edges that are positive
        data = data[data[2] > 0]
        # print(data.shape)

        # print the first few rows to see what it is like
        print(data.head(10))
        # make sure that the edges are sorted according to time
        data.sort_values(by=3)
        nodes = []
        for index, row in data.iterrows():
            if row[0] not in nodes:
                nodes.append(row[0])
            if row[1] not in nodes:
                nodes.append(row[1])
        node_dict = {nodes[i]: i for i in range(len(nodes))}
        edges = []
        for index, row in data.iterrows():
            edges.append([node_dict[row[0]], node_dict[row[1]]])
        output_name = '{}/bitcoin.txt'.format(subsample_direc)
        np.savetxt(output_name, edges, fmt='%s')

    return



if __name__ == "__main__":

    '''
    # code for loading gplus (non temporal)
    # no need to re-generate subsample files because the sizes of the ego networks are good
    direc = 'gplus'
    all_networks = glob.glob("./data/{}/*.edges".format(direc))
    for network in all_networks:
        load_data(network)
    
    # code for loading reddit (temporal) and design subsamples
    filename = './reddit.tsv'
    subsample_direc = './data/reddit'
    
    # change num_edges into any value
    num_edges = 50000
    num_samples = 5
    preprocess_temporal_data(filename, subsample_direc, num_edges, num_samples)
    
    '''
    # code for loading bitcoin (temporal), no subsampling needed since the size is good
    # but still need to tidy up the dataset, such as changing the node IDs
    filename = './data/bitcoin.csv'
    data_direc = './data/bitcoin'
    preprocess_temporal_data(filename, data_direc)