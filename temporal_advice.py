# see if the ML learned advice really works:
# what happens if we use old data for the new datasets

import glob
import random
import pandas as pd
from datetime import datetime, timedelta
# need this for printing pandas data framework in a more detailed way
pd.set_option("display.max_rows", None, "display.max_columns", None)
import numpy as np
import math
import os

np.random.seed(0)
random.seed(0)

from pivot import get_nodes_info, adversarial_seq, pivot, get_disagreement


# re-order the nodes and give them IDs as 0, 1, ..., n - 1
# and change the representation of edges accordingly
# this function is different from reorder() function in data_preprocess
def reorder(nodes, edges):
    n = len(nodes)
    node_dict = {nodes[i]: i for i in range(n)}
    new_edges = [[node_dict[p], node_dict[q]] for [p, q] in edges]
    return new_edges, node_dict

# this is different from load_data() function in data_preprocess
def load_temporal_data(filename):
    f = open(filename, 'r')
    edges = []
    nodes = set()
    for line in f.readlines():
        edge = [x for x in line.strip().split()]
        edges.append(edge)
        for x in edge:
            nodes.add(x)
    print("network id: {}  nodes: {}  edges: {}".format(filename.split('\\')[-1].split('-')[-1].split('.')[0], len(nodes), len(edges)))
    f.close()
    nodes = list(nodes)
    return len(nodes), edges

# compute how many nodes in set2 also appears in set1 (not symmetric!)
def overlap_ratio(set1, set2):
    n = len(set2)
    rep = 0.0
    for node in set2:
        if node in set1:
            rep += 1
    return rep / n

# from list1, delete all nodes that are not in set2
# the remaining nodes in list1 still preserves the same ordering
# list1: has to be a list
# set2: can be a list or a set
def filter(list1, set2):
    bad_indices = [i for i in range(len(list1)) if list1[i] not in set2]
    for j in sorted(bad_indices, reverse=True):
        del list1[j]
    return list1


# used to gather statistics for the following two datasets:
# https://snap.stanford.edu/data/soc-RedditHyperlinks.html (soc-redditHyperlinks-body.tsv, renamed as reddit.tsv)
# https://snap.stanford.edu/data/soc-sign-bitcoin-otc.html (soc-sign-bitcoinotc.csv, renamed as bitcoin.csv)
# the new dataset produced has duration = period
# the old dataset produced has duration = old_period
def gather_temporal_data_stats(filename, subsample_direc, period, old_period):

    if not os.path.exists(subsample_direc):
        os.makedirs(subsample_direc)

    if filename == './reddit.tsv':
        data = pd.read_csv(filename, sep='\t')
        print(data.columns)
        data = data.sort_values(by=['TIMESTAMP'])

        # select only the positive edges
        data = data[data['LINK_SENTIMENT'] > 0]
        data = data.reset_index(drop=True)
        # print out the first and last times
        datetime_begin = data.iloc[0]['TIMESTAMP']
        datetime_end = data.iloc[-1]['TIMESTAMP']
        print(datetime_begin, datetime_end)

        # define the datetime format
        date_format = '%Y-%m-%d %H:%M:%S'
        first_day = datetime.strptime(datetime_begin, date_format)
        last_day = datetime.strptime(datetime_end, date_format)
        delta = last_day - first_day
        total_days = delta.days

        print("duration: %d days" % total_days)
        if old_period is not None:
            print("old dataset: %d days, new dataset: %s" % (old_period, period))
        else:
            print("new dataset: %s" % period)

        # put the edges into equal-length edge groups according to time
        # time_length is the number of days for each group
        data['TIMESTAMP'] = pd.to_datetime(data['TIMESTAMP'], format='%Y-%m-%d %H:%M:%S')
        record_groups = data.groupby([pd.Grouper(key='TIMESTAMP', freq=period)])

        time_groups = list(record_groups.groups)
        # leave out the first one because the size is too small
        del time_groups[0]
        all_nodes_list = []

        for i in range(1, len(time_groups)):
            time = time_groups[i]
            # data format of time: '%Y-%m-%d %H:%M:%S'

            # create the new dataset
            subsamples = record_groups.get_group(time)[['SOURCE_SUBREDDIT', 'TARGET_SUBREDDIT', 'TIMESTAMP']]

            node_timestamp_dict = {}
            count = 0
            for index, row in subsamples.iterrows():
                if row['SOURCE_SUBREDDIT'] not in node_timestamp_dict:
                    count += 1
                    node_timestamp_dict[row[0]] = count
                if row['TARGET_SUBREDDIT'] not in node_timestamp_dict:
                    count += 1
                    node_timestamp_dict[row[1]] = count

            nodes = list(node_timestamp_dict.keys())
            all_nodes_list.append(nodes)

            print("%s: %d nodes, %d edges" % (time, len(nodes), subsamples.shape[0]))

            # write everything to output files
            edges = []
            for index, row in subsamples.iterrows():
                edges.append([row['SOURCE_SUBREDDIT'], row['TARGET_SUBREDDIT']])
            output_name = '{}/period-{}-{}.txt'.format(subsample_direc, str(period), str(i))
            np.savetxt(output_name, edges, fmt='%s')

            if old_period is None:
                continue
            # create the old dataset which we use to generate advice
            old_end = datetime.strptime(str(time_groups[i - 1]), date_format)
            old_begin = old_end - timedelta(days=old_period)


            mask = (data['TIMESTAMP'] >= old_begin) & (data['TIMESTAMP'] < old_end)
            old_data = data.loc[mask]

            old_node_timestamp_dict = {}
            count = 0
            for index, row in old_data.iterrows():
                if row['SOURCE_SUBREDDIT'] not in old_node_timestamp_dict:
                    count += 1
                    old_node_timestamp_dict[row[0]] = count
                if row['TARGET_SUBREDDIT'] not in old_node_timestamp_dict:
                    count += 1
                    old_node_timestamp_dict[row[1]] = count

            old_nodes = list(old_node_timestamp_dict.keys())

            '''
            if i == 0:
                print("%s: %d nodes, %d edges" % (time, len(nodes), subsamples.shape[0]))
                print("old dataset: %d nodes, %d edges" % (len(old_nodes), old_data.shape[0]))
            else:
                ov_ratio = overlap_ratio(old_nodes, nodes)
                print("%s: %d nodes, %d edges, %f reused" % (time, len(nodes), subsamples.shape[0], ov_ratio))
            '''
            print("old dataset: %d nodes, %d edges" % (len(old_nodes), old_data.shape[0]))
            ov_ratio = overlap_ratio(old_nodes, nodes)
            print("%f reused" % ov_ratio)


            old_edges = []
            for index, row in old_data.iterrows():
                old_edges.append([row['SOURCE_SUBREDDIT'], row['TARGET_SUBREDDIT']])
            output_name = '{}/period-{}-{}-{}days.txt'.format(subsample_direc, str(period), str(i), str(old_period))
            np.savetxt(output_name, old_edges, fmt='%s')



    elif filename == './bitcoin.csv':
        data = pd.read_csv(filename, header=None)
        # print(data.columns)

        # select only the edges that are positive
        data = data[data[2] > 0]
        # print(data.shape)

        num_rows = data.shape[0]

        # print the first few rows to see what it is like
        print(data.head(5))
        # make sure that the edges are sorted according to time
        data.sort_values(by=3)
        # time unit is measured in seconds since the start of the epoch

        data = data.reset_index(drop=True)
        # print out the first and last times
        seconds_begin = data.iloc[0][3]
        seconds_end = data.iloc[-1][3]
        delta = seconds_end - seconds_begin
        print("duration: %f seconds, %f years" % (delta, delta / (12 * 30 * 24 * 3600)))
        if old_period is not None:
            print("old dataset: %d days, new dataset: %s" %(old_period, period))
        else:
            print("new dataset: %s" % period)

        # change the timestmap format for easier data preprocessing later

        # define the datetime format
        date_format = '%Y-%m-%d %H:%M:%S'

        data[3].replace({data.iloc[j][3] : datetime.fromtimestamp(int(data.iloc[j][3])) for j in range(num_rows)}, inplace=True)
        record_groups = data.groupby([pd.Grouper(key=3, freq=period)])

        time_groups = list(record_groups.groups)
        # leave out the first one because the size is too small
        del time_groups[0]
        all_nodes_list = []


        for i in range(1, len(time_groups)):
            time = time_groups[i]
            subsamples = record_groups.get_group(time)

            node_timestamp_dict = {}
            count = 0
            for index, row in subsamples.iterrows():
                if row[0] not in node_timestamp_dict:
                    count += 1
                    node_timestamp_dict[row[0]] = count
                if row[1] not in node_timestamp_dict:
                    count += 1
                    node_timestamp_dict[row[1]] = count
            # print(node_timestamp_dict)
            nodes = node_timestamp_dict.keys()
            # print(nodes)
            all_nodes_list.append(nodes)

            print("%s: %d nodes, %d edges" % (time, len(nodes), subsamples.shape[0]))

            # write everything to output files
            edges = []
            for index, row in subsamples.iterrows():
                edges.append([row[0], row[1]])
            output_name = '{}/period-{}-{}.txt'.format(subsample_direc, str(period), str(i))
            np.savetxt(output_name, edges, fmt='%s')

            # create the old dataset which we use to generate advice
            if old_period is None:
                continue

            old_end = datetime.strptime(str(time_groups[i - 1]), date_format)
            old_begin = old_end - timedelta(days=old_period)

            mask = (data[3] >= old_begin) & (data[3] < old_end)
            old_data = data.loc[mask]

            old_node_timestamp_dict = {}
            count = 0
            for index, row in old_data.iterrows():
                if row[0] not in old_node_timestamp_dict:
                    count += 1
                    old_node_timestamp_dict[row[0]] = count
                if row[1] not in old_node_timestamp_dict:
                    count += 1
                    old_node_timestamp_dict[row[1]] = count

            old_nodes = list(old_node_timestamp_dict.keys())

            print("old dataset: %d nodes, %d edges" % (len(old_nodes), old_data.shape[0]))
            ov_ratio = overlap_ratio(old_nodes, nodes)
            print("%f reused" % ov_ratio)

            old_edges = []
            for index, row in old_data.iterrows():
                old_edges.append([row[0], row[1]])
            output_name = '{}/period-{}-{}-{}days.txt'.format(subsample_direc, str(period), str(i), str(old_period))
            np.savetxt(output_name, old_edges, fmt='%s')

    return

def advice_effect_test(old_filename, filename, output_direc, sequence_list, num_trial=5):

    _, old_edges = load_temporal_data(old_filename)
    old_nodes = set()
    for edge in old_edges:
        for node in edge:
            old_nodes.add(node)

    n, edges = load_temporal_data(filename)
    node_timestamp_dict = {}
    count = 0
    for edge in edges:
        if edge[0] not in node_timestamp_dict:
            count += 1
            node_timestamp_dict[edge[0]] = count
        if edge[1] not in node_timestamp_dict:
            count += 1
            node_timestamp_dict[edge[1]] = count

    nodes = list(node_timestamp_dict.keys())

    # take only the set of old nodes that appear in the new data set
    old_nodes = filter(list(old_nodes), nodes)
    print(len(nodes), len(old_nodes), float(len(old_nodes)) / len(nodes))
    # check the eps value
    f = open(output_direc, 'w')
    f.write("eps={}\n".format(float(len(old_nodes)) / len(nodes)))

    # prepare data
    edges, node_dict = reorder(nodes, edges)
    advice = [node_dict[node] for node in old_nodes]

    neighbor_dict = {i:[] for i in range(n)}

    for [p, q] in edges:
        if q not in neighbor_dict[p]:
            neighbor_dict[p].append(q)
        if p not in neighbor_dict[q]:
            neighbor_dict[q].append(p)

    f.write('trial\t')
    f.write('fully random\t')

    for rule in sequence_list:
        f.write('{}\t'.format(rule))
        f.write('{} w. advice\t'.format(rule))
    f.write('\n')


    nodes = get_nodes_info(neighbor_dict)

    # construct the sequences in adversarial order according to different rules
    dis_arbs = []
    seq_arbs = []
    for rule in sequence_list:
        seq_arb = adversarial_seq(nodes, rule=rule)
        seq_arbs.append(seq_arb)
        pivots_arb, clusters_arb = pivot(seq_arb, neighbor_dict)
        dis_arb = get_disagreement(n, neighbor_dict, clusters_arb)
        dis_arbs.append(dis_arb)

    for i in range(num_trial):
        print("trial:", i)
        f.write(str(i) + '  ')

        # completely random pivot

        seq_rdm = list(range(n))
        random.shuffle(seq_rdm)
        pivots_rdm, clusters_rdm = pivot(seq_rdm, neighbor_dict)
        dis_rdm = get_disagreement(n, neighbor_dict, clusters_rdm)
        print("random: {}, ratio: {}".format(dis_rdm, float(dis_rdm) * 2 / (n * (n - 1))))
        f.write(str(dis_rdm) + '  ')

        for j in range(len(sequence_list)):
            rule = sequence_list[j]
            print("sorting according to {}...".format(rule))
            print("{}: {}".format(rule, dis_arbs[j]))
            f.write(str(dis_arbs[j]) + '  ')

            np.random.shuffle(advice)
            remaining_nodes = [node for node in seq_arbs[j] if node not in advice]
            seq_advice = advice + remaining_nodes
            pivots_advice, clusters_advice = pivot(seq_advice, neighbor_dict)
            dis_advice = get_disagreement(n, neighbor_dict, clusters_advice)
            print("advice: {}".format(dis_advice))
            f.write(str(dis_advice) + '  ')
            f.flush()

        f.write(str(n * (n - 1) / 2) + '\n')
    f.close()
    return


if __name__ == "__main__":

    # produce all the old datasets and new datasets
    filename = './reddit.tsv'
    # filename = './bitcoin.csv'

    direc = filename.split('/')[-1].split('.')[0]

    period = '6M' if direc == 'reddit' else '2Y'
    instance_id = 5 if direc == 'reddit' else 1
    subsample_direc = './data/{}'.format(direc)

    # list of the time duration of the old datasets that we want to use
    # reddit: 5, 10, 15, 20, 25, 30 days
    # bitcoin: 20, 40, 60, 80, 100, 120 days
    old_periods = np.linspace(5, 5 * 6, num=6) if direc == 'reddit' else np.linspace(20, 20 * 6, num=6)
    print(old_periods)


    for old_period in old_periods:
        gather_temporal_data_stats(filename, subsample_direc, period, int(old_period))

    filename = './data/{}/period-{}-{}.txt'.format(direc, period, instance_id)


    for old_period in old_periods:
        old_filename = './data/{}/period-{}-{}-{}days.txt'.format(direc, period, instance_id, int(old_period))
        output_file = './outputs/{}/period-{}-{}-{}days.txt'.format(direc, period, instance_id, int(old_period))
        # old_filename = './data/bitcoin/period-Y-2.txt'

        sequence_list = ['degree', 'triangle', 'time']
        advice_effect_test(old_filename, filename, output_file, sequence_list, num_trial=100)
