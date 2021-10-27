import random
import math
import glob
import os
import io
import numpy as np
random.seed(0)
np.random.seed(0)
# for profiling code to see where the bottleneck is
import cProfile, pstats
from temporal_advice import load_temporal_data, gather_temporal_data_stats, reorder
from collections import namedtuple
from pivot import Node, get_corrupted_advice, adversarial_seq, get_clust_coef, pivot, pt_to_cluster, get_disagreement, get_nodes_info

# for givend dataset, test different sequence constructions and the effect of advice sets
def test_temporal(filename, sequence_list, output_direc, eps, alpha, num_trial=5):
    print("eps = {}, alpha = {} ".format(eps, alpha))
    n, edges = load_temporal_data(filename)
    neighbor_dict = {i:[] for i in range(n)}

    f = open(output_direc, 'w')

    # sort the nodes according to how early they appear in al edges
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
    edges, node_dict = reorder(nodes, edges)

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
    # f.write('trial    fully random    temporal   temporal w. advice    degree    degree w. advice   clust_coef   clust_coef w. advice  bound\n')

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
        f.write(str(i) + '\t')

        # completely random pivot

        seq_rdm = list(range(n))
        random.shuffle(seq_rdm)
        pivots_rdm, clusters_rdm = pivot(seq_rdm, neighbor_dict)
        dis_rdm = get_disagreement(n, neighbor_dict, clusters_rdm)
        print("random: {}, ratio: {}".format(dis_rdm, float(dis_rdm) * 2 / (n * (n - 1))))
        f.write(str(dis_rdm) + '\t')

        for j in range(len(sequence_list)):
            rule = sequence_list[j]
            print("sorting according to {}...".format(rule))
            print("{}: {}".format(rule, dis_arbs[j]))
            f.write(str(dis_arbs[j]) + '\t')

            advice, remaining_nodes = get_corrupted_advice(seq_arbs[j], eps=eps, alpha=alpha)
            np.random.shuffle(advice)
            seq_advice = advice + remaining_nodes
            pivots_advice, clusters_advice = pivot(seq_advice, neighbor_dict)
            dis_advice = get_disagreement(n, neighbor_dict, clusters_advice)
            print("advice: {}".format(dis_advice))
            f.write(str(dis_advice) + '\t')
            f.flush()

        f.write(str(n * (n - 1) / 2) + '\n')
    f.close()
    return

if __name__ == "__main__":
    filename = './reddit.tsv'
    # filename = './bitcoin.csv'

    direc = filename.split('/')[-1].split('.')[0]
    subsample_direc = './data/{}'.format(direc)
    output_folder = 'outputs'

    if direc == 'reddit':
        periods = ['1M', '2M', '4M', '6M', '8M', '10M', '1Y']
    else:
        periods = ['4M', '8M', '10M', '1Y', '2Y']


    # we test 2 networks for each time length
    num_networks = 2
    num_trial = 30
    sequence_list = ['degree', 'triangle', 'time']

    eps_list = [0.5, 0.4, 0.3, 0.2, 0.15, 0.10, 0.08, 0.05, 0.04, 0.02, 0.01]
    eps_alpha_list = [(0.18, 0.02), (0.15, 0.05), (0.12, 0.08), (0.1, 0.1), (0.08, 0.12), (0.05, 0.15), (0.02, 0.18)]


    for period in periods:
        # gather_temporal_data_stats(filename, subsample_direc, period, None)
        # print out the size of each network

        all_networks = ['{}/period-{}-{}.txt'.format(subsample_direc, str(period), str(i)) for i in range(1, num_networks + 1)]

        for i in range(num_networks):
            network = all_networks[i]
            name = '{}-{}'.format(period, str(i + 1))
            output_direc = './{}/{}'.format(output_folder, direc)
            if not os.path.exists(output_direc):
                os.makedirs(output_direc)

            for eps in eps_list:
                test_temporal(network, sequence_list, '{}/{}_eps={}_alpha={}.txt'.format(output_direc, name, eps, 0), eps, 0, num_trial)

            for (eps, alpha) in eps_alpha_list:
                test_temporal(network, sequence_list, '{}/{}_eps={}_alpha={}.txt'.format(output_direc, name, eps, alpha), eps, alpha, num_trial)

