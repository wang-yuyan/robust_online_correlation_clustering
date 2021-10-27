# this code is used to test Pivot on non-temporal datasets
# it is slightly different from the code used for temporal datasets

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
from data_preprocess import load_data
from collections import namedtuple
# for temporal data we have already sorted the nodes' IDs according to time
Node = namedtuple('Node', 'id degree bad_triangles')

# given a sequence of nodes, eps and alpha, generate a set of advice with corruption
# notice: unlike in the main body, here eps is the fraction of uniformly random samples, not eps - alpha
def get_corrupted_advice(seq, eps=0.2, alpha=0.2):
    n = len(seq)
    remaining_nodes = seq.copy()
    n_rdm = math.ceil(eps * n)
    n_adv = math.ceil(alpha * n)
    advice = []
    for i in range(n_adv):
        advice.append(remaining_nodes.pop(0))
    n -= n_adv
    for i in range(n_rdm):
        index = random.randint(0, n - 1)
        advice.append(remaining_nodes.pop(index))
        n -= 1
    return advice, remaining_nodes

'''
# don't really need this any more since one can just call get_corrupted_advice() with alpha=0.
def get_advice(seq, eps=0.2):
    n = len(seq)
    remaining_nodes = seq.copy()
    n_prime = math.ceil(eps * n)
    advice = []
    for i in range(n_prime):
        index = random.randint(0, n-1)
        advice.append(remaining_nodes.pop(index))
        n -= 1
    return advice, remaining_nodes
'''

# construct a list of nodes according to some rule
# input: nodes is a list of namedtuple "Node"
# output: a list of IDs, a permutation of 0,1,..., n-1
def adversarial_seq(nodes, rule):
    if rule == 'degree':
        seq = sorted(nodes, key=lambda x: x.degree, reverse=True)
    elif rule == 'triangle':
        seq = sorted(nodes, key=lambda x: x.bad_triangles, reverse=True)
    elif rule == 'time':
        seq = nodes.copy()

    return [node.id for node in seq]

# given a graph, computes the local clustering coefficient of a given node
def get_clust_coef(p, neighbor_dict):
    k = len(neighbor_dict[p])
    count = 0
    for i in range(k):
        u = neighbor_dict[p][i]
        for j in range(i + 1, k):
            v = neighbor_dict[p][j]
            if u in neighbor_dict[v]:
                count += 1
    if k > 1:
        return float(count) * 2 / (k * (k - 1))
    else:
        return 0.0


# a naive implementation of pivot
# input: a non-repetitive sequence of all the nodes' IDs in the graph (guaranteed to be 0,1,2..., n-1)
#        a graph, represented as a dictionary of (node, list of neighbors)
# output: a sequence of pivots, and a sequence of clusters
def pivot(seq, neighbor_dict):
    assert len(seq) == len(neighbor_dict)
    pivots = []
    clusters = []
    for i in seq:
        new_pivot = True
        for j in range(len(pivots)):
            p = pivots[j]
            if i in neighbor_dict[p]:
                clusters[j].append(i)
                new_pivot = False
                break
        if new_pivot:
            pivots.append(i)
            clusters.append([i])

    return pivots, clusters

# check if the output is indeed a clustering of the input sequence and if it is the result of pivoting
def check_output(seq, neighbor_dict, pivots, clusters):
    m = len(pivots)
    flat_nodes = [p for cluster in clusters for p in cluster]
    assert len(flat_nodes) == len(seq)
    for p in flat_nodes:
        assert p in seq
    for p in seq:
        assert p in flat_nodes

    # check if this is the result of pivot
    for i in range(m):
        for j in range(i):
            for p in clusters[i]:
                assert p not in neighbor_dict[pivot[i]]
    return


# check if the inputs has the correct form
def check_input(seq, neighbor_dict):
    # check if the nodes are the same set
    V = neighbor_dict.keys()
    assert len(seq) == len(V)
    for p in seq:
        assert p in V
    for q in V:
        assert q in seq
    # check if the graph input has the correct set of nodes and is symmetric
    for p in V:
        for q in neighbor_dict[p]:
            assert q in V
            assert p in neighbor_dict[q]

    n = len(seq)
    for i in range(n):
        assert i in neighbor_dict

    return

# create a dictionary of {node_ID: cluster_ID}
def pt_to_cluster(clusters):
    m = len(clusters)
    pt_cluster_dict = {}
    for i in range(m):
        cluster = clusters[i]
        for p in cluster:
            pt_cluster_dict[p] = i
    return pt_cluster_dict

# commented out the old implementation which was inefficient
# computes the corr.clust obj func
def get_disagreement(n, neighbor_dict, clusters):

    pt_cluster_dict = pt_to_cluster(clusters)
    assert len(pt_cluster_dict) == n
    cluster_sizes = [len(cluster) for cluster in clusters]
    correct_positive = 0
    num_edges = 0
    for p in neighbor_dict:
        for q in neighbor_dict[p]:
            # avoid counting repetitively
            if p >= q:
                continue
            num_edges += 1
            if pt_cluster_dict[p] == pt_cluster_dict[q]:
                correct_positive += 1
    dis = sum([k * (k - 1) / 2 for k in cluster_sizes]) + num_edges - 2 * correct_positive

    '''
    dis = 0
    for i in range(n):
        for j in range(i + 1, n):
            if (pt_cluster_dict[i] == pt_cluster_dict[j] and i not in neighbor_dict[j]) or (pt_cluster_dict[i] != pt_cluster_dict[j] and i in neighbor_dict[j]):
                dis += 1
    print(dis)
    '''
    return dis

# get the information for each node including: degree, # of triangles
# input: a graph in the format of neighboring dictionary {node id: list of its neighbors}
def get_nodes_info(neighbor_dict):
    n = len(neighbor_dict)
    Nodes = []
    for i in range(n):
        k = len(neighbor_dict[i])
        clust_coef = get_clust_coef(i, neighbor_dict)
        bad_triangles = (1.0 - clust_coef) * k * (k - 1) / 2
        Nodes.append(Node(id=i, degree=k, bad_triangles=bad_triangles))
    return Nodes

# for givend dataset, test different sequence constructions and the effect of advice sets
def test(filename, sequence_list, output_direc, eps, alpha, num_trial=5):
    print("eps = {}, alpha = {} ".format(eps, alpha))
    n, edges = load_data(filename)
    neighbor_dict = {i:[] for i in range(n)}

    f = open(output_direc, 'w')

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
    direc = 'gplus'
    output_folder = 'outputs'

    # change this to your local directory
    all_networks = glob.glob("./data/{}/*.edges".format(direc))

    # we can change this into min(len(all_networks), n) for any n to test limited datasets in the same folder
    num_networks = len(all_networks)

    # print out the size of each network
    for i in range(num_networks):
        n, edges = load_data(all_networks[i])

    num_trial = 30

    sequence_list = ['degree', 'triangle']

    eps_list = [0.5, 0.4, 0.3, 0.2, 0.15, 0.10, 0.08, 0.05, 0.04, 0.02, 0.01]
    eps_alpha_list = [(0.18, 0.02), (0.15, 0.05), (0.12, 0.08), (0.1, 0.1), (0.08, 0.12), (0.05, 0.15), (0.02, 0.18)]

    if direc == 'gplus':
    # include this line if it is gplus because gplus has too many networks
    # otherwise if it is data set facebook, just use all ego-networks
        IDs = [100129275726588145876, 100329698645326486178, 100500197140377336562, 100518419853963396365, 100668989009254813743, 101133961721621664586, 100720409235366385249]
        all_networks = []
        for id in IDs:
            all_networks.append("./data/{}/{}.edges".format(direc, str(id)))

    '''
    # for profiling code
    pr = cProfile.Profile()
    pr.enable()
    '''

    for i in range(num_networks):
        network = all_networks[i]
        name = network.split('\\')[-1].split('.')[0] if direc == 'facebook' else str(IDs[i])
        output_direc = './{}/{}'.format(output_folder, direc)
        if not os.path.exists(output_direc):
            os.makedirs(output_direc)
        
        for eps in eps_list:
            test(network, sequence_list, '{}/{}_eps={}_alpha={}.txt'.format(output_direc, name, eps, 0), eps, 0, num_trial)

        for (eps, alpha) in eps_alpha_list:
            test(network, sequence_list, '{}/{}_eps={}_alpha={}.txt'.format(output_direc, name, eps, alpha), eps, alpha, num_trial)

    '''
    pr.disable()
    pr.dump_stats("profile")
    s = io.StringIO()
    p = pstats.Stats('profile', stream=s)

    p.strip_dirs().sort_stats('cumulative').print_stats()
    with open('pivot_profile.txt', 'w') as f:
        f.write(s.getvalue())
        f.close()
    '''