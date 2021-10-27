import statistics as stats
import glob
import numpy as np
from pivot_temporal import load_temporal_data

# for tidying up the data, taking averages and stdev over all trials and generate plots

if __name__ == "__main__":
    # change the setting here.

    direc = 'gplus'
    output_folder = 'outputs'
    eps = 0.1
    alpha = 0

    analysis_file = './{}/{}/pivot_result_eps={}.txt'.format(output_folder, direc, eps)
    all_data_f = open(analysis_file, 'w')
    all_data_f.write('ID  Random Degree   Deg. Adv   Bad Triangles   Tri. Adv   Upper Bound\n')

    all_results = glob.glob("./{}/{}/*_eps={}_alpha={}.txt".format(output_folder, direc, eps, alpha))
    # write at the end of the output file
    for file in all_results:
        ID = file.split('\\')[-1].split('_')[0]
        all_data_f.write('{}  '.format(ID))
        f = open(file, 'r')
        lines = f.readlines()
        results = []
        for line in lines[1:]:
            results.append([int(float(x)) for x in line.strip('\n').split()[1:]])
        results = np.array(results)

        disagreements = np.mean(results, axis=0)
        stdev = np.std(results, axis=0)
        for j in range(results.shape[1] - 1):
            all_data_f.write(str(round(disagreements[j] / disagreements[0] - 1.0, 4)) + '/' + str(round(stdev[j] / disagreements[0], 4)) + '  ')
        all_data_f.write('\n')
        f.close()

    all_data_f.close()


    direc = 'bitcoin'
    data_folder = 'data'
    output_folder = 'outputs'

    all_data = glob.glob("./{}/{}/period-*.txt".format(data_folder, direc, eps, alpha))
    data_info_folder = './{}/{}_networks_info.txt'.format(data_folder, direc)
    info_f = open(data_info_folder, 'w')
    for network in all_data:
        n, edges = load_temporal_data(network)
        id = network.split('\\')[-1].split('.')[0]
        info_f.write('network id: {}  nodes: {}  edges: {}\n'.format(id, n, len(edges)))
    info_f.close()


    all_results = glob.glob("./{}/{}/*_eps={}_alpha={}.txt".format(output_folder, direc, eps, alpha))

    analysis_file = './{}/{}/pivot_result_eps={}.txt'.format(output_folder, direc, eps)
    all_data_f = open(analysis_file, 'w')
    all_data_f.write('Days  Degree   Deg. Adv   Bad Triangles   Tri. Adv  \n')

    # write at the end of the output file
    for file in all_results:
        ID = file.split('\\')[-1].split('_')[0]
        all_data_f.write('{}  '.format(ID))
        f = open(file, 'r')
        lines = f.readlines()
        results = []
        for line in lines[1:]:
            results.append([int(float(x)) for x in line.strip('\n').split()[1:]])
        results = np.array(results)

        disagreements = np.mean(results, axis=0)
        stdev = np.std(results, axis=0)
        for j in range(results.shape[1] - 1):
            all_data_f.write(str(round(disagreements[j] / disagreements[0] - 1.0, 4)) + '/' + str(round(stdev[j] / disagreements[0], 4))+ '  ')
        all_data_f.write('\n')
        f.close()

    all_data_f.close()


