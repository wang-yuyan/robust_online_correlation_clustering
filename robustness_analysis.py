import glob
import numpy as np
import matplotlib.pyplot as plt
from pivot_temporal import load_temporal_data

if __name__ == "__main__":
    direc = 'gplus'
    output_folder = 'outputs'

    if direc == 'gplus':
        period = 100668989009254813743
    elif direc == 'facebook':
        period = 1912
    elif direc == 'reddit':
        period = '1Y'
    else:
        period = '2Y'

    if direc == 'reddit' or 'bitcoin':
        seq_list = ['Degree', 'Bad Triangles', 'Time']
    else:
        seq_list = ['Degree', 'Bad Triangles']
    trial_id = 1
    eps_list = [0.5, 0.4, 0.3, 0.2, 0.15, 0.10, 0.08, 0.05, 0.04, 0.02, 0.01]
    eps_alpha_list = [(0.2, 0), (0.18, 0.02), (0.15, 0.05), (0.12, 0.08), (0.1, 0.1), (0.08, 0.12), (0.05, 0.15), (0.02, 0.18)]

    # roustness against eps values

    # this is the file we will use to store the results
    analysis_file = './{}/{}/eps_robustness_result.txt'.format(output_folder, direc)
    robustness_f = open(analysis_file, 'w')
    for eps in eps_list:
        robustness_f.write('{}  '.format(eps))
    robustness_f.write('\n')

    ratios = []
    # write at the end of the output file
    for eps in eps_list:
        # output_file = './{}/{}/{}-{}_eps={}_alpha={}.txt'.format(output_folder, direc, period, trial_id, eps, 0)
        output_file = './{}/{}/{}_eps={}_alpha={}.txt'.format(output_folder, direc, period, eps, 0)
        f = open(output_file, 'r')
        lines = f.readlines()
        results = []
        for line in lines[1:]:
            results.append([int(float(x)) for x in line.strip('\n').split()[1:]])
        results = np.array(results)

        disagreements = np.mean(results, axis=0)
        print(disagreements)
        m = int((results.shape[1] - 2) / 2)

        ratios.append([disagreements[2 * j + 2] / disagreements[0] - 1.0 for j in range(m)])

        f.close()
    ratios = np.transpose(np.array(ratios))
    for row in ratios:
        for entry in row:
            robustness_f.write('{}  '.format(entry))
        robustness_f.write('\n')

    robustness_f.close()

    # draw the plots, degradation v.s. eps value
    plt.xlabel("eps")
    plt.ylabel("degradation")
    for j in range(ratios.shape[0]):
        plt.plot(eps_list, ratios[j], label=seq_list[j])
    plt.legend()
    plt.savefig('./{}/{}_{}_eps_robustness.pdf'.format(output_folder, direc, period))

    plt.clf()

    # do the same thing for alpha value

    analysis_file = './{}/{}/alpha_robustness_result.txt'.format(output_folder, direc)
    robustness_f = open(analysis_file, 'w')
    for (eps, alpha) in eps_alpha_list:
        robustness_f.write('{}  '.format(alpha))
    robustness_f.write('\n')

    ratios = []
    # write at the end of the output file
    for (eps, alpha) in eps_alpha_list:
        # output_file = './{}/{}/{}-{}_eps={}_alpha={}.txt'.format(output_folder, direc, period, trial_id, eps, alpha)
        output_file = './{}/{}/{}_eps={}_alpha={}.txt'.format(output_folder, direc, period, eps, alpha)
        f = open(output_file, 'r')
        lines = f.readlines()
        results = []
        for line in lines[1:]:
            results.append([int(float(x)) for x in line.strip('\n').split()[1:]])
        results = np.array(results)

        disagreements = np.mean(results, axis=0)
        print(disagreements)
        m = int((results.shape[1] - 2) / 2)

        ratios.append([disagreements[2 * j + 2] / disagreements[0] - 1.0 for j in range(m)])

        f.close()
    ratios = np.transpose(np.array(ratios))
    for row in ratios:
        for entry in row:
            robustness_f.write('{}  '.format(entry))
        robustness_f.write('\n')

    robustness_f.close()

    plt.xlabel("alpha")
    plt.ylabel("degradation")
    for j in range(ratios.shape[0]):
        plt.plot([alpha for (eps, alpha) in eps_alpha_list], ratios[j], label=seq_list[j])
    plt.legend()
    plt.savefig('./{}/{}_{}_alpha_robustness.pdf'.format(output_folder, direc, period))

    plt.clf()



