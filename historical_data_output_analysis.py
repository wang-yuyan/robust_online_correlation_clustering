# for tidying up the data for temporal datasets
# and analyzing the effect of using historical data

import numpy as np



if __name__ == "__main__":
    # change the setting here
    # direc: name of the dataset, also name of the folder that stores the data
    # direc could be 'bitcoin' or 'reddit'
    direc = 'bitcoin'
    output_folder = 'outputs'

    # these settings depends on the setting we use to generate historical data outputs
    period = '1Y' if direc == 'reddit' else '2Y'
    instance_id = 5 if direc == 'reddit' else 1

    # analysis_file:
    analysis_file = './{}/{}/historical_data_{}-{}.txt'.format(output_folder, direc, period, instance_id)
    all_data_f = open(analysis_file, 'w')
    all_data_f.write(
        'Days   eps   Random   Degree   Deg. Adv   Bad Triangles   Tri. Adv   Time   Time. Adv   Upper Bound\n')

    # this list also depends on previous settings when we obtained the data
    old_periods = [5, 10, 15, 20, 25] if direc == 'reddit' else [20, 40, 60, 80, 100]

    all_results = []

    # for random Pivot, we can only take the first
    random_pivot = - 1.0

    for old_period in old_periods:
        all_data_f.write('{}  '.format(old_period))
        file = './{}/{}/{}-{}-{}-{}days.txt'.format(output_folder, direc, 'period', period, instance_id, old_period)
        f = open(file, 'r')

        lines = f.readlines()

        eps = lines[0].split('=')[-1]
        all_data_f.write('{}  '.format(eps))

        results = []
        for line in lines[2:]:
            results.append([int(float(x)) for x in line.strip('\n').split()[1:]])
        results = np.array(results)

        print(results)

        disagreements = np.mean(results, axis=0)
        if random_pivot < 0.0:
            random_pivot = disagreements[0]
        stdev = np.std(results, axis=0)
        for j in range(results.shape[1] - 1):
            all_data_f.write(str(round(disagreements[j] / random_pivot - 1.0, 4)) + '/' + str(round(stdev[j] / random_pivot, 4)) + '  ')
        all_data_f.write('\n')

    all_data_f.close()