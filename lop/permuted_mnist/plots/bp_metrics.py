import sys
import json
import pickle
import argparse
from lop.utils.miscellaneous import *
from lop.utils.plot_online_performance import generate_online_performance_plot


def add_cfg_performance(cfg='', setting_idx=0, m=2*10*1000, num_runs=30, metric='accuracy'):
    with open(cfg, 'r') as f:
        params = json.load(f)
    list_params, param_settings = get_configurations(params=params)
    per_param_setting_performance = []
    for idx in range(num_runs):
        file = '../' + params['data_dir'] + str(setting_idx) + '/' + str(idx)
        with open(file, 'rb') as f:
            data = pickle.load(f)

        if metric == 'weight':
            num_weights = 9588000
            per_param_setting_performance.append(np.array(bin_m_errs(errs=data['weight_mag_sum'].sum(dim=1)/num_weights, m=m)))
        elif metric == 'dead_neurons':
            num_units = 3*2000
            per_param_setting_performance.append(np.array(bin_m_errs(errs=data['dead_neurons'].sum(dim=1), m=m)))
        elif metric == 'effective_rank':
            rank_normlization = 3*2000/100
            per_param_setting_performance.append(np.array(bin_m_errs(errs=data['effective_ranks'], m=m)))
        else:
            per_param_setting_performance.append(np.array(bin_m_errs(errs=data['accuracies'] * 100, m=m)))
            print('Accuracy:', data['accuracies'].mean())
    print(param_settings[setting_idx], setting_idx, np.array(per_param_setting_performance).mean())
    return np.array(per_param_setting_performance)


def main(arguments):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--cfg_file', help="Path of the file containing the parameters of the experiment", type=str,
                            default='../cfg/bp/std_net.json')
    parser.add_argument('--metric', help="Specify the metric you want to plot, the options are: accuracy, weight,"
                                         " dead_neurons, and effective_rank", type=str, default='accuracy')

    args = parser.parse_args(arguments)
    cfg_file = args.cfg_file
    metric = args.metric

    with open(cfg_file, 'r') as f:
        params = json.load(f)
    list_params, param_settings = get_configurations(params=params)

    performances = []
    m = {'weight': 60*1000, 'accuracy': 60*1000, 'dead_neurons': 1, 'effective_rank': 1}[metric]
    num_runs = params['num_runs']

    indices = [i for i in range(4)]
    for i in indices:
        performances.append(add_cfg_performance(cfg=cfg_file, setting_idx=i, m=m, num_runs=num_runs, metric=metric))

    yticks = {'weight': [0, 0.02, 0.04, 0.06, 0.08, 0.10], 'accuracy': [82, 84, 86, 88, 90, 92, 94],
              'dead_neurons': [0, 50, 100], 'effective_rank': [0, 10, 20, 30, 40, 50, 60, 70, 80]}[metric]
    generate_online_performance_plot(
        performances=performances,
        colors=['C1', 'C3', 'C5', 'C2', 'C4', 'C6'],
        yticks=yticks,
        xticks=[0, 100*m],
        xticks_labels=['0', '100'],
        m=m,
        fontsize=18,
        labels=param_settings,
    )


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))

