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
            num_weights = 479400
            per_param_setting_performance.append(np.array(bin_m_errs(errs=data['weight_mag_sum'].sum(dim=1)/num_weights, m=m)))
        elif metric == 'dead_neurons':
            num_units = 3*2000
            per_param_setting_performance.append(np.array(bin_m_errs(errs=data['dead_neurons'], m=m)))
        elif metric == 'effective_rank':
            rank_normlization = 3*2000/100
            per_param_setting_performance.append(np.array(bin_m_errs(errs=data['effective_ranks'], m=m)))
        else:
            per_param_setting_performance.append(np.array(bin_m_errs(errs=data['accuracies'] * 100, m=m)))
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

    bp_cfg, bp_setting_idx = '../cfg/task800/bp/std_net.json', 0
    l2_cfg, l2_setting_idx = '../cfg/task800/l2/std_net.json', 0
    erowd_cfg, erowd_setting_idx = '../cfg/task800/erowd/std_net.json', 0
    snp_cfg, snp_setting_idx = '../cfg/task800/snp/std_net.json', 0
    cbp_cfg, cbp_setting_idx = '../cfg/task800/cbp/std_net.json', 0
    er_cfg, er_setting_idx = '../cfg/task800/er/std_net.json', 0
    performances.append(add_cfg_performance(cfg=bp_cfg, setting_idx=bp_setting_idx, m=m, num_runs=num_runs, metric=metric))
    performances.append(add_cfg_performance(cfg=l2_cfg, setting_idx=l2_setting_idx, m=m, num_runs=num_runs, metric=metric))
    performances.append(add_cfg_performance(cfg=erowd_cfg, setting_idx=erowd_setting_idx, m=m, num_runs=num_runs, metric=metric))
    performances.append(add_cfg_performance(cfg=snp_cfg, setting_idx=snp_setting_idx, m=m, num_runs=num_runs, metric=metric))
    performances.append(add_cfg_performance(cfg=cbp_cfg, setting_idx=cbp_setting_idx, m=m, num_runs=num_runs, metric=metric))
    performances.append(add_cfg_performance(cfg=er_cfg, setting_idx=er_setting_idx, m=m, num_runs=num_runs, metric=metric))


    yticks = {'weight': [0, 0.02, 0.04, 0.06, 0.08, 0.10], 'accuracy': [91,92, 93, 94, 95, 96],
              'dead_neurons': [0, 20, 40, 60, 80, 100], 'effective_rank': [0, 10, 20, 30, 40, 50, 60, 70, 80]}[metric]
    generate_online_performance_plot(
        performances=performances,
        colors=['C3', 'C9', 'C4', 'C1', 'C2', 'C0'],
        yticks=yticks,
        xticks=[0, 200*m, 400*m, 600*m, 800*m],
        xticks_labels=['0', '200', '400', '600', '800'],
        m=m,
        fontsize=18,
        labels=['bp', 'l2', 'erowd', 'snp', 'cbp', 'er'],
    )


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))

