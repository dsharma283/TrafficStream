from matplotlib.ticker import FormatStrFormatter
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import matplotlib
import argparse
import sys, os

matplotlib.use('Agg')
sys.path.insert(0, '.')

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--files', nargs='+', default=None,
                        help='List of paths to result files separated by space.')
    parser.add_argument('--plots-path', default=None, required = True,
                        help='Output plot images')
    parser.add_argument('--loss-name', default='mse', required = True,
                        help='Loss function name used to generate numbers')
    args = parser.parse_args()
    args.plots_path = Path(args.plots_path)
    args.plots_path.mkdir(parents=True, exist_ok=True)
    return args

def main():
    args = parse_args()

    files_list = get_files_list(args)

    agg_pdata = {
            'mae' : {15: {'expansible': [], 'static': [], 'retrained': [], 'trafficStream': []} ,
                30: {'expansible': [], 'static': [], 'retrained': [], 'trafficStream': []},
                60: {'expansible': [], 'static': [], 'retrained': [], 'trafficStream': []}},
            'rmse': {15: {'expansible': [], 'static': [], 'retrained': [], 'trafficStream': []},
                30: {'expansible': [], 'static': [], 'retrained': [], 'trafficStream': []},
                60: {'expansible': [], 'static': [], 'retrained': [], 'trafficStream': []}},
            'mape': {15: {'expansible': [], 'static': [], 'retrained': [], 'trafficStream': []},
                30: {'expansible': [], 'static': [], 'retrained': [], 'trafficStream': []},
                60: {'expansible': [], 'static': [], 'retrained': [], 'trafficStream': []}}
            }

    for file in files_list:
        sn = file.name.split('.')[0]
        with open(file, 'r') as f:
            if sn == 'static':
                lines = f.readlines()[-9:]
            else:
                lines = f.readlines()[-16:][:9]
        for line in lines:
            line = line.strip('\n').split('-')[-1].split('\t')
            #print(line)
            typkey = line[1].strip(' ')
            tstamp = int(line[0].strip(' ')) * 5
            vals = [float(x) for x in line[2:-1]]
            agg_pdata[typkey][tstamp][sn].append(vals)
    #print(aggregated_plot_data.items())
    methods = ['expansible', 'static', 'retrained', 'trafficStream']
    horizons = sorted(agg_pdata['mae'].keys())

    # Create subplots for each metric
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    metrics = list(agg_pdata.keys())#['mae', 'rmse', 'mape']

    for i, metric in enumerate(metrics):
        ax = axs[i]
        for method in methods:
            y_values = [np.mean(agg_pdata[metric][h][method]) if agg_pdata[metric][h][method] else 0 for h in horizons]
            ax.plot(horizons, y_values, marker='o', label=method)
            ax.set_title(f'{metric.upper()}-{args.loss_name.upper()}')
            ax.set_xlabel("Prediction Time (mins)")
            ax.set_ylabel(metric)
            ax.legend()
            ax.grid(True)

    plt.tight_layout()
    ofile = os.path.join(args.plots_path, f'prediction-plot-{args.loss_name}.jpg')
    plt.savefig(ofile)
    

def get_files_list(args):
    #print(f'getfiles args.model_dirs = {args.model_dirs}')
    if args.files is not None:
        print('processing files')
        files_list = [Path(file) for file in args.files]
    files_list.sort()
    return files_list


if __name__ == '__main__':
    main()
