import json
import os
import matplotlib.pyplot as plt
import math
import data_preprocessing
import argparse
import matplotlib.ticker as ticker
import pandas as pd


to_exclude_dataset = False


def main(args):
    path = './results/figures'
    model_types = []
    logs = []

    if args.recalculate_counts:
        logs_dir = './logs/'

        with open(os.path.join('config', 'logs_meta.json')) as f:
            logs_meta = json.load(f)

        data_preprocessing.download_logs(logs_meta, logs_dir)
        distributions, _ = data_preprocessing.create_distributions(logs_dir, log_name='file_name')

        dls_counts = data_preprocessing.count_nb_traces_longer_than_prefix(trace_length_distributions=distributions)
        with open(os.path.join(path, 'nb_traces_longer_than_prefix.json'), 'w') as fp:
            json.dump(dls_counts, fp)
    else:
        with open(os.path.join(path, 'nb_traces_longer_than_prefix.json')) as f:
            dls_counts = json.load(f)

    for file_name in os.listdir(path):
        if file_name.startswith('suffix_evaluation_result_dls_'):
            dls_json_file_name = os.path.join(path, file_name)

            with open(dls_json_file_name) as f:
                dls_results = json.load(f)

            # extracting all model types:
            for model_type in dls_results.keys():
                if model_type not in model_types:
                    model_types.append(str(model_type))

                # extracting all logs:
                for log in dls_results[model_type]:
                    if log not in logs:
                        logs.append(str(log))

    if to_exclude_dataset:
        new_logs = []
        for log in logs:
            if log not in DATASET_EXLUDED:
                new_logs.append(log)
        logs = new_logs

    model_types.sort()
    logs.sort()

    b = 3  # number of columns
    a = math.ceil(len(logs) / b)  # number of rows
    c = 1  # initialize plot counter
    subplots = {}
    fig = plt.figure(figsize=(18, 12))
    fig.tight_layout()
    for log in logs:
        subplots[log] = fig.add_subplot(a, b, c)
        subplots[log].set_title('{}'.format(log))
        #subplots[log].set_xlabel('prefix length')
        subplots[log].set_ylim(0.0, 1.1)
        #subplots[log].set_ylabel('DLS')
        subplots[log].get_xaxis().set_major_locator(plt.MaxNLocator(integer=True))
        subplots[log].set_zorder(1)
        subplots[log].set_frame_on(False)
        c += 1

    # the table:
    table_rows = []
    for file_name in os.listdir(path):
        if file_name.startswith('suffix_evaluation_result_dls_mae_'):
            result_json_file_name = os.path.join(path, file_name)

            with open(result_json_file_name) as f:
                results = json.load(f)

            table_row = []
            for model_type in model_types:
                for log in logs:
                    if model_type in results.keys():
                        if log in results[model_type].keys():
                            table_row.append('{:.4f}'.format(float(results[model_type][log]['dls'])))
                            current_model_type = model_type
            table_rows.append([current_model_type] + table_row)

    table = pd.DataFrame(table_rows, columns=['model_type'] + logs).set_index('model_type')
    with open(os.path.join(path, 'table_dls.tex'), 'w') as tf:
        tf.write(table.to_latex())

    # table 2:
    table_rows = []
    for file_name in os.listdir(path):
        if file_name.startswith('suffix_evaluation_result_dls_mae_'):
            result_json_file_name = os.path.join(path, file_name)

            with open(result_json_file_name) as f:
                results = json.load(f)

            table_row = []
            for model_type in model_types:
                for log in logs:
                    if model_type in results.keys():
                        if log in results[model_type].keys():
                            table_row.append('{:.4f}'.format(float(results[model_type][log]['dls'])))
                            current_model_type = model_type
            table_rows.append([current_model_type] + table_row)

    table = pd.DataFrame(table_rows, columns=['model_type'] + logs).set_index('model_type')
    with open(os.path.join(path, 'table_transpose_dls.tex'), 'w') as tf:
        tf.write(table.transpose().to_latex())

    twin_subplots = {}
    for log in logs:
        twin_subplots[log] = subplots[log].twinx()
        #twin_subplots[log].set_ylabel('# traces')
        twin_subplots[log].tick_params(axis='x',          # changes apply to the x-axis
                                       which='both',      # both major and minor ticks are affected
                                       bottom=False,      # ticks along the bottom edge are off
                                       top=False,         # ticks along the top edge are off
                                       labelbottom=False) # labels along the bottom edge are off
        twin_subplots[log].yaxis.set_major_formatter(
            ticker.FuncFormatter(lambda x, pos: '{:.0f}'.format(x / 1000) + 'K'))

    for log in logs:
        if log in dls_counts.keys():
            d = data_preprocessing.key_string_to_int(dls_counts[log])
            twin_subplots[log].bar(d.keys(),
                                   d.values(),
                                   color='lightgray')

    for file_name in os.listdir(path):
        if file_name.startswith('suffix_evaluation_result_dls_'):
            dls_json_file_name = os.path.join(path, file_name)

            with open(dls_json_file_name) as f:
                dls_results = json.load(f)

            for log in logs:
                for model_type in model_types:
                    if model_type in dls_results.keys():
                        if log in dls_results[model_type].keys():
                            suffix_evaluation_sum_result = data_preprocessing.suffix_evaluation_sum_dls(dls_results, model_type)
                            d = data_preprocessing.key_string_to_int(suffix_evaluation_sum_result[model_type][log]['dls_per_prefix'])
                            subplots[log].plot(d.keys(),
                                               d.values(),
                                               label=model_type)
                            subplots[log].yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, pos: '{:.1f}'.format(y)))

    fig.subplots_adjust(hspace=0.2)

    subplots[logs[len(logs) - 1]].legend(loc='center left',
                                         bbox_to_anchor=(-1.5, -0.2),
                                         ncol=7,
                                         fancybox=False,
                                         shadow=False)

    for log in logs:
        subplots[log].set_ylim(bottom=0)

    fig.savefig(os.path.join(path, 'all_models_dls.png'), dpi=fig.dpi)
    fig.savefig(os.path.join(path, 'all_models_dls.eps'), format='eps')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--recalculate_counts', help='recalculate_counts', default=True, type=bool)
    args = parser.parse_args()
    main(args)
