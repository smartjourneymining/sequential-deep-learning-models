import json
from tqdm import tqdm
from urllib.request import urlopen
from urllib.request import urlretrieve
import cgi
import os
import gzip
import shutil
from pm4py.objects.log.importer.xes import importer as xes_importer
import pm4py
import matplotlib.pyplot as plt
import math
import pandas as pd
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.ticker as ticker


to_exclude_dataset = False

LOG_DIR = './logs/'
if not os.path.exists(LOG_DIR): os.makedirs(LOG_DIR)

path = './results/figures'

with open(os.path.join('config', 'logs_meta.json')) as f:
    logs_meta = json.load(f)

for log_name in tqdm(logs_meta):
    remotefile = urlopen(logs_meta[log_name])
    blah = remotefile.info()['Content-Disposition']
    value, params = cgi.parse_header(blah)
    filename = params["filename"]
    urlretrieve(logs_meta[log_name], os.path.join(LOG_DIR, filename))

for file_name in os.listdir(LOG_DIR):
    if file_name.endswith('.gz'):
        gz_file_name = os.path.join(LOG_DIR, file_name)
        with gzip.open(gz_file_name, 'rb') as f_in:
            with open(gz_file_name[:-3], 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)

trace_length_distributions = {}
event_frequency_distributions = {}
event_frequency_distributions_colors = {}

for file_name in sorted(os.listdir(LOG_DIR)):
    if file_name.endswith('.xes'):
        xes_file_name = os.path.join(LOG_DIR, file_name)
        log = xes_importer.apply(xes_file_name)
        event_frequency_distribution = {}
        event_frequency_distribution_colors = {}
        trace_length_distribution = {}
        longest_trace_length = 0
        amount_of_padding = 0
        for trace in log:
            if len(trace) > longest_trace_length:
                longest_trace_length = len(trace)
            if len(trace) not in trace_length_distribution:
                trace_length_distribution[len(trace)] = 1
            else:
                trace_length_distribution[len(trace)] += 1
            for event in trace:
                if str(event['concept:name']) not in event_frequency_distribution:
                    event_frequency_distribution[str(event['concept:name'])] = 1
                    event_frequency_distribution_colors[str(event['concept:name'])] = 'tab:gray'
                else:
                    event_frequency_distribution[str(event['concept:name'])] += 1
        for trace in log:
            amount_of_padding += longest_trace_length - len(trace)
        event_frequency_distribution['[PAD]'] = amount_of_padding
        event_frequency_distribution_colors['[PAD]'] = 'tab:red'
        trace_length_distribution = dict(sorted(trace_length_distribution.items()))
        trace_length_distributions[str(file_name)] = trace_length_distribution
        event_frequency_distributions[str(file_name)] = event_frequency_distribution
        event_frequency_distributions_colors[str(file_name)] = event_frequency_distribution_colors
    elif file_name.endswith('.csv'):
        log = pm4py.format_dataframe(pd.read_csv(os.path.join(LOG_DIR, file_name), sep=','),
                                     case_id='CaseID',
                                     activity_key='ActivityID',
                                     timestamp_key='CompleteTimestamp')
        log = pm4py.convert_to_event_log(log)
        event_frequency_distribution = {}
        event_frequency_distribution_colors = {}
        trace_length_distribution = {}
        longest_trace_length = 0
        amount_of_padding = 0
        for trace in log:
            if len(trace) > longest_trace_length:
                longest_trace_length = len(trace)
            if len(trace) not in trace_length_distribution:
                trace_length_distribution[len(trace)] = 1
            else:
                trace_length_distribution[len(trace)] += 1
            for event in trace:
                if str(event['concept:name']) not in event_frequency_distribution:
                    event_frequency_distribution[str(event['concept:name'])] = 1
                    event_frequency_distribution_colors[str(event['concept:name'])] = 'tab:gray'
                else:
                    event_frequency_distribution[str(event['concept:name'])] += 1
        for trace in log:
            amount_of_padding += longest_trace_length - len(trace)
        event_frequency_distribution['[PAD]'] = amount_of_padding
        event_frequency_distribution_colors['[PAD]'] = 'tab:red'
        trace_length_distribution = dict(sorted(trace_length_distribution.items()))
        trace_length_distributions[str(file_name)] = trace_length_distribution
        event_frequency_distributions[str(file_name)] = event_frequency_distribution
        event_frequency_distributions_colors[str(file_name)] = event_frequency_distribution_colors

logs = []
for log_name in trace_length_distributions.keys():
    logs.append(log_name)
logs.sort()

if to_exclude_dataset:
    new_logs = []
    for log in logs:
        if log not in DATASET_EXLUDED:
            new_logs.append(log)
    logs = new_logs

b = 3  # number of columns
a = math.ceil(len(logs) / b)  # number of rows
c = 1  # initialize plot counter
subplots = {}
fig = plt.figure(figsize=(18, 12))
fig.tight_layout()
for log in logs:
    subplots[log] = fig.add_subplot(a, b, c)
    subplots[log].set_title('{}'.format(log))
    #subplots[log].set_xlabel('trace length')
    #subplots[log].set_ylabel('frequency')
    subplots[log].get_xaxis().set_major_locator(plt.MaxNLocator(integer=True))
    c += 1

for log in logs:
    subplots[log].bar(trace_length_distributions[log].keys(), trace_length_distributions[log].values(), color='tab:blue')

axins = {}
for log in logs:
    axins[log] = inset_axes(subplots[log], width="30%", height="70%")
    axins[log].bar(event_frequency_distributions[log].keys(), event_frequency_distributions[log].values(), color=event_frequency_distributions_colors[log].values())
    axins[log].text(0.4,
                    0.9,
                    'vocab size: ' + str(len(event_frequency_distributions[log])),
                    horizontalalignment='center',
                    verticalalignment='center',
                    transform=axins[log].transAxes,
                    size=8)
    axins[log].set_xlabel('activity inc. [PAD]', fontsize=8)
    axins[log].set_ylabel('frequency', fontsize=8)
    axins[log].yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{:.0f}'.format(x / 1000) + 'K'))

for log in logs:
    axins[log].set_ylim(bottom=0)
    axins[log].tick_params(axis='x', length=0)
    axins[log].tick_params(axis='y', which='major', labelsize=6)
    axins[log].tick_params(axis='y', which='minor', labelsize=6)
    plt.setp(axins[log].get_xticklabels(), visible=False)

fig.subplots_adjust(hspace=0.2)
fig.savefig(os.path.join(path, 'case_length_statistics.png'), dpi=fig.dpi)
fig.savefig(os.path.join(path, 'case_length_statistics.eps'), format='eps')
