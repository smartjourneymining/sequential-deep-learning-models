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
import random
from copy import deepcopy
import torch
from torch.utils.data import DataLoader, TensorDataset


def key_string_to_int(d):
    return {int(k): v for k, v in d.items()}


def create_split_log(log, validation_ratio=0.2):
    nb_training = math.ceil(len(log['traces']) * (1 - validation_ratio))
    random.shuffle(log['traces'])

    augmented_log = deepcopy(log)
    del augmented_log['traces']
    augmented_log['training_traces'] = log['traces'][:nb_training]
    augmented_log['validation_traces'] = log['traces'][nb_training:]

    return augmented_log


# TODO having a parameter & del training prefixes during eval
# Prefixes for the sequential encoder-decoder models
def create_prefixes(log,
                    min_prefix=2,
                    create_tensors=True,
                    add_special_tokens=True,
                    pad_sequences=True,
                    pad_token=0,
                    to_wrap_into_torch_dataset=True,
                    training_batch_size=None,
                    validation_batch_size=None,
                    single_position_target=False):
    augmented_log = deepcopy(log)
    augmented_log['training_prefixes_and_suffixes'] = {'ids': {},
                                                       'activities': {'prefixes': {}, 'suffixes': {'input': {}, 'target': {}}},
                                                       'times': {'prefixes': {}, 'suffixes': {'input': {}, 'target': {}}}}
    augmented_log['validation_prefixes_and_suffixes'] = {'ids': {},
                                                         'activities': {'prefixes': {}, 'suffixes': {'input': {}, 'target': {}}},
                                                         'times': {'prefixes': {}, 'suffixes': {'input': {}, 'target': {}}}}

    def iterate_over_traces(log,
                            subset='training',
                            create_tensors=True,
                            add_special_tokens=True,
                            pad_sequences=True,
                            pad_token=0):

        if create_tensors:
            dynamic_tensification = torch.tensor
        else:
            dynamic_tensification = lambda x: x

        # Defining tokens:
        sos_token = log['vocabulary_size'] + 1
        eos_token = log['vocabulary_size'] + 2
        mask_token = log['vocabulary_size'] + 3
        log['sos_token'] = sos_token
        log['eos_token'] = eos_token
        log['mask_token'] = mask_token
        log['pad_token'] = pad_token

        # Very interesting research question:
        time_attribute_padding_value = 0.0

        # For each original trace in the log:
        for trace in tqdm(log[subset + '_traces'], desc='creating ' + subset + ' prefixes of ' + augmented_log['id'] + ' for ae'):
            if single_position_target:
                max_prefix = len(trace['activities']) + 1
            else:
                # to be compatible with other models' total prefix combinations:
                max_prefix = len(trace['activities']) + 1

            for prefix in range(min_prefix, max_prefix):
                if single_position_target:
                    # These two are just workaround to have anything for nothing:
                    activities_suffix_sequence_input = [sos_token]
                    times_suffix_sequence_input = [time_attribute_padding_value]

                    if prefix < len(trace['activities']):
                        activities_suffix_sequence_target = [trace['activities'][prefix]]
                        times_suffix_sequence_target = [trace['times'][prefix]]
                    else:
                        activities_suffix_sequence_target = [eos_token]
                        times_suffix_sequence_target = [time_attribute_padding_value]
                else:
                    if add_special_tokens:
                        activities_suffix_sequence_input = [sos_token] + trace['activities'][prefix:]
                        times_suffix_sequence_input = [time_attribute_padding_value] + trace['times'][prefix:]
                        activities_suffix_sequence_target = trace['activities'][prefix:] + [eos_token]
                        times_suffix_sequence_target = trace['times'][prefix:] + [time_attribute_padding_value]
                    else:
                        activities_suffix_sequence_input = trace['activities'][prefix:]
                        times_suffix_sequence_input = trace['times'][prefix:]
                        activities_suffix_sequence_target = trace['activities'][prefix:]
                        times_suffix_sequence_target = trace['times'][prefix:]

                with torch.no_grad():
                    if prefix not in log[subset + '_prefixes_and_suffixes']['activities']['prefixes'].keys():
                        log[subset + '_prefixes_and_suffixes']['activities']['prefixes'][prefix] = [
                            dynamic_tensification(trace['activities'][:prefix])]
                        log[subset + '_prefixes_and_suffixes']['activities']['suffixes']['input'][prefix] = [
                            dynamic_tensification(activities_suffix_sequence_input)]
                        log[subset + '_prefixes_and_suffixes']['activities']['suffixes']['target'][prefix] = [
                            dynamic_tensification(activities_suffix_sequence_target)]
                        log[subset + '_prefixes_and_suffixes']['times']['prefixes'][prefix] = [
                            dynamic_tensification(trace['times'][:prefix])]
                        log[subset + '_prefixes_and_suffixes']['times']['suffixes']['input'][prefix] = [
                            dynamic_tensification(times_suffix_sequence_input)]
                        log[subset + '_prefixes_and_suffixes']['times']['suffixes']['target'][prefix] = [
                            dynamic_tensification(times_suffix_sequence_target)]
                        log[subset + '_prefixes_and_suffixes']['ids'][prefix] = [trace['id']]
                    else:
                        log[subset + '_prefixes_and_suffixes']['activities']['prefixes'][prefix].append(
                            dynamic_tensification(trace['activities'][:prefix]))
                        log[subset + '_prefixes_and_suffixes']['activities']['suffixes']['input'][prefix].append(
                            dynamic_tensification(activities_suffix_sequence_input))
                        log[subset + '_prefixes_and_suffixes']['activities']['suffixes']['target'][prefix].append(
                            dynamic_tensification(activities_suffix_sequence_target))
                        log[subset + '_prefixes_and_suffixes']['times']['prefixes'][prefix].append(
                            dynamic_tensification(trace['times'][:prefix]))
                        log[subset + '_prefixes_and_suffixes']['times']['suffixes']['input'][prefix].append(
                            dynamic_tensification(times_suffix_sequence_input))
                        log[subset + '_prefixes_and_suffixes']['times']['suffixes']['target'][prefix].append(
                            dynamic_tensification(times_suffix_sequence_target))
                        log[subset + '_prefixes_and_suffixes']['ids'][prefix].append(trace['id'])

        if create_tensors:
            if pad_sequences:
                with torch.no_grad():
                    if not single_position_target:
                        # Create a suffix tensor (in each prefix list) which has the max length for sure:
                        for prefix in log[subset + '_prefixes_and_suffixes']['activities']['prefixes'].keys():
                            an_activity_suffix_input = log[subset + '_prefixes_and_suffixes']['activities']['suffixes']['input'][prefix][0]
                            a_time_suffix_input = log[subset + '_prefixes_and_suffixes']['times']['suffixes']['input'][prefix][0]
                            an_activity_suffix_target = log[subset + '_prefixes_and_suffixes']['activities']['suffixes']['target'][prefix][0]
                            a_time_suffix_target = log[subset + '_prefixes_and_suffixes']['times']['suffixes']['target'][prefix][0]

                            # Max length (for suffix) is extended by one to cover [EOS] (target) and [SOS] (input)
                            if add_special_tokens:
                                max_length = log['longest_trace_length'] + 1
                            else:
                                max_length = log['longest_trace_length']

                            extension = pad_token * torch.ones((max_length - prefix - an_activity_suffix_input.size(0)))
                            log[subset + '_prefixes_and_suffixes']['activities']['suffixes']['input'][prefix][0] = torch.cat(
                                (an_activity_suffix_input, extension))
                            extension = time_attribute_padding_value * torch.ones((max_length - prefix - a_time_suffix_input.size(0)))
                            log[subset + '_prefixes_and_suffixes']['times']['suffixes']['input'][prefix][0] = torch.cat(
                                (a_time_suffix_input, extension))
                            extension = pad_token * torch.ones((max_length - prefix - an_activity_suffix_target.size(0)))
                            log[subset + '_prefixes_and_suffixes']['activities']['suffixes']['target'][prefix][0] = torch.cat(
                                (an_activity_suffix_target, extension))
                            extension = time_attribute_padding_value * torch.ones((max_length - prefix - a_time_suffix_target.size(0)))
                            log[subset + '_prefixes_and_suffixes']['times']['suffixes']['target'][prefix][0] = torch.cat(
                                (a_time_suffix_target, extension))

                    for prefix in log[subset + '_prefixes_and_suffixes']['activities']['prefixes'].keys():
                        log[subset + '_prefixes_and_suffixes']['activities']['prefixes'][
                            prefix] = torch.nn.utils.rnn.pad_sequence(
                            log[subset + '_prefixes_and_suffixes']['activities']['prefixes'][prefix],
                            batch_first=True,
                            padding_value=pad_token)
                        log[subset + '_prefixes_and_suffixes']['activities']['suffixes']['input'][
                            prefix] = torch.nn.utils.rnn.pad_sequence(
                            log[subset + '_prefixes_and_suffixes']['activities']['suffixes']['input'][prefix],
                            batch_first=True,
                            padding_value=pad_token)
                        log[subset + '_prefixes_and_suffixes']['activities']['suffixes']['target'][
                            prefix] = torch.nn.utils.rnn.pad_sequence(
                            log[subset + '_prefixes_and_suffixes']['activities']['suffixes']['target'][prefix],
                            batch_first=True,
                            padding_value=pad_token)
                        log[subset + '_prefixes_and_suffixes']['times']['prefixes'][
                            prefix] = torch.nn.utils.rnn.pad_sequence(
                            log[subset + '_prefixes_and_suffixes']['times']['prefixes'][prefix],
                            batch_first=True,
                            padding_value=pad_token)
                        log[subset + '_prefixes_and_suffixes']['times']['suffixes']['input'][
                            prefix] = torch.nn.utils.rnn.pad_sequence(
                            log[subset + '_prefixes_and_suffixes']['times']['suffixes']['input'][prefix],
                            batch_first=True,
                            padding_value=pad_token)
                        log[subset + '_prefixes_and_suffixes']['times']['suffixes']['target'][
                            prefix] = torch.nn.utils.rnn.pad_sequence(
                            log[subset + '_prefixes_and_suffixes']['times']['suffixes']['target'][prefix],
                            batch_first=True,
                            padding_value=pad_token)

        return log

    augmented_log = iterate_over_traces(log=augmented_log,
                                        subset='training',
                                        create_tensors=create_tensors,
                                        add_special_tokens=add_special_tokens,
                                        pad_sequences=pad_sequences,
                                        pad_token=pad_token)
    augmented_log = iterate_over_traces(log=augmented_log,
                                        subset='validation',
                                        create_tensors=create_tensors,
                                        add_special_tokens=add_special_tokens,
                                        pad_sequences=pad_sequences,
                                        pad_token=pad_token)

    # Transform the log in place
    def wrap_into_torch_dataset(log, subset, batch_size):
        for prefix in log[subset + '_prefixes_and_suffixes']['activities']['prefixes'].keys():
            a_p = log[subset + '_prefixes_and_suffixes']['activities']['prefixes'][prefix].unsqueeze(2)
            t_p = log[subset + '_prefixes_and_suffixes']['times']['prefixes'][prefix].unsqueeze(2)
            a_s_i = log[subset + '_prefixes_and_suffixes']['activities']['suffixes']['input'][prefix].unsqueeze(2)
            t_s_i = log[subset + '_prefixes_and_suffixes']['times']['suffixes']['input'][prefix].unsqueeze(2)
            a_s_t = log[subset + '_prefixes_and_suffixes']['activities']['suffixes']['target'][prefix].long()
            t_s_t = log[subset + '_prefixes_and_suffixes']['times']['suffixes']['target'][prefix].unsqueeze(2)

            if subset == 'training':
                d_l = DataLoader(dataset=TensorDataset(a_p, t_p, a_s_i, t_s_i, a_s_t, t_s_t),
                                 pin_memory=True,
                                 shuffle=True,
                                 batch_size=batch_size)
            else:
                d_l = DataLoader(dataset=TensorDataset(a_p, t_p, a_s_i, t_s_i, a_s_t, t_s_t),
                                 pin_memory=True,
                                 shuffle=False,
                                 batch_size=batch_size)

            log[subset + '_torch_data_loaders'][prefix] = d_l

        keys = list(log[subset + '_prefixes_and_suffixes']['activities']['prefixes'].keys())
        for prefix in keys:
            del log[subset + '_prefixes_and_suffixes']['activities']['prefixes'][prefix]
            del log[subset + '_prefixes_and_suffixes']['times']['prefixes'][prefix]
            del log[subset + '_prefixes_and_suffixes']['activities']['suffixes']['input'][prefix]
            del log[subset + '_prefixes_and_suffixes']['times']['suffixes']['input'][prefix]
            del log[subset + '_prefixes_and_suffixes']['activities']['suffixes']['target'][prefix]
            del log[subset + '_prefixes_and_suffixes']['times']['suffixes']['target'][prefix]

    if to_wrap_into_torch_dataset:
        augmented_log['training_torch_data_loaders']= {}
        augmented_log['validation_torch_data_loaders'] = {}
        wrap_into_torch_dataset(log=augmented_log, subset='training', batch_size=training_batch_size)
        wrap_into_torch_dataset(log=augmented_log, subset='validation', batch_size=validation_batch_size)

    return augmented_log


# Sequences for the transformer models
def create_transformer_augmentation(log,
                                    pad_token=0,
                                    training_batch_size=None,
                                    validation_batch_size=None):

    augmented_log = deepcopy(log)
    augmented_log['training_augmented_traces'] = {'ids': [],
                                                  'activities': {'input': [], 'target': []},
                                                  'times': {'input': [], 'target': []}}
    augmented_log['validation_augmented_traces'] = {'ids': [],
                                                    'activities': {'input': [], 'target': []},
                                                    'times': {'input': [], 'target': []}}

    def iterate_over_traces(log,
                            subset='training',
                            pad_token=0):

        with torch.no_grad():
            dynamic_tensification = torch.tensor

            # Defining the tokens for [SOS] and [EOS]:
            sos_token = log['vocabulary_size'] + 1
            eos_token = log['vocabulary_size'] + 2
            log['sos_token'] = sos_token
            log['eos_token'] = eos_token
            log['pad_token'] = pad_token

            # Very interesting research question:
            time_attribute_padding_value = 0.0

            # For each original trace in the log:
            for trace in tqdm(log[subset + '_traces'], desc='creating ' + subset + ' prefixes of ' + augmented_log['id'] + ' for transformer'):
                max_prefix = len(trace['activities'])

                activities_sequence_input = [sos_token] + trace['activities']
                times_sequence_input = [time_attribute_padding_value] + trace['times']
                activities_sequence_target = trace['activities'] + [eos_token]
                times_sequence_target = trace['times'] + [time_attribute_padding_value]

                log[subset + '_augmented_traces']['activities']['input'].append(
                    dynamic_tensification(activities_sequence_input))
                log[subset + '_augmented_traces']['activities']['target'].append(
                    dynamic_tensification(activities_sequence_target))
                log[subset + '_augmented_traces']['times']['input'].append(
                    dynamic_tensification(times_sequence_input))
                log[subset + '_augmented_traces']['times']['target'].append(
                    dynamic_tensification(times_sequence_target))
                log[subset + '_augmented_traces']['ids'].append(trace['id'])

            # Create a suffix tensor (in each prefix list) which has the max length for sure:
            an_activities_sequence_input = log[subset + '_augmented_traces']['activities']['input'][0]
            a_times_sequence_input = log[subset + '_augmented_traces']['times']['input'][0]
            an_activities_sequence_target = log[subset + '_augmented_traces']['activities']['target'][0]
            a_times_sequence_target = log[subset + '_augmented_traces']['times']['target'][0]

            # Max length is extended by one to cover [EOS] (target) and [SOS] (input)
            max_length = log['longest_trace_length'] + 1

            extension = pad_token * torch.ones((max_length - an_activities_sequence_input.size(0)))
            log[subset + '_augmented_traces']['activities']['input'][0] = torch.cat((an_activities_sequence_input, extension))
            extension = time_attribute_padding_value * torch.ones((max_length - a_times_sequence_input.size(0)))
            log[subset + '_augmented_traces']['times']['input'][0] = torch.cat((a_times_sequence_input, extension))
            extension = pad_token * torch.ones((max_length - an_activities_sequence_target.size(0)))
            log[subset + '_augmented_traces']['activities']['target'][0] = torch.cat((an_activities_sequence_target, extension))
            extension = time_attribute_padding_value * torch.ones((max_length - a_times_sequence_target.size(0)))
            log[subset + '_augmented_traces']['times']['target'][0] = torch.cat((a_times_sequence_target, extension))

            log[subset + '_augmented_traces']['activities']['input'] = torch.nn.utils.rnn.pad_sequence(
                log[subset + '_augmented_traces']['activities']['input'],
                batch_first=True,
                padding_value=pad_token)
            log[subset + '_augmented_traces']['activities']['target'] = torch.nn.utils.rnn.pad_sequence(
                log[subset + '_augmented_traces']['activities']['target'],
                batch_first=True,
                padding_value=pad_token)
            log[subset + '_augmented_traces']['times']['input'] = torch.nn.utils.rnn.pad_sequence(
                log[subset + '_augmented_traces']['times']['input'],
                batch_first=True,
                padding_value=pad_token)
            log[subset + '_augmented_traces']['times']['target'] = torch.nn.utils.rnn.pad_sequence(
                log[subset + '_augmented_traces']['times']['target'],
                batch_first=True,
                padding_value=pad_token)

            return log

    augmented_log = iterate_over_traces(log=augmented_log,
                                        subset='training',
                                        pad_token=pad_token)
    augmented_log = iterate_over_traces(log=augmented_log,
                                        subset='validation',
                                        pad_token=pad_token)

    # Transform the log in place
    def wrap_into_torch_dataset(log, subset, batch_size):
        a_s_i = log[subset + '_augmented_traces']['activities']['input'].unsqueeze(2)
        t_s_i = log[subset + '_augmented_traces']['times']['input'].unsqueeze(2)
        a_s_t = log[subset + '_augmented_traces']['activities']['target'].long()
        t_s_t = log[subset + '_augmented_traces']['times']['target'].unsqueeze(2)

        if subset == 'training':
            d_l = DataLoader(dataset=TensorDataset(a_s_i, t_s_i, a_s_t, t_s_t),
                             pin_memory=True,
                             shuffle=True,
                             batch_size=batch_size)
                             #persistent_workers=True,
                             #num_workers=2,
                             #prefetch_factor=4)
        else:
            d_l = DataLoader(dataset=TensorDataset(a_s_i, t_s_i, a_s_t, t_s_t),
                             pin_memory=True,
                             shuffle=False,
                             batch_size=batch_size)
                             #persistent_workers=True,
                             #num_workers=2,
                             #prefetch_factor=4)

        log[subset + '_torch_data_loaders'] = d_l

        del log[subset + '_augmented_traces']['activities']['input']
        del log[subset + '_augmented_traces']['times']['input']
        del log[subset + '_augmented_traces']['activities']['target']
        del log[subset + '_augmented_traces']['times']['target']

    augmented_log['training_torch_data_loaders']= {}
    augmented_log['validation_torch_data_loaders'] = {}
    wrap_into_torch_dataset(log=augmented_log, subset='training', batch_size=training_batch_size)
    wrap_into_torch_dataset(log=augmented_log, subset='validation', batch_size=validation_batch_size)

    return augmented_log


def create_structured_log(log, log_name=None, to_normalise=True):
    processed_log = {'id': str(log_name),
                     'longest_trace_length': int(0),
                     'traces': [],
                     'activity_label_to_category_index': {},
                     'category_index_to_activity_label': {},
                     'nb_traces': len(log),
                     'max_time_value': float("-inf"),
                     'min_time_value': float("inf"),
                     'vocabulary_size': None}

    # It will create categories starting form 1 on
    # Category 0 is always reserved for [PAD]
    # Additional special tokens will be added later
    activity_index = 0

    for trace in log:
        processed_trace = {'id': trace.attributes['concept:name'], 'activities': [], 'times': []}
        last_datetime = None

        if len(trace) > processed_log['longest_trace_length']: processed_log['longest_trace_length'] = len(trace)

        for event in trace:
            if event['concept:name'] in processed_log['activity_label_to_category_index'].keys():
                processed_trace['activities'].append(
                    processed_log['activity_label_to_category_index'][event['concept:name']])
            else:
                activity_index += 1
                processed_log['activity_label_to_category_index'][event['concept:name']] = activity_index
                processed_log['category_index_to_activity_label'][activity_index] = event['concept:name']
                processed_trace['activities'].append(
                    processed_log['activity_label_to_category_index'][event['concept:name']])

            if last_datetime is not None:
                diff = (event['time:timestamp'] - last_datetime).total_seconds() # decided to be on a second scale
                processed_trace['times'].append(diff)
                last_datetime = event['time:timestamp']

                if processed_log['max_time_value'] < diff:
                    processed_log['max_time_value'] = diff
                if processed_log['min_time_value'] > diff:
                    processed_log['min_time_value'] = diff
            else:
                init_value = 0.0
                if processed_log['max_time_value'] < init_value:
                    processed_log['max_time_value'] = init_value
                if processed_log['min_time_value'] > init_value:
                    processed_log['min_time_value'] = init_value
                processed_trace['times'].append(init_value)
                last_datetime = event['time:timestamp']

        processed_log['traces'].append(processed_trace)
    processed_log['vocabulary_size'] = activity_index

    if to_normalise:
        # in-place normalisation:
        normalise(processed_log)

    return processed_log


# in-place normalisation:
def normalise(processed_log):
    for trace in processed_log['traces']:
        normalized_times = []
        for time in trace['times']:
            normalized_times.append((time - processed_log['min_time_value']) / (
                    processed_log['max_time_value'] - processed_log['min_time_value']))
        trace['times'] = normalized_times


# in-place denormalisation:
def denormalise(prediction, in_days=True):
    # in-place denormalisation:
    def denorm(s):
        if in_days:
            return (float(s) * (float(prediction['max_time_value']) - float(prediction['min_time_value'])) + float(prediction['min_time_value'])) / 60 / 60 / 24
        else:
            return float(s) * (float(prediction['max_time_value']) - float(prediction['min_time_value'])) + float(prediction['min_time_value'])

    def rec_walk_list(l):
        for index, item in enumerate(l):
            if isinstance(l[index], list): # is a list
                rec_walk_list(l[index])
            else: # is a scalar
                l[index] = denorm(l[index])

    def rec_walk_dict(d):
        for k, v in d.items():
            if isinstance(v, dict):
                rec_walk_dict(v)
            else: # is a list
                rec_walk_list(v)

    prediction['times_denormalised'] = deepcopy(prediction['times'])
    rec_walk_dict(prediction['times_denormalised'])


def download_logs(logs_meta, logs_dir):
    if not os.path.exists(logs_dir): os.makedirs(logs_dir)

    for log_name in tqdm(logs_meta, desc="downloading logs"):
        remotefile = urlopen(logs_meta[log_name])
        blah = remotefile.info()['Content-Disposition']
        value, params = cgi.parse_header(blah)
        filename = params["filename"]
        urlretrieve(logs_meta[log_name], os.path.join(logs_dir, filename))

    for file_name in os.listdir(logs_dir):
        if file_name.endswith('.gz'):
            gz_file_name = os.path.join(logs_dir, file_name)
            with gzip.open(gz_file_name, 'rb') as f_in:
                with open(gz_file_name[:-3], 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)


def create_distributions(logs_dir, log_name=None):
    distributions = {}
    logs = {}

    for file_name in sorted(os.listdir(logs_dir)):
        if file_name.endswith('.xes'):
            xes_file_name = os.path.join(logs_dir, file_name)
            log = xes_importer.apply(xes_file_name)
            distribution = {}
            for i in log:
                if len(i) not in distribution:
                    distribution[len(i)] = 1
                else:
                    distribution[len(i)] += 1
            distribution = dict(sorted(distribution.items()))
            if log_name is not None:
                if log_name == 'file_name':
                    distributions[str(file_name)] = distribution
                else:
                    pass
            else:
                distributions[log.attributes['concept:name']] = distribution
            logs[str(file_name)] = log
        elif file_name.endswith('.csv'):
            log = pm4py.format_dataframe(pd.read_csv(os.path.join(logs_dir, file_name), sep=','),
                                         case_id='CaseID',
                                         activity_key='ActivityID',
                                         timestamp_key='CompleteTimestamp')
            log = pm4py.convert_to_event_log(log)
            distribution = {}
            for i in log:
                if len(i) not in distribution:
                    distribution[len(i)] = 1
                else:
                    distribution[len(i)] += 1
            distribution = dict(sorted(distribution.items()))
            distributions[str(file_name)] = distribution
            logs[str(file_name)] = log

    return dict(sorted(distributions.items())), dict(sorted(logs.items()))


def create_length_distribution_figure(distributions):
    b = 3  # number of columns
    a = math.ceil(len(distributions) / b)  # number of rows
    c = 1  # initialize plot counter

    distributions = dict(sorted(distributions.items()))

    fig = plt.figure(figsize=(18, 12))
    fig.tight_layout()

    for log_name, log_distribution in distributions.items():
        plt.subplot(a, b, c)
        plt.title('{}'.format(log_name))
        plt.xlabel('trace length')
        plt.ylabel('# traces')
        plt.bar(log_distribution.keys(), log_distribution.values())
        fig.gca().get_xaxis().set_major_locator(plt.MaxNLocator(integer=True))
        c = c + 1

    fig.subplots_adjust(wspace=0.2)
    fig.subplots_adjust(hspace=0.6)
    fig.savefig('case_legth_statistics.png', dpi=fig.dpi)


def create_count_figure(counts):
    b = 3  # number of columns
    a = math.ceil(len(counts) / b)  # number of rows
    c = 1  # initialize plot counter

    fig = plt.figure(figsize=(18, 12))
    fig.tight_layout()

    counts = dict(sorted(counts.items()))

    for log_name, count in counts.items():
        plt.subplot(a, b, c)
        plt.title('{}'.format(log_name))
        plt.xlabel('prefix')
        plt.ylabel('# traces longer than prefix')
        plt.bar(count.keys(), count.values())
        # plt.plot(count.keys(), count.values())
        fig.gca().get_xaxis().set_major_locator(plt.MaxNLocator(integer=True))
        c = c + 1

    fig.subplots_adjust(wspace=0.2)
    fig.subplots_adjust(hspace=0.6)

    return fig


def count_nb_traces_longer_than_prefix(trace_length_distributions, min_prefix=2, max_prefix=200, delete_zero_prefixes=True):
    counts = {}

    for log_name, log_distribution in trace_length_distributions.items():
        counts[log_name] = {}
        log_distribution = dict(sorted(log_distribution.items()))

        for prefix in range(min_prefix, max_prefix):
            counts[log_name][prefix] = 0
            keys = log_distribution.keys()

            for key in keys:
                if key > prefix:
                    counts[log_name][prefix] += log_distribution[key]

    # delete zero-count prefixes:
    if delete_zero_prefixes:
        for log_name in counts.keys():
            prefixes_to_delete =[]
            for prefix in counts[log_name].keys():
                if counts[log_name][prefix] == 0: prefixes_to_delete.append(prefix)
            for prefix in prefixes_to_delete:
                del counts[log_name][prefix]

    return counts


def suffix_evaluation_sum_dls(suffix_evaluation_result, model_type):
    suffix_evaluation_sum_dls_result = {model_type: {}}

    # have a fix order of event logs on the figure:
    suffix_evaluation_result[model_type] = dict(sorted(suffix_evaluation_result[model_type].items()))

    for log_name, prefix_dls_distribution in suffix_evaluation_result[model_type].items():
        suffix_evaluation_sum_dls_result[model_type][log_name] = {'dls_per_prefix': {},
                                                                  'dls': prefix_dls_distribution['dls'],
                                                                  'nb_worst_situs': prefix_dls_distribution[
                                                                      'nb_worst_situs'],
                                                                  'nb_all_situs': prefix_dls_distribution['nb_all_situs']}

        for prefix, dls_scores in prefix_dls_distribution['dls_per_prefix'].items():
            if len(dls_scores):
                suffix_evaluation_sum_dls_result[model_type][log_name]['dls_per_prefix'][prefix] = sum(dls_scores) / len(
                    dls_scores)
            else:
                # for now passing:
                pass
                # suffix_evaluation_sum_dls_result[model_type][log_name]['dls_per_prefix'][prefix] = 0.0

    return suffix_evaluation_sum_dls_result


def suffix_evaluation_sum_mae(suffix_evaluation_result, model_type):
    suffix_evaluation_sum_mae_result = {model_type: {}}

    # have a fix order of event logs on the figure:
    suffix_evaluation_result[model_type] = dict(sorted(suffix_evaluation_result[model_type].items()))

    for log_name, prefix_dls_distribution in suffix_evaluation_result[model_type].items():
        suffix_evaluation_sum_mae_result[model_type][log_name] = {'mae_per_prefix': {},
                                                                  'mae': prefix_dls_distribution['mae'],
                                                                  'nb_worst_situs': prefix_dls_distribution['nb_worst_situs'],
                                                                  'nb_all_situs': prefix_dls_distribution['nb_all_situs']}

        for prefix, dls_scores in prefix_dls_distribution['mae_per_prefix'].items():
            if len(dls_scores):
                suffix_evaluation_sum_mae_result[model_type][log_name]['mae_per_prefix'][prefix] = sum(dls_scores) / len(
                    dls_scores)
            else:
                # for now passing:
                pass
                # suffix_evaluation_sum_mae_result[model_type][log_name]['dls_per_prefix'][prefix] = 0.0

    return suffix_evaluation_sum_mae_result


def suffix_evaluation_sum_mae_denormalised(suffix_evaluation_result, model_type):
    suffix_evaluation_sum_mae_result = {model_type: {}}

    # have a fix order of event logs on the figure:
    suffix_evaluation_result[model_type] = dict(sorted(suffix_evaluation_result[model_type].items()))

    for log_name, prefix_dls_distribution in suffix_evaluation_result[model_type].items():
        suffix_evaluation_sum_mae_result[model_type][log_name] = {'mae_denormalised_per_prefix': {},
                                                                  'mae_denormalised': prefix_dls_distribution['mae_denormalised'],
                                                                  'nb_worst_situs': prefix_dls_distribution['nb_worst_situs'],
                                                                  'nb_all_situs': prefix_dls_distribution['nb_all_situs']}

        for prefix, dls_scores in prefix_dls_distribution['mae_denormalised_per_prefix'].items():
            if len(dls_scores):
                suffix_evaluation_sum_mae_result[model_type][log_name]['mae_denormalised_per_prefix'][prefix] = sum(dls_scores) / len(dls_scores)
            else:
                # for now passing:
                pass
                # suffix_evaluation_sum_mae_result[model_type][log_name]['dls_per_prefix'][prefix] = 0.0

    return suffix_evaluation_sum_mae_result


def create_prefix_dls_distribution_figure(suffix_evaluation_result, model_type):
    suffix_evaluation_sum_result = suffix_evaluation_sum_dls(suffix_evaluation_result, model_type)
    
    b = 3  # number of columns
    a = math.ceil(len(suffix_evaluation_sum_result[model_type]) / b)  # number of rows
    c = 1  # initialize plot counter

    fig = plt.figure(figsize=(18, 12))
    fig.tight_layout()

    for log_name, prefix_dls_distribution in suffix_evaluation_sum_result[model_type].items():
        plt.subplot(a, b, c)
        plt.title('{}'.format(log_name))
        plt.xlabel('prefix length')
        plt.ylim(0.0, 1.0)
        plt.ylabel('dls')
        plt.text(0.05,
                 0.85,
                 "model:" + model_type + ",dls:" + prefix_dls_distribution['dls'] + ",worst:" + str(prefix_dls_distribution['nb_worst_situs']) + ",all:" + str(prefix_dls_distribution['nb_all_situs']) + ",w/a:" + "{:.2f}".format(prefix_dls_distribution['nb_worst_situs']/prefix_dls_distribution['nb_all_situs']))
        plt.bar(prefix_dls_distribution['dls_per_prefix'].keys(), prefix_dls_distribution['dls_per_prefix'].values())
        fig.gca().get_xaxis().set_major_locator(plt.MaxNLocator(integer=True))
        c += 1

    fig.subplots_adjust(wspace=0.2)
    fig.subplots_adjust(hspace=0.6)
    
    return fig
