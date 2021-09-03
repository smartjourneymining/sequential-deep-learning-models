import os
import json
import data_preprocessing
import models
import numpy as np
import torch
import random
import datetime
import argparse
import socket
from similarity.damerau import Damerau # pip install strsim
import utils
from fast_transformers.masking import TriangularCausalMask
from copy import deepcopy


def iterate_over_generated_suffixes(predictions=None):
    damerau = Damerau()
    nb_worst_situs = 0
    nb_all_situs = 0
    average_damerau_levenshtein_similarity = 0
    average_mae = 0.0
    average_mae_denormalised = 0.0
    eos_token = predictions['eos_token']
    predictions['dls'] = {}
    predictions['mae'] = {}
    predictions['mae_denormalised'] = {}
    predictions['evaluated_ids'] = {}

    for prefix in predictions['ids'].keys():
        predictions['dls'][prefix] = []
        predictions['mae'][prefix] = []
        predictions['mae_denormalised'][prefix] = []
        predictions['evaluated_ids'][prefix] = []

        for i in range(len(predictions['activities']['suffixes']['target'][prefix])):
            target_activity_suffix_padded = predictions['activities']['suffixes']['target'][prefix][i]
            target_activity_suffix = []
            target_time_suffix_padded = predictions['times']['suffixes']['target'][prefix][i]
            target_time_suffix = []
            target_time_denormalised_suffix_padded = predictions['times_denormalised']['suffixes']['target'][prefix][i]
            target_time_denormalised_suffix = []

            # The situ when the original trace was shorter than the given prefix is:
            # prepared for an expanded dimensionality too:
            if (len(target_activity_suffix_padded) > 0) and isinstance(target_activity_suffix_padded[0], list):
                if target_activity_suffix_padded[0][0] == predictions['pad_token']:
                    continue
            else:
                if target_activity_suffix_padded[0] == predictions['pad_token']:
                    continue

            # prepared for an expanded dimensionality too:
            # for the activities:
            target_length = 0
            if (len(target_activity_suffix_padded) > 0) and isinstance(target_activity_suffix_padded[0], list):
                for j in target_activity_suffix_padded:
                    if j[0] != eos_token:
                        target_activity_suffix.append(j[0])
                        target_length += 1
                    else:
                        break
            else:
                for j in target_activity_suffix_padded:
                    if j != eos_token:
                        target_activity_suffix.append(j)
                        target_length += 1
                    else:
                        break
            # for the times:
            if (len(target_time_suffix_padded) > 0) and isinstance(target_time_suffix_padded[0], list):
                for j in range(target_length):
                    target_time_suffix.append(target_time_suffix_padded[j][0])
                    target_time_denormalised_suffix.append(target_time_denormalised_suffix_padded[j][0])
            else:
                for j in range(target_length):
                    target_time_suffix.append(target_time_suffix_padded[j])
                    target_time_denormalised_suffix.append(target_time_denormalised_suffix_padded[j])

            is_the_worst_situ = True
            prediction_activity_suffix_padded = predictions['activities']['suffixes']['prediction'][prefix][i]
            prediction_activity_suffix = []
            prediction_time_suffix_padded = predictions['times']['suffixes']['prediction'][prefix][i]
            prediction_time_suffix = []
            prediction_time_denormalised_suffix_padded = predictions['times_denormalised']['suffixes']['prediction'][prefix][i]
            prediction_time_denormalised_suffix = []

            # In the worst case it stops at the length of the longest suffix
            # prepared for an expanded dimensionality too:
            # for the activities:
            prediction_length = 0
            if (len(prediction_activity_suffix_padded) > 0) and isinstance(prediction_activity_suffix_padded[0], list):
                for j in prediction_activity_suffix_padded:
                    if j[0] != eos_token:
                        prediction_activity_suffix.append(j[0])
                        prediction_length += 1
                    else:
                        is_the_worst_situ = False
                        break
            else:
                for j in prediction_activity_suffix_padded:
                    if j != eos_token:
                        prediction_activity_suffix.append(j)
                        prediction_length += 1
                    else:
                        is_the_worst_situ = False
                        break
            # for the times:
            if (len(prediction_time_suffix_padded) > 0) and isinstance(prediction_time_suffix_padded[0], list):
                for j in range(prediction_length):
                    prediction_time_suffix.append(prediction_time_suffix_padded[j][0])
                    prediction_time_denormalised_suffix.append(prediction_time_denormalised_suffix_padded[j][0])
            else:
                for j in range(prediction_length):
                    prediction_time_suffix.append(prediction_time_suffix_padded[j])
                    prediction_time_denormalised_suffix.append(prediction_time_denormalised_suffix_padded[j])

            if is_the_worst_situ:
                nb_worst_situs += 1

            # The situ when the suffix had an [EOS] position only and that is perfectly predicted:
            if len(prediction_activity_suffix) == len(target_activity_suffix) == 0:
                damerau_levenshtein_similarity = 1.0
                mae = 0.0
                mae_denormalised = 0.0
            else:
                damerau_levenshtein_similarity = 1.0 - damerau.distance(prediction_activity_suffix, target_activity_suffix) / max(len(prediction_activity_suffix), len(target_activity_suffix))
                if len(target_time_suffix) == 0:
                    sum_target_time_suffix = 0.0
                elif len(target_time_suffix) == 1:
                    sum_target_time_suffix = target_time_suffix[0]
                elif len(target_time_suffix) > 1:
                    sum_target_time_suffix = sum(target_time_suffix)
                if len(prediction_time_suffix) == 0:
                    sum_prediction_time_suffix = 0.0
                elif len(prediction_time_suffix) == 1:
                    sum_prediction_time_suffix = prediction_time_suffix[0]
                elif len(prediction_time_suffix) > 1:
                    sum_prediction_time_suffix = sum(prediction_time_suffix)
                if len(target_time_denormalised_suffix) == 0:
                    sum_target_time_denormalised_suffix = 0.0
                elif len(target_time_denormalised_suffix) == 1:
                    sum_target_time_denormalised_suffix = target_time_denormalised_suffix[0]
                elif len(target_time_denormalised_suffix) > 1:
                    sum_target_time_denormalised_suffix = sum(target_time_denormalised_suffix)
                if len(prediction_time_denormalised_suffix) == 0:
                    sum_prediction_time_denormalised_suffix = 0.0
                elif len(prediction_time_denormalised_suffix) == 1:
                    sum_prediction_time_denormalised_suffix = prediction_time_denormalised_suffix[0]
                elif len(prediction_time_denormalised_suffix) > 1:
                    sum_prediction_time_denormalised_suffix = sum(prediction_time_denormalised_suffix)
                mae = abs(sum_target_time_suffix - sum_prediction_time_suffix)
                mae_denormalised = abs(sum_target_time_denormalised_suffix - sum_prediction_time_denormalised_suffix)

            trace_id = predictions['ids'][prefix][i]
            average_damerau_levenshtein_similarity += damerau_levenshtein_similarity
            average_mae += mae
            average_mae_denormalised += mae_denormalised
            predictions['dls'][prefix].append(damerau_levenshtein_similarity) # in-place editing of the input dictionary
            predictions['mae'][prefix].append(mae)  # in-place editing of the input dictionary
            predictions['mae_denormalised'][prefix].append(mae_denormalised)  # in-place editing of the input dictionary
            predictions['evaluated_ids'][prefix].append(trace_id)  # in-place editing of the input dictionary
            nb_all_situs += 1

    # the list (of the prefix) is created in the beggining hence if it remains empty it is deleted now:
    prefix_to_delete = []
    for prefix in predictions['dls'].keys():
        if len(predictions['dls'][prefix]) == 0:
            prefix_to_delete.append(prefix)
    for prefix in prefix_to_delete:
        del predictions['dls'][prefix]
        del predictions['mae'][prefix]
        del predictions['mae_denormalised'][prefix]
        del predictions['evaluated_ids'][prefix]

    average_damerau_levenshtein_similarity /= nb_all_situs
    average_mae /= nb_all_situs
    average_mae_denormalised /= nb_all_situs
    return average_damerau_levenshtein_similarity, average_mae, average_mae_denormalised, nb_worst_situs, nb_all_situs


def seq_ae_predict(seq_ae_teacher_forcing_ratio, model, model_input_x, model_input_y, temperature=1.0, top_k=None, sample=False):
    prediction = ()

    # TODO prepare it for activity labels only
    # Semi open loop:
    encoder_hidden = model.encoder(model_input_x)[1]
    decoder_hidden = encoder_hidden
    input_sos = (model_input_y[0][:, 0, :].unsqueeze(-1), model_input_y[1][:, 0, :].unsqueeze(-1))
    input_position = input_sos

    for i in range(model_input_y[0].size(1)):
        inter_position, decoder_hidden = model.decoder.cell(model.decoder.value_embedding(input_position), decoder_hidden)
        output_position = model.decoder.readout(inter_position)

        if i == 0:
            a_decoded = utils.generate(output_position[0], temperature=temperature, top_k=top_k, sample=sample)
            prediction = (a_decoded, output_position[1])
        elif i > 0:
            a_decoded = utils.generate(output_position[0], temperature=temperature, top_k=top_k, sample=sample)
            a_pred = torch.cat((prediction[0], a_decoded), dim=1)
            t_pred = torch.cat((prediction[1], output_position[1]), dim=1)
            prediction = (a_pred, t_pred)

        a_inp = a_decoded
        t_inp = output_position[1]
        input_position = (a_inp, t_inp)

    return prediction


def transformer_full_predict(seq_ae_teacher_forcing_ratio, model, model_input_x, model_input_y, temperature=1.0, top_k=None, sample=False):
    # TODO prepare it for activity labels only
    # Semi open loop:
    model_input_x = model.value_embedding(model_input_x)
    model_input_x = model.position_embedding(model_input_x)
    memory = model.self_attentional_block.encoder(model_input_x)
    input_sos = (model_input_y[0][:, 0, :].unsqueeze(-1), model_input_y[1][:, 0, :].unsqueeze(-1))
    input_positions = input_sos

    for i in range(model_input_y[0].size(1)):
        input_positions_embedded = model.value_embedding(input_positions)
        input_positions_embedded = model.position_embedding(input_positions_embedded)
        inter_positions = model.self_attentional_block.decoder(input_positions_embedded, memory, tgt_mask=model.target_lookahead_mask[:input_positions[0].size(1), :input_positions[0].size(1)])
        output_positions = model.readout(inter_positions)
        a_decoded = utils.generate(output_positions[0][:, -1, :].unsqueeze(1), temperature=temperature, top_k=top_k, sample=sample)
        a_pred = torch.cat((input_positions[0], a_decoded), dim=1)
        t_pred = torch.cat((input_positions[1], output_positions[1][:, -1, :].unsqueeze(1)), dim=1)
        input_positions = (a_pred, t_pred)

    prediction = (input_positions[0][:, 1:, :], input_positions[1][:, 1:, :])  # Cut the starting [SOS] position

    return prediction


def rnn_predict(seq_ae_teacher_forcing_ratio, model, model_input_x, model_input_y, temperature=1.0, top_k=None, sample=False, max_length=None):
    # TODO prepare it for activity labels only
    # init model wiht prefix input:
    prefix = model_input_x[0].size(1)
    inp_p = (model_input_x[0], model_input_x[1])
    output = model(inp_p)
    a_decoded = utils.generate(output[0][:, -1, :].unsqueeze(1), temperature=temperature, top_k=top_k, sample=sample)
    prediction = (a_decoded, output[1][:, -1, :].unsqueeze(1))
    input_position = prediction

    # Semi open loop:
    for i in range(prefix, max_length - 1):
        output_position = model(input_position)
        a_decoded = utils.generate(output_position[0], temperature=temperature, top_k=top_k, sample=sample)
        a_pred = torch.cat((prediction[0], a_decoded), dim=1)
        t_pred = torch.cat((prediction[1], output_position[1]), dim=1)
        prediction = (a_pred, t_pred)
        a_inp = a_decoded
        t_inp = output_position[1]
        input_position = (a_inp, t_inp)

    return prediction


def gpt_predict(model, inp, target, ids, temperature=1.0, top_k=None, sample=False, min_prefix=2):
    gpt_min_prefix = min_prefix + 1 # For fair comparison with other model architectures due to the [SOS] at first position

    batch_of_traces = {'activities': {'prefixes': {}, 'suffixes': {'target': {}, 'prediction': {}}},
                       'times': {'prefixes': {}, 'suffixes': {'target': {}, 'prediction': {}}},
                       'ids': {}}

    for prefix in range(gpt_min_prefix, inp[0].size(1) + 1): # Plus one: we want the very last [EOS] predictions too
        # TODO prepare it for activity labels only

        # Initial input condition:
        input_condition = (inp[0][:, :prefix, :], inp[1][:, :prefix, :])

        # Open-loop inference:
        for i in range(inp[0].size(1) + 1 - prefix): # very important -> the complete output is: [SOS] trace [EOS]
            causal_mask = TriangularCausalMask(input_condition[0].size(1), device=next(model.parameters()).device)
            output = model(input_condition, attn_mask=causal_mask)
            a_decoded = utils.generate(output[0][:, -1, :].unsqueeze(1), temperature=temperature, top_k=top_k, sample=sample)
            a_pred = torch.cat((input_condition[0], a_decoded), dim=1)
            t_pred = torch.cat((input_condition[1], output[1][:, -1, :].unsqueeze(1)), dim=1)
            input_condition = (a_pred, t_pred)

            del causal_mask
            del output
            del input_condition

        prediction = (input_condition[0][:, 1:, :], input_condition[1][:, 1:, :]) # Cut the starting [SOS] position
        real_prefix = prefix - 1 # without the [SOS] position

        batch_of_traces['activities']['prefixes'][real_prefix] = prediction[0][:, :real_prefix, :].tolist()
        batch_of_traces['times']['prefixes'][real_prefix] = prediction[1][:, :real_prefix, :].tolist()
        batch_of_traces['activities']['suffixes']['prediction'][real_prefix] = prediction[0][:, real_prefix:, :].tolist()
        batch_of_traces['times']['suffixes']['prediction'][real_prefix] = prediction[1][:, real_prefix:, :].tolist()
        batch_of_traces['activities']['suffixes']['target'][real_prefix] = target[0][:, real_prefix:].tolist()
        batch_of_traces['times']['suffixes']['target'][real_prefix] = target[1][:, real_prefix:, :].tolist()
        # real ids:
        batch_of_traces['ids'][real_prefix] = ids

        del prediction

    return batch_of_traces


def wavenet_predict(model, inp, target, ids, temperature=1.0, top_k=None, sample=False, min_prefix=2):
    gpt_min_prefix = min_prefix + 1 # For fair comparison with other model architectures due to the [SOS] at first position

    batch_of_traces = {'activities': {'prefixes': {}, 'suffixes': {'target': {}, 'prediction': {}}},
                       'times': {'prefixes': {}, 'suffixes': {'target': {}, 'prediction': {}}},
                       'ids': {}}

    receptive_field = model.left_padding + 1

    for prefix in range(gpt_min_prefix, inp[0].size(1) + 1): # Plus one: we want the very last [EOS] predictions too
        # TODO prepare it for activity labels only

        # Initial input condition:
        input_condition = (inp[0][:, :prefix, :], inp[1][:, :prefix, :])

        # Open-loop inference:
        for i in range(inp[0].size(1) + 1 - prefix): # very important -> the complete output is: [SOS] trace [EOS]
            if input_condition[0].size(1) < receptive_field:
                left_padding = receptive_field - input_condition[0].size(1)
            else:
                left_padding = 0
            output = model((input_condition[0][:, -receptive_field:, :], input_condition[1][:, -receptive_field:, :]), left_padding=left_padding)
            a_decoded = utils.generate(output[0][:, -1, :].unsqueeze(1), temperature=temperature, top_k=top_k, sample=sample)
            a_pred = torch.cat((input_condition[0], a_decoded), dim=1)
            t_pred = torch.cat((input_condition[1], output[1][:, -1, :].unsqueeze(1)), dim=1)
            input_condition = (a_pred, t_pred)

            del output

        prediction = (input_condition[0][:, 1:, :], input_condition[1][:, 1:, :]) # Cut the starting [SOS] position
        real_prefix = prefix - 1 # without the [SOS] position

        batch_of_traces['activities']['prefixes'][real_prefix] = prediction[0][:, :real_prefix, :].tolist()
        batch_of_traces['times']['prefixes'][real_prefix] = prediction[1][:, :real_prefix, :].tolist()
        batch_of_traces['activities']['suffixes']['prediction'][real_prefix] = prediction[0][:, real_prefix:, :].tolist()
        batch_of_traces['times']['suffixes']['prediction'][real_prefix] = prediction[1][:, real_prefix:, :].tolist()
        batch_of_traces['activities']['suffixes']['target'][real_prefix] = target[0][:, real_prefix:].tolist()
        batch_of_traces['times']['suffixes']['target'][real_prefix] = target[1][:, real_prefix:, :].tolist()
        # real ids:
        batch_of_traces['ids'][real_prefix] = deepcopy(ids)

        del prediction
        del input_condition

    return batch_of_traces


def bert_predict(model, inp, target, ids, temperature=1.0, top_k=None, sample=False, min_prefix=2, order='random'):
    bert_min_prefix = min_prefix

    batch_of_traces = {'activities': {'prefixes': {}, 'suffixes': {'target': {}, 'prediction': {}}},
                       'times': {'prefixes': {}, 'suffixes': {'target': {}, 'prediction': {}}},
                       'ids': {}}

    for prefix in range(bert_min_prefix, inp[0].size(1)): # TODO validation of upper limit
        # TODO prepare it for activity labels only

        if order == 'random' or order == 'l2r':
            if order == 'random':
                suffix_indexes = torch.randperm(n=inp[0].size(1) - prefix, device=inp[0].device, dtype=torch.long)
                suffix_indexes = torch.add(suffix_indexes, prefix)
            elif order == 'l2r':
                suffix_indexes = torch.arange(start=prefix, end=inp[0].size(1), device=inp[0].device, dtype=torch.long)

            a_input_condition = inp[0].detach().clone()
            t_input_condition = inp[1].detach().clone()

            # fully mask the suffix as step 0:
            a_input_condition[:, suffix_indexes, :] = float(model.mask_token) * torch.ones(
                (a_input_condition.size(0), suffix_indexes.size(0), a_input_condition.size(2)), device=a_input_condition.device)
            t_input_condition[:, suffix_indexes, :] = float(model.attributes_meta[1]['min_value']) * torch.ones(
                (t_input_condition.size(0), suffix_indexes.size(0), t_input_condition.size(2)), device=t_input_condition.device)

            for suffix_index in suffix_indexes:
                model.mlm.method = 'fix_masks'
                model.mlm.fix_masks = suffix_index

                output = model((a_input_condition, t_input_condition), attn_mask=None)
                a_decoded = utils.generate(output[0][:, suffix_index, :], temperature=temperature, top_k=top_k, sample=sample)
                a_input_condition[:, suffix_index, :] = a_decoded
                t_input_condition[:, suffix_index, :] = output[1][:, suffix_index, :]

            del suffix_indexes

        prediction = (a_input_condition, t_input_condition)
        real_prefix = prefix

        batch_of_traces['activities']['prefixes'][real_prefix] = prediction[0][:, :real_prefix, :].tolist()
        batch_of_traces['times']['prefixes'][real_prefix] = prediction[1][:, :real_prefix, :].tolist()
        batch_of_traces['activities']['suffixes']['prediction'][real_prefix] = prediction[0][:, real_prefix:, :].tolist()
        batch_of_traces['times']['suffixes']['prediction'][real_prefix] = prediction[1][:, real_prefix:, :].tolist()
        batch_of_traces['activities']['suffixes']['target'][real_prefix] = target[0][:, real_prefix:].tolist()
        batch_of_traces['times']['suffixes']['target'][real_prefix] = target[1][:, real_prefix:, :].tolist()
        # real ids:
        batch_of_traces['ids'][real_prefix] = ids

        del a_input_condition
        del t_input_condition
        del prediction

    return batch_of_traces


# loop for transformer models:
def iterate_over_traces_transformer(log_with_traces,
                                    model=None,
                                    device=None,
                                    subset=None):

    # TODO prepare it for activity labels only
    # TODO encode trace ids as integers then that could be tracked

    predictions = {'activities': {'prefixes': {}, 'suffixes': {'target': {}, 'prediction': {}}},
                   'times': {'prefixes': {}, 'suffixes': {'target': {}, 'prediction': {}}},
                   'eos_token': log_with_traces['eos_token'],
                   'sos_token': log_with_traces['sos_token'],
                   'pad_token': log_with_traces['pad_token'],
                   'ids': {},
                   'max_time_value': log_with_traces['max_time_value'],
                   'min_time_value': log_with_traces['min_time_value']}

    data_loader = log_with_traces[subset + '_torch_data_loaders']
    i = 0
    for mini_batch in iter(data_loader):
        if device == 'GPU':
            if model.architecture == 'GPT':
                a_s_i = mini_batch[0].cuda()
                t_s_i = mini_batch[1].cuda()
                a_s_t = mini_batch[2]
                t_s_t = mini_batch[3]
            elif model.architecture == 'BERT':
                a_s_t = mini_batch[2].cuda()
                t_s_t = mini_batch[3].cuda()
        else:
            if model.architecture == 'GPT':
                a_s_i = mini_batch[0]
                t_s_i = mini_batch[1]
                a_s_t = mini_batch[2]
                t_s_t = mini_batch[3]
            elif model.architecture == 'BERT':
                a_s_t = mini_batch[2]
                t_s_t = mini_batch[3]

        if model.architecture == 'GPT':
            prediction = gpt_predict(model=model,
                                     inp=(a_s_i, t_s_i),
                                     target=(a_s_t, t_s_t),
                                     ids=log_with_traces[subset + '_augmented_traces']['ids'][i * mini_batch[0].size(0):(i+1) * mini_batch[0].size(0)])
        elif model.architecture == 'BERT':
            # The variable categorical target is an unsqueezed long so it has to be transformed into an unsqueezed float input:
            prediction = bert_predict(model=model,
                                      inp=(a_s_t.unsqueeze(2).float(), t_s_t),
                                      target=(a_s_t.unsqueeze(2).float(), t_s_t),
                                      ids=log_with_traces[subset + '_augmented_traces']['ids'][
                                          i * mini_batch[0].size(0):(i + 1) * mini_batch[0].size(0)],
                                      order=args.bert_order)

        for prefix in prediction['activities']['prefixes'].keys():
            if prefix not in predictions['activities']['prefixes'].keys():
                predictions['activities']['prefixes'][prefix] = []
                predictions['times']['prefixes'][prefix] = []
                predictions['activities']['suffixes']['prediction'][prefix] = []
                predictions['times']['suffixes']['prediction'][prefix] = []
                predictions['activities']['suffixes']['target'][prefix] = []
                predictions['times']['suffixes']['target'][prefix] = []
                predictions['ids'][prefix] = []

            predictions['activities']['prefixes'][prefix] += deepcopy(prediction['activities']['prefixes'][prefix])
            predictions['times']['prefixes'][prefix] += deepcopy(prediction['times']['prefixes'][prefix])
            predictions['activities']['suffixes']['prediction'][prefix] += deepcopy(prediction['activities']['suffixes']['prediction'][prefix])
            predictions['times']['suffixes']['prediction'][prefix] += deepcopy(prediction['times']['suffixes']['prediction'][prefix])
            predictions['activities']['suffixes']['target'][prefix] += deepcopy(prediction['activities']['suffixes']['target'][prefix])
            predictions['times']['suffixes']['target'][prefix] += deepcopy(prediction['times']['suffixes']['target'][prefix])
            predictions['ids'][prefix] += deepcopy(prediction['ids'][prefix])

        del prediction
        if model.architecture == 'GPT':
            del a_s_i
            del t_s_i
            del a_s_t
            del t_s_t
        elif model.architecture == 'BERT':
            del a_s_t
            del t_s_t
        del mini_batch

    return predictions


def iterate_over_traces_wavenet(log_with_traces,
                                model=None,
                                device=None,
                                subset=None):

    # TODO prepare it for activity labels only
    # TODO encode trace ids as integers then that could be tracked

    predictions = {'activities': {'prefixes': {}, 'suffixes': {'target': {}, 'prediction': {}}},
                   'times': {'prefixes': {}, 'suffixes': {'target': {}, 'prediction': {}}},
                   'eos_token': log_with_traces['eos_token'],
                   'sos_token': log_with_traces['sos_token'],
                   'pad_token': log_with_traces['pad_token'],
                   'ids': {},
                   'max_time_value': log_with_traces['max_time_value'],
                   'min_time_value': log_with_traces['min_time_value']}

    data_loader = log_with_traces[subset + '_torch_data_loaders']
    i = 0
    for mini_batch in iter(data_loader):
        if device == 'GPU':
            if model.architecture == 'GPT':
                a_s_i = mini_batch[0].cuda()
                t_s_i = mini_batch[1].cuda()
                a_s_t = mini_batch[2]
                t_s_t = mini_batch[3]
            elif model.architecture == 'BERT':
                a_s_t = mini_batch[2].cuda()
                t_s_t = mini_batch[3].cuda()
        else:
            if model.architecture == 'GPT':
                a_s_i = mini_batch[0]
                t_s_i = mini_batch[1]
                a_s_t = mini_batch[2]
                t_s_t = mini_batch[3]
            elif model.architecture == 'BERT':
                a_s_t = mini_batch[2]
                t_s_t = mini_batch[3]

        if model.architecture == 'GPT':
            prediction = wavenet_predict(model=model,
                                         inp=(a_s_i, t_s_i),
                                         target=(a_s_t, t_s_t),
                                         ids=log_with_traces[subset + '_augmented_traces']['ids'][i * mini_batch[0].size(0):(i+1) * mini_batch[0].size(0)])

        for prefix in prediction['activities']['prefixes'].keys():
            if prefix not in predictions['activities']['prefixes'].keys():
                predictions['activities']['prefixes'][prefix] = []
                predictions['times']['prefixes'][prefix] = []
                predictions['activities']['suffixes']['prediction'][prefix] = []
                predictions['times']['suffixes']['prediction'][prefix] = []
                predictions['activities']['suffixes']['target'][prefix] = []
                predictions['times']['suffixes']['target'][prefix] = []
                predictions['ids'][prefix] = []

            predictions['activities']['prefixes'][prefix] += deepcopy(prediction['activities']['prefixes'][prefix])
            predictions['times']['prefixes'][prefix] += deepcopy(prediction['times']['prefixes'][prefix])
            predictions['activities']['suffixes']['prediction'][prefix] += deepcopy(
                prediction['activities']['suffixes']['prediction'][prefix])
            predictions['times']['suffixes']['prediction'][prefix] += deepcopy(
                prediction['times']['suffixes']['prediction'][prefix])
            predictions['activities']['suffixes']['target'][prefix] += deepcopy(
                prediction['activities']['suffixes']['target'][prefix])
            predictions['times']['suffixes']['target'][prefix] += deepcopy(
                prediction['times']['suffixes']['target'][prefix])
            predictions['ids'][prefix] += deepcopy(prediction['ids'][prefix])

        del prediction
        if model.architecture == 'GPT':
            del a_s_i
            del t_s_i
            del a_s_t
            del t_s_t
        elif model.architecture == 'BERT':
            del a_s_t
            del t_s_t
        del mini_batch

    return predictions


# loop for encoder-decoder models:
def iterate_over_prefixes_ae(log_with_prefixes,
                             model=None,
                             device=None,
                             subset=None,
                             to_wrap_into_torch_dataset=None):

    # TODO prepare it for activity labels only
    predictions = {'activities': {'prefixes': {}, 'suffixes': {'target': {}, 'prediction': {}}},
                   'times': {'prefixes': {}, 'suffixes': {'target': {}, 'prediction': {}}},
                   'ids': {},
                   'eos_token': log_with_prefixes['eos_token'],
                   'sos_token': log_with_prefixes['sos_token'],
                   'pad_token': log_with_prefixes['pad_token'],
                   'max_time_value': log_with_prefixes['max_time_value'],
                   'min_time_value': log_with_prefixes['min_time_value']}

    if not to_wrap_into_torch_dataset:
        # Not implemented
        pass
    else:
        prefixes = list(log_with_prefixes[subset + '_torch_data_loaders'].keys())

        for prefix in prefixes:
            data_loader = log_with_prefixes[subset + '_torch_data_loaders'][prefix]

            a_p_mini_batches = []
            t_p_mini_batches = []
            a_s_t_mini_batches = []
            t_s_t_mini_batches = []
            prediction_a_mini_batches = []
            prediction_t_mini_batches = []

            for mini_batch in iter(data_loader):
                if device == 'GPU':
                    a_p = mini_batch[0].cuda()
                    t_p = mini_batch[1].cuda()
                    a_s_i = mini_batch[2].cuda()
                    t_s_i = mini_batch[3].cuda()
                    a_s_t = mini_batch[4]
                    t_s_t = mini_batch[5]
                else:
                    a_p = mini_batch[0]
                    t_p = mini_batch[1]
                    a_s_i = mini_batch[2]
                    t_s_i = mini_batch[3]
                    a_s_t = mini_batch[4]
                    t_s_t = mini_batch[5]

                prediction = seq_ae_predict(seq_ae_teacher_forcing_ratio=0.0,
                                            model=model,
                                            model_input_x=(a_p, t_p),
                                            model_input_y=(a_s_i, t_s_i))

                a_p_mini_batches += a_p.tolist()
                t_p_mini_batches += t_p.tolist()
                a_s_t_mini_batches += a_s_t.tolist()
                t_s_t_mini_batches += t_s_t.tolist()
                prediction_a_mini_batches += prediction[0].tolist()
                prediction_t_mini_batches += prediction[1].tolist()

                del a_p
                del t_p
                del a_s_i
                del t_s_i
                del a_s_t
                del t_s_t
                del mini_batch
                del prediction

            del data_loader

            predictions['activities']['prefixes'][prefix] = a_p_mini_batches
            predictions['times']['prefixes'][prefix] = t_p_mini_batches
            predictions['activities']['suffixes']['target'][prefix] = a_s_t_mini_batches
            predictions['times']['suffixes']['target'][prefix] = t_s_t_mini_batches
            predictions['activities']['suffixes']['prediction'][prefix] = prediction_a_mini_batches
            predictions['times']['suffixes']['prediction'][prefix] = prediction_t_mini_batches
            predictions['ids'][prefix] = log_with_prefixes[subset + '_prefixes_and_suffixes']['ids'][prefix]

        return predictions


# loop for transformer_full model:
def iterate_over_prefixes_transformer_full(log_with_prefixes,
                                           model=None,
                                           device=None,
                                           subset=None,
                                           to_wrap_into_torch_dataset=None):

    # TODO prepare it for activity labels only
    predictions = {'activities': {'prefixes': {}, 'suffixes': {'target': {}, 'prediction': {}}},
                   'times': {'prefixes': {}, 'suffixes': {'target': {}, 'prediction': {}}},
                   'ids': {},
                   'eos_token': log_with_prefixes['eos_token'],
                   'sos_token': log_with_prefixes['sos_token'],
                   'pad_token': log_with_prefixes['pad_token'],
                   'max_time_value': log_with_prefixes['max_time_value'],
                   'min_time_value': log_with_prefixes['min_time_value']}

    if not to_wrap_into_torch_dataset:
        # Not implemented
        pass
    else:
        prefixes = list(log_with_prefixes[subset + '_torch_data_loaders'].keys())

        for prefix in prefixes:
            data_loader = log_with_prefixes[subset + '_torch_data_loaders'][prefix]

            a_p_mini_batches = []
            t_p_mini_batches = []
            a_s_t_mini_batches = []
            t_s_t_mini_batches = []
            prediction_a_mini_batches = []
            prediction_t_mini_batches = []

            for mini_batch in iter(data_loader):
                if device == 'GPU':
                    a_p = mini_batch[0].cuda()
                    t_p = mini_batch[1].cuda()
                    a_s_i = mini_batch[2].cuda()
                    t_s_i = mini_batch[3].cuda()
                    a_s_t = mini_batch[4]
                    t_s_t = mini_batch[5]
                else:
                    a_p = mini_batch[0]
                    t_p = mini_batch[1]
                    a_s_i = mini_batch[2]
                    t_s_i = mini_batch[3]
                    a_s_t = mini_batch[4]
                    t_s_t = mini_batch[5]

                prediction = transformer_full_predict(seq_ae_teacher_forcing_ratio=0.0,
                                                      model=model,
                                                      model_input_x=(a_p, t_p),
                                                      model_input_y=(a_s_i, t_s_i))

                a_p_mini_batches += a_p.tolist()
                t_p_mini_batches += t_p.tolist()
                a_s_t_mini_batches += a_s_t.tolist()
                t_s_t_mini_batches += t_s_t.tolist()
                prediction_a_mini_batches += prediction[0].tolist()
                prediction_t_mini_batches += prediction[1].tolist()

                del a_p
                del t_p
                del a_s_i
                del t_s_i
                del a_s_t
                del t_s_t
                del mini_batch
                del prediction

            del data_loader

            predictions['activities']['prefixes'][prefix] = a_p_mini_batches
            predictions['times']['prefixes'][prefix] = t_p_mini_batches
            predictions['activities']['suffixes']['target'][prefix] = a_s_t_mini_batches
            predictions['times']['suffixes']['target'][prefix] = t_s_t_mini_batches
            predictions['activities']['suffixes']['prediction'][prefix] = prediction_a_mini_batches
            predictions['times']['suffixes']['prediction'][prefix] = prediction_t_mini_batches
            predictions['ids'][prefix] = log_with_prefixes[subset + '_prefixes_and_suffixes']['ids'][prefix]

        return predictions


# loop for the rnn model:
def iterate_over_prefixes_rnn(log_with_prefixes,
                              model=None,
                              device=None,
                              subset=None,
                              to_wrap_into_torch_dataset=None,
                              max_length=None):

    # TODO prepare it for activity labels only
    predictions = {'activities': {'prefixes': {}, 'suffixes': {'target': {}, 'prediction': {}}},
                   'times': {'prefixes': {}, 'suffixes': {'target': {}, 'prediction': {}}},
                   'ids': {},
                   'eos_token': log_with_prefixes['eos_token'],
                   'sos_token': log_with_prefixes['sos_token'],
                   'pad_token': log_with_prefixes['pad_token'],
                   'max_time_value': log_with_prefixes['max_time_value'],
                   'min_time_value': log_with_prefixes['min_time_value']}

    if not to_wrap_into_torch_dataset:
        # Not implemented
        pass
    else:
        prefixes = list(log_with_prefixes[subset + '_torch_data_loaders'].keys())

        for prefix in prefixes:
            data_loader = log_with_prefixes[subset + '_torch_data_loaders'][prefix]

            a_p_mini_batches = []
            t_p_mini_batches = []
            a_s_t_mini_batches = []
            t_s_t_mini_batches = []
            prediction_a_mini_batches = []
            prediction_t_mini_batches = []

            for mini_batch in iter(data_loader):
                if device == 'GPU':
                    a_p = mini_batch[0].cuda()
                    t_p = mini_batch[1].cuda()
                    a_s_i = mini_batch[2].cuda()
                    t_s_i = mini_batch[3].cuda()
                    a_s_t = mini_batch[4]
                    t_s_t = mini_batch[5]
                else:
                    a_p = mini_batch[0]
                    t_p = mini_batch[1]
                    a_s_i = mini_batch[2]
                    t_s_i = mini_batch[3]
                    a_s_t = mini_batch[4]
                    t_s_t = mini_batch[5]

                prediction = rnn_predict(seq_ae_teacher_forcing_ratio=0.0,
                                         model=model,
                                         model_input_x=(a_p, t_p),
                                         model_input_y=(a_s_i, t_s_i),
                                         max_length=max_length)

                a_p_mini_batches += a_p.tolist()
                t_p_mini_batches += t_p.tolist()
                a_s_t_mini_batches += a_s_t.tolist()
                t_s_t_mini_batches += t_s_t.tolist()
                prediction_a_mini_batches += prediction[0].tolist()
                prediction_t_mini_batches += prediction[1].tolist()

                del a_p
                del t_p
                del a_s_i
                del t_s_i
                del a_s_t
                del t_s_t
                del mini_batch
                del prediction

            del data_loader

            predictions['activities']['prefixes'][prefix] = a_p_mini_batches
            predictions['times']['prefixes'][prefix] = t_p_mini_batches
            predictions['activities']['suffixes']['target'][prefix] = a_s_t_mini_batches
            predictions['times']['suffixes']['target'][prefix] = t_s_t_mini_batches
            predictions['activities']['suffixes']['prediction'][prefix] = prediction_a_mini_batches
            predictions['times']['suffixes']['prediction'][prefix] = prediction_t_mini_batches
            predictions['ids'][prefix] = log_with_prefixes[subset + '_prefixes_and_suffixes']['ids'][prefix]

        return predictions


def generate_suffixes_ae(checkpoint_file, log_file, args, path):
    log_with_prefixes = data_preprocessing.create_prefixes(log_file,
                                                           min_prefix=2,
                                                           create_tensors=True,
                                                           add_special_tokens=True,
                                                           pad_sequences=True,
                                                           pad_token=args.pad_token,
                                                           to_wrap_into_torch_dataset=args.to_wrap_into_torch_dataset,
                                                           training_batch_size=args.training_batch_size,
                                                           validation_batch_size=args.validation_batch_size,
                                                           single_position_target=False)

    del log_with_prefixes['training_torch_data_loaders']

    # [EOS], [SOS], [PAD]
    nb_special_tokens = 3
    attributes_meta = {
        0: {'nb_special_tokens': nb_special_tokens, 'vocabulary_size': log_with_prefixes['vocabulary_size']},
        1: {'min_value': 0.0, 'max_value': 1.0}}

    vars(args)['sos_token'] = log_with_prefixes['sos_token']
    vars(args)['eos_token'] = log_with_prefixes['eos_token']
    vars(args)['nb_special_tokens'] = nb_special_tokens
    vars(args)['vocabulary_size'] = log_with_prefixes['vocabulary_size']
    vars(args)['longest_trace_length'] = log_with_prefixes['longest_trace_length']

    # All traces are longer by one position due to the closing [EOS]:
    max_length = log_with_prefixes['longest_trace_length'] + 1
    vars(args)['max_length'] = max_length

    with open(os.path.join(path, 'evaluation_parameters.json'), 'a') as fp:
        json.dump(vars(args), fp)
        fp.write('\n')

    model = models.SequentialAutoEncoder(hidden_size=args.hidden_dim,
                                         num_layers=args.n_layers,
                                         dropout_prob=args.dropout_prob,
                                         vocab_size=attributes_meta[0]['vocabulary_size'],
                                         attributes_meta=attributes_meta,
                                         time_attribute_concatenated=args.time_attribute_concatenated,
                                         pad_token=args.pad_token,
                                         nb_special_tokens=attributes_meta[0]['nb_special_tokens'])

    device = torch.device('cpu')
    model.load_state_dict(torch.load(checkpoint_file, map_location=device)['model_state_dict'])

    if args.device == 'GPU':
        model.cuda()

    model.eval()
    with torch.no_grad():
        predictions = iterate_over_prefixes_ae(log_with_prefixes=log_with_prefixes,
                                               model=model,
                                               device=args.device,
                                               subset='validation',
                                               to_wrap_into_torch_dataset=args.to_wrap_into_torch_dataset)
        return predictions


def generate_suffixes_transformer_full(checkpoint_file, log_file, args, path):
    log_with_prefixes = data_preprocessing.create_prefixes(log_file,
                                                           min_prefix=2,
                                                           create_tensors=True,
                                                           add_special_tokens=True,
                                                           pad_sequences=True,
                                                           pad_token=args.pad_token,
                                                           to_wrap_into_torch_dataset=args.to_wrap_into_torch_dataset,
                                                           training_batch_size=args.training_batch_size,
                                                           validation_batch_size=args.validation_batch_size,
                                                           single_position_target=False)

    del log_with_prefixes['training_torch_data_loaders']

    # [EOS], [SOS], [PAD]
    nb_special_tokens = 3
    attributes_meta = {
        0: {'nb_special_tokens': nb_special_tokens, 'vocabulary_size': log_with_prefixes['vocabulary_size']},
        1: {'min_value': 0.0, 'max_value': 1.0}}

    vars(args)['sos_token'] = log_with_prefixes['sos_token']
    vars(args)['eos_token'] = log_with_prefixes['eos_token']
    vars(args)['nb_special_tokens'] = nb_special_tokens
    vars(args)['vocabulary_size'] = log_with_prefixes['vocabulary_size']
    vars(args)['longest_trace_length'] = log_with_prefixes['longest_trace_length']

    # All traces are longer by one position due to the closing [EOS]:
    max_length = log_with_prefixes['longest_trace_length'] + 1
    vars(args)['max_length'] = max_length

    with open(os.path.join(path, 'evaluation_parameters.json'), 'a') as fp:
        json.dump(vars(args), fp)
        fp.write('\n')

    model = models.TransformerAutoEncoder(d_model=args.hidden_dim,
                                          sequence_length=max_length,
                                          n_layers=args.n_layers,
                                          n_heads=args.n_heads,
                                          d_query=int(args.hidden_dim / args.n_heads),
                                          dropout_prob=args.dropout_prob,
                                          vocab_size=attributes_meta[0]['vocabulary_size'],
                                          pad_token=args.pad_token,
                                          attributes_meta=attributes_meta,
                                          time_attribute_concatenated=args.time_attribute_concatenated,
                                          nb_special_tokens=attributes_meta[0]['nb_special_tokens'])

    device = torch.device('cpu')
    model.load_state_dict(torch.load(checkpoint_file, map_location=device)['model_state_dict'])

    if args.device == 'GPU':
        model.cuda()

    model.eval()
    with torch.no_grad():
        predictions = iterate_over_prefixes_transformer_full(log_with_prefixes=log_with_prefixes,
                                                             model=model,
                                                             device=args.device,
                                                             subset='validation',
                                                             to_wrap_into_torch_dataset=args.to_wrap_into_torch_dataset)
        del log_with_prefixes
        return predictions


def generate_suffixes_rnn(checkpoint_file, log_file, args, path):
    log_with_prefixes = data_preprocessing.create_prefixes(log_file,
                                                           min_prefix=2,
                                                           create_tensors=True,
                                                           add_special_tokens=True,
                                                           pad_sequences=True,
                                                           pad_token=args.pad_token,
                                                           to_wrap_into_torch_dataset=args.to_wrap_into_torch_dataset,
                                                           training_batch_size=args.training_batch_size,
                                                           validation_batch_size=args.validation_batch_size,
                                                           single_position_target=False)

    del log_with_prefixes['training_torch_data_loaders']

    # [EOS], [SOS], [PAD]
    nb_special_tokens = 3
    attributes_meta = {
        0: {'nb_special_tokens': nb_special_tokens, 'vocabulary_size': log_with_prefixes['vocabulary_size']},
        1: {'min_value': 0.0, 'max_value': 1.0}}

    vars(args)['sos_token'] = log_with_prefixes['sos_token']
    vars(args)['eos_token'] = log_with_prefixes['eos_token']
    vars(args)['nb_special_tokens'] = nb_special_tokens
    vars(args)['vocabulary_size'] = log_with_prefixes['vocabulary_size']
    vars(args)['longest_trace_length'] = log_with_prefixes['longest_trace_length']

    # All traces are longer by one position due to the closing [EOS]:
    max_length = log_with_prefixes['longest_trace_length'] + 1
    vars(args)['max_length'] = max_length

    with open(os.path.join(path, 'evaluation_parameters.json'), 'a') as fp:
        json.dump(vars(args), fp)
        fp.write('\n')

    model = models.SequentialDecoder(hidden_size=args.hidden_dim,
                                     num_layers=args.n_layers,
                                     dropout_prob=args.dropout_prob,
                                     vocab_size=attributes_meta[0]['vocabulary_size'],
                                     attributes_meta=attributes_meta,
                                     time_attribute_concatenated=args.time_attribute_concatenated,
                                     pad_token=args.pad_token,
                                     nb_special_tokens=attributes_meta[0]['nb_special_tokens'])

    device = torch.device('cpu')
    model.load_state_dict(torch.load(checkpoint_file, map_location=device)['model_state_dict'])

    if args.device == 'GPU':
        model.cuda()

    model.eval()
    with torch.no_grad():
        predictions = iterate_over_prefixes_rnn(log_with_prefixes=log_with_prefixes,
                                                model=model,
                                                device=args.device,
                                                subset='validation',
                                                to_wrap_into_torch_dataset=args.to_wrap_into_torch_dataset,
                                                max_length=max_length)
        del log_with_prefixes
        return predictions


def generate_suffixes_rnn_full(checkpoint_file, log_file, args, path):
    log_with_prefixes = data_preprocessing.create_prefixes(log_file,
                                                           min_prefix=2,
                                                           create_tensors=True,
                                                           add_special_tokens=True,
                                                           pad_sequences=True,
                                                           pad_token=args.pad_token,
                                                           to_wrap_into_torch_dataset=args.to_wrap_into_torch_dataset,
                                                           training_batch_size=args.training_batch_size,
                                                           validation_batch_size=args.validation_batch_size,
                                                           single_position_target=False)

    del log_with_prefixes['training_torch_data_loaders']

    # [EOS], [SOS], [PAD], [MASK]
    nb_special_tokens = 4
    attributes_meta = {
        0: {'nb_special_tokens': nb_special_tokens, 'vocabulary_size': log_with_prefixes['vocabulary_size']},
        1: {'min_value': 0.0, 'max_value': 1.0}}

    vars(args)['sos_token'] = log_with_prefixes['sos_token']
    vars(args)['eos_token'] = log_with_prefixes['eos_token']
    vars(args)['mask_token'] = log_with_prefixes['mask_token']
    vars(args)['nb_special_tokens'] = nb_special_tokens
    vars(args)['vocabulary_size'] = log_with_prefixes['vocabulary_size']
    vars(args)['longest_trace_length'] = log_with_prefixes['longest_trace_length']

    # All traces are longer by one position due to the closing [EOS]:
    max_length = log_with_prefixes['longest_trace_length'] + 1
    vars(args)['max_length'] = max_length

    with open(os.path.join(path, 'evaluation_parameters.json'), 'a') as fp:
        json.dump(vars(args), fp)
        fp.write('\n')

    model = models.SequentialDecoder(hidden_size=args.hidden_dim,
                                     num_layers=args.n_layers,
                                     dropout_prob=args.dropout_prob,
                                     vocab_size=attributes_meta[0]['vocabulary_size'],
                                     attributes_meta=attributes_meta,
                                     time_attribute_concatenated=args.time_attribute_concatenated,
                                     pad_token=args.pad_token,
                                     nb_special_tokens=attributes_meta[0]['nb_special_tokens'],
                                     architecture='GPT')

    device = torch.device('cpu')
    model.load_state_dict(torch.load(checkpoint_file, map_location=device)['model_state_dict'])

    if args.device == 'GPU':
        model.cuda()

    model.eval()
    with torch.no_grad():
        predictions = iterate_over_prefixes_rnn(log_with_prefixes=log_with_prefixes,
                                                model=model,
                                                device=args.device,
                                                subset='validation',
                                                to_wrap_into_torch_dataset=args.to_wrap_into_torch_dataset,
                                                max_length=max_length)
        del log_with_prefixes
        return  predictions


def generate_suffixes_transformer(checkpoint_file, log_file, args, path):
    augmented_log = data_preprocessing.create_transformer_augmentation(log_file,
                                                                       pad_token=args.pad_token,
                                                                       training_batch_size=args.training_batch_size,
                                                                       validation_batch_size=args.validation_batch_size)

    del augmented_log['training_torch_data_loaders']

    # [EOS], [SOS], [PAD] and [MASK]
    nb_special_tokens = 4
    attributes_meta = {0: {'nb_special_tokens': nb_special_tokens, 'vocabulary_size': augmented_log['vocabulary_size']},
                       1: {'min_value': 0.0, 'max_value': 1.0}}

    # [MASK] special token is about to be the largest integer of our vocabulary:
    mask_token = max(augmented_log['sos_token'], augmented_log['eos_token']) + 1
    vars(args)['mask_token'] = mask_token
    vars(args)['sos_token'] = augmented_log['sos_token']
    vars(args)['eos_token'] = augmented_log['eos_token']
    vars(args)['nb_special_tokens'] = nb_special_tokens
    vars(args)['vocabulary_size'] = augmented_log['vocabulary_size']
    vars(args)['longest_trace_length'] = augmented_log['longest_trace_length']

    # All traces are longer by one position due to the closing [EOS]:
    max_length = augmented_log['longest_trace_length'] + 1
    vars(args)['max_length'] = max_length

    with open(os.path.join(path, 'evaluation_parameters.json'), 'a') as fp:
        json.dump(vars(args), fp)
        fp.write('\n')

    model = models.Transformer(d_model=args.hidden_dim,
                               sequence_length=max_length,
                               n_layers=args.n_layers,
                               n_heads=args.n_heads,
                               d_query=int(args.hidden_dim / args.n_heads),
                               dropout_prob=args.dropout_prob,
                               attention_dropout_prob=args.dropout_prob,
                               mlm_prob=args.mlm_masking_prob,
                               vocab_size=attributes_meta[0]['vocabulary_size'],
                               pad_token=args.pad_token,
                               mask_token=mask_token,
                               sos_token=augmented_log['sos_token'],
                               eos_token=augmented_log['eos_token'],
                               architecture=args.architecture,
                               attributes_meta=attributes_meta,
                               time_attribute_concatenated=args.time_attribute_concatenated,
                               nb_special_tokens=attributes_meta[0]['nb_special_tokens'])

    device = torch.device('cpu')
    model.load_state_dict(torch.load(checkpoint_file, map_location=device)['model_state_dict'])

    if args.device == 'GPU':
        model.cuda()

    model.eval()
    with torch.no_grad():
        predictions = iterate_over_traces_transformer(log_with_traces=augmented_log,
                                                      model=model,
                                                      device=args.device,
                                                      subset='validation')
        del augmented_log
        return predictions


def generate_suffixes_wavenet(checkpoint_file, log_file, args, path):
    augmented_log = data_preprocessing.create_transformer_augmentation(log_file,
                                                                       pad_token=args.pad_token,
                                                                       training_batch_size=args.training_batch_size,
                                                                       validation_batch_size=args.validation_batch_size)

    del augmented_log['training_torch_data_loaders']

    # [EOS], [SOS], [PAD] and [MASK]
    nb_special_tokens = 4
    attributes_meta = {0: {'nb_special_tokens': nb_special_tokens, 'vocabulary_size': augmented_log['vocabulary_size']},
                       1: {'min_value': 0.0, 'max_value': 1.0}}

    # [MASK] special token is about to be the largest integer of our vocabulary:
    mask_token = max(augmented_log['sos_token'], augmented_log['eos_token']) + 1
    vars(args)['mask_token'] = mask_token
    vars(args)['sos_token'] = augmented_log['sos_token']
    vars(args)['eos_token'] = augmented_log['eos_token']
    vars(args)['nb_special_tokens'] = nb_special_tokens
    vars(args)['vocabulary_size'] = augmented_log['vocabulary_size']
    vars(args)['longest_trace_length'] = augmented_log['longest_trace_length']

    # All traces are longer by one position due to the closing [EOS]:
    max_length = augmented_log['longest_trace_length'] + 1
    vars(args)['max_length'] = max_length

    with open(os.path.join(path, 'evaluation_parameters.json'), 'a') as fp:
        json.dump(vars(args), fp)
        fp.write('\n')

    model = models.WaveNet(hidden_size=args.hidden_dim,
                           n_layers=args.n_layers,
                           dropout_prob=args.dropout_prob,
                           pad_token=args.pad_token,
                           sos_token=augmented_log['sos_token'],
                           eos_token=augmented_log['eos_token'],
                           mask_token=mask_token,
                           vocab_size=attributes_meta[0]['vocabulary_size'],
                           attributes_meta=attributes_meta,
                           time_attribute_concatenated=args.time_attribute_concatenated,
                           nb_special_tokens=attributes_meta[0]['nb_special_tokens'],
                           architecture='GPT')

    device = torch.device('cpu')
    model.load_state_dict(torch.load(checkpoint_file, map_location=device)['model_state_dict'])

    if args.device == 'GPU':
        model.cuda()

    model.eval()
    with torch.no_grad():
        predictions = iterate_over_traces_wavenet(log_with_traces=augmented_log,
                                                  model=model,
                                                  device=args.device,
                                                  subset='validation')
        del augmented_log
        return predictions


def generate(datetime, model_type, args):
    path = os.path.join('results', model_type)

    # Walk through all the log-result directories:
    for log_directory in sorted(os.scandir(path), key=lambda x: x.name):
        if not os.path.isdir(log_directory): continue

        log_path = log_directory.path
        checkpoint_path = os.path.join(log_path, 'checkpoints')
        checkpoint_files = [f for f in os.listdir(checkpoint_path) if os.path.isfile(os.path.join(checkpoint_path, f))]
        matching_checkpoint_files = [s for s in checkpoint_files if datetime in s]
        checkpoint_file = matching_checkpoint_files[0]
        split_log_file_path = os.path.join(log_path, 'split_log_' + datetime + '.json')

        with open(split_log_file_path) as f_in:
            log_file = json.load(f_in)

        if model_type == 'ae':
            predictions = generate_suffixes_ae(os.path.join(checkpoint_path, checkpoint_file), log_file, args, log_path)
        elif model_type == 'GPT' or model_type == 'BERT':
            predictions = generate_suffixes_transformer(os.path.join(checkpoint_path, checkpoint_file), log_file, args, log_path)
        elif model_type == 'rnn':
            predictions = generate_suffixes_rnn(os.path.join(checkpoint_path, checkpoint_file), log_file, args, log_path)
        elif model_type == 'ae_gan':
            predictions = generate_suffixes_ae(os.path.join(checkpoint_path, checkpoint_file), log_file, args, log_path)
        elif model_type == 'rnn_full':
            predictions = generate_suffixes_rnn_full(os.path.join(checkpoint_path, checkpoint_file), log_file, args, log_path)
        elif model_type == 'transformer_full':
            predictions = generate_suffixes_transformer_full(os.path.join(checkpoint_path, checkpoint_file), log_file, args, log_path)
        elif model_type == 'wavenet':
            predictions = generate_suffixes_wavenet(os.path.join(checkpoint_path, checkpoint_file), log_file, args, log_path)

        del log_file

        predictions['datetime'] = datetime
        predictions['log'] = log_directory.name

        with open(os.path.join(log_path, 'suffix_generation_result_' + str(datetime) + '.json'), 'w') as fp:
            json.dump(predictions, fp)

        del predictions


def evaluate_generation(datetime, model_type):
    path = os.path.join('results', model_type)

    # Walk through all the log-result directories:
    for log_directory in sorted(os.scandir(path), key=lambda x: x.name):
        if not os.path.isdir(log_directory): continue

        log_path = log_directory.path
        print('This is evaluation of: ' + log_path)
        with open(os.path.join(log_path, 'suffix_generation_result_' + str(datetime) + '.json')) as f_in:
            predictions = json.load(f_in)

        # in-place time attribute denormalisation:
        data_preprocessing.denormalise(predictions)

        results = {model_type: {}}
        results[model_type][predictions['log']] = {}

        average_damerau_levenshtein_similarity, average_mae, average_mae_denormalised, nb_worst_situs, nb_all_situs = iterate_over_generated_suffixes(predictions=predictions)

        results[model_type][predictions['log']]['dls'] = "{:.4f}".format(average_damerau_levenshtein_similarity)
        results[model_type][predictions['log']]['mae'] = "{:.4f}".format(average_mae)
        results[model_type][predictions['log']]['mae_denormalised'] = "{:.4f}".format(average_mae_denormalised)
        results[model_type][predictions['log']]['nb_worst_situs'] = nb_worst_situs
        results[model_type][predictions['log']]['nb_all_situs'] = nb_all_situs
        results[model_type][predictions['log']]['dls_per_prefix'] = predictions['dls']
        results[model_type][predictions['log']]['mae_per_prefix'] = predictions['mae']
        results[model_type][predictions['log']]['mae_denormalised_per_prefix'] = predictions['mae_denormalised']
        results[model_type][predictions['log']]['id_per_prefix'] = predictions['evaluated_ids']

        with open(os.path.join(log_path, 'suffix_evaluation_result_dls_mae_' + str(datetime) + '.json'), 'w') as fp:
            json.dump(results[model_type][predictions['log']], fp)

        del results
        del predictions

    # Merge evaluation results (per model type):
    suffix_evaluation_results = {model_type: {}}
    for log_directory in os.scandir(path):
        if not os.path.isdir(log_directory): continue

        log_path = log_directory.path
        with open(os.path.join(log_path, 'suffix_evaluation_result_dls_mae_' + str(datetime) + '.json')) as f_in:
            suffix_evaluation_result = json.load(f_in)

        suffix_evaluation_results[model_type][log_directory.name] = suffix_evaluation_result

    with open(os.path.join(path, 'suffix_evaluation_result_dls_mae_' + str(datetime) + '.json'), 'w') as fp:
        json.dump(suffix_evaluation_results, fp)

    del suffix_evaluation_results
    del suffix_evaluation_result


def main(args, dt_object):
    if not args.random:
        # RANDOM SEEDs:
        random_seed = args.random_seed
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        rng = np.random.default_rng(seed=random_seed)
        torch.backends.cudnn.deterministic = True
        random.seed(a=args.random_seed)

    '''
    model_type = 'ae'
    datetime = '202107022200'
    vars(args)['datetime'] = datetime
    print('This is evaluation of: ' + args.datetime)
    print('This is evaluation of: ' + model_type)
    if args.suffix_generation:
        generate(datetime=datetime, model_type=model_type, args=args)
    if args.dls_evaluation:
        evaluate_generation(datetime, model_type)
    
    model_type = 'ae_gan'
    datetime = '202108061546'
    vars(args)['datetime'] = datetime
    print('This is evaluation of: ' + args.datetime)
    print('This is evaluation of: ' + model_type)
    if args.suffix_generation:
        generate(datetime=datetime, model_type=model_type, args=args)
    if args.dls_evaluation:
        evaluate_generation(datetime, model_type)
    
    model_type = 'GPT'
    datetime = '202107211858'
    vars(args)['datetime'] = datetime
    vars(args)['architecture'] = model_type
    print('This is evaluation of: ' + args.datetime)
    print('This is evaluation of: ' + model_type)
    if args.suffix_generation:
        generate(datetime=datetime, model_type=model_type, args=args)
    if args.dls_evaluation:
        evaluate_generation(datetime, model_type)
    
    model_type = 'BERT'
    datetime = '202107211809'
    vars(args)['datetime'] = datetime
    vars(args)['architecture'] = model_type
    print('This is evaluation of: ' + args.datetime)
    print('This is evaluation of: ' + model_type)
    if args.suffix_generation:
        generate(datetime=datetime, model_type=model_type, args=args)
    if args.dls_evaluation:
        evaluate_generation(datetime, model_type)
    
    
    model_type = 'rnn'
    datetime = '202107230132'
    vars(args)['datetime'] = datetime
    vars(args)['architecture'] = model_type
    print('This is evaluation of: ' + args.datetime)
    print('This is evaluation of: ' + model_type)
    if args.suffix_generation:
        generate(datetime=datetime, model_type=model_type, args=args)
    if args.dls_evaluation:
        evaluate_generation(datetime, model_type)
    
    model_type = 'transformer_full'
    datetime = '202108072316'
    vars(args)['datetime'] = datetime
    vars(args)['architecture'] = model_type
    print('This is evaluation of: ' + args.datetime)
    print('This is evaluation of: ' + model_type)
    if args.suffix_generation:
        generate(datetime=datetime, model_type=model_type, args=args)
    if args.dls_evaluation:
        evaluate_generation(datetime, model_type)
    '''

    model_type = 'wavenet'
    datetime = '202108140006'
    vars(args)['datetime'] = datetime
    vars(args)['architecture'] = model_type
    print('This is evaluation of: ' + args.datetime)
    print('This is evaluation of: ' + model_type)
    if args.suffix_generation:
        generate(datetime=datetime, model_type=model_type, args=args)
    if args.dls_evaluation:
        evaluate_generation(datetime, model_type)


if __name__ == '__main__':
    dt_object = datetime.datetime.now()
    print(dt_object)

    parser = argparse.ArgumentParser()
    parser.add_argument('--datetime', help='datetime', default=dt_object.strftime("%Y%m%d%H%M"), type=str)
    parser.add_argument('--hidden_dim', help='hidden state dimensions', default=128, type=int)
    parser.add_argument('--n_layers', help='number of layers', default=4, type=int)
    parser.add_argument('--n_heads', help='number of heads', default=4, type=int)
    parser.add_argument('--training_batch_size', help='number of training samples in mini-batch', default=512, type=int)
    parser.add_argument('--validation_batch_size', help='number of validation samples in mini-batch', default=512, type=int)
    parser.add_argument('--training_mlm_method', help='training MLM method', default='BERT', type=str)
    parser.add_argument('--validation_mlm_method', help='validation MLM method', default='fix_masks', type=str)  # we would like to end up with some non-stochastic & at least pseudo likelihood metric
    parser.add_argument('--mlm_masking_prob', help='mlm_masking_prob', default=0.15, type=float)
    parser.add_argument('--dropout_prob', help='dropout_prob', default=0.1, type=float)
    parser.add_argument('--validation_split', help='validation_split', default=0.2, type=float)
    parser.add_argument('--dataset', help='dataset', default='', type=str)
    parser.add_argument('--random_seed', help='random_seed', default=1982, type=int)
    parser.add_argument('--random', help='if random', default=False, type=bool)
    parser.add_argument('--gpu', help='gpu', default=0, type=int)
    parser.add_argument('--validation_indexes', help='list of validation_indexes NO SPACES BETWEEN ITEMS!', default='[0,1,4,10,15]', type=str)
    parser.add_argument('--ground_truth_p', help='ground_truth_p', default=0.0, type=float)
    parser.add_argument('--time_attribute_concatenated', help='time_attribute_concatenated', default=False, type=bool)
    parser.add_argument('--device', help='GPU or CPU', default='GPU', type=str)
    parser.add_argument('--pad_token', help='pad_token', default=0, type=int)
    parser.add_argument('--to_wrap_into_torch_dataset', help='to_wrap_into_torch_dataset', default=True, type=bool)
    parser.add_argument('--seq_ae_teacher_forcing_ratio', help='seq_ae_teacher_forcing_ratio', default=0.0, type=float)
    parser.add_argument('--single_position_target', help='single_position_target', default=False, type=bool)
    parser.add_argument('--bert_order', help='BERT inference order: random or l2r', default='random', type=str)
    parser.add_argument('--dls_evaluation', help='DLS', default=True, type=bool)
    parser.add_argument('--suffix_generation', help='suffix generation', default=True, type=bool)

    args = parser.parse_args()

    vars(args)['hostname'] = str(socket.gethostname())

    if args.device == 'GPU':
        torch.cuda.set_device(args.gpu)
        print('This is evaluation at gpu: ' + str(args.gpu))

    main(args, dt_object)
