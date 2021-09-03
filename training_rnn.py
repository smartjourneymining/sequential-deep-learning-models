import models
import torch.nn as nn
import torch
import numpy as np
import datetime
import socket
import json
import argparse
import data_preprocessing
import random
import os
import math
import copy
import utils


def seq_ae_predict(seq_ae_teacher_forcing_ratio, model, model_input_x, model_input_y, temperature=1.0, top_k=None, sample=False):
    prediction = ()
    use_teacher_forcing = True if random.random() < seq_ae_teacher_forcing_ratio else False

    if use_teacher_forcing:
        prediction = model(model_input_x, model_input_y)
    else:
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
                prediction = output_position
            elif i > 0:
                a_p = torch.cat((prediction[0], output_position[0]), dim=1)
                t_p = torch.cat((prediction[1], output_position[1]), dim=1)
                prediction = (a_p, t_p)

            a_i = output_position[0].detach()
            t_i = output_position[1].detach()

            input_position = (utils.generate(a_i, temperature=temperature, top_k=top_k, sample=sample), t_i)

    return prediction


# Training loop for encoder-decoder models:
def iterate_over_prefixes(log_with_prefixes,
                          batch_size=128,
                          model=None,
                          device=None,
                          categorical_criterion=None,
                          regression_criterion=None,
                          subset=None,
                          optimizer=None,
                          lagrange_a=None,
                          to_wrap_into_torch_dataset=None):

    summa_categorical_loss = 0.0
    summa_regression_loss = 0.0
    steps = 0

    if not to_wrap_into_torch_dataset:
        for prefix in log_with_prefixes[subset + '_prefixes_and_suffixes']['activities']['prefixes'].keys():
            activities_prefixes = log_with_prefixes[subset + '_prefixes_and_suffixes']['activities']['prefixes'][prefix]
            times_prefixes = log_with_prefixes[subset + '_prefixes_and_suffixes']['times']['prefixes'][prefix]
            activities_suffixes_target = log_with_prefixes[subset + '_prefixes_and_suffixes']['activities']['suffixes']['target'][prefix]
            times_suffixes_target = log_with_prefixes[subset + '_prefixes_and_suffixes']['times']['suffixes']['target'][prefix]

            if isinstance(activities_prefixes, torch.Tensor):
                nb_iterations = math.ceil(activities_prefixes.size(0) / batch_size)

            for i in range(nb_iterations):
                if subset == 'training':
                    optimizer.zero_grad()

                if device == 'GPU':
                    activities_prefixes_batch = activities_prefixes[i*batch_size:i*batch_size+batch_size, :].unsqueeze(2).cuda()
                    times_prefixes_batch = times_prefixes[i*batch_size:i*batch_size+batch_size, :].unsqueeze(2).cuda()
                    activities_suffixes_target_batch = activities_suffixes_target[i*batch_size:i*batch_size+batch_size, :].squeeze(-1).long().cuda()
                    times_suffixes_target_batch = times_suffixes_target[i * batch_size:i * batch_size + batch_size, :].unsqueeze(1).cuda()
                else:
                    activities_prefixes_batch = activities_prefixes[i * batch_size:i * batch_size + batch_size, :].unsqueeze(2)
                    times_prefixes_batch = times_prefixes[i * batch_size:i * batch_size + batch_size, :].unsqueeze(2)
                    activities_suffixes_target_batch = activities_suffixes_target[i*batch_size:i*batch_size+batch_size, :].squeeze(-1).long()
                    times_suffixes_target_batch = times_suffixes_target[i * batch_size:i * batch_size + batch_size, :].unsqueeze(1)

                prediction = model((activities_prefixes_batch, times_prefixes_batch))

                categorical_criterion.reduction = 'mean'
                # Only the last position
                categorical_loss = categorical_criterion(prediction[0][:, -1, :], activities_suffixes_target_batch)

                # If time attribute and time prediction present:
                if len(prediction) > 1:
                    regression_criterion.reduction = 'mean'
                    # Only the last position
                    regression_loss = regression_criterion(prediction[1][:, -1, :], times_suffixes_target_batch)

                    if subset == 'training':
                        (categorical_loss + lagrange_a * regression_loss).backward()

                    summa_categorical_loss += categorical_loss
                    summa_regression_loss += regression_loss
                else:
                    if subset == 'training':
                        categorical_loss.backward()

                    summa_categorical_loss += categorical_loss

                steps += 1

                if subset == 'training':
                    optimizer.step()

        if len(prediction) > 1:
            return summa_categorical_loss.item() / steps, summa_regression_loss.item() / steps
        else:
            return (summa_categorical_loss.item() / steps, )
    else:
        if subset == 'training':
            prefixes = list(log_with_prefixes[subset + '_torch_data_loaders'].keys())
            random.shuffle(prefixes)
        else:
            prefixes = list(log_with_prefixes[subset + '_torch_data_loaders'].keys())

        for prefix in prefixes:
            data_loader = log_with_prefixes[subset + '_torch_data_loaders'][prefix]
            for mini_batch in iter(data_loader):
                if subset == 'training':
                    optimizer.zero_grad()

                if device == 'GPU':
                    a_p = mini_batch[0].cuda()
                    t_p = mini_batch[1].cuda()
                    a_s_t = mini_batch[4].cuda()
                    t_s_t = mini_batch[5].cuda()
                else:
                    a_p = mini_batch[0]
                    t_p = mini_batch[1]
                    a_s_t = mini_batch[4]
                    t_s_t = mini_batch[5]

                prediction = model(x=(a_p, t_p))

                categorical_criterion.reduction = 'mean'
                # Only the last position
                categorical_loss = categorical_criterion(prediction[0][:, -1, :], a_s_t.squeeze(-1))

                # If time attribute and time prediction present:
                if len(prediction) > 1:
                    regression_criterion.reduction = 'mean'
                    # Only the last position
                    regression_loss = regression_criterion(prediction[1][:, -1, :], t_s_t.squeeze(-1))

                    if subset == 'training':
                        (categorical_loss + lagrange_a * regression_loss).backward()

                    summa_categorical_loss += categorical_loss
                    summa_regression_loss += regression_loss
                else:
                    if subset == 'training':
                        categorical_loss.backward()

                    summa_categorical_loss += categorical_loss

                steps += 1

                if subset == 'training':
                    optimizer.step()

        if len(prediction) > 1:
            return summa_categorical_loss.item() / steps, summa_regression_loss.item() / steps
        else:
            return (summa_categorical_loss.item() / steps, )


def main(args, dt_object):
    if not args.random:
        # RANDOM SEEDs:
        random_seed = args.random_seed
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        rng = np.random.default_rng(seed=random_seed)
        torch.backends.cudnn.deterministic = True
        random.seed(a=args.random_seed)

    # Data prep
    logs_dir = './logs/'

    with open(os.path.join('config', 'logs_meta.json')) as f:
        logs_meta = json.load(f)

    # data_preprocessing.download_logs(logs_meta, logs_dir)
    distributions, logs = data_preprocessing.create_distributions(logs_dir)

    for log_name in logs:
        if args.device == 'GPU':
            print('total GPU memory: ' + str(torch.cuda.get_device_properties(device=args.gpu).total_memory))
            print('allocated GPU memory: ' + str(torch.cuda.memory_allocated(device=args.gpu)))

        processed_log = data_preprocessing.create_structured_log(logs[log_name], log_name=log_name)

        path = os.path.join('results', 'rnn', str(processed_log['id']))
        if not os.path.exists(path): os.makedirs(path)

        vars(args)['dataset'] = str(processed_log['id'])

        if os.path.isdir(os.path.join('split_logs', log_name)):
            for file_name in sorted(os.listdir(os.path.join('split_logs', log_name))):
                if file_name.startswith('split_log_'):
                    split_log_file_name = os.path.join('split_logs', log_name, file_name)
                    with open(split_log_file_name) as f_in:
                        split_log = json.load(f_in)
                    print(split_log_file_name + ' is used as common data')
            del processed_log
        else:
            split_log = data_preprocessing.create_split_log(processed_log, validation_ratio=args.validation_split)

        with open(os.path.join(path, 'split_log_' + dt_object.strftime("%Y%m%d%H%M") + '.json'), 'w') as f:
            json.dump(split_log, f)

        log_with_prefixes = data_preprocessing.create_prefixes(split_log,
                                                               min_prefix=2,
                                                               create_tensors=True,
                                                               add_special_tokens=True,
                                                               pad_sequences=True,
                                                               pad_token=args.pad_token,
                                                               to_wrap_into_torch_dataset=args.to_wrap_into_torch_dataset,
                                                               training_batch_size=args.training_batch_size,
                                                               validation_batch_size=args.validation_batch_size,
                                                               single_position_target=args.single_position_target)

        # [EOS], [SOS], [PAD]
        nb_special_tokens = 3
        attributes_meta = {0: {'nb_special_tokens': nb_special_tokens, 'vocabulary_size': log_with_prefixes['vocabulary_size']},
                           1: {'min_value': 0.0, 'max_value': 1.0}}

        vars(args)['sos_token'] = log_with_prefixes['sos_token']
        vars(args)['eos_token'] = log_with_prefixes['eos_token']
        vars(args)['nb_special_tokens'] = nb_special_tokens
        vars(args)['vocabulary_size'] = log_with_prefixes['vocabulary_size']
        vars(args)['longest_trace_length'] = log_with_prefixes['longest_trace_length']

        # All traces are longer by one position due to the closing [EOS]:
        max_length = log_with_prefixes['longest_trace_length'] + 1
        vars(args)['max_length'] = max_length

        with open(os.path.join(path, 'experiment_parameters.json'), 'a') as fp:
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
                                         architecture='Niek')

        if args.device == 'GPU':
            model.cuda()

        categorical_criterion = nn.CrossEntropyLoss()
        regression_criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=args.training_learning_rate,
                                     weight_decay=args.training_gaussian_process)

        training_log_filename = "training_figures_" + dt_object.strftime("%Y%m%d%H%M") + ".csv"
        with open(os.path.join(path, training_log_filename), "a") as myfile:
            myfile.write('datetime'
                         ',epoch'
                         ',training_loss_activity'
                         ',training_loss_time'
                         ',training_loss'
                         ',validation_loss_activity'
                         ',validation_loss_time'
                         ',validation_loss'
                         ',elapsed_seconds\n')

        # not saving all version of model:
        min_loss_threshold = args.save_criterion_threshold

        if not os.path.exists(os.path.join(path, 'checkpoints')): os.makedirs(os.path.join(path, 'checkpoints'))

        model_to_save = {}
        total_validation_losses = []

        for e in range(args.nb_epoch):
            if not e % 10:
                print('training epoch ' + str(e) + '/' + str(args.nb_epoch) + ' of ' + str(log_with_prefixes['id']))

            model.train()
            dt_object_training_start = datetime.datetime.now()

            training_loss = iterate_over_prefixes(log_with_prefixes=log_with_prefixes,
                                                  batch_size=args.training_batch_size,
                                                  model=model,
                                                  device=args.device,
                                                  categorical_criterion=categorical_criterion,
                                                  regression_criterion=regression_criterion,
                                                  subset='training',
                                                  optimizer=optimizer,
                                                  lagrange_a=args.lagrange_a,
                                                  to_wrap_into_torch_dataset=args.to_wrap_into_torch_dataset)

            dt_object_training_end = datetime.datetime.now()

            training_duration = (dt_object_training_end - dt_object_training_start).total_seconds()

            model.eval()
            with torch.no_grad():
                validation_loss = iterate_over_prefixes(log_with_prefixes=log_with_prefixes,
                                                        batch_size=args.validation_batch_size,
                                                        model=model,
                                                        device=args.device,
                                                        categorical_criterion=categorical_criterion,
                                                        regression_criterion=regression_criterion,
                                                        subset='validation',
                                                        to_wrap_into_torch_dataset=args.to_wrap_into_torch_dataset)

            # an arbritary value:
            validation_loss_fix_masks = (99, 99)
            total_validation_loss_fix_masks = 99

            if len(validation_loss) > 1:
                total_validation_loss = validation_loss[0] + args.lagrange_a * validation_loss[1]
                with open(os.path.join(path, training_log_filename), "a") as myfile:
                    myfile.write(dt_object.strftime("%Y%m%d%H%M")
                                 + ',' + str(e)
                                 + ',' + "{:.4f}".format(training_loss[0])
                                 + ',' + "{:.4f}".format(training_loss[1])
                                 + ',' + "{:.4f}".format(training_loss[0] + args.lagrange_a * training_loss[1])
                                 + ',' + "{:.4f}".format(validation_loss[0])
                                 + ',' + "{:.4f}".format(validation_loss[1])
                                 + ',' + "{:.4f}".format(total_validation_loss)
                                 + ',' + "{:.3f}".format(training_duration)
                                 + ',' + "{:.4f}".format(validation_loss_fix_masks[0])
                                 + ',' + "{:.4f}".format(validation_loss_fix_masks[1])
                                 + ',' + "{:.4f}".format(total_validation_loss_fix_masks)
                                 + '\n')
            else:
                total_validation_loss = validation_loss[0]
                with open(os.path.join(path, training_log_filename), "a") as myfile:
                    myfile.write(dt_object.strftime("%Y%m%d%H%M")
                                 + ',' + str(e)
                                 + ',' + "{:.4f}".format(training_loss[0])
                                 + ',' + 'NA'
                                 + ',' + "{:.4f}".format(training_loss[0])
                                 + ',' + "{:.4f}".format(validation_loss[0])
                                 + ',' + 'NA'
                                 + ',' + "{:.4f}".format(total_validation_loss)
                                 + ',' + "{:.3f}".format(training_duration)
                                 + ',' + "{:.4f}".format(validation_loss_fix_masks[0])
                                 + ',' + "{:.4f}".format(validation_loss_fix_masks[1])
                                 + ',' + "{:.4f}".format(total_validation_loss_fix_masks)
                                 + '\n')

            total_validation_losses.append(total_validation_loss)

            if args.early_stopping:
                early_stopping_var = 50
                if len(total_validation_losses) > early_stopping_var:
                    if np.all(np.array(total_validation_losses)[-(early_stopping_var - 1):] >
                              np.array(total_validation_losses)[-early_stopping_var]):
                        print("early stopping")
                        break

            if total_validation_loss < min_loss_threshold:
                model_to_save['model_state_dict'] = copy.deepcopy(model.state_dict())
                model_to_save['optimizer_state_dict'] = copy.deepcopy(optimizer.state_dict())
                model_to_save['loss'] = copy.deepcopy(total_validation_loss)
                model_to_save['epoch'] = copy.deepcopy(e)
                model_to_save['total_validation_loss'] = copy.deepcopy(total_validation_loss)
                min_loss_threshold = total_validation_loss

        if len(model_to_save) > 0:
            checkpoint_name = 'model-' + "{:.4f}".format(model_to_save['total_validation_loss']) + '_epoch-' + str(model_to_save['epoch']) + '_date-' + dt_object.strftime("%Y%m%d%H%M") + '.pt'

            torch.save({
                'model_state_dict': model_to_save['model_state_dict'],
                'optimizer_state_dict': model_to_save['optimizer_state_dict'],
                'loss': model_to_save['loss'],
            }, os.path.join(path, 'checkpoints', checkpoint_name))


if __name__ == '__main__':
    dt_object = datetime.datetime.now()
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--datetime', help='datetime', default=dt_object.strftime("%Y%m%d%H%M"), type=str)
    parser.add_argument('--hidden_dim', help='hidden state dimensions', default=128, type=int)
    parser.add_argument('--n_layers', help='number of layers', default=4, type=int)
    parser.add_argument('--n_heads', help='number of heads', default=4, type=int)
    parser.add_argument('--nb_epoch', help='training iterations', default=400, type=int)
    parser.add_argument('--training_batch_size', help='number of training samples in mini-batch', default=2560, type=int)
    parser.add_argument('--validation_batch_size', help='number of validation samples in mini-batch', default=2560, type=int)
    parser.add_argument('--training_mlm_method', help='training MLM method', default='BERT', type=str)
    parser.add_argument('--validation_mlm_method', help='validation MLM method', default='fix_masks', type=str) # we would like to end up with some non-stochastic & at least pseudo likelihood metric
    parser.add_argument('--mlm_masking_prob', help='mlm_masking_prob', default=0.15, type=float)
    parser.add_argument('--dropout_prob', help='dropout_prob', default=0.3, type=float)
    parser.add_argument('--training_learning_rate', help='GD learning rate', default=1e-4, type=float)
    parser.add_argument('--training_gaussian_process', help='GP', default=1e-5, type=float)
    parser.add_argument('--validation_split', help='validation_split', default=0.2, type=float)
    parser.add_argument('--dataset', help='dataset', default='', type=str)
    parser.add_argument('--random_seed', help='random_seed', default=1982, type=int)
    parser.add_argument('--random', help='if random', default=True, type=bool)
    parser.add_argument('--gpu', help='gpu', default=6, type=int)
    parser.add_argument('--validation_indexes', help='list of validation_indexes NO SPACES BETWEEN ITEMS!', default='[0,1,4,10,15]', type=str)
    parser.add_argument('--ground_truth_p', help='ground_truth_p', default=0.0, type=float)
    parser.add_argument('--architecture', help='BERT or GPT', default='BERT', type=str)
    parser.add_argument('--time_attribute_concatenated', help='time_attribute_concatenated', default=False, type=bool)
    parser.add_argument('--device', help='GPU or CPU', default='GPU', type=str)
    parser.add_argument('--lagrange_a', help='Langrange multiplier', default=1.0, type=float)
    parser.add_argument('--save_criterion_threshold', help='save_criterion_threshold', default=4.0, type=float)
    parser.add_argument('--pad_token', help='pad_token', default=0, type=int)
    parser.add_argument('--to_wrap_into_torch_dataset', help='to_wrap_into_torch_dataset', default=True, type=bool)
    parser.add_argument('--seq_ae_teacher_forcing_ratio', help='seq_ae_teacher_forcing_ratio', default=1.0, type=float)
    parser.add_argument('--early_stopping', help='early_stopping', default=True, type=bool)
    parser.add_argument('--single_position_target', help='single_position_target', default=True, type=bool)

    args = parser.parse_args()
    
    vars(args)['hostname'] = str(socket.gethostname())
    
    print('This is training of: ' + dt_object.strftime("%Y%m%d%H%M"))

    if args.device == 'GPU':
        torch.cuda.set_device(args.gpu)
        print('This is training at gpu: ' + str(args.gpu))

    main(args, dt_object)