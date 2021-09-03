import torch.nn as nn
import torch
import math
from fast_transformers.builders import TransformerEncoderBuilder
import torch.nn.init as init


# http://nlp.seas.harvard.edu/2018/04/03/attention.html#positional-encoding
# https://github.com/wzlxjtu/PositionalEncoding2D/blob/master/positionalembedding2d.py
class PositionalEncoding(nn.Module):
    def __init__(self, max_len, d_model, dropout_prob=0.0, series_dimensions=1):
        global pe
        super().__init__()
        self.dropout = nn.Dropout(p=dropout_prob)
        self.d_model = d_model
        self.max_len = max_len
        self.series_dimensions = series_dimensions
        
        if self.series_dimensions == 1:
            if d_model % 2 != 0:
                raise ValueError("Cannot use sin/cos positional encoding with "
                                 "odd dim (got dim={:d})".format(d_model))
            pe = torch.zeros(self.max_len, d_model).float()
            pe.require_grad = False
            position = torch.arange(0, self.max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
        elif self.series_dimensions > 1:
            if d_model % 4 != 0:
                raise ValueError("Cannot use sin/cos positional encoding with "
                                 "odd dim (got dim={:d})".format(d_model))
            height = self.series_dimensions
            width = self.max_len
            pe = torch.zeros(d_model, height, width).float()
            pe.require_grad = False
            # Each dimension use half of d_model
            d_model = int(d_model / 2)
            div_term = torch.exp(torch.arange(0., d_model, 2) * -(math.log(10000.0) / d_model))
            pos_w = torch.arange(0., width).unsqueeze(1)
            pos_h = torch.arange(0., height).unsqueeze(1)
            pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
            pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
            pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
            pe[d_model + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
            pe = pe.view(2*d_model, height * width, -1).squeeze(-1) # Flattening it back to 1D series
            pe = pe.transpose(0, 1)
            
        pe = pe.unsqueeze(0) # Extending it by an extra leading dim for the batches
        self.register_buffer('pe', pe)

    # Expecting a flattened (1D) series
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# http://nlp.seas.harvard.edu/2018/04/03/attention.html#embeddings-and-softmax
# TODO study the example with padding_idx part at https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html
# TODO https://github.com/pytorch/pytorch/blob/bac4cfd54d44aa0fbc574e6561b878cb406762cc/torch/nn/modules/sparse.py#L22
# From now on input/output is always a tuple!
# or further attributes should be concatenated as an extra (last) dim of input tensor x
# https://walkwithfastai.com/tab.ae
class Embedding(nn.Module):
    def __init__(self, d_model, vocab_size, dropout_prob=0.0, attributes_meta=None, time_attribute_concatenated=False, pad_token=None):
        super().__init__()

        self.d_model = d_model
        self.attributes_meta = attributes_meta
        self.time_attribute_concatenated = time_attribute_concatenated

        if 1 in self.attributes_meta.keys() and not self.time_attribute_concatenated:
            self.activity_label = nn.Embedding(vocab_size, self.d_model, padding_idx=pad_token)
            self.time_attribute = nn.Linear(1, self.d_model)
        elif 1 in self.attributes_meta.keys() and self.time_attribute_concatenated:
            self.activity_label = nn.Embedding(vocab_size, self.d_model-1, padding_idx=pad_token)
        elif 1 not in self.attributes_meta.keys():
            self.activity_label = nn.Embedding(vocab_size, self.d_model, padding_idx=pad_token)

        self.dropout = nn.Dropout(p=dropout_prob)

    def forward(self, x): # input is always a tuple
        if 1 in self.attributes_meta.keys() and not self.time_attribute_concatenated:
            return self.dropout(self.activity_label(x[0].long()).squeeze(2) + self.time_attribute(x[1])) * math.sqrt(self.d_model)
        elif 1 in self.attributes_meta.keys() and self.time_attribute_concatenated:
            return self.dropout(torch.cat((self.activity_label(x[0].long()).squeeze(2), x[1]), dim=-1)) * math.sqrt(self.d_model)
        elif 1 not in self.attributes_meta.keys():
            return self.dropout(self.activity_label(x[0].long()).squeeze(2)) * math.sqrt(self.d_model)


class ManualEmbedding(nn.Module):
    def __init__(self, d_model, vocab_size, dropout_prob=0.0, attributes_meta=None, time_attribute_concatenated=False):
        super().__init__()

        self.d_model = d_model
        self.attributes_meta = attributes_meta
        self.time_attribute_concatenated = time_attribute_concatenated

        if 1 in self.attributes_meta.keys() and not self.time_attribute_concatenated:
            self.activity_label = nn.Linear(vocab_size, self.d_model)
            self.time_attribute = nn.Linear(1, self.d_model)
        elif 1 in self.attributes_meta.keys() and self.time_attribute_concatenated:
            self.activity_label = nn.Linear(vocab_size, self.d_model-1)
        elif 1 not in self.attributes_meta.keys():
            self.activity_label = nn.Linear(vocab_size, self.d_model)

        self.dropout = nn.Dropout(p=dropout_prob)

    def forward(self, x): # input is always a tuple
        if 1 in self.attributes_meta.keys() and not self.time_attribute_concatenated:
            return self.dropout(self.activity_label(x[0]).squeeze(2) + self.time_attribute(x[1])) * math.sqrt(self.d_model)
        elif 1 in self.attributes_meta.keys() and self.time_attribute_concatenated:
            return self.dropout(torch.cat((self.activity_label(x[0]).squeeze(2), x[1]), dim=-1)) * math.sqrt(self.d_model)
        elif 1 not in self.attributes_meta.keys():
            return self.dropout(self.activity_label(x[0]).squeeze(2)) * math.sqrt(self.d_model)


# http://nlp.seas.harvard.edu/2018/04/03/attention.html#model-architecture
# TODO Weight sharing https://arxiv.org/abs/1608.05859 & https://arxiv.org/abs/1706.03762
# TODO Wrapping sigmoid() could be beneficial for the time_attribute
class Readout(nn.Module):
    def __init__(self, d_model, vocab_size, dropout_prob=0.0, attributes_meta=None, time_attribute_concatenated=False):
        super().__init__()

        self.attributes_meta = attributes_meta
        self.time_attribute_concatenated = time_attribute_concatenated

        if 1 in self.attributes_meta.keys() and not self.time_attribute_concatenated:
            self.activity_label = nn.Linear(d_model, vocab_size)
            self.time_attribute = nn.Linear(d_model, 1)
        elif 1 in self.attributes_meta.keys() and self.time_attribute_concatenated:
            self.activity_label = nn.Linear(d_model-1, vocab_size)
        elif 1 not in self.attributes_meta.keys():
            self.activity_label = nn.Linear(d_model, vocab_size)

        self.dropout = nn.Dropout(p=dropout_prob)

    def forward(self, x):
        if 1 in self.attributes_meta.keys() and not self.time_attribute_concatenated:
            return self.dropout(self.activity_label(x)), self.dropout(self.time_attribute(x))
        elif 1 in self.attributes_meta.keys() and self.time_attribute_concatenated:
            return self.dropout(self.activity_label(x[:, :, :-1])), self.dropout(x[:, :, -1:])
        elif 1 not in self.attributes_meta.keys():
            return (self.dropout(self.activity_label(x)),)  # output is always a tuple


class SequentialEncoder(nn.Module):
    def __init__(self,
                 hidden_size,
                 num_layers,
                 dropout_prob,
                 vocab_size,
                 attributes_meta,
                 time_attribute_concatenated,
                 pad_token,
                 nb_special_tokens):
        super().__init__()

        self.vocab_size = vocab_size + nb_special_tokens

        self.value_embedding = Embedding(d_model=hidden_size,
                                         vocab_size=self.vocab_size,
                                         dropout_prob=dropout_prob,
                                         attributes_meta=attributes_meta,
                                         time_attribute_concatenated=time_attribute_concatenated,
                                         pad_token=pad_token)

        self.cell = nn.LSTM(input_size=hidden_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True,
                            dropout=dropout_prob)

    def forward(self, x):
        return self.cell(self.value_embedding(x))


class SequentialDecoder(nn.Module):
    def __init__(self,
                 hidden_size,
                 num_layers,
                 dropout_prob,
                 vocab_size,
                 attributes_meta,
                 time_attribute_concatenated,
                 pad_token,
                 nb_special_tokens,
                 architecture=None):
        super().__init__()

        self.vocab_size = vocab_size + nb_special_tokens

        if architecture is not None:
            self.architecture = architecture

        self.value_embedding = Embedding(d_model=hidden_size,
                                         vocab_size=self.vocab_size,
                                         dropout_prob=dropout_prob,
                                         attributes_meta=attributes_meta,
                                         time_attribute_concatenated=time_attribute_concatenated,
                                         pad_token=pad_token)

        self.cell = nn.LSTM(input_size=hidden_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True,
                            dropout=dropout_prob)

        self.readout = Readout(d_model=hidden_size,
                               vocab_size=self.vocab_size,
                               dropout_prob=dropout_prob,
                               attributes_meta=attributes_meta,
                               time_attribute_concatenated=time_attribute_concatenated)

    def forward(self, x, init_hidden=None):
        if init_hidden is not None:
            return self.readout(self.cell(self.value_embedding(x), init_hidden)[0])
        else:
            return self.readout(self.cell(self.value_embedding(x))[0])


class SequentialDiscriminator(nn.Module):
    def __init__(self,
                 hidden_size,
                 num_layers,
                 dropout_prob,
                 vocab_size,
                 attributes_meta,
                 time_attribute_concatenated,
                 pad_token,
                 nb_special_tokens):
        super().__init__()

        self.vocab_size = vocab_size + nb_special_tokens

        self.value_embedding = ManualEmbedding(d_model=hidden_size,
                                               vocab_size=self.vocab_size,
                                               dropout_prob=dropout_prob,
                                               attributes_meta=attributes_meta,
                                               time_attribute_concatenated=time_attribute_concatenated)

        self.cell = nn.LSTM(input_size=hidden_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True,
                            dropout=dropout_prob)

        self.readout = nn.Linear(hidden_size, 1)

        self.dropout = nn.Dropout(p=dropout_prob)

    def forward(self, x):
        return self.dropout(self.readout(self.cell(self.value_embedding(x))[0]))


class SequentialAutoEncoder(nn.Module):
    # TODO implement SequentialDecoder.Readout.weights = SequentialDecoder.Embedding.weights = SequentialEncoder.Embedding.weights
    def __init__(self,
                 hidden_size,
                 num_layers,
                 dropout_prob,
                 vocab_size,
                 attributes_meta,
                 time_attribute_concatenated,
                 pad_token,
                 nb_special_tokens):
        super().__init__()

        self.encoder = SequentialEncoder(hidden_size,
                                         num_layers,
                                         dropout_prob,
                                         vocab_size,
                                         attributes_meta,
                                         time_attribute_concatenated,
                                         pad_token,
                                         nb_special_tokens)
        self.decoder = SequentialDecoder(hidden_size,
                                         num_layers,
                                         dropout_prob,vocab_size,
                                         attributes_meta,
                                         time_attribute_concatenated,
                                         pad_token,
                                         nb_special_tokens)

    def forward(self, prefix, suffix):
        # During training it is teacher forcing / supervised learning / closed loop
        # During inference it is open loop
        return self.decoder(suffix, self.encoder(prefix)[1])


class SelfAttentionalBlock(nn.Module):
    def __init__(self,
                 d_model,
                 attention_type="full",
                 n_layers=4,
                 n_heads=4,
                 d_query=32,
                 dropout_prob=0.0,
                 attention_dropout_prob=0.0,
                 intra_transformer_activation='gelu'):
        super().__init__()

        self.d_model = d_model
        self.dropout_prob = dropout_prob
        self.d_query = d_query
        self.n_heads = n_heads
        self.hidden_size = self.n_heads * self.d_query

        self.self_attentional_block = TransformerEncoderBuilder.from_kwargs(
            attention_type=attention_type,
            n_layers=n_layers,
            n_heads=self.n_heads,
            feed_forward_dimensions=self.hidden_size * 4,
            query_dimensions=self.d_query,
            value_dimensions=self.d_query,
            dropout=self.dropout_prob,
            attention_dropout=attention_dropout_prob,
            activation=intra_transformer_activation
        ).get()

    def forward(self, x, attn_mask=None):
        return self.self_attentional_block(x, attn_mask=attn_mask)


class Transformer(nn.Module):
    def __init__(self,
                 d_model,
                 sequence_length,
                 attention_type="full",
                 n_layers=4,
                 n_heads=4,
                 d_query=32,
                 dropout_prob=0.0,
                 attention_dropout_prob=0.0,
                 series_dimensions=1,
                 pad_token=None,
                 mask_token=None,
                 sos_token=None,
                 eos_token=None,
                 mlm_prob=0.15,
                 vocab_size=None,
                 intra_transformer_activation='gelu',
                 architecture=None,
                 attributes_meta=None,
                 time_attribute_concatenated=False,
                 nb_special_tokens=None):
        super().__init__()

        self.d_model = d_model
        self.dropout_prob = dropout_prob
        self.d_query = d_query
        self.n_heads = n_heads
        self.hidden_size = self.n_heads * self.d_query
        self.series_dimensions = series_dimensions
        self.sequence_length = sequence_length
        self.mlm_prob = mlm_prob
        self.mask_token = mask_token
        self.pad_token = pad_token
        self.sos_token = sos_token
        self.eos_token = eos_token
        self.architecture = architecture
        self.attributes_meta = attributes_meta
        self.time_attribute_concatenated = time_attribute_concatenated
        self.nb_special_tokens = nb_special_tokens
        self.vocab_size = vocab_size + self.nb_special_tokens

        if self.architecture == 'BERT':
            self.mlm = DynamicMLM(prob=self.mlm_prob,
                                  mask_token=self.mask_token,
                                  vocab_size=self.vocab_size,
                                  nb_special_tokens=self.nb_special_tokens,
                                  method='u-PMLM',
                                  attributes_meta=self.attributes_meta)
        self.position_embedding = PositionalEncoding(max_len=self.sequence_length,
                                                     d_model=self.d_model,
                                                     dropout_prob=self.dropout_prob,
                                                     series_dimensions=self.series_dimensions)
        self.value_embedding = Embedding(d_model=self.d_model,
                                         vocab_size=self.vocab_size,
                                         dropout_prob=self.dropout_prob,
                                         attributes_meta = self.attributes_meta,
                                         time_attribute_concatenated=self.time_attribute_concatenated,
                                         pad_token=self.pad_token)
        self.self_attentional_block = SelfAttentionalBlock(d_model=self.d_model,
                                                           attention_type=attention_type,
                                                           n_layers=n_layers,
                                                           n_heads=self.n_heads,
                                                           d_query=self.d_query,
                                                           dropout_prob=self.dropout_prob,
                                                           attention_dropout_prob=attention_dropout_prob,
                                                           intra_transformer_activation=intra_transformer_activation)
        self.readout = Readout(d_model=self.d_model,
                               vocab_size=self.vocab_size,
                               dropout_prob=self.dropout_prob,
                               attributes_meta = self.attributes_meta,
                               time_attribute_concatenated=self.time_attribute_concatenated)

    def forward(self, x, attn_mask=None):
        if self.architecture == 'BERT': # BERT or GPT
            x_new = []
            if 0 in self.attributes_meta.keys():
                x_new.append(x[0].detach().clone())
            if 1 in self.attributes_meta.keys():
                x_new.append(x[1].detach().clone())
            x = self.mlm(tuple(x_new))

        x = self.value_embedding(x)
        x = self.position_embedding(x)
        x = self.self_attentional_block(x, attn_mask=attn_mask)

        return self.readout(x) # it is a tuple


# Motivated by https://www.aclweb.org/anthology/N19-1423/ but dynamic masking as per https://arxiv.org/abs/1907.11692
# Performed with tensors on device
class DynamicMLM(nn.Module):
    def __init__(self,
                 prob,
                 mask_token,
                 vocab_size,
                 nb_special_tokens,
                 fix_masks=None,
                 to_noise=True,
                 method='u-PMLM',
                 attributes_meta=None):
        super().__init__()
        self.prob = prob
        self.mask_token = mask_token

        self.mlm_indexes = None
        self.masked_indexes = None
        self.randomized_indexes = None

        self.vocab_size = vocab_size
        self.nb_special_tokens = nb_special_tokens
        self.fix_masks = fix_masks
        self.to_noise = to_noise
        self.method = method
        self.attributes_meta = attributes_meta

    def forward(self, x): # input is always a tuple
        with torch.no_grad():
            # Probabilistic MLM with a uniform prior
            # https://www.aclweb.org/anthology/2020.acl-main.24/
            # http://proceedings.mlr.press/v97/stern19a.html
            if self.method == 'u-PMLM':
                sampled_prob = torch.rand(1)  # sampling form uniform prior
                permutation = torch.randperm(x[0].size(1), device=x[0].device)
                self.mlm_indexes = permutation[:int(permutation.size(0) * sampled_prob) + 1]
                self.mlm_indexes = torch.sort(self.mlm_indexes)[0]
                self.masked_indexes = self.mlm_indexes  # All mlm_indexes are with the [MASK] token (u-PMLM)

                if self.to_noise:
                    # activity label:
                    if 0 in self.attributes_meta.keys():
                        x[0][:, self.masked_indexes, :] = float(self.mask_token) * torch.ones(
                            (x[0].size(0), self.masked_indexes.size(0), x[0].size(2)), device=x[0].device)

                    # time attribute:
                    # Since all the other special tokens come with value 0:
                    if 1 in self.attributes_meta.keys():
                        min_values = float(self.attributes_meta[1]['min_value']) * torch.ones(
                            (x[1].size(0), self.masked_indexes.size(0), x[1].size(2)), device=x[1].device)
                        x[1][:, self.masked_indexes, :] = min_values
                    '''
                    # Salt-and-pepper noise https://www.jmlr.org/papers/volume11/vincent10a/vincent10a.pdf
                    if 1 in self.attributes_meta.keys():
                        min_values = float(self.attributes_meta[1]['min_value']) * torch.ones(
                            (x[1].size(0), self.masked_indexes.size(0), x[1].size(2)), device=x[1].device)
                        max_values = float(self.attributes_meta[1]['max_value']) * torch.ones(
                             (x[1].size(0), self.masked_indexes.size(0), x[1].size(2)), device=x[1].device)
                        probability = torch.tensor(0.5, device=x[1].device)
                        bernoulli = torch.distributions.bernoulli.Bernoulli(probability).sample(x[1].size())
                        x[1][:, self.masked_indexes, :] = torch.where(bernoulli[:, self.masked_indexes, :] == 1, max_values, min_values)
                    '''

            # Pseudo log-likelihood calculation for MLM
            # https://arxiv.org/abs/1902.04094
            # https://arxiv.org/abs/2106.02736
            elif self.method == 'fix_masks':
                self.mlm_indexes = self.fix_masks

                if len(self.mlm_indexes.size()) == 1:
                    self.masked_indexes = self.mlm_indexes  # All mlm_indexes are with the [MASK] token

                    if self.to_noise:
                        # activity label
                        if 0 in self.attributes_meta.keys():
                            x[0][:, self.masked_indexes, :] = float(self.mask_token) * torch.ones(
                                (x[0].size(0), self.masked_indexes.size(0), x[0].size(2)), device=x[0].device)

                        # time attribute:
                            # Since all the other special tokens come with value 0:
                            if 1 in self.attributes_meta.keys():
                                min_values = float(self.attributes_meta[1]['min_value']) * torch.ones(
                                    (x[1].size(0), self.masked_indexes.size(0), x[1].size(2)), device=x[1].device)
                                x[1][:, self.masked_indexes, :] = min_values
                        '''
                        if 1 in self.attributes_meta.keys():
                            min_values = float(self.attributes_meta[1]['min_value']) * torch.ones(
                                (x[1].size(0), self.masked_indexes.size(0), x[1].size(2)), device=x[1].device)
                            max_values = float(self.attributes_meta[1]['max_value']) * torch.ones(
                                (x[1].size(0), self.masked_indexes.size(0), x[1].size(2)), device=x[1].device)
                            probability = torch.tensor(0.5, device=x[1].device)
                            bernoulli = torch.distributions.bernoulli.Bernoulli(probability).sample(x[1].size())
                            x[1][:, self.masked_indexes, :] = torch.where(bernoulli[:, self.masked_indexes, :] == 1,
                                                                          max_values, min_values)
                        '''

            return x # always a tuple


class TransformerAutoEncoder(nn.Module):
    def __init__(self,
                 d_model,
                 sequence_length,
                 n_layers=4,
                 n_heads=4,
                 d_query=32,
                 dropout_prob=0.0,
                 series_dimensions=1,
                 pad_token=None,
                 sos_token=None,
                 eos_token=None,
                 vocab_size=None,
                 intra_transformer_activation='gelu',
                 attributes_meta=None,
                 time_attribute_concatenated=False,
                 nb_special_tokens=None):
        super().__init__()

        self.d_model = d_model
        self.dropout_prob = dropout_prob
        self.d_query = d_query
        self.n_heads = n_heads
        self.hidden_size = self.n_heads * self.d_query
        self.series_dimensions = series_dimensions
        self.sequence_length = sequence_length
        self.pad_token = pad_token
        self.attributes_meta = attributes_meta
        self.time_attribute_concatenated = time_attribute_concatenated
        self.nb_special_tokens = nb_special_tokens
        self.vocab_size = vocab_size + self.nb_special_tokens

        target_lookahead_mask = (torch.triu(torch.ones(self.sequence_length, self.sequence_length)) == 1).transpose(0, 1)
        self.register_buffer('target_lookahead_mask', target_lookahead_mask.float().masked_fill(target_lookahead_mask == 0, float('-inf')).masked_fill(target_lookahead_mask == 1, float(0.0)))

        self.position_embedding = PositionalEncoding(max_len=self.sequence_length,
                                                     d_model=self.d_model,
                                                     dropout_prob=self.dropout_prob,
                                                     series_dimensions=self.series_dimensions)
        self.value_embedding = Embedding(d_model=self.d_model,
                                         vocab_size=self.vocab_size,
                                         dropout_prob=self.dropout_prob,
                                         attributes_meta = self.attributes_meta,
                                         time_attribute_concatenated=self.time_attribute_concatenated,
                                         pad_token=self.pad_token)
        self.self_attentional_block = nn.Transformer(d_model=self.d_model,
                                                     num_encoder_layers=n_layers,
                                                     num_decoder_layers=n_layers,
                                                     nhead=self.n_heads,
                                                     dropout=self.dropout_prob,
                                                     activation=intra_transformer_activation,
                                                     batch_first=True)
        self.readout = Readout(d_model=self.d_model,
                               vocab_size=self.vocab_size,
                               dropout_prob=self.dropout_prob,
                               attributes_meta = self.attributes_meta,
                               time_attribute_concatenated=self.time_attribute_concatenated)

    def forward(self, x, y, attn_mask=None):
        x = self.value_embedding(x)
        x = self.position_embedding(x)

        y = self.value_embedding(y)
        y = self.position_embedding(y)

        return self.readout(self.self_attentional_block(x, y, tgt_mask=self.target_lookahead_mask[:y.size(1), :y.size(1)])) # it is a tuple


# credits to https://github.com/litanli/wavenet-time-series-forecasting/blob/master/wavenet_pytorch.py
class DilatedCausalConv1d(nn.Module):
    def __init__(self, hyperparams: dict, dilation_factor: int, in_channels: int):
        super().__init__()

        def weights_init(m):
            if isinstance(m, nn.Conv1d):
                init.kaiming_normal_(m.weight.data)
                init.zeros_(m.bias.data)

        self.dilation_factor = dilation_factor
        self.dilated_causal_conv = nn.Conv1d(in_channels=in_channels,
                                             out_channels=hyperparams['nb_filters'],
                                             kernel_size=hyperparams['kernel_size'],
                                             dilation=dilation_factor)
        self.dilated_causal_conv.apply(weights_init)

        self.skip_connection = nn.Conv1d(in_channels=in_channels,
                                         out_channels=hyperparams['nb_filters'],
                                         kernel_size=1)
        self.skip_connection.apply(weights_init)
        self.leaky_relu = nn.LeakyReLU(0.1)
        self.layer_norm = nn.LayerNorm(hyperparams['nb_filters'])

    def forward(self, x):
        x1 = self.leaky_relu(self.dilated_causal_conv(x))
        x2 = x[:, :, self.dilation_factor:]
        x2 = self.skip_connection(x2)
        return self.layer_norm((x1 + x2).transpose(1, 2)).transpose(1, 2)


class WaveNet(nn.Module):
    def __init__(self,
                 hidden_size,
                 n_layers=4,
                 dropout_prob=0.0,
                 pad_token=None,
                 sos_token=None,
                 eos_token=None,
                 mask_token=None,
                 vocab_size=None,
                 attributes_meta=None,
                 time_attribute_concatenated=False,
                 nb_special_tokens=None,
                 architecture=None):
        super().__init__()

        def weights_init(m):
            if isinstance(m, nn.Conv1d):
                init.kaiming_normal_(m.weight.data)
                init.zeros_(m.bias.data)

        in_channels = hidden_size
        hyperparams = {'nb_layers': n_layers,
                       'nb_filters': hidden_size,
                       'kernel_size': 2}

        if architecture is not None:
            self.architecture = architecture

        self.dropout_prob = dropout_prob
        self.hidden_size = hidden_size
        self.pad_token = pad_token
        self.sos_token = sos_token
        self.eos_token = eos_token
        self.mask_token = mask_token
        self.attributes_meta = attributes_meta
        self.time_attribute_concatenated = time_attribute_concatenated
        self.nb_special_tokens = nb_special_tokens
        self.vocab_size = vocab_size + self.nb_special_tokens

        self.dilation_factors = [2 ** i for i in range(0, hyperparams['nb_layers'])]
        self.in_channels = [in_channels] + [hyperparams['nb_filters'] for _ in range(hyperparams['nb_layers'])]
        self.dilated_causal_convs = nn.Sequential(
            *[DilatedCausalConv1d(hyperparams, self.dilation_factors[i], self.in_channels[i]) for i in
              range(hyperparams['nb_layers'])])
        for dilated_causal_conv in self.dilated_causal_convs:
            dilated_causal_conv.apply(weights_init)

        self.output_layer = nn.Conv1d(in_channels=self.in_channels[-1],
                                      out_channels=self.hidden_size,
                                      kernel_size=1)
        self.output_layer.apply(weights_init)
        self.leaky_relu = nn.LeakyReLU(0.1)

        self.value_embedding = Embedding(d_model=self.hidden_size,
                                         vocab_size=self.vocab_size,
                                         dropout_prob=self.dropout_prob,
                                         attributes_meta=self.attributes_meta,
                                         time_attribute_concatenated=self.time_attribute_concatenated,
                                         pad_token=self.pad_token)
        self.readout = Readout(d_model=self.hidden_size,
                               vocab_size=self.vocab_size,
                               dropout_prob=self.dropout_prob,
                               attributes_meta=self.attributes_meta,
                               time_attribute_concatenated=self.time_attribute_concatenated)

        receptive_field = 2 ** (hyperparams['nb_layers'] - 1) * hyperparams['kernel_size']
        print('receptive_field: ' + str(receptive_field))
        self.left_padding = receptive_field - 1

    def forward(self, x, left_padding=None):
        x = self.value_embedding(x)
        x = x.transpose(1, 2)

        if left_padding is None:
            x = nn.functional.pad(x, (self.left_padding, 0), mode='constant', value=0)
        else:
            if left_padding > 0:
                x = nn.functional.pad(x, (left_padding, 0), mode='constant', value=0)

        x = self.dilated_causal_convs(x)
        x = self.leaky_relu(self.output_layer(x))

        x = x.transpose(1, 2)
        x = self.readout(x)
        return x
