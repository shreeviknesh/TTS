import torch
from .utils import sequence_mask

# layers.attention.py
class BahdanauAttention(torch.nn.Module):
    def __init__(self, annot_dim, query_dim, attn_dim):
        super(BahdanauAttention, self).__init__()
        self.query_layer = torch.nn.Linear(query_dim, attn_dim, bias=True)
        self.annot_layer = torch.nn.Linear(annot_dim, attn_dim, bias=True)
        self.v = torch.nn.Linear(attn_dim, 1, bias=False)

    def forward(self, annots, query):
        """
        Shapes:
            - annots: (batch, max_time, dim)
            - query: (batch, 1, dim) or (batch, dim)
        """
        if query.dim() == 2:
            query = query.unsqueeze(1)
        processed_query = self.query_layer(query)
        processed_annots = self.annot_layer(annots)
        alignment = self.v(torch.tanh(processed_query + processed_annots))
        return alignment.squeeze(-1)

class LocationSensitiveAttention(torch.nn.Module):
    """Location sensitive attention following
    https://arxiv.org/pdf/1506.07503.pdf"""

    def __init__(self,
                 annot_dim,
                 query_dim,
                 attn_dim,
                 kernel_size=31,
                 filters=32):
        super(LocationSensitiveAttention, self).__init__()
        self.kernel_size = kernel_size
        self.filters = filters
        padding = [(kernel_size - 1) // 2, (kernel_size - 1) // 2]
        self.loc_conv = torch.nn.Sequential(
            torch.nn.ConstantPad1d(padding, 0),
            torch.nn.Conv1d(
                2,
                filters,
                kernel_size=kernel_size,
                stride=1,
                padding=0,
                bias=False))
        self.loc_linear = torch.nn.Linear(filters, attn_dim, bias=True)
        self.query_layer = torch.nn.Linear(query_dim, attn_dim, bias=True)
        self.annot_layer = torch.nn.Linear(annot_dim, attn_dim, bias=True)
        self.v = torch.nn.Linear(attn_dim, 1, bias=False)
        self.processed_annots = None

    def init_layers(self):
        torch.torch.nn.init.xavier_uniform_(
            self.loc_linear.weight,
            gain=torch.torch.nn.init.calculate_gain('tanh'))
        torch.torch.nn.init.xavier_uniform_(
            self.query_layer.weight,
            gain=torch.torch.nn.init.calculate_gain('tanh'))
        torch.torch.nn.init.xavier_uniform_(
            self.annot_layer.weight,
            gain=torch.torch.nn.init.calculate_gain('tanh'))
        torch.torch.nn.init.xavier_uniform_(
            self.v.weight,
            gain=torch.torch.nn.init.calculate_gain('linear'))

    def reset(self):
        self.processed_annots = None

    def forward(self, annot, query, loc):
        """
        Shapes:
            - annot: (batch, max_time, dim)
            - query: (batch, 1, dim) or (batch, dim)
            - loc: (batch, 2, max_time)
        """
        if query.dim() == 2:
            query = query.unsqueeze(1)
        processed_loc = self.loc_linear(self.loc_conv(loc).transpose(1, 2))
        processed_query = self.query_layer(query)
        if self.processed_annots is None:
            self.processed_annots = self.annot_layer(annot)
        alignment = self.v(
            torch.tanh(processed_query + self.processed_annots + processed_loc))
        del processed_loc
        del processed_query
        return alignment.squeeze(-1)

class AttentionRNNCell(torch.nn.Module):
    def __init__(self, out_dim, rnn_dim, annot_dim, memory_dim, align_model, windowing=False):
        r"""
        General Attention RNN wrapper

        Args:
            out_dim (int): context vector feature dimension.
            rnn_dim (int): rnn hidden state dimension.
            annot_dim (int): annotation vector feature dimension.
            memory_dim (int): memory vector (decoder output) feature dimension.
            align_model (str): 'b' for Bahdanau, 'ls' Location Sensitive alignment.
            windowing (bool): attention windowing forcing monotonic attention.
                It is only active in eval mode.
        """
        super(AttentionRNNCell, self).__init__()
        self.align_model = align_model
        self.rnn_cell = torch.nn.GRUCell(annot_dim + memory_dim, rnn_dim)
        self.windowing = windowing
        if self.windowing:
            self.win_back = 3
            self.win_front = 6
            self.win_idx = None

        if align_model == 'b':
            self.alignment_model = BahdanauAttention(annot_dim, rnn_dim,
                                                     out_dim)
        if align_model == 'ls':
            self.alignment_model = LocationSensitiveAttention(
                annot_dim, rnn_dim, out_dim)
        else:
            raise RuntimeError(" Wrong alignment model name: {}. Use\
                'b' (Bahdanau) or 'ls' (Location Sensitive).".format(
                align_model))

    def forward(self, memory, context, rnn_state, annots, atten, mask, t):
        """
        Shapes:
            - memory: (batch, 1, dim) or (batch, dim)
            - context: (batch, dim)
            - rnn_state: (batch, out_dim)
            - annots: (batch, max_time, annot_dim)
            - atten: (batch, 2, max_time)
            - mask: (batch,)
        """
        if t == 0:
            self.alignment_model.reset()
            self.win_idx = 0
        rnn_output = self.rnn_cell(torch.cat((memory, context), -1), rnn_state)
        if self.align_model == 'b':
            alignment = self.alignment_model(annots, rnn_output)
        else:
            alignment = self.alignment_model(annots, rnn_output, atten)
        if mask is not None:
            mask = mask.view(memory.size(0), -1)
            alignment.masked_fill_(1 - mask, -float("inf"))
        if not self.training and self.windowing:
            back_win = self.win_idx - self.win_back
            front_win = self.win_idx + self.win_front
            if back_win > 0:
                alignment[:, :back_win] = -float("inf")
            if front_win < memory.shape[1]:
                alignment[:, front_win:] = -float("inf")
            self.win_idx = torch.argmax(alignment,1).long()[0].item()
        alignment = torch.sigmoid(alignment) / torch.sigmoid(alignment).sum(dim=1).unsqueeze(1)
        context = torch.bmm(alignment.unsqueeze(1), annots)
        context = context.squeeze(1)
        return rnn_output, context, alignment

# layers.losses.py
class L1LossMasked(torch.nn.Module):
    def __init__(self):
        super(L1LossMasked, self).__init__()

    def forward(self, input, target, length):
        """
        Args:
            input: A Variable containing a FloatTensor of size
                (batch, max_len, dim) which contains the
                unnormalized probability for each class.
            target: A Variable containing a LongTensor of size
                (batch, max_len, dim) which contains the index of the true
                class for each corresponding step.
            length: A Variable containing a LongTensor of size (batch,)
                which contains the length of each data in a batch.
        Returns:
            loss: An average loss value masked by the length.
        """
        # mask: (batch, max_len, 1)
        mask = sequence_mask(
            sequence_length=length, max_len=target.size(1)).unsqueeze(2).float()
        mask = mask.expand_as(input)
        loss = functional.l1_loss(
            input * mask, target * mask, reduction="sum")
        loss = loss / mask.sum()
        return loss

class MSELossMasked(torch.nn.Module):
    def __init__(self):
        super(MSELossMasked, self).__init__()

    def forward(self, input, target, length):
        """
        Args:
            input: A Variable containing a FloatTensor of size
                (batch, max_len, dim) which contains the
                unnormalized probability for each class.
            target: A Variable containing a LongTensor of size
                (batch, max_len, dim) which contains the index of the true
                class for each corresponding step.
            length: A Variable containing a LongTensor of size (batch,)
                which contains the length of each data in a batch.
        Returns:
            loss: An average loss value masked by the length.
        """
        input = input.contiguous()
        target = target.contiguous()

        # logits_flat: (batch * max_len, dim)
        input = input.view(-1, input.shape[-1])
        # target_flat: (batch * max_len, dim)
        target_flat = target.view(-1, target.shape[-1])
        # losses_flat: (batch * max_len, dim)
        losses_flat = functional.mse_loss(
            input, target_flat, size_average=False, reduce=False)
        # losses: (batch, max_len, dim)
        losses = losses_flat.view(*target.size())

        # mask: (batch, max_len, 1)
        mask = sequence_mask(
            sequence_length=length, max_len=target.size(1)).unsqueeze(2)
        losses = losses * mask.float()
        loss = losses.sum() / (length.float().sum() * float(target.shape[2]))
        return loss

# layers.tacotron.py
class Prenet(torch.nn.Module):
    r""" Prenet as explained at https://arxiv.org/abs/1703.10135.
    It creates as many layers as given by 'out_features'

    Args:
        in_features (int): size of the input vector
        out_features (int or list): size of each output sample.
            If it is a list, for each value, there is created a new layer.
    """

    def __init__(self, in_features, out_features=[256, 128]):
        super(Prenet, self).__init__()
        in_features = [in_features] + out_features[:-1]
        self.layers = torch.nn.ModuleList([
            torch.nn.Linear(in_size, out_size)
            for (in_size, out_size) in zip(in_features, out_features)
        ])
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(0.5)

    def init_layers(self):
        for layer in self.layers:
            torch.torch.nn.init.xavier_uniform_(
                layer.weight, gain=torch.torch.nn.init.calculate_gain('relu'))

    def forward(self, inputs):
        for linear in self.layers:
            inputs = self.dropout(self.relu(linear(inputs)))
        return inputs

class BatchNormConv1d(torch.nn.Module):
    r"""A wrapper for Conv1d with BatchNorm. It sets the activation
    function between Conv and BatchNorm layers. BatchNorm layer
    is initialized with the TF default values for momentum and eps.

    Args:
        in_channels: size of each input sample
        out_channels: size of each output samples
        kernel_size: kernel size of conv filters
        stride: stride of conv filters
        padding: padding of conv filters
        activation: activation function set b/w Conv1d and BatchNorm

    Shapes:
        - input: batch x dims
        - output: batch x dims
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding,
                 activation=None):

        super(BatchNormConv1d, self).__init__()
        self.padding = padding
        self.padder = torch.nn.ConstantPad1d(padding, 0)
        self.conv1d = torch.nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            bias=False)
        # Following tensorflow's default parameters
        self.bn = torch.nn.BatchNorm1d(out_channels, momentum=0.99, eps=1e-3)
        self.activation = activation

    def init_layers(self):
        if type(self.activation) == torch.torch.nn.ReLU:
            w_gain = 'relu'
        elif type(self.activation) == torch.torch.nn.Tanh:
            w_gain = 'tanh'
        elif self.activation is None:
            w_gain = 'linear'
        else:
            raise RuntimeError('Unknown activation function')
        torch.torch.nn.init.xavier_uniform_(
            self.conv1d.weight, gain=torch.torch.nn.init.calculate_gain(w_gain))

    def forward(self, x):
        x = self.padder(x)
        x = self.conv1d(x)
        x = self.bn(x)
        if self.activation is not None:
            x = self.activation(x)
        return x

class Highway(torch.nn.Module):
    def __init__(self, in_size, out_size):
        super(Highway, self).__init__()
        self.H = torch.nn.Linear(in_size, out_size)
        self.H.bias.data.zero_()
        self.T = torch.nn.Linear(in_size, out_size)
        self.T.bias.data.fill_(-1)
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()

    def init_layers(self):
        torch.torch.nn.init.xavier_uniform_(
            self.H.weight, gain=torch.torch.nn.init.calculate_gain('relu'))
        torch.torch.nn.init.xavier_uniform_(
            self.T.weight, gain=torch.torch.nn.init.calculate_gain('sigmoid'))

    def forward(self, inputs):
        H = self.relu(self.H(inputs))
        T = self.sigmoid(self.T(inputs))
        return H * T + inputs * (1.0 - T)

class CBHG(torch.nn.Module):
    """CBHG module: a recurrent neural network composed of:
        - 1-d convolution banks
        - Highway networks + residual connections
        - Bidirectional gated recurrent units

        Args:
            in_features (int): sample size
            K (int): max filter size in conv bank
            projections (list): conv channel sizes for conv projections
            num_highways (int): number of highways layers

        Shapes:
            - input: batch x time x dim
            - output: batch x time x dim*2
    """

    def __init__(self,
                 in_features,
                 K=16,
                 conv_bank_features=128,
                 conv_projections=[128, 128],
                 highway_features=128,
                 gru_features=128,
                 num_highways=4):
        super(CBHG, self).__init__()
        self.in_features = in_features
        self.conv_bank_features = conv_bank_features
        self.highway_features = highway_features
        self.gru_features = gru_features
        self.conv_projections = conv_projections
        self.relu = torch.nn.ReLU()
        self.conv1d_banks = torch.nn.ModuleList([
            BatchNormConv1d(
                in_features,
                conv_bank_features,
                kernel_size=k,
                stride=1,
                padding=[(k - 1) // 2, k // 2],
                activation=self.relu) for k in range(1, K + 1)
        ])
        self.max_pool1d = torch.nn.Sequential(
            torch.nn.ConstantPad1d([0, 1], value=0),
            torch.nn.MaxPool1d(kernel_size=2, stride=1, padding=0))
        out_features = [K * conv_bank_features] + conv_projections[:-1]
        activations = [self.relu] * (len(conv_projections) - 1)
        activations += [None]

        layer_set = []
        for (in_size, out_size, ac) in zip(out_features, conv_projections,
                                           activations):
            layer = BatchNormConv1d(
                in_size,
                out_size,
                kernel_size=3,
                stride=1,
                padding=[1, 1],
                activation=ac)
            layer_set.append(layer)
        self.conv1d_projections = torch.nn.ModuleList(layer_set)

        if self.highway_features != conv_projections[-1]:
            self.pre_highway = torch.nn.Linear(
                conv_projections[-1], highway_features, bias=False)
        self.highways = torch.nn.ModuleList([
            Highway(highway_features, highway_features)
            for _ in range(num_highways)
        ])

        self.gru = torch.nn.GRU(
            gru_features,
            gru_features,
            1,
            batch_first=True,
            bidirectional=True)

    def forward(self, inputs):
        x = inputs
        if x.size(-1) == self.in_features:
            x = x.transpose(1, 2)
        T = x.size(-1)
        outs = []
        for conv1d in self.conv1d_banks:
            out = conv1d(x)
            outs.append(out)
        x = torch.cat(outs, dim=1)
        assert x.size(1) == self.conv_bank_features * len(self.conv1d_banks)
        x = self.max_pool1d(x)
        for conv1d in self.conv1d_projections:
            x = conv1d(x)
        x = x.transpose(1, 2)
        x += inputs
        if self.highway_features != self.conv_projections[-1]:
            x = self.pre_highway(x)
        for highway in self.highways:
            x = highway(x)
        self.gru.flatten_parameters()
        outputs, _ = self.gru(x)
        return outputs

class EncoderCBHG(torch.nn.Module):
    def __init__(self):
        super(EncoderCBHG, self).__init__()
        self.cbhg = CBHG(
            128,
            K=16,
            conv_bank_features=128,
            conv_projections=[128, 128],
            highway_features=128,
            gru_features=128,
            num_highways=4)

    def forward(self, x):
        return self.cbhg(x)

class Encoder(torch.nn.Module):
    r"""Encapsulate Prenet and CBHG modules for encoder"""

    def __init__(self, in_features):
        super(Encoder, self).__init__()
        self.prenet = Prenet(in_features, out_features=[256, 128])
        self.cbhg = EncoderCBHG()

    def forward(self, inputs):
        r"""
        Args:
            inputs (FloatTensor): embedding features

        Shapes:
            - inputs: batch x time x in_features
            - outputs: batch x time x 128*2
        """
        inputs = self.prenet(inputs)
        return self.cbhg(inputs)

class PostCBHG(torch.nn.Module):
    def __init__(self, mel_dim):
        super(PostCBHG, self).__init__()
        self.cbhg = CBHG(
            mel_dim,
            K=8,
            conv_bank_features=128,
            conv_projections=[256, mel_dim],
            highway_features=128,
            gru_features=128,
            num_highways=4)

    def forward(self, x):
        return self.cbhg(x)

class Decoder(torch.nn.Module):
    r"""Decoder module.

    Args:
        in_features (int): input vector (encoder output) sample size.
        memory_dim (int): memory vector (prev. time-step output) sample size.
        r (int): number of outputs per time step.
        memory_size (int): size of the past window. if <= 0 memory_size = r
    """

    def __init__(self, in_features, memory_dim, r, memory_size, attn_windowing):
        super(Decoder, self).__init__()
        self.r = r
        self.in_features = in_features
        self.max_decoder_steps = 500
        self.memory_size = memory_size if memory_size > 0 else r
        self.memory_dim = memory_dim
        self.prenet = Prenet(memory_dim * self.memory_size, out_features=[256, 128])
        self.attention_rnn = AttentionRNNCell(
            out_dim=128,
            rnn_dim=256,
            annot_dim=in_features,
            memory_dim=128,
            align_model='ls',
            windowing=attn_windowing)
        self.project_to_decoder_in = torch.nn.Linear(256 + in_features, 256)
        self.decoder_rnns = torch.nn.ModuleList([torch.nn.GRUCell(256, 256) for _ in range(2)])
        self.proj_to_mel = torch.nn.Linear(256, memory_dim * r)
        self.attention_rnn_init = torch.nn.Embedding(1, 256)
        self.memory_init = torch.nn.Embedding(1, self.memory_size * memory_dim)
        self.decoder_rnn_inits = torch.nn.Embedding(2, 256)
        self.stopnet = StopNet(256 + memory_dim * r)

    def init_layers(self):
        torch.torch.nn.init.xavier_uniform_(
            self.project_to_decoder_in.weight,
            gain=torch.torch.nn.init.calculate_gain('linear'))
        torch.torch.nn.init.xavier_uniform_(
            self.proj_to_mel.weight,
            gain=torch.torch.nn.init.calculate_gain('linear'))

    def _reshape_memory(self, memory):
        """
        Reshape the spectrograms for given 'r'
        """
        B = memory.shape[0]
        if memory.size(-1) == self.memory_dim:
            memory = memory.contiguous()
            memory = memory.view(B, memory.size(1) // self.r, -1)
        memory = memory.transpose(0, 1)
        return memory

    def _init_states(self, inputs):
        """
        Initialization of decoder states
        """
        B = inputs.size(0)
        T = inputs.size(1)
        initial_memory = self.memory_init(inputs.data.new_zeros(B).long())

        attention_rnn_hidden = self.attention_rnn_init(inputs.data.new_zeros(B).long())
        decoder_rnn_hiddens = [
            self.decoder_rnn_inits(inputs.data.new_tensor([idx]*B).long())
            for idx in range(len(self.decoder_rnns))
        ]
        current_context_vec = inputs.data.new(B, self.in_features).zero_()

        attention = inputs.data.new(B, T).zero_()
        attention_cum = inputs.data.new(B, T).zero_()
        return (initial_memory, attention_rnn_hidden, decoder_rnn_hiddens,
            current_context_vec, attention, attention_cum)

    def forward(self, inputs, memory=None, mask=None):
        """
        Decoder forward step.

        If decoder inputs are not given (e.g., at testing time), as noted in
        Tacotron paper, greedy decoding is adapted.

        Args:
            inputs: Encoder outputs.
            memory (None): Decoder memory (autoregression. If None (at eval-time),
              decoder outputs are used as decoder inputs. If None, it uses the last
              output as the input.
            mask (None): Attention mask for sequence padding.

        Shapes:
            - inputs: batch x time x encoder_out_dim
            - memory: batch x #mel_specs x mel_spec_dim
        """
        greedy = not self.training
        if memory is not None:
            memory = self._reshape_memory(memory)
            T_decoder = memory.size(0)
        outputs = []
        attentions = []
        stop_tokens = []
        t = 0
        memory_input, attention_rnn_hidden, decoder_rnn_hiddens,\
            current_context_vec, attention, attention_cum = self._init_states(inputs)
        while True:
            if t > 0:
                if memory is None:
                    new_memory = outputs[-1]
                else:
                    new_memory = memory[t - 1]
                # Queuing if memory size defined else use previous prediction only.
                if self.memory_size > 0:
                    memory_input = torch.cat([memory_input[:, self.r * self.memory_dim:].clone(), new_memory], dim=-1)
                else:
                    memory_input = new_memory
            # Prenet
            processed_memory = self.prenet(memory_input)
            # Attention RNN
            attention_cat = torch.cat(
                (attention.unsqueeze(1), attention_cum.unsqueeze(1)), dim=1)
            attention_rnn_hidden, current_context_vec, attention = self.attention_rnn(
                processed_memory, current_context_vec, attention_rnn_hidden,
                inputs, attention_cat, mask, t)
            del attention_cat
            attention_cum += attention
            # Concat RNN output and attention context vector
            decoder_input = self.project_to_decoder_in(
                torch.cat((attention_rnn_hidden, current_context_vec), -1))
            # Pass through the decoder RNNs
            for idx in range(len(self.decoder_rnns)):
                decoder_rnn_hiddens[idx] = self.decoder_rnns[idx](
                    decoder_input, decoder_rnn_hiddens[idx])
                # Residual connection
                decoder_input = decoder_rnn_hiddens[idx] + decoder_input
            decoder_output = decoder_input
            del decoder_input
            # predict mel vectors from decoder vectors
            output = self.proj_to_mel(decoder_output)
            output = torch.sigmoid(output)
            # predict stop token
            stopnet_input = torch.cat([decoder_output, output], -1)
            del decoder_output
            stop_token = self.stopnet(stopnet_input)
            del stopnet_input
            outputs += [output]
            attentions += [attention]
            stop_tokens += [stop_token]
            del output
            t += 1
            if memory is not None:
                if t >= T_decoder:
                    break
            else:
                if t > inputs.shape[1] / 4 and (stop_token > 0.6 or
                                                attention[:, -1].item() > 0.6):
                    break
                elif t > self.max_decoder_steps:
                    print("   | > Decoder stopped with 'max_decoder_steps")
                    break
        # Back to batch first
        attentions = torch.stack(attentions).transpose(0, 1)
        outputs = torch.stack(outputs).transpose(0, 1).contiguous()
        stop_tokens = torch.stack(stop_tokens).transpose(0, 1)
        return outputs, attentions, stop_tokens

class StopNet(torch.nn.Module):
    r"""
    Predicting stop-token in decoder.

    Args:
        in_features (int): feature dimension of input.
    """

    def __init__(self, in_features):
        super(StopNet, self).__init__()
        self.dropout = torch.nn.Dropout(0.1)
        self.linear = torch.nn.Linear(in_features, 1)
        self.sigmoid = torch.nn.Sigmoid()
        torch.torch.nn.init.xavier_uniform_(
            self.linear.weight, gain=torch.torch.nn.init.calculate_gain('linear'))

    def forward(self, inputs):
        outputs = self.dropout(inputs)
        outputs = self.linear(outputs)
        outputs = self.sigmoid(outputs)
        return outputs

# models.tacotron.py
class Tacotron(torch.nn.Module):
    def __init__(self,
                 num_chars,
                 embedding_dim=256,
                 linear_dim=1025,
                 mel_dim=80,
                 r=5,
                 padding_idx=None,
                 memory_size=5,
                 attn_windowing=False):
        super(Tacotron, self).__init__()
        self.r = r
        self.mel_dim = mel_dim
        self.linear_dim = linear_dim
        self.embedding = torch.nn.Embedding(num_chars, embedding_dim, padding_idx=padding_idx)
        print("\t   | > Number of characters : {}".format(num_chars))
        self.embedding.weight.data.normal_(0, 0.3)
        self.encoder = Encoder(embedding_dim)
        self.decoder = Decoder(256, mel_dim, r, memory_size, attn_windowing)
        self.postnet = PostCBHG(mel_dim)
        self.last_linear = torch.nn.Sequential(torch.nn.Linear(self.postnet.cbhg.gru_features * 2, linear_dim), torch.nn.Sigmoid())

    def forward(self, characters, mel_specs=None, mask=None):
        B = characters.size(0)
        inputs = self.embedding(characters)
        encoder_outputs = self.encoder(inputs)
        mel_outputs, alignments, stop_tokens = self.decoder(encoder_outputs, mel_specs, mask)
        mel_outputs = mel_outputs.view(B, -1, self.mel_dim)
        linear_outputs = self.postnet(mel_outputs)
        linear_outputs = self.last_linear(linear_outputs)
        return mel_outputs, linear_outputs, alignments, stop_tokens
