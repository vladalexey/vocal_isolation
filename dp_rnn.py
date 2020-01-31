import numpy as np 
import os
import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch.autograd import Variable
import sys

class SingleRNN(nn.Module):

    """
    Container module for a single RNN layer.
    
    args:
        rnn_type: string, select from 'RNN', 'LSTM' and 'GRU'.
        input_size: int, dimension of the input feature. The input should have shape 
                    (batch, seq_len, input_size).
        hidden_size: int, dimension of the hidden state. 
        dropout: float, dropout ratio. Default is 0.
        bidirectional: bool, whether the RNN layers are bidirectional. Default is False.
    """

    def __init__(self, rnn_type, input_size, 
        hidden_size, dropout=0, bidirectional=False):

        super(SingleRNN, self).__init__()

        self.rnn_type = rnn_type
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_direction = int(bidirectional) + 1

        self.rnn = getattr(nn, rnn_type)(input_size, hidden_size, 1, 
            dropout=dropout, batch_first=True, bidirectional=bidirectional)

        # linear projection layer
        self.proj = nn.Linear(hidden_size*self.num_direction, input_size)

    def forward(self, input):
        # input_shape = (n_batch, seq_len, dim)

        rnn_output, _ = self.rnn(input)
        rnn_output = self.proj(rnn_output.contiguous().view(-1, rnn_output.shape[2])).view(input.shape)

        return rnn_output

class DPRNN(nn.Module):

    def __init__(self, rnn_type, input_size, hidden_size, output_size,
        dropout=0, num_layers=1, bidirectional=True):
        super(DPRNN, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size

        self.row_rnn = nn.ModuleList([])
        self.col_rnn = nn.ModuleList([])
        self.row_norm = nn.ModuleList([])
        self.col_norm = nn.ModuleList([])

        for _ in range(num_layers):

            # intra chunk RNN is always non-causal
            self.row_rnn.append(SingleRNN(rnn_type, input_size, hidden_size, 
                dropout=dropout, bidirectional=True))


            self.col_rnn.append(SingleRNN(rnn_type, input_size, hidden_size,
                dropout=dropout, bidirectional=bidirectional))

            self.row_norm.append(nn.GroupNorm(1, input_size, eps=1e-8))
            self.col_norm.append(nn.GroupNorm(1, input_size, eps=1e-8))

        self.output = nn.Sequential(
            nn.PReLU(),
            nn.Conv2d(input_size, output_size, 1)
        )

    def forward(self, input):
        # input_shape = (n_batch, N, dim1, dim2)
        # apply RNN on dim1 -> then dim2

        bs, _, dim1, dim2 = input.shape

        for idx in range(len(self.row_rnn)):

            row_inp = input.permute(0, 3, 2, 1).contiguous().view(bs * dim2, dim1, -1) # bs * dim2, dim1, N
            row_out = self.row_rnn[idx](row_inp)
            row_out = row_out.view(bs, dim2, dim1, -1).permute(0, 3, 2, 1).contiguous()
            row_out = self.row_norm[idx](row_out)

            output = input + row_out

            col_inp = input.permute(0, 2, 3, 1).contiguous().view(bs * dim2, dim1, -1) # bs * dim2, dim1, N
            col_out = self.col_rnn[idx](col_inp)
            col_out = col_out.view(bs, dim2, dim1, -1).permute(0, 3, 1, 2).contiguous()
            col_out = self.col_norm[idx](col_out)

            output = output + col_out

        output = self.output(output)

        return output




        


