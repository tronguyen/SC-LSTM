import torch
import torch.nn as nn
import torch.functional as F
from torch.autograd import Variable
import numpy as np
import gc

class sLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, isbidirection=False):
        """Scoring LSTM"""
        super(sLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, bidirectional=isbidirection)



    def forward(self, features, init_hidden=None):
        """
        Args:
            features: [seq_len, 1, 100] (compressed pool5 features)
        Return:
            scores [seq_len, 1]
        """
        self.lstm.flatten_parameters()

        # [seq_len, 1, hidden_size * 2]
        c_features, (h_n, c_n) = self.lstm(features)
        # c_features = self.linear(c_features_)

        return c_features[-1]

class mLinear(nn.Linear):
    def __init__(self, input_size, output_size, bias=False):
        super(mLinear, self).__init__(input_size,output_size,bias=bias)
        # self.weight = Variable(torch.zeros(output_size, input_size), requires_grad=True)
        # self.register_parameter('weight',self.weight)

    def forward(self, input):
        # return torch.functional.matmul()linear(input, torch.exp(self.weight), self.bias)
        output = torch.matmul(input,torch.exp(self.weight).t())
        return output

class dLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, topic_size, max_step, batch_size, vocab_size, i2w, num_layers=1):
        """Scoring LSTM"""
        super(dLSTM, self).__init__()
        self.rnn_cell = nn.LSTMCell(input_size, hidden_size, num_layers)
        self.max_step = max_step
        self.linear = nn.Sequential(
            nn.Linear(hidden_size + hidden_size*2, vocab_size),
            nn.Tanh()
        )
        self.loss = nn.CrossEntropyLoss()
        self._pad = Variable(torch.zeros(batch_size*topic_size, hidden_size), requires_grad=False)
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.topic_size = topic_size
        self.hidden_size = hidden_size
        self.i2w = i2w

    def expose_topic(self, temb, wemb):
        _pad = Variable(torch.zeros(self.topic_size, self.hidden_size), requires_grad=False)
        hx, cx = _pad, _pad
        w_i = torch.LongTensor(np.asarray([0]*self.topic_size))
        print(temb.data[:,:8])
        for i in range(5):
            z_x = torch.cat([temb, wemb(w_i)],dim=-1)
            hx, cx = self.rnn_cell(z_x, (hx, cx))
            i_output = nn.Softmax(dim=-1)(self.linear(torch.cat([hx, z_x], dim=-1)))
            w_i = torch.argmax(i_output, dim=-1)
            w_np = list(w_i.data.numpy())
            print(' - '.join([self.i2w[w] if w != 0 else 'PAD' for w in w_np]))


    def forward(self, temb, wemb, tdistr, target):
        w_i = torch.LongTensor(np.asarray([0]*self.batch_size*self.topic_size))
        # output = []
        # for t in range(temb.shape[0]):
        hx, cx = self._pad, self._pad
        step_loss_total = torch.zeros([self.batch_size*self.topic_size, 1])
        b_temb = temb.repeat(1, self.batch_size).reshape(self.topic_size * self.batch_size, -1)
        for i in range(self.max_step):
            z_x = torch.cat([b_temb,wemb(w_i)],dim=-1)
            hx, cx = self.rnn_cell(z_x, (hx, cx))
            i_output = nn.Softmax(dim=-1)(self.linear(torch.cat([hx, z_x], dim=-1))) #* tdistr[:,t].unsqueeze(-1).expand(-1,self.vocab_size)
            w_i = torch.argmax(i_output, dim=-1)
            # print(w_i)
            mask_loss = Variable((target[:, i] != 0).type(torch.FloatTensor).unsqueeze(-1).repeat(self.topic_size,1), requires_grad=False)
            step_loss = torch.log(i_output).gather(1, target[:, i].unsqueeze(1).repeat(self.topic_size,1)) * mask_loss
            step_loss_total += step_loss

        output = step_loss_total.reshape(self.topic_size, self.batch_size)*torch.transpose(tdistr,1,0)
        # output.append(step_loss_total)

            # step_loss = self.loss(step_output.view(-1,step_output.shape[-1]),target[:,i].view(-1)) * mask_loss

        # output = torch.cat(output,dim=-1)


        # gc.collect()
        return torch.transpose(output,1,0)


