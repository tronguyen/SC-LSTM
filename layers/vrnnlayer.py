import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.distributions.one_hot_categorical import OneHotCategorical

class vrnn(nn.Module):
    def __init__(self, input_size, rnn_size, output_size):
        """Scoring LSTM"""
        super(vrnn, self).__init__()
        self.lstmcell = nn.LSTMCell(input_size, rnn_size)
        # self.linear = nn.Sequential(
        #     # nn.Linear(rnn_size, rnn_size),
        #     # nn.Tanh(),
        #     nn.Linear(rnn_size, rnn_size),
        #     nn.Tanh(),
        #     nn.Linear(rnn_size, output_size),
        #     nn.Tanh()
        # )


    # def forward(self,lin_q, lin_p, rnn_w, config):
    #     pad_h_s = Variable(torch.zeros(config.batch_size,config.hidden_size), requires_grad=False)
    #     seq_len = rnn_w.shape[0]
    #     h_0, c_0 = pad_h_s, pad_h_s
    #
    #     p_list, q_list = [], []
    #     for i in range(seq_len):
    #         # h of i-th layer
    #         try:
    #             q_i = lin_q(torch.cat([h_0, rnn_w[i]], dim=-1))
    #             p_i = lin_p(h_0)
    #             cat_distr = OneHotCategorical(q_i[-1])
    #             z_i = cat_distr.sample().unsqueeze_(0)
    #             # x = torch.cat([rnn_w[i],z_i], dim=-1)
    #             x = z_i
    #             h_i, c_i = self.lstmcell(x, (h_0, c_0))
    #             h_0, c_0 = h_i, c_i
    #
    #             q_list += [q_i]
    #             p_list += [p_i]
    #         except RuntimeError:
    #             print('...')
    #
    #     # rnn_s = torch.cat([pad_h_s,rnn_s[:-1]],dim=0)
    #     # _p = lin_p(rnn_s) # remove last P
    #     # pack_q = torch.cat([rnn_s,rnn_w], dim=-1)
    #     # _q = lin_q(pack_q)
    #     _p = torch.stack(p_list)
    #     _q = torch.stack(q_list)
    #     return _p, _q
    def gauss_sampling(self, mu, log_variance):

        std = torch.exp(0.5 * log_variance)

        # e ~ N(0,1)
        epsilon = Variable(torch.randn(std.shape), requires_grad=False)

        # [num_layers, 1, hidden_size]
        return mu + epsilon * std

    def kldist(self, q_mu, q_logvar, p_mu, p_logvar):
        kl = torch.sum(0.5 * (p_logvar - q_logvar +
                          (torch.exp(q_logvar) + (q_mu - p_mu) ** 2) /
                          torch.exp(p_logvar) - 1), dim=-1)
        return kl

    def forward(self,lin_q, lin_p, rnn_w, config, q_mu, q_var, p_mu, p_var, temb):
        pad_h_s = Variable(torch.zeros(config.batch_size,config.hidden_size), requires_grad=False)
        seq_len = rnn_w.shape[0]
        h_0, c_0 = pad_h_s, pad_h_s

        p_list, q_list, z_list, kl_list = [], [], [], []
        for i in range(seq_len):
            # h of i-th layer
            try:
#                 q_i = lin_q(torch.cat([h_0, rnn_w[i]], dim=-1))
                q_i = lin_q(rnn_w[i])
                q_mu_val = q_mu(q_i)
                q_logvar_val = torch.log(q_var(q_i))
                z_i = self.gauss_sampling(q_mu_val, q_logvar_val)

                p_i = lin_p(h_0)
                p_mu_val = p_mu(p_i)
                p_logvar_val = torch.log(p_var(p_i))

                kl_i = self.kldist(q_mu_val,q_logvar_val,p_mu_val,p_logvar_val)

                # cat_distr = OneHotCategorical(q_i[-1])
                # z_i = cat_distr.sample().unsqueeze_(0)
                vz_i = torch.argmax(nn.Softmax(dim=-1)(z_i), dim=-1)
#                 x = torch.cat([rnn_w[i], temb[vz_i]], dim=-1)
                x = temb[vz_i]

                # x = torch.cat([rnn_w[i],z_i], dim=-1)
                # x = nn.Softmax(dim=-1)(z_i)
                # x = z_i
                h_i, c_i = self.lstmcell(x, (h_0, c_0))
                h_0, c_0 = h_i, c_i

                # q_list += [q_i]
                # p_list += [p_i]
                z_list += [z_i]
                kl_list += [kl_i]
            except RuntimeError as e:
                print(e)

        # rnn_s = torch.cat([pad_h_s,rnn_s[:-1]],dim=0)
        # _p = lin_p(rnn_s) # remove last P
        # pack_q = torch.cat([rnn_s,rnn_w], dim=-1)
        # _q = lin_q(pack_q)
        # _p = torch.stack(p_list)
        # _q = torch.stack(q_list)
        _z = torch.stack(z_list)
        _kl = torch.stack(kl_list)
        return _z, _kl