import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.distributions.one_hot_categorical import OneHotCategorical

class vaenc(nn.Module):
    def __init__(self, input_size, hidden_size, config):
        """Scoring LSTM"""
        super(vaenc, self).__init__()
        self.config = config
        self.lstmcell = nn.LSTMCell(hidden_size, hidden_size)
#         self.lin_q = nn.Sequential(
#             nn.Linear(self.config.mlp_size, hidden_size),
#             nn.Tanh()
#         )
        
        self.lin_x = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
#             nn.Linear(hidden_size, hidden_size),
#             nn.ReLU()
        )
        
        self.lin_xh = nn.Sequential(
            nn.Linear(self.config.mlp_size*2 + hidden_size, self.config.mlp_size*2),
            nn.ReLU(),
#             nn.Linear(self.config.mlp_size, self.config.mlp_size),
#             nn.ReLU()
        )
        
    
    def gauss_sampling(self, mu, log_variance):

        std = torch.exp(0.5 * log_variance)

        # e ~ N(0,1)
        epsilon = Variable(torch.randn(std.shape), requires_grad=False).cuda()

        # [num_layers, 1, hidden_size]
        return mu + epsilon * std

    def kldist(self, q_mu, q_logvar, p_mu, p_logvar):
        kl = torch.sum(0.5 * (p_logvar - q_logvar +
                          (torch.exp(q_logvar) + (q_mu - p_mu) ** 2) /
                          torch.exp(p_logvar) - 1), dim=-1)
        return kl

    def forward(self, sen_t, lin_q, lin_p, q_mu, q_var, p_mu, p_var):
        pad_h_s = Variable(torch.zeros(self.config.batch_size,self.config.hidden_size), requires_grad=False).cuda()
        seq_len = sen_t.shape[0]
        h_i, c_i = pad_h_s, pad_h_s

        p_list, q_list, z_list, kl_list = [], [], [], []
        for i in range(seq_len):
            try:
#                 in_q = self.lin_xh(torch.cat([sen_t[i], h_i], dim=-1))
                q_i = lin_q(sen_t[i])
                q_mu_val = q_mu(q_i)
                q_logvar_val = torch.log(q_var(q_i))
                z_i = self.gauss_sampling(q_mu_val, q_logvar_val)

                p_i = lin_p(h_i)
                p_mu_val = p_mu(p_i)
                p_logvar_val = torch.log(p_var(p_i))

                kl_i = self.kldist(q_mu_val,q_logvar_val,p_mu_val,p_logvar_val)
                
#                 x = torch.cat([q_i, z_i], dim=-1)
                x = q_i
                
                x = self.lin_x(x)
                
                h_i, c_i = self.lstmcell(x, (h_i, c_i))

                # q_list += [q_i]
                # p_list += [p_i]
                z_list += [z_i]
                kl_list += [kl_i]
            except RuntimeError as e:
                print(e)

        _z = torch.stack(z_list)
        _kl = torch.stack(kl_list)
        return _z, _kl
    
class simple_dec(nn.Module):
    def __init__(self, hidden_size, output_size):
        """Scoring LSTM"""
        super(simple_dec, self).__init__()
        self.w_distr = nn.Linear(hidden_size, output_size, bias=False)
#         self.w_distr.weight.data = torch.div(torch.exp(self.w_distr.weight), 
#                         torch.sum(torch.exp(self.w_distr.weight), dim=0, keepdim=True).expand_as(self.w_distr.weight))
        self.linear = nn.Sequential(
            self.w_distr,
            nn.LogSoftmax(dim=-1)
        )
        
        
    def forward(self, z_list, x_list):
        assert len(z_list)==len(x_list)
        act = self.linear(z_list)
        logprob = act * x_list
        logprob = torch.sum(torch.sum(logprob, -1), 0)
        
#         logprob_neg = act * neg_x_list
#         logprob_neg = torch.sum(torch.sum(logprob_neg, -1), 0)
        
        return -torch.mean(logprob) # mean by batches
        
        
        
    