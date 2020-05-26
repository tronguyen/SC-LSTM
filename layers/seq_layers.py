import torch
import torch.nn as nn
from torch.nn import Parameter
from torch.autograd import Variable

class SCLSTM_Cell(nn.Module):
    def __init__(self, input_size, hidden_size, kwd_vocab_sz, dropout=0.0):
        super(SCLSTM_Cell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
 
        self.weight = Parameter(torch.Tensor(input_size + hidden_size, hidden_size * 4))
        self.bias = Parameter(torch.Tensor(hidden_size * 4))
        self.wd = Parameter(torch.Tensor(kwd_vocab_sz, hidden_size))
    
    def forward(self, x, h_c, d):
        bs, _ = x.size()
        hs = self.hidden_size
#         if h_c is None:
#             h_t, c_t = (torch.zeros(bs, self.hidden_size).to(x.device), 
#                         torch.zeros(bs, self.hidden_size).to(x.device))
#         else:
        h_t, c_t = h_c
        
        x_h = torch.cat([x, h_t], dim=-1)
        lin_xh = torch.mm(x_h, self.weight) + self.bias
        i_t, f_t, g_t, o_t = (
            torch.sigmoid(lin_xh[:, :hs]),
            torch.sigmoid(lin_xh[:, hs:hs*2]),
            torch.tanh(lin_xh[:, hs*2:hs*3]),
            torch.sigmoid(lin_xh[:, hs*3:])
        )

        c_t = f_t*c_t + i_t*g_t + torch.tanh(torch.mm(d, self.wd))
        h_t = o_t * torch.tanh(c_t)
#         hidden_seq.append(h_t.unsqueeze(0))
#         hidden_seq = torch.cat(hidden_seq, dim=0) # [s x b x f]
#         hidden_seq = hidden_seq.transpose(0, 1)
        return h_t, c_t

class SCLSTM_MultiCell(nn.Module):

    def __init__(self, num_layers, input_size, hidden_size, kwd_vocab_sz, dropout=0.0):
        super(SCLSTM_MultiCell, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        nxt_size = input_size

        for i in range(num_layers):
            _cell = SCLSTM_Cell(nxt_size, hidden_size, kwd_vocab_sz)
            self.layers.append(_cell)
            nxt_size = hidden_size + input_size

    def forward(self, x, h_c, d):
        """
        Args:
            x: [batch_size, input_size]
            h_c: [2, num_layers, batch_size, hidden_size]
        Return:
            last_h_c: [2, batch_size, hidden_size] (h from last layer)
            h_c_list: [2, num_layers, batch_size, hidden_size] (h and c from all layers)
        """
        bs, _ = x.size()
        h_0, c_0 = h_c
        h_list, c_list = [], []
        for i, layer in enumerate(self.layers):
            if i>0:
                x_in = torch.cat([x, h_i], dim=-1)
            else:
                x_in = x
                
            h_i, c_i = layer(x_in, (h_0[i], c_0[i]), d)
            h_list += [h_i]
            c_list += [c_i]
            
            # x for next layer
            if i + 1 != self.num_layers:
                h_i = self.dropout(h_i)
                        
#         last_h_c = (h_list[-1], c_list[-1])
        flat_h = torch.cat(h_list, dim=-1)
        h_list = torch.stack(h_list) # [n x b x f]
        c_list = torch.stack(c_list)
        h_c_list = (h_list, c_list)

        return flat_h, h_c_list

    
    
        
        
    