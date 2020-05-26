# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

import numpy as np
from tqdm import tqdm, trange

from layers.metalayer import vaenc, simple_dec

from utils import TensorboardWriter

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator
from colorama import Fore
from colorama import Style
import os
from datetime import datetime
import shutil
import pickle
# import scipy.optimize

np.random.seed(112312)
class Solver(object):
    def __init__(self, config=None, train_loader=None, test_loader=None, valid_loader=None):
        """Class that Builds, Trains and Evaluates SUM-GAN model"""
        self.config = config
        self.train_loader = train_loader
        os.environ["CUDA_VISIBLE_DEVICES"] = self.config.gpu
        f_vocab = './output/vocab.dat'
        self.rev_vocab = dict()
        with open(f_vocab, 'r') as f:
            for line in f:
                wi = line.split('\t')
                self.rev_vocab[int(wi[1])] = wi[0]

    def build(self):
        # Build Modules
        self.vaenc = vaenc(self.config.mlp_size, self.config.hidden_size, self.config).cuda()
#         self.vaenc = vaenc(self.config.topic_size, self.config.hidden_size, self.config)
        
        self.dec = simple_dec(self.config.topic_size, self.config.vocab_size).cuda()
        
        self.enc_x = nn.Sequential(
            nn.Linear(self.config.vocab_size, self.config.mlp_size*2),
#             nn.ReLU(),
#             nn.Linear(self.config.mlp_size*2, self.config.mlp_size),
#             nn.ReLU()
        ).cuda()

        self.prob_q = nn.Sequential(
            nn.Linear(self.config.mlp_size*2, self.config.mlp_size),
            nn.ReLU(),
#             nn.Linear(self.config.hidden_size, self.config.hidden_size),
#             nn.ReLU()
        ).cuda()
        self.Q_mu = nn.Linear(self.config.mlp_size, self.config.topic_size).cuda()
        self.Q_var = nn.Sequential(
            nn.Linear(self.config.mlp_size, self.config.topic_size),
            nn.Softplus()
        ).cuda()

        self.prob_p = nn.Sequential(
            nn.Linear(self.config.hidden_size, self.config.hidden_size),
            nn.ReLU(),
            nn.Linear(self.config.hidden_size, self.config.hidden_size),
            nn.ReLU(),
            nn.Linear(self.config.hidden_size, self.config.hidden_size),
            nn.ReLU()
        ).cuda()
        self.P_mu = nn.Linear(self.config.hidden_size, self.config.topic_size).cuda()
        self.P_var = nn.Sequential(
            nn.Linear(self.config.hidden_size, self.config.topic_size),
            nn.Softplus()
        ).cuda()

        self.model = nn.ModuleList([
            self.vaenc, self.dec, self.prob_q, self.prob_p,
            self.Q_mu, self.Q_var, self.P_mu, self.P_var, self.enc_x
        ])

        if self.config.mode == 'train':

            # Overview Parameters
            print('Init Model Parameters')
            for name, param in self.model.named_parameters():
                print('\t' + name + '\t', list(param.size()))
                if 'weight' in name and 'bnorm' not in name:
#                     torch.nn.init.xavier_normal_(param)
                    torch.nn.init.xavier_normal_(param)
                if 'bias' in name:
                    torch.nn.init.constant_(param, 0.)

            # Tensorboard
            # if self.config.write_model:
            self.current_time = datetime.now().strftime('%b%d_%H-%M-%S')
            self.writer = TensorboardWriter(self.config.logdir + '/' + self.current_time)
            # Add emb-layer
            self.model.train()
            # create dir
            os.makedirs('./Result/fullads/{}'.format(self.current_time))
            os.makedirs('./model/{}'.format(self.current_time))

        # Build Optimizers
        self.optimizer = optim.Adam(
            list(self.model.parameters()),
            lr=self.config.lr
        )
#         print(self.model)

    def load_model(self):
        _fname = './{}/chk_point_{}.pth'.format(self.config.modeldir, self.config.resume_ep)
        if os.path.isfile(_fname):
            print("=> loading checkpoint '{}'".format(_fname))
            if int(self.config.gpu) < 0:
                checkpoint = torch.load(_fname, map_location=lambda storage, loc: storage) # load into cpu-mode
            else:
                checkpoint = torch.load(_fname) # gpu-mode
            self.start_epoch = checkpoint['epoch']
            # checkpoint['state_dict'].pop('1.s_lstm.out.0.bias',None) # remove bias in selector
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'][0])
            self.topicemb = checkpoint['topic_emb']
        else:
            print("=> no checkpoint found at '{}'".format(_fname))

    def reconstruction_loss(self, h_origin, h_fake):
        """L2 loss between original-regenerated features at cLSTM's last hidden layer"""
        return torch.mean(torch.pow(h_origin - h_fake, 2))

    def prior_loss(self, mu, log_variance):
        """KL( q(e|x) || N(0,1) )"""
        kl = -1 + log_variance.exp() + mu.pow(2) - log_variance
        kl_t = torch.transpose(kl,0,1).contiguous().view(mu.shape[1], -1)

        return 0.5 * torch.mean(torch.sum(kl_t,-1))

    def entropy_loss(self, P, Q): # increasing entropy for scores
        en_p = torch.sum(P * torch.log(P), dim=-1)
        en_q = torch.sum(Q * torch.log(Q), dim=-1)
        return -(torch.mean(en_p) + torch.mean(en_q))

    def _zero_grads(self):
        self.optimizer.zero_grad()

    def save_checkpoint(self, state, filename):
        torch.save(state, filename)

    def update_param(self, module):
        for name, param in module.named_parameters():
            temp = torch.exp(param.data)
            param.data = torch.log(temp / torch.sum(temp,0).view(1,-1))

    def KLDist2DistBuck(self, Q, P):
        kl = torch.mean(torch.sum(Q * torch.log(Q / P), dim=-1))
        # kl = torch.sum(torch.sum(Q * torch.log(Q / P), dim=-1))
        # kl = T.mean(T.sum(T.switch(T.neq(Q,0) & T.neq(P,0),Q * T.log(Q/P),0), axis=1))
        return kl

    def get_norm_grad(self, module, norm_type=2):
        total_norm = 0
        for name, param in module.named_parameters():
            if param.grad is not None:
                total_norm += torch.sum(torch.pow(param.grad.view(-1),2))
        return torch.sqrt(total_norm)


    def clear_grad_pad(self):
        grad = self.embedding.weight.grad
        grad[0] = torch.zeros_like(grad[0])

    def diversify(self, mxt): # mxt [nv x nt]
        mxt = torch.div(mxt, torch.norm(mxt, p=2, dim=0, keepdim=True).expand_as(mxt))
        mxt_t = torch.transpose(mxt, 0, 1)
        _mm = torch.mm(mxt_t, mxt) # [nt x nt]
        _msk = Variable(torch.ByteTensor(np.tril(np.ones([mxt.shape[1], mxt.shape[1]]), 1))).cuda()
        _val = torch.masked_select(_mm, _msk)
        d_mean = torch.mean(_val) #/Variable(int(mxt.shape[1]*(mxt.shape[1]-1)/2))
        d_var = torch.mean(torch.pow(_val - d_mean, 2))
        return d_mean + d_var
        
    def train_rnn(self):
        kl_list = []
        nllk_list = []
        loss_list = []
        l1_list = []
        dv_list = []
        for batch_i, doc_features in enumerate(tqdm(
                self.train_loader, desc='Batch-RNN', dynamic_ncols=True, ascii=True)):

            self._zero_grads()

            _x, _y, corpus_id = doc_features
            corpus_id = corpus_id.numpy()
            var_x_ = Variable(_x, requires_grad=False).cuda()
            var_y = Variable(_y, requires_grad=False).cuda()
            
            # [b_len x s_len x vocab]
            var_x = torch.squeeze(var_x_, 0).transpose(0, 1)     
            enc_x = self.enc_x(var_x)
            zQ, kl_seq = self.vaenc(enc_x, self.prob_q, self.prob_p, self.Q_mu, self.Q_var, 
                                    self.P_mu, self.P_var)
            
            nllk = self.dec(zQ, var_x)
            kl_loss = torch.mean(torch.sum(kl_seq,0)) * self.config.kl
            l1_reg = torch.mean(torch.sum(torch.sum(torch.abs(zQ), -1),0)) * self.config.lbd
            dv_loss = self.diversify(self.dec.w_distr.weight) * self.config.dv
            
            batch_loss = kl_loss + nllk + l1_reg + dv_loss
    
            batch_loss.backward(retain_graph=True)
            self.optimizer.step()

            loss_list.append(batch_loss.cpu().data.numpy())
            nllk_list.append(nllk.cpu().data.numpy())
            kl_list.append(kl_loss.cpu().data.numpy())
            l1_list.append(l1_reg.cpu().data.numpy())
            dv_list.append(dv_loss.cpu().data.numpy())
        
        Q = nn.Softmax(dim=-1)(zQ)
        argmax = torch.argmax(Q[:,-1,:], -1)
        print('\n-Doc: '.format(corpus_id[-1]))
        print(argmax.cpu().data.numpy().flatten())
#         print(Q[-3:,-1,:].data.numpy())
        
        return loss_list, nllk_list, kl_list, l1_list, dv_list


    def train(self):
        print('***Init all variables ...')
        # self.initFunc()

        print('***Start training ...')
        for epoch_i in trange(self.config.n_epoch, desc='Epoch', ncols=80):
            loss_list, nllk_list, kl_list, l1_list, dv_list = self.train_rnn()
            # Save parameters at checkpoint
            if (epoch_i + 1) % self.config.eval_rate == 0:
                self.evaluate(epoch_i + 1)
                if self.config.write_model:
                    # save model
                    self.save_checkpoint({
                        'epoch': epoch_i + 1,
                        'state_dict': self.model.state_dict(),
                        'total_loss': np.mean(loss_list),
                        'optimizer': [self.optimizer.state_dict()],
                    }, filename='./model/{}/chk_point_{}.pth'.format(self.current_time, epoch_i + 1))
#                     self.save_checkpoint(self.topicemb, filename='./model/chk_point_topic_{}.pth'.format(epoch_i + 1))

            print('\n***Ep-{} | Total_loss: {}| KL: {}| NLLK(P/N): {}/0| L1: {}| DV: {}| NORM: {}'.format(
                        epoch_i,np.mean(loss_list),np.mean(kl_list),np.mean(nllk_list),
                        np.mean(l1_list), np.mean(dv_list),
                        self.get_norm_grad(self.model).data
                        ))

            self.writer.update_parameters(self.model, epoch_i)
            self.writer.update_loss(np.mean(loss_list), epoch_i, 'total_loss')
            self.writer.update_loss(np.mean(kl_list), epoch_i, 'kl_loss')
            self.writer.update_loss(np.mean(l1_list), epoch_i, 'l1_reg')
            self.writer.update_loss(np.mean(nllk_list), epoch_i, 'nllk')


    def evaluate(self, epoch_i):
        def myplot(distr_list, words):
            labels = list(words)
            xs = range(len(labels))
            def format_fn(tick_val, tick_pos):
                if int(tick_val) in xs:
                    return labels[int(tick_val)]
                else:
                    return ''
            fig, axs = plt.subplots(len(distr_list), 1, sharey=True, figsize=(50,100))

            for i in range(len(distr_list)):
                axs[i].xaxis.set_major_formatter(FuncFormatter(format_fn))
                axs[i].xaxis.set_major_locator(MaxNLocator(nbins=50,integer=True))
                # axs[i].xaxis.set_xticks()
                values = list(distr_list[i])
                # axs[0].bar(names, values)
                # axs[1].scatter(names, values)
                axs[i].plot(xs, values)
            fig.suptitle('Categorical Plotting')
            fig.savefig('./Result/fullads/{}/distr-viz-ep{}.eps'.format(self.current_time,epoch_i),format='eps', dpi=1000)

        self.model.eval()
        f_vocab = './output/vocab.dat'
        f_out = open('./Result/fullads/{}/distr-view-ep{}.dat'.format(self.current_time,epoch_i), 'w')
        rev_vocab = dict()
        vocab = []
        with open(f_vocab,'r') as f:
            for line in f:
                wi = line.split('\t')
                rev_vocab[int(wi[1])]= wi[0]
                vocab.append(wi[0])
                
        w_distr = self.dec.w_distr.weight.detach().cpu().numpy()
        w_distr = w_distr[1:,:]
        w_distr = np.exp(w_distr)/np.sum(np.exp(w_distr),axis=0)
        nv = w_distr.shape[0]
        nt = w_distr.shape[1]

        # plot
        myplot(list(w_distr[:500,:].transpose()), vocab[:500])

        for t in range(nt):
            _, sortedvb = zip(*sorted(zip(w_distr[:,t], vocab), reverse=True))
            sortedvb = list(sortedvb)
            newline = ''
            for w in sortedvb:
                newline = newline + '\t' + w
            f_out.write(newline.strip() + '\n')

        f_out.close()
        self.model.train()

        np.savetxt('./Result/fullads/{}/beta-view-ep{}.dat'.format(self.current_time,epoch_i), w_distr, delimiter='\t')
        # np.savetxt('./output/bg-view-ep{}.dat'.format(epoch_i), self.backgroundWords, delimiter='\t')


if __name__ == '__main__':
    pass
