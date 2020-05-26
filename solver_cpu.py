# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

import numpy as np
from tqdm import tqdm, trange

from layers.seqlayer import sLSTM, mLinear
from layers.vrnnlayer import vrnn

from utils import TensorboardWriter

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator
from colorama import Fore
from colorama import Style
import os
from datetime import datetime
from __init__ import LOGGER
import shutil
import pickle

class Solver(object):
    def __init__(self, config=None, train_loader=None, test_loader=None, valid_loader=None):
        """Class that Builds, Trains and Evaluates SUM-GAN model"""
        self.config = config
        self.train_loader = train_loader
        os.environ["CUDA_VISIBLE_DEVICES"] = self.config.gpu

    def build(self):
        # Build Modules
        self.embedding = nn.Embedding(self.config.vocab_size, self.config.input_size, padding_idx=0)
        if False:
            weights_matrix = pickle.load(open('./output/glove_weight.dat','rb'))
            self.embedding.weight.data = torch.Tensor(weights_matrix)
            self.embedding.weight.requires_grad = False

        self.beta = mLinear(self.config.topic_size, self.config.vocab_size)
        self.bg_distr = mLinear(1, self.config.vocab_size)
        # nn.init.uniform(self.beta.weight)

        self.rnn_w = sLSTM(self.config.input_size, self.config.hidden_size)
        # self.rnn_s = sLSTM(self.config.hidden_size+self.config.topic_size, self.config.hidden_size, isbidirection=False)

        self.prob_q = nn.Sequential(
            nn.Linear(self.config.hidden_size*2,self.config.hidden_size),
            nn.ReLU(),
            nn.Linear(self.config.hidden_size, int(self.config.hidden_size/2)),
            nn.ReLU(),
            nn.Linear(int(self.config.hidden_size/2), int(self.config.hidden_size / 4)),
            nn.Tanh(),
            nn.Linear(int(self.config.hidden_size / 4), self.config.topic_size),
            nn.Softmax(dim=-1)
        )

        self.prob_p = nn.Sequential(
            nn.Linear(self.config.hidden_size, self.config.hidden_size),
            nn.Tanh(),
            nn.Linear(self.config.hidden_size, int(self.config.hidden_size/2)),
            nn.Tanh(),
            nn.Linear(int(self.config.hidden_size/2), int(self.config.hidden_size / 4)),
            nn.Tanh(),
            nn.Linear(int(self.config.hidden_size / 4), self.config.topic_size),
            nn.Softmax(dim=-1)
        )

        # self.vrnn = vrnn(self.config.hidden_size+self.config.topic_size,self.config.hidden_size)
        self.vrnn = vrnn(self.config.topic_size, self.config.hidden_size)

        self.model = nn.ModuleList([
            self.embedding, self.rnn_w, self.vrnn, self.prob_q, self.prob_p, self.beta, self.bg_distr])

        # Init for LDA
        self.wordTopicCounts = np.zeros([self.config.vocab_size, self.config.topic_size])
        self.topicCounts = np.zeros(self.config.topic_size)


        if self.config.mode == 'train':
            self.model.train()
            # self.model.apply(apply_weight_norm)

            # Overview Parameters
            LOGGER.info('Init Model Parameters')
            for name, param in self.model.named_parameters():
                print('\t' + name + '\t', list(param.size()))
                if 'weight' in name and 'bnorm' not in name:
                    torch.nn.init.xavier_normal_(param)
                if 'bias' in name:
                    torch.nn.init.constant_(param, 0.)

            # Tensorboard
            # if self.config.write_model:
            current_time = datetime.now().strftime('%b%d_%H-%M-%S')
            self.writer = TensorboardWriter(self.config.logdir + '/' + current_time)

        self.update_param(self.beta)
        self.update_param(self.bg_distr)

        # Build Optimizers
        self.optimizer = optim.Adam(
            list(self.model.parameters()),
            lr=self.config.lr
        )
        LOGGER.info(self.model)

    def load_model(self):
        _fname = './{}/improv_chk_point_{}.pth'.format(self.config.modeldir, self.config.resume_ep)
        if os.path.isfile(_fname):
            LOGGER.info("=> loading checkpoint '{}'".format(_fname))
            if int(self.config.gpu) < 0:
                checkpoint = torch.load(_fname, map_location=lambda storage, loc: storage) # load into cpu-mode
            else:
                checkpoint = torch.load(_fname) # gpu-mode
            self.start_epoch = checkpoint['epoch']
            # checkpoint['state_dict'].pop('1.s_lstm.out.0.bias',None) # remove bias in selector
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'][0])
        else:
            LOGGER.info("=> no checkpoint found at '{}'".format(_fname))

    def reconstruction_loss(self, h_origin, h_fake):
        """L2 loss between original-regenerated features at cLSTM's last hidden layer"""
        return torch.mean(torch.pow(h_origin - h_fake, 2))

    def reconstruction_loss_(self, f_org, f_gen):
        return torch.mean(torch.sum(torch.pow(f_org - f_gen, 2), -1))

    def prior_loss(self, mu, log_variance):
        """KL( q(e|x) || N(0,1) )"""
        kl = -1 + log_variance.exp() + mu.pow(2) - log_variance
        kl_t = torch.transpose(kl,0,1).contiguous().view(mu.shape[1], -1)

        return 0.5 * torch.mean(torch.sum(kl_t,-1))

    def sparsity_loss(self, scores):
        """Summary-Length Regularization"""
        return torch.mean(torch.abs(torch.mean(scores,0) - self.config.summary_rate))

    def gan_loss(self, original_prob, fake_prob, uniform_prob):
        """Typical GAN loss + Classify uniformly scored features"""

        gan_loss = torch.mean(torch.log(original_prob) + torch.log(1. - fake_prob)
                              + torch.log(1. - uniform_prob))  # Discriminate uniform score

        return gan_loss

    def entropy_loss(self, P, Q): # increasing entropy for scores
        en_p = torch.sum(P * torch.log(P), dim=-1)
        en_q = torch.sum(Q * torch.log(Q), dim=-1)
        return -(torch.mean(en_p) + torch.mean(en_q))

    def entropy_topic(self):
        _e = Variable(torch.eye(self.config.topic_size), requires_grad=False)
        beta = self.beta(_e)
        kl_list = []
        for i in range(self.config.topic_size):
            for j in range(i+1, self.config.topic_size):
                Q = beta[i]
                P = beta[j]
                kl_list.append(torch.sigmoid(-torch.sum(Q * torch.log(Q / P),dim=-1)))
        return sum(kl_list)/(self.config.topic_size*(self.config.topic_size-1)/2)


    def L1_loss(self, scores):
        return torch.mean(torch.sum(torch.abs(scores),0))

    def _zero_grads(self):
        self.optimizer.zero_grad()

    def track_grad(self, module, mode, printable=True):
        _mn = 0.
        for name, param in module.named_parameters():
            _i = torch.mean(param.grad.data)
            _mn += _i
            if printable:
                print('\nMODE-{} \t Name: {} - MeanGrad: {}'.format(mode, name, _i))
        return _mn

    def track_param(self, module, mode, printable=True):
        _mn = 0.
        for name, param in module.named_parameters():
            _i = torch.mean(param.data)
            _mn += _i
            if printable:
                print('\nMODE-{} \t Name: {} - MeanParam: {}'.format(mode, name, _i))
        return _mn

    def criterion(self, pred, target):
        return torch.mean(torch.pow(pred-target,2))

    def cont_loss(self, scores):
        return torch.mean(torch.sum(torch.abs(scores[1:]-scores[:-1]),0))

    def save_checkpoint(self, state, filename):
        torch.save(state, filename)

    def update_param(self, module):
        for name, param in module.named_parameters():
            temp = torch.exp(param.data)
            param.data = torch.log(temp / torch.sum(temp,0).view(1,-1))

    def get_norm_param(self, module):
        _mn = 0
        for name, param in module.named_parameters():
            _mn += torch.sum(torch.pow(param.view(-1),2))
        return torch.sqrt(_mn)

    def get_norm_grad(self, module, norm_type=2):
        total_norm = 0
        for name, param in module.named_parameters():
            if param.grad is not None:
                total_norm += torch.sum(torch.pow(param.grad.view(-1),2))
        return torch.sqrt(total_norm)

    def KLDist2DistBuck(self, Q, P):
        kl = torch.mean(torch.sum(Q * torch.log(Q / P), dim=-1))
        # kl = T.mean(T.sum(T.switch(T.neq(Q,0) & T.neq(P,0),Q * T.log(Q/P),0), axis=1))
        return kl

    def logLK(self, var_x, Q, stop_tensor):
        Q_one = Variable(torch.ones(Q.shape[0], Q.shape[1], 1), requires_grad=False)
        prob_x = (1.-self.config.eps_reg)*self.beta(Q) + self.config.eps_reg*self.bg_distr(Q_one)
        log_prob_x = torch.log(prob_x)
        mask_log = Variable(torch.zeros(log_prob_x.shape), requires_grad=False)
        var_x_ = torch.unsqueeze(var_x,1)
        mask_log = mask_log.scatter_(-1,var_x_,1)

        mask_log = mask_log.scatter_(-1, stop_tensor.repeat(mask_log.shape[0],mask_log.shape[1],1), 0) # remove stop-words
        log_prob_x_ = (log_prob_x * mask_log)[:,:,1:] # remove pad-word
        llk = torch.sum(log_prob_x_,-1)
        return torch.mean(llk) # for len of doc = num of sentences

    # Apply L-BFGS algorithm
    def train_topic(self):
        pass


    def train(self):
        LOGGER.info('***Start training ...')
        stopword = pickle.load(open('./output/stopword_index.dat','rb'))
        stop_tensor = Variable(torch.LongTensor(stopword), requires_grad=False)

        for epoch_i in trange(self.config.n_epoch, desc='Epoch', ncols=80):
            total_loss = []
            kl_list = []
            nllk_list = []
            entropy_loss_list = []
            entropy_topic_list = []
            for batch_i, doc_features in enumerate(tqdm(
                    self.train_loader, desc='Batch', ncols=80, leave=False)):

                self._zero_grads()
                _x, _y = doc_features
                var_x_ = Variable(_x,requires_grad=False)
                var_y = Variable(_y,requires_grad=False)
                # [s_len x w_len]
                var_x = torch.squeeze(var_x_,0)
                emb_x = self.embedding(var_x).transpose_(0,1)
                sen_x_ = self.rnn_w(emb_x)[-1,:,:]
                sen_x = torch.unsqueeze(sen_x_, 1)
                # doc_x = self.rnn_s(sen_x)
                P, Q = self.vrnn(self.prob_q, self.prob_p, sen_x, self.config)

                # calculate all losses
                kl_loss = self.config.kl_reg*self.KLDist2DistBuck(Q,P)
                llk = self.logLK(var_x,Q,stop_tensor)
                entro_loss = self.config.entropy_reg*self.entropy_loss(P,Q)
                entro_topic = self.entropy_topic()

                bat_loss = kl_loss - llk + entro_loss + entro_topic
                bat_loss.backward(retain_graph=True)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.clip)
                self.optimizer.step()
                self.update_param(self.beta)
                self.update_param(self.bg_distr)
                # print('\n-batch_loss: {}'.format(bat_loss.data[0]))
                kl_list.append(kl_loss.data)
                nllk_list.append(-llk.data)
                entropy_topic_list.append(entro_topic.data)
                entropy_loss_list.append(entro_loss.data)

                total_loss.append(bat_loss.data)

            # Save parameters at checkpoint
            if (epoch_i + 1) % self.config.eval_rate == 0:
                self.evaluate(epoch_i + 1)
                if self.config.write_model:
                    # save model
                    self.save_checkpoint({
                        'epoch': epoch_i + 1,
                        'state_dict': self.model.state_dict(),
                        'total_loss': np.mean(total_loss),
                        'optimizer': [self.optimizer.state_dict()],
                    }, filename='./model/improv_chk_point_{}.pth'.format(epoch_i + 1))

            tqdm.write('\n***Ep-{} | Total_loss : {} | KL: {} | NLLK: {} | ENTRO: {} | NORM: {}'.format(
                        epoch_i,np.mean(total_loss),np.mean(kl_list),np.mean(nllk_list),np.mean(entropy_loss_list),
                        self.get_norm_grad(self.model).data
                        ))

            self.writer.update_parameters(self.model, epoch_i)
            self.writer.update_loss(np.mean(total_loss), epoch_i, 'total_loss')
            self.writer.update_loss(np.mean(kl_list), epoch_i, 'kl_loss')
            self.writer.update_loss(np.mean(nllk_list), epoch_i, 'nllk')
            self.writer.update_loss(np.mean(entropy_loss_list), epoch_i, 'entropy_loss')
            self.writer.update_loss(np.mean(entropy_topic_list), epoch_i, 'entropy_topic')

    def evaluate(self, epoch_i):
        def myplot(distr_list, words):
            labels = list(words)
            xs = range(len(labels))
            def format_fn(tick_val, tick_pos):
                if int(tick_val) in xs:
                    return labels[int(tick_val)]
                else:
                    return ''
            fig, axs = plt.subplots(len(distr_list), 1, sharey=True)

            for i in range(len(distr_list)):
                axs[i].xaxis.set_minor_formatter(FuncFormatter(format_fn))
                axs[i].xaxis.set_minor_locator(MaxNLocator(integer=True))
                values = list(distr_list[i])
                # axs[0].bar(names, values)
                # axs[1].scatter(names, values)
                axs[i].plot(xs, values)
            fig.suptitle('Categorical Plotting')
            fig.savefig('./output/distr-viz.png')

        self.model.eval()
        f_vocab = './output/vocab.dat'
        f_out = open('./output/distr-view-ep{}.dat'.format(epoch_i), 'w')
        rev_vocab = dict()
        vocab = []
        with open(f_vocab,'r') as f:
            for line in f:
                wi = line.split('\t')
                rev_vocab[int(wi[1])]= wi[0]
                vocab.append(wi[0])

        w_distr = self.beta.weight.data.numpy()
        b_distr = self.bg_distr.weight.data.numpy()
        w_distr = np.concatenate([w_distr, b_distr], axis=1)
        nv = w_distr.shape[0]
        nt = w_distr.shape[1]

        # plot
        myplot(list(np.exp(w_distr).transpose()), ['pad_w']+ vocab)

        for t in range(nt):
            _, sortedvb = zip(*sorted(zip(w_distr[:,t], vocab), reverse=True))
            sortedvb = list(sortedvb)
            newline = ''
            for w in sortedvb:
                newline = newline + '\t' + w
            f_out.write(newline.strip() + '\n')

        f_out.close()
        self.model.train()



if __name__ == '__main__':
    pass
