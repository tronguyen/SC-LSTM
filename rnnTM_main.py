# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

import numpy as np
from tqdm import tqdm, trange

from layers.seqlayer import sLSTM, mLinear, dLSTM
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
# from __init__ import LOGGER
import shutil
import pickle
# import scipy.optimize

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
        self.embedding = nn.Embedding(self.config.vocab_size+1, self.config.input_size, padding_idx=0)
        self.topicemb = Variable(torch.FloatTensor(self.config.topic_size, self.config.input_size), requires_grad=True)

        if False:
            weights_matrix = pickle.load(open('./output/glove_weight_50d.dat','rb'), encoding='latin1')
            self.embedding.weight.data = torch.Tensor(weights_matrix)
            self.embedding.weight.requires_grad = False

        self.rnn_w = sLSTM(self.config.input_size, self.config.hidden_size, num_layers=1)
        self.dLSTM = dLSTM(self.config.input_size*2, self.config.d_hidden_size,
                           self.config.topic_size, self.config.MAX_WORD, self.config.batch_size, self.config.vocab_size, self.rev_vocab)

        self.prob_q = nn.Sequential(
            nn.Linear(self.config.hidden_size, self.config.hidden_size),
            nn.ReLU(),
            nn.Linear(self.config.hidden_size, self.config.hidden_size),
            nn.ReLU(),
            # nn.Linear(self.config.hidden_size, self.config.hidden_size),
            # nn.ReLU()
        )
        self.Q_mu = nn.Linear(self.config.hidden_size, self.config.topic_size)
        self.Q_var = nn.Sequential(
            nn.Linear(self.config.hidden_size, self.config.topic_size),
            nn.Softplus()
        )

        self.prob_p = nn.Sequential(
            nn.Linear(self.config.hidden_size, self.config.hidden_size),
            nn.ReLU(),
            nn.Linear(self.config.hidden_size, self.config.hidden_size),
            nn.ReLU(),
            # nn.Linear(self.config.hidden_size, self.config.hidden_size),
            # nn.ReLU()
        )
        self.P_mu = nn.Linear(self.config.hidden_size, self.config.topic_size)
        self.P_var = nn.Sequential(
            nn.Linear(self.config.hidden_size, self.config.topic_size),
            nn.Softplus()
        )

        # self.vrnn = vrnn(self.config.hidden_word_size+self.config.topic_size,self.config.hidden_size)
        # self.vrnn = vrnn(self.config.hidden_size + self.config.topic_size, self.config.hidden_size, self.config.topic_size)
        self.vrnn = vrnn(self.config.hidden_size, self.config.hidden_size,
                         self.config.topic_size)
        # self.vrnn = vrnn(self.config.topic_size, self.config.hidden_size,
        #                  self.config.topic_size)

        self.model = nn.ModuleList([
            self.rnn_w, self.vrnn, self.prob_q, self.prob_p,
            self.Q_mu, self.Q_var, self.P_mu, self.P_var, self.dLSTM
        ])

        # Init for LDA
        self.wordTopicCounts = np.zeros([self.config.vocab_size, self.config.topic_size])
        self.topicCounts = np.zeros(self.config.topic_size)
        self.backgroundWords = np.zeros(self.config.vocab_size)
        self.topicWords = np.zeros([self.config.vocab_size, self.config.topic_size])
        self.kappa = 0.01 #np.random.rand()
        self.wordTopics = dict()

        # self.stopword = pickle.load(open('./output/stopword_index.dat', 'rb'))
        self.stopword = []
        self.stopword.append(0)

        # self._lambda = 1.0

        if self.config.mode == 'train':

            # Overview Parameters
            print('Init Model Parameters')
            torch.nn.init.xavier_uniform_(self.topicemb.data)
            # torch.nn.init.xavier_uniform_(self.embedding.weight.data[1:])
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
            # Add emb-layer
            self.model.train()

        self.model.append(self.embedding)
        # self.embedding.weight[0] = self.embedding.weight[0]*0.
        # Build Optimizers
        self.optimizer = optim.Adam(
            list(self.model.parameters()) + [self.topicemb],
            lr=self.config.lr
        )
        print(self.model)

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
    def fn(self, X):
        # res = 0
        # dtopicWords = np.zeros_like(self.topicWords)
        X = np.reshape(X,self.topicWords.shape)
        bg_topicword = np.expand_dims(self.backgroundWords,1) + X
        Z = np.sum(np.exp(bg_topicword),axis=0)
        lZ = np.log(Z)
        res = -np.sum((bg_topicword - lZ) * self.wordTopicCounts)

        # dtopicWord
        dtopicWords = -(self.wordTopicCounts - self.topicCounts/Z * np.exp(bg_topicword))

        return res, dtopicWords.flatten()

    def sumZ(self):
        bg_topicword = np.expand_dims(self.backgroundWords, 1) + self.topicWords
        Z = np.sum(np.exp(bg_topicword), axis=0)
        return Z

    def normalizeTopicWords(self):
        av_topic = np.mean(self.topicWords, axis=1)
        self.topicWords -= np.expand_dims(av_topic,1)
        self.backgroundWords = av_topic

    def get_norm_grad(self, module, norm_type=2):
        total_norm = 0
        for name, param in module.named_parameters():
            if param.grad is not None:
                total_norm += torch.sum(torch.pow(param.grad.view(-1),2))
        return torch.sqrt(total_norm)


    def clear_grad_pad(self):
        grad = self.embedding.weight.grad
        grad[0] = torch.zeros_like(grad[0])

    def initFunc(self):
        for batch_i, doc_features in enumerate(tqdm(
                self.train_loader, desc='Batch', ncols=80, leave=False)):
            _xb, _yb, corpus_idb = doc_features
            for b in range(self.config.batch_size):
                _x, _y, corpus_id = _xb[b,:,:], _yb[b,:], corpus_idb[b]
                cor_x = _x.squeeze_(0).numpy()
                corpus_id = np.asscalar(corpus_id.numpy())
                if corpus_id not in self.wordTopics:
                    self.wordTopics[corpus_id] = dict()
                    for doc_id, doc_x in enumerate(cor_x):
                        self.wordTopics[corpus_id][doc_id] = dict()
                        doc_x_clean = [w for w in doc_x if w not in self.stopword]
                        temp = np.random.choice(range(self.config.topic_size), len(doc_x_clean))
                        self.wordTopics[corpus_id][doc_id] = temp
                        for wp, wi_0 in enumerate(doc_x_clean):
                            wi = wi_0 - 1
                            self.wordTopicCounts[wi, temp[wp]] += 1
                            self.topicCounts[temp[wp]] += 1


    def train_rnn(self, maxiter=1):
        kl_list = []
        nllk_list = []
        total_loss_value = []
        sp_list = []
        for batch_i, doc_features in enumerate(tqdm(
                self.train_loader, desc='Batch-RNN', ncols=80, leave=False)):

            self._zero_grads()

            _x, _y, corpus_id = doc_features
            corpus_id = corpus_id.numpy()
            var_x_ = Variable(_x, requires_grad=False)
            var_y = Variable(_y, requires_grad=False)
            # [b_len x s_len x w_len]
            var_x = torch.squeeze(var_x_, 0)
            # emb_x = self.embedding(var_x).reshape(-1, var_x.shape[-1], self.config.input_size).transpose(0, 1)
            # sen_x = self.rnn_w(emb_x).reshape(var_x.shape[0], var_x.shape[1], -1).transpose(0,1)
            emb_x = self.embedding(var_x)
            sen_x = torch.sum(emb_x, 2).transpose(0, 1)
            # sen_x = torch.unsqueeze(sen_x_, 1)
            # [len x batch x wdim]
            zQ, kl_seq = self.vrnn(self.prob_q, self.prob_p,
                              sen_x, self.config,
                              self.Q_mu, self.Q_var, self.P_mu, self.P_var, self.topicemb)
            inv_var_x = Variable(torch.LongTensor(np.asarray(var_x.numpy()[:,:,::-1],dtype=float).transpose(1,0,-1)), requires_grad=False)

            Q = nn.Softmax(dim=-1)(zQ)
            sloss_lst = []
            for s in range(zQ.shape[0]):
                sprob = self.dLSTM(self.topicemb, self.embedding, Q[s], inv_var_x[s])
                sloss = torch.mean(torch.sum(sprob,-1))
                sloss_lst.append(sloss)

            # Sampling Z
            # Q = nn.Softmax(dim=-1)(zQ * self.kappa)
            # np_Q = zQ.data.numpy()
            # np_Q = np.squeeze(np_Q, 1)
            # topicCorpus = []
            # for b in range(Q.shape[1]):
            #     cor_x = _x[b].numpy()
            #     mask = np.zeros([cor_x.shape[0],self.config.topic_size])
            #     for doc_id, doc_x in enumerate(cor_x):
            #         topics = self.wordTopics[np.asscalar(corpus_id[b])][doc_id]
            #         if len(topics)>0:
            #             countTopic = np.zeros(self.config.topic_size)
            #             # for wp, wi in enumerate(doc_x[doc_x!=0]):
            #             doc_x_clean = [w for w in doc_x if w not in self.stopword]
            #             for wp, _ in enumerate(doc_x_clean):
            #                 countTopic[topics[wp]] += 1
            #
            #             # concat all topics-count
            #             # countTopicList.append(countTopic)
            #             mask[doc_id] = countTopic
            #
            #     countTopicList = np.expand_dims(mask, 1)
            #     topicCorpus.append(countTopicList)
            #
            # topicCorpus = Variable(torch.FloatTensor(np.concatenate(topicCorpus,1)),
            #                           requires_grad=False)
            pQ = torch.log(Q)
            llk = sum(sloss_lst)
            kl_loss = torch.mean(torch.sum(kl_seq,0))
            sp = torch.mean(torch.sum(torch.sum(pQ,-1),0))*0.

            batch_loss = -llk + kl_loss -sp

            batch_loss.backward(retain_graph=True)
            # self.clear_grad_pad()
#             torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.clip)
#             torch.nn.utils.clip_grad_norm_(self.topicemb.data, self.config.clip)
            self.optimizer.step()

            # print(torch.sqrt(torch.sum(torch.pow(self.topicemb.grad.view(-1),2))))

            total_loss_value.append(batch_loss.data.numpy())
            nllk_list.append(-llk.data.numpy())
            kl_list.append(kl_loss.data.numpy())
            sp_list.append(-sp.data.numpy())

        argmax = torch.argmax(Q[:,-1,:], -1)
        print('\n-Doc: {}'.format(corpus_id[-1]))
        print(argmax.data.numpy().flatten())

        return total_loss_value, nllk_list, kl_list, sp_list

    def sample_Z(self, sample=False):
        # sZ = self.sumZ()
        for batch_i, doc_features in enumerate(tqdm(
                self.train_loader, desc='Batch-TM', ncols=80, leave=False)):

            _x, _y, corpus_idb = doc_features
            corpus_idb = corpus_idb.numpy()
            var_x_ = Variable(_x, requires_grad=False)
            var_y = Variable(_y, requires_grad=False)
            # [s_len x w_len]
            var_x = torch.squeeze(var_x_, 0)
            emb_x = self.embedding(var_x).reshape(-1,var_x.shape[-1], self.config.input_size).transpose_(0, 1)
            sen_x = self.rnn_w(emb_x).reshape(var_x.shape[1], var_x.shape[0],-1)
            # sen_x = torch.unsqueeze(sen_x_, 1)
            # [len x batch x nTopic]
            zQ, _ = self.vrnn(self.prob_q, self.prob_p, sen_x, self.config,
                              self.Q_mu, self.Q_var, self.P_mu, self.P_var)

            # calculate all losses
            # kl_loss = self.config.kl_reg*self.KLDist2DistBuck(Q,P)

            # kl_loss = torch.sum(kl_seq)
            # kl_list.append(kl_loss.data.numpy())

            # llk = self.logLK(var_x,Q,stop_tensor)
            # entro_loss = self.config.entropy_reg*self.entropy_loss(P,Q)
            # entro_topic = self.entropy_topic()

            # Sampling Z
            # Q = nn.Softmax(dim=-1)(zQ*self.kappa)
            np_Q = zQ.data.numpy()
            # np_Q = np.squeeze(np_Q, 1)
            cor_x = _x.squeeze_(0).numpy()

            for b in range(cor_x.shape[0]):
                corpus_id = np.asscalar(corpus_idb[b])
                if corpus_id==217:
                    print('..')
                for doc_id, doc_x in enumerate(cor_x[b]):
                    topics = self.wordTopics[corpus_id][doc_id]
                    if len(topics) >0:
                        countTopic = np.zeros(self.config.topic_size)
                        # for wp, wi in enumerate(doc_x[doc_x!=0]):
                        doc_x_clean = [w for w in doc_x if w not in self.stopword]
                        for wp, wi_0 in enumerate(doc_x_clean):
                            wi = wi_0 - 1
                            topicScores = np.zeros(self.config.topic_size)

                            for k in range(self.config.topic_size):
                                # topicScores[k] = np.exp(self.kappa * np_Q[doc_id, k]
                                #                         + self.backgroundWords[wi] + self.topicWords[wi, k])/sZ[k]
                                topicScores[k] = np.exp(self.kappa * np_Q[doc_id,b, k]
                                                        + self.backgroundWords[wi] + self.topicWords[wi, k])

                            topicScores = topicScores / np.sum(topicScores)
                            # sampling Z_wp
                            if sample:
                                mul_drawn = np.random.multinomial(1, topicScores)
                            else:
                                mul_drawn = np.zeros_like(topicScores)
                                mul_drawn[np.argmax(topicScores)] = 1

                            # countTopic += mul_drawn
                            newTopic = np.where(mul_drawn == 1)[0][0]
                            if newTopic != topics[wp]:
                                t = topics[wp]
                                self.wordTopicCounts[wi, t] -= 1
                                self.wordTopicCounts[wi, newTopic] += 1
                                self.topicCounts[t] -= 1
                                self.topicCounts[newTopic] += 1
                                topics[wp] = newTopic


    def train(self):
        print('***Init all variables ...')
        # self.initFunc()

        print('***Start training ...')
        # stop_tensor = Variable(torch.LongTensor(stopword), requires_grad=False)

        for epoch_i in trange(self.config.n_epoch, desc='Epoch', ncols=80):
            # total_loss = []
            # kl_list = []
            # nllk_list = []
            # total_loss_value = []
            # entropy_loss_list = []
            # entropy_topic_list = []
            # self._zero_grads()

            # _maxiter_ = 1
            # for _ in range(_maxiter_):
            # self.sample_Z()
                # # for _ in range(_maxiter_):
                # for batch_i, doc_features in enumerate(tqdm(
                #         self.train_loader, desc='Batch-TM', ncols=80, leave=False)):
                #
                #     _x, _y, corpus_id = doc_features
                #     corpus_id = corpus_id.numpy()[0]
                #     var_x_ = Variable(_x,requires_grad=False)
                #     var_y = Variable(_y,requires_grad=False)
                #     # [s_len x w_len]
                #     var_x = torch.squeeze(var_x_,0)
                #     emb_x = self.embedding(var_x).transpose_(0,1)
                #     sen_x_ = self.rnn_w(emb_x)[-1,:,:]
                #     sen_x = torch.unsqueeze(sen_x_, 1)
                #     # [len x batch x nTopic]
                #     zQ, _ = self.vrnn(self.prob_q, self.prob_p, sen_x, self.config,
                #                            self.Q_mu, self.Q_var, self.P_mu, self.P_var)
                #
                #     # calculate all losses
                #     # kl_loss = self.config.kl_reg*self.KLDist2DistBuck(Q,P)
                #
                #     # kl_loss = torch.sum(kl_seq)
                #     # kl_list.append(kl_loss.data.numpy())
                #
                #     # llk = self.logLK(var_x,Q,stop_tensor)
                #     # entro_loss = self.config.entropy_reg*self.entropy_loss(P,Q)
                #     # entro_topic = self.entropy_topic()
                #
                #     # Sampling Z
                #     # Q = nn.Softmax(dim=-1)(zQ*self.kappa)
                #     np_Q = zQ.data.numpy()
                #     np_Q = np.squeeze(np_Q,1)
                #     cor_x = _x.squeeze_(0).numpy()
                #     countTopicList = []
                #
                #     for doc_id, doc_x in enumerate(cor_x):
                #         topics = self.wordTopics[corpus_id][doc_id]
                #         countTopic = np.zeros(self.config.topic_size)
                #         # for wp, wi in enumerate(doc_x[doc_x!=0]):
                #         doc_x_clean = [w for w in doc_x if w not in self.stopword]
                #         for wp, wi_0 in enumerate(doc_x_clean):
                #             wi = wi_0-1
                #             topicScores = np.zeros(self.config.topic_size)
                #             for k in range(self.config.topic_size):
                #                 topicScores[k] = np.exp(self.kappa * np_Q[doc_id,k]
                #                                         + self.backgroundWords[wi] + self.topicWords[wi,k])
                #
                #             topicScores = topicScores/np.sum(topicScores)
                #             # sampling Z_wp
                #             mul_drawn = np.random.multinomial(1,topicScores)
                #             # countTopic += mul_drawn
                #             newTopic = np.where(mul_drawn==1)[0][0]
                #             if newTopic != topics[wp]:
                #                 t = topics[wp]
                #                 self.wordTopicCounts[wi,t]-=1
                #                 self.wordTopicCounts[wi,newTopic]+=1
                #                 self.topicCounts[t]-=1
                #                 self.topicCounts[newTopic] +=1
                #                 topics[wp] = newTopic
                        # concat all topics-count
                        # countTopicList.append(countTopic)

                    # countTopicList = Variable(torch.FloatTensor(np.expand_dims(np.stack(countTopicList),1)), requires_grad=False)
                    # pQ = torch.log(Q)
                    # llk = torch.sum(torch.sum(countTopicList*pQ, -1))
                    # batch_loss = -llk + kl_loss
                    # total_loss.append(batch_loss)
                    # total_loss_value.append(batch_loss.data.numpy())
                    # nllk_list.append(-llk.data.numpy())

            print('\n- Topic:')
            self.dLSTM.expose_topic(self.topicemb, self.embedding)
            total_loss_value, nllk_list, kl_list, sp_list = self.train_rnn(maxiter=1)


            # Update LDA-part by quasi-newton
            # newTopicWords, fvalue, _ = scipy.optimize.fmin_l_bfgs_b(func=self.fn,x0=self.topicWords.flatten(),maxiter=5)
            # self.topicWords = np.reshape(newTopicWords, self.topicWords.shape)
            # self.normalizeTopicWords()

            # Backprop NN

            # for _loss in total_loss:
            # (sum(total_loss)/len(total_loss)).backward(retain_graph=True)
            # (sum(total_loss)).backward(retain_graph=True)
                # _loss.backward(retain_graph=True)
            # torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.clip)
            # self.optimizer.step()

            # Save parameters at checkpoint
            if (epoch_i + 1) % self.config.eval_rate == 0:
                self.evaluate(epoch_i + 1)
                if self.config.write_model:
                    # save model
                    self.save_checkpoint({
                        'epoch': epoch_i + 1,
                        'state_dict': self.model.state_dict(),
                        'total_loss': np.mean(total_loss_value),
                        'optimizer': [self.optimizer.state_dict()],
                        'topic_emb': self.topicemb
                    }, filename='./model/chk_point_{}.pth'.format(epoch_i + 1))
                    # self.save_checkpoint(self.topicemb, filename='./model/chk_point_topic_{}.pth'.format(epoch_i + 1))

            tqdm.write('\n***Ep-{} | Total_loss: {} | KL: {} | NLLK: {} | Sparsity: {} | NORM: {}'.format(
                        epoch_i,np.mean(total_loss_value),np.mean(kl_list),np.mean(nllk_list), np.mean(sp_list),
                        self.get_norm_grad(self.model).data
                        ))

            self.writer.update_parameters(self.model, epoch_i)
            self.writer.update_loss(np.sum(total_loss_value), epoch_i, 'total_loss')
            self.writer.update_loss(np.sum(kl_list), epoch_i, 'kl_loss')
            self.writer.update_loss(np.sum(nllk_list), epoch_i, 'nllk')
            # self.writer.update_loss(fvalue, epoch_i, 'Fn')


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
            fig.savefig('./Result/fullads/distr-viz-ep{}.eps'.format(epoch_i),format='eps', dpi=1000)

        self.model.eval()
        f_vocab = './output/vocab.dat'
        f_out = open('./Result/fullads/distr-view-ep{}.dat'.format(epoch_i), 'w')
        bg_out = open('./Result/fullads/bg-view-ep{}.dat'.format(epoch_i), 'w')
        rev_vocab = dict()
        vocab = []
        with open(f_vocab,'r') as f:
            for line in f:
                wi = line.split('\t')
                rev_vocab[int(wi[1])]= wi[0]
                vocab.append(wi[0])
                bg_out.write('{}\t{}\n'.format(self.backgroundWords[int(wi[1])-1],wi[0]))

        bg_out.close()
        w_distr = np.exp(self.topicWords)/np.sum(np.exp(self.topicWords),axis=0)
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

        np.savetxt('./Result/fullads/beta-view-ep{}.dat'.format(epoch_i), self.topicWords, delimiter='\t')
        # np.savetxt('./output/bg-view-ep{}.dat'.format(epoch_i), self.backgroundWords, delimiter='\t')


if __name__ == '__main__':
    pass
