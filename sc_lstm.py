# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

import numpy as np
from tqdm import tqdm, trange

from layers.seq_layers import SCLSTM_MultiCell

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
import path_config as p

np.random.seed(112312)
class Solver(object):
    def __init__(self, config=None, train_loader=None, test_loader=None, valid_loader=None):
        """Class that Builds, Trains and Evaluates SCLSTM model"""
        self.config = config
        self.train_loader = train_loader
        self.test_loader = test_loader
        os.environ["CUDA_VISIBLE_DEVICES"] = self.config.gpu
        self.vocab = pickle.load(open(p.word_vocab_pkl, 'rb'))
        self.kvoc = pickle.load(open(p.kwd_pkl, 'rb'))
        self.i2w = {i:w for i, w in enumerate(self.vocab)} # index to vocab
        self.i2k = {i:k for i, k in enumerate(self.kvoc)} # index to keyword
        self.w2i = {w:i for i, w in self.i2w.items()}

    def build(self):
        # Build Modules
        self.device = torch.device('cuda:0,1')
        self.embedding = nn.Embedding(self.config.vocab_size, self.config.wemb_size, padding_idx=0).cuda()
        
        if True:
            weights_matrix = torch.FloatTensor(pickle.load(open(p.word_vec_pkl,'rb')))
            self.embedding.from_pretrained(weights_matrix, freeze=False)
            self.embedding.weight.requires_grad = True
        
        self.w_hr_fw = nn.ModuleList(self.config.num_layers*
                                  [nn.Linear(self.config.hidden_size, self.config.kwd_size, bias=False).cuda()])
        self.w_hr_bw = nn.ModuleList(self.config.num_layers*
                                  [nn.Linear(self.config.hidden_size, self.config.kwd_size, bias=False).cuda()])
        
        self.w_wr = nn.Linear(self.config.wemb_size, self.config.kwd_size, bias=False).cuda()
        self.w_ho_fw = nn.Sequential(
                nn.Linear(self.config.hidden_size*self.config.num_layers, self.config.vocab_size),
    #             nn.LogSoftmax(dim=-1)
        ).cuda()
        self.w_ho_bw = nn.Linear(self.config.hidden_size*self.config.num_layers, self.config.vocab_size).cuda()
        self.sc_rnn_fw = SCLSTM_MultiCell(self.config.num_layers, self.config.wemb_size, 
                                       self.config.hidden_size, self.config.kwd_size, dropout=self.config.drop_rate).cuda()
        
        self.sc_rnn_bw = SCLSTM_MultiCell(self.config.num_layers, self.config.wemb_size, 
                                       self.config.hidden_size, self.config.kwd_size, dropout=self.config.drop_rate).cuda()
        
        self.model = nn.ModuleList([
            self.w_hr_fw, self.w_hr_bw, self.w_wr, self.w_ho_fw, self.w_ho_bw, 
            self.sc_rnn_fw, self.sc_rnn_bw
        ])
        
        self.criterion = nn.CrossEntropyLoss(reduction='none')
        
        with torch.no_grad():
            self.hc_list_init = (
                Variable(torch.zeros(self.config.num_layers, self.config.batch_size, self.config.hidden_size), 
                     requires_grad=False).cuda(),
                Variable(torch.zeros(self.config.num_layers, self.config.batch_size, self.config.hidden_size), 
                     requires_grad=False).cuda()
            )
            
        #--- Init dirs for output ---
        self.current_time = datetime.now().strftime('%b%d_%H-%M-%S')
        if self.config.mode == 'train':
            # Overview Parameters
            print('Init Model Parameters')
            for name, param in self.model.named_parameters():
                print('\t' + name + '\t', list(param.size()))
                if param.data.ndimension() >= 2:
                    nn.init.xavier_uniform_(param.data)
                else:
                    nn.init.zeros_(param.data)

            # Tensorboard
            self.writer = TensorboardWriter(p.tb_dir + self.current_time)
            # Add emb-layer
            self.model.train()
            # create dir
#             self.res_dir = p.result_path.format(p.dataname, self.current_time) # result dir
            self.cp_dir = p.check_point.format(p.dataname, self.current_time) # checkpoint dir
#             os.makedirs(self.res_dir)
            os.makedirs(self.cp_dir)
        
        #--- Setup output file ---
        self.out_file = open(p.out_result_dir.format(p.dataname, self.current_time), 'w')
        
        self.model.append(self.embedding)
#         self.model.to(self.device)
#         self.model = nn.DataParallel(self.model)
        # Build Optimizers
        self.optimizer = optim.Adam(
            list(self.model.parameters()),
            lr=self.config.lr
        )
        print(self.model)

    def load_model(self, ep):
        _fname = (self.cp_dir if self.config.mode=='train' else self.config.resume_dir) + 'chk_point_{}.pth'.format(ep)
        if os.path.isfile(_fname):
            print("=> loading checkpoint '{}'".format(_fname))
            if self.config.load_cpu:
                checkpoint = torch.load(_fname, map_location=lambda storage, loc: storage) # load into cpu-mode
            else:
                checkpoint = torch.load(_fname) # gpu-mode
            self.start_epoch = checkpoint['epoch']
            # checkpoint['state_dict'].pop('1.s_lstm.out.0.bias',None) # remove bias in selector
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'][0])
        else:
            print("=> no checkpoint found at '{}'".format(_fname))

    def _zero_grads(self):
        self.optimizer.zero_grad()

    def save_checkpoint(self, state, filename):
        torch.save(state, filename)
        
    def get_norm_grad(self, module, norm_type=2):
        total_norm = 0
        for name, param in module.named_parameters():
            if param.grad is not None:
                total_norm += torch.sum(torch.pow(param.grad.view(-1),2))
        return torch.sqrt(total_norm).data
        
    def one_step_fw(self, w_t, y_t, hc_list, d_t, rnn_model, w_hr, w_ho):
        h_tm1, _ = hc_list
        #--- Keyword detector ---
        res_hr = sum([w_hr[l](h_tm1[l]) for l in range(self.config.num_layers)])
        r_t = torch.sigmoid(self.w_wr(w_t) + self.config.alpha*res_hr)
        d_t = r_t*d_t
        flat_h, hc_list = rnn_model(w_t, hc_list, d_t)
        
        with torch.no_grad():
            mask = Variable((y_t!=0).float(), requires_grad=False)
            assert not torch.isnan(mask).any()
        pred = w_ho(flat_h)
        llk_step = torch.mean(self.criterion(pred, y_t) * mask)
        l1_step = torch.mean(torch.sum(torch.abs(d_t), dim=-1))
        assert not torch.isnan(llk_step).any()
        assert not torch.isnan(l1_step).any()
        return llk_step, l1_step, pred, hc_list, d_t
    
    def train_epoch(self):
        loss_list = []
        l1_list = []
        fw_list, bw_list = [], []
        for batch_i, doc_features in enumerate(tqdm(
                self.train_loader, desc='Batch', dynamic_ncols=True, ascii=True)):
            self._zero_grads()
            doc, kwd = doc_features
            with torch.no_grad():
                var_doc = Variable(doc, requires_grad=False).cuda()
                var_kwd = Variable(kwd, requires_grad=False).cuda()
            
            doc_emb = self.embedding(var_doc) # get word-emb
            
            #--- Word generation ---           
            step_loss = []
            step_l1 = []
            
            #--- FW Stage ---
            hc_list = self.hc_list_init
            d_t = var_kwd
            for t in range(p.MAX_DOC_LEN-1):
                w_t = doc_emb[:,t,:]
                y_t = var_doc[:,t+1]
#                 h_tm1, _ = hc_list
                
#                 #--- Keyword detector ---
#                 res_hr = sum([self.w_hr[l](h_tm1[l]) for l in range(self.config.num_layers)])
#                 r_t = torch.sigmoid(self.w_wr(w_t) + self.config.alpha*res_hr)
#                 d_t = r_t*d_t
# #                 print hc_list[0].shape, w_t.shape, d_t.shape
#                 flat_h, hc_list = self.sc_rnn(w_t, hc_list, d_t)
                
#                 #--- Log LLK ---
#                 with torch.no_grad():
#                     mask = Variable((y_t!=0).float(), requires_grad=False)
#                     assert not torch.isnan(mask).any()
#                 pred = self.w_ho(flat_h)
#                 llk_step = torch.mean(self.criterion(pred, y_t) * mask)
#                 l1_step = torch.mean(torch.sum(torch.abs(d_t), dim=-1))
                
#                 assert not torch.isnan(llk_step).any()
#                 assert not torch.isnan(l1_step).any()
                llk_step, l1_step, pred, hc_list, d_t = self.one_step_fw(w_t, y_t, hc_list, d_t, self.sc_rnn_fw,
                                                                        self.w_hr_fw, self.w_ho_fw)
                p_pred, w_pred = torch.max(nn.LogSoftmax(dim=-1)(pred), dim=-1)
#                 print [(self.i2w[i], v) for i, v in zip(w_pred.detach().cpu().numpy(), p_pred.detach().cpu().numpy())]
                
                step_loss.append(llk_step)
                step_l1.append(l1_step)
            
            fw_loss = sum(step_loss)
            fw_l1 = sum(step_l1)*self.config.eta
            batch_loss = fw_loss + fw_l1
            batch_loss.backward(retain_graph=True)
            
            #--- BW Stage ---
            torch.cuda.empty_cache()
            step_loss = []
            step_l1 = []
            hc_list = self.hc_list_init
            d_t = var_kwd
            for t in range(p.MAX_DOC_LEN-1,0,-1):
                w_t = doc_emb[:,t,:]
                y_t = var_doc[:,t-1]
                llk_step, l1_step, pred, hc_list, d_t = self.one_step_fw(w_t, y_t, hc_list, d_t, self.sc_rnn_bw,
                                                                        self.w_hr_bw, self.w_ho_bw)
                step_loss.append(llk_step)
                step_l1.append(l1_step)
            
            bw_loss = sum(step_loss)
            bw_l1 = sum(step_l1)*self.config.eta
            
            #--- BW for learning ---
#             _loss = (fw_loss + bw_loss)/2.
#             _l1 = (fw_l1 + bw_l1)/2.
            batch_loss = bw_loss + bw_l1
            batch_loss.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.clip)
            self.optimizer.step()
            
            #--- tracking loss ---
            loss_list.append(0.5*(fw_loss + bw_loss).cpu().data.numpy())
            l1_list.append(0.5*(fw_l1 + bw_l1).cpu().data.numpy())
            fw_list.append(fw_loss.cpu().data.numpy())
            bw_list.append(bw_loss.cpu().data.numpy())
        
        return loss_list, l1_list, fw_list, bw_list

    def train(self):
        print('***Start training ...')
        for epoch_i in tqdm(range(self.config.n_epoch), desc='Epoch', dynamic_ncols=True, ascii=True):
            loss_list, l1_list, fw_list, bw_list = self.train_epoch()
            # Save parameters at checkpoint
            if (epoch_i + 1) % self.config.eval_rate == 0:
                
                #--- Dump model ---
                if self.config.write_model:
                    # save model
                    self.save_checkpoint({
                        'epoch': epoch_i + 1,
                        'state_dict': self.model.state_dict(),
                        'total_loss': np.mean(loss_list),
                        'optimizer': [self.optimizer.state_dict()],
                    }, filename=self.cp_dir + 'chk_point_{}.pth'.format(epoch_i + 1))
                
                #--- Eval each step ---
                if self.config.is_eval:
                    self.evaluate(epoch_i + 1)

            print('\n***Ep-{} | Total_loss: {} [FW/BW {}/{}] | D-L1: {} | NORM: {}'.format(
                        epoch_i,np.mean(loss_list),np.mean(fw_list),np.mean(bw_list),np.mean(l1_list),
                        self.get_norm_grad(self.model)
                        ))

#             self.writer.update_parameters(self.model, epoch_i)
            self.writer.update_loss(np.mean(loss_list), epoch_i, 'total_loss')
            self.writer.update_loss(np.mean(l1_list), epoch_i, 'l1_reg')
            self.writer.update_loss(np.mean(fw_list), epoch_i, 'fw_loss')
            self.writer.update_loss(np.mean(bw_list), epoch_i, 'bw_loss')
        
        
    def gen_one_step(self, x, hc_list, d_t, rnn_model, w_hr, w_ho):
        with torch.no_grad():
            var_x = Variable(torch.LongTensor(x), requires_grad=False).cuda()
            d_t = Variable(d_t, requires_grad=False).cuda()
            hc_list = self.to_gpu(hc_list)
        
        w_t = self.embedding(var_x)
        h_tm1, _ = hc_list
        res_hr = sum([w_hr[l](h_tm1[l]) for l in range(self.config.num_layers)])
        r_t = torch.sigmoid(self.w_wr(w_t) + self.config.alpha*res_hr)
        d_t = r_t*d_t
        flat_h, hc_list = rnn_model(w_t, hc_list, d_t)
        _prob = nn.LogSoftmax(dim=-1)(w_ho(flat_h))
        return _prob.detach().cpu().numpy().squeeze(), self.to_cpu(hc_list), d_t.detach().cpu()
    
    def get_top_index(self, _prob):
        # [b, vocab]
        _prob = np.exp(_prob)
        if self.config.is_sample:
            top_indices = np.random.choice(self.config.vocab_size, self.config.beam_size, replace=False, p=_prob.reshape(-1))
        else:
            top_indices = np.argsort(-_prob)
        
        return top_indices
    
    def to_cpu(self, _list):
        return tuple([m.detach().cpu() for m in _list])
    
    def to_gpu(self, _list):
        return tuple([Variable(m, requires_grad=False).cuda() for m in _list])
    
    def rerank(self, beams, d_t):
        def add_bw_score(w_list, d_t):
#             import pdb; pdb.set_trace()
            with torch.no_grad():
                hc_list = (
                    torch.zeros(self.config.num_layers, 1, self.config.hidden_size),
                    torch.zeros(self.config.num_layers, 1, self.config.hidden_size)
                )
            w_list = [self.w2i[w] for w in w_list[::-1]]
            llk = 0.
            for i, w in enumerate(w_list[:-1]):
                _prob, hc_list, d_t = self.gen_one_step([w], hc_list, d_t, self.sc_rnn_bw, self.w_hr_bw, self.w_ho_bw)
                llk += _prob[w_list[i+1]]
            return llk/(len(w_list)-1)
        
        for i, b in enumerate(beams):
#             import pdb; pdb.set_trace()
            beams[i] = tuple([0.5 * (b[0] + add_bw_score(b[1], d_t))]) + tuple(b[1:])
            
        return beams
    
    def evaluate(self, epoch_i):
        #--- load model ---
        self.load_model(epoch_i)
        self.model.eval()
        for r_id, doc_features in enumerate(tqdm(
                self.test_loader, desc='Test', dynamic_ncols=True, ascii=True)):
            _, d_t = doc_features 
            try:
                if torch.sum(d_t)==0:
                    continue
                #--- Gen 1st step ---
                with torch.no_grad():
                    hc_list = (
                        torch.zeros(self.config.num_layers, 1, self.config.hidden_size),
                        torch.zeros(self.config.num_layers, 1, self.config.hidden_size)
                    )

                b = (0.0, [self.i2w[1]], [1], hc_list, d_t)
                _prob, hc_list, d_t = self.gen_one_step(b[2], b[3], b[4], self.sc_rnn_fw, self.w_hr_fw, self.w_ho_fw)
                top_indices = self.get_top_index(_prob)
                beam_candidates = []        
                for i in range(self.config.beam_size):
                    wordix = top_indices[i]
                    beam_candidates.append((b[0] + _prob[wordix], b[1] + [self.i2w[wordix]], [wordix], hc_list, d_t))

                #--- Gen the whole sentence ---
                beams = beam_candidates[:self.config.beam_size]
                for t in range(self.config.gen_size - 1):
                    beam_candidates = []
                    for b in beams:
                        _prob, hc_list, d_t = self.gen_one_step(b[2], b[3], b[4], self.sc_rnn_fw, self.w_hr_fw, self.w_ho_fw)
                        top_indices = self.get_top_index(_prob)

                        for i in range(self.config.beam_size):
                            #--- already EOS ---
                            if b[2]==[2]: 
                                beam_candidates.append(b)
                                break
                            wordix = top_indices[i]
                            beam_candidates.append((b[0] + _prob[wordix], b[1] + [self.i2w[wordix]], [wordix], hc_list, d_t))

                    beam_candidates.sort(key=lambda x:x[0]/(len(x[1])-1), reverse = True) # decreasing order
                    beams = beam_candidates[:self.config.beam_size] # truncate to get new beams

                #--- RERANK beams ---
                beams = self.rerank(beams, doc_features[1])
                beams.sort(key=lambda x: x[0], reverse=True)

                res = "[*]EP_{}_KW_[{}]_SENT_[{}]\n".format(
                    epoch_i,
                    ' '.join([self.i2k[int(j)] for j in torch.flatten(torch.nonzero(doc_features[1][0])).numpy()]),
                    ' '.join(beams[0][1])
                )
                print(res)
                self.out_file.write(res)
                self.out_file.flush()
            except Exception as e:
                print('Exception: ', str(e))
                pass
#         self.out_file.close()
        
        self.model.train()

if __name__ == '__main__':
    pass










