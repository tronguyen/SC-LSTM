# -*- coding: utf-8 -*-

import torch
from torch.utils.data import Dataset, DataLoader
import pickle
import numpy as np
import sys, os
sys.path.append('../')
import path_config as p

class CorpusData(Dataset):
    def __init__(self, config, get_kw_onehot=False):
        self.doc_list = pickle.load(open(config.data_pkl, 'rb'))
        self.config = config
        self.kw_onehot = get_kw_onehot

    def __len__(self):
        return len(self.doc_list[:self.config.max_docs]) if self.config.max_docs > -1 else len(self.doc_list)

    def __getitem__(self, index):
        index = index if not self.config.sub_sample else np.random.choice(range(len(self.doc_list)), 1)[0]
        doc_vec, kwd_vec = self.doc_list[index]
#         print doc_vec
#         print kwd_vec
        if self.kw_onehot:
            kwd_vec = kwd_vec*0
            kwd_vec[index] = 1
        return [torch.LongTensor(doc_vec), torch.FloatTensor(kwd_vec)]

def get_loader(config):
    mode = config.mode
    batch_size = config.batch_size
    if mode.lower() == 'train':
        return DataLoader(CorpusData(config), batch_size=batch_size, shuffle=config.is_shuffle, drop_last=True, num_workers=4)
    else:
        return DataLoader(CorpusData(config, get_kw_onehot=True), batch_size=batch_size, shuffle=config.is_shuffle)


if __name__ == '__main__':
    pass
