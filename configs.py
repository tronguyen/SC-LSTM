# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import pprint


def str2bool(v):
    """string to boolean"""
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def str2float(v):
    return float(v)

def str2int(v):
    return int(v)

def str2arr(v):
    return [float(v) for v in v.split(':') if v != '']

class Config(object):
    def __init__(self, **kwargs):
        """Configuration Class: set kwargs as class attributes with setattr"""
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __repr__(self):
        """Pretty-print configurations in alphabetical order"""
        config_str = 'Configurations\n'
        config_str += pprint.pformat(self.__dict__)
        return config_str

def get_config(parse=True, **optional_kwargs):
    """
    Get configurations as attributes of class
    """
    parser = argparse.ArgumentParser()

    # Mode
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--verbose', type=str2bool, default='False')
    parser.add_argument('--preprocessed', type=str2bool, default='True')
    parser.add_argument('--pkl_list', type=str, default='') # path to pickled data for training

    # Model
    parser.add_argument('--hidden_size', type=int, default=64)
    parser.add_argument('--vocab_size', type=int, default=400010)
    parser.add_argument('--wemb_size', type=int, default=50)
    parser.add_argument('--kwd_size', type=int, default=9105)
    parser.add_argument('--num_layers', type=int, default=2)

    # Train
    parser.add_argument('--eval_run', type=str2bool, default='False')
    parser.add_argument('--n_epoch', type=int, default=500)    # epoch for training
    parser.add_argument('--clip', type=float, default=1.)   # grad-clipping
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=50) # batch-size for training
    parser.add_argument('--beam_size', type=int, default=10)
    parser.add_argument('--gen_size', type=int, default=15)
    
    parser.add_argument('--alpha', type=float, default=1.)
    parser.add_argument('--eta', type=float, default=1.)
    parser.add_argument('--drop_rate', type=float, default=0.1)
    parser.add_argument('--max_docs', type=int, default=-1)
    parser.add_argument('--is_shuffle', type=str2bool, default='True')
    
    parser.add_argument('--eval_rate', type=int, default=50)
    parser.add_argument('--write_model', type=str2bool, default='True')
    parser.add_argument('--sub_sample', type=str2bool, default='False')
    parser.add_argument('--is_eval', type=str2bool, default='False')
    
    parser.add_argument('--pretrain', type=str2bool, default='True')
    parser.add_argument('--isdebug', type=str2bool, default='False')
    parser.add_argument('--is_sample', type=str2bool, default='True')
    parser.add_argument('--load_cpu', type=str2bool, default='False')
#     parser.add_argument('--logdir', type=str, default='checkpoint') # where to store checkpoints for visualize
    

    # Test config
    parser.add_argument('--resume_dir', type=str, default='')
    parser.add_argument('--resume_ep', type=int, default=500)   # test at ep-400
    parser.add_argument('--gpu', type=str, default='3,2') # gpu-index use; -1 for cpu

    if parse:
        kwargs = parser.parse_args()
    else:
        kwargs = parser.parse_known_args()[0]

    # Namespace => Dictionary
    kwargs = vars(kwargs)
    kwargs.update(optional_kwargs)

    return Config(**kwargs)


if __name__ == '__main__':
    config = get_config()
    import ipdb
    ipdb.set_trace()
