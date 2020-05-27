import os
import path_config as p
from configs import get_config
from preprocessor.data_loader import get_loader
from sc_lstm import Solver

if __name__ == '__main__':
    try:
        test_config = get_config(mode='test', data_pkl=p.train_data_pkl, batch_size=1, max_docs=9105, is_shuffle=False)
        print(test_config)
        
        test_loader = get_loader(test_config)

        solver = Solver(config=test_config, test_loader=test_loader)
        solver.build()
        solver.evaluate(test_config.resume_ep)

    except Exception as err:
        raise Exception('\n\n<ERROR> Main exception: {}'.format(err))

