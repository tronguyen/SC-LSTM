from configs import get_config
from sc_lstm import Solver
from preprocessor.data_loader import get_loader
import path_config as p

if __name__ == '__main__':
    train_config = get_config(mode='train', data_pkl=p.train_data_pkl)
    train_loader = get_loader(train_config)
    print(train_config)
    
    test_config = get_config(mode='test', data_pkl=p.train_data_pkl, batch_size=1, max_docs=20, sub_sample=True)
    test_loader = get_loader(test_config)
    
    solver = Solver(train_config, train_loader=train_loader, test_loader=test_loader)

    solver.build()
    solver.train()
