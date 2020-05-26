dataname = 'slogan'

root = '/ldap_home/trong.nguyen/NLG/'
data_dir = '/data/trong.nguyen/NLG/'

vec_file = root + 'data/glove.6B.50d.txt'
text_file = root + 'data/slogan/TrainingData_Text.txt'
kwd_file = root + 'data/slogan/TrainingData_Keywords.txt'

train_data_pkl = root + 'output/data/slogan/train_data.pkl'
kwd_pkl = root + 'output/data/slogan/kwd_vocab.pkl'
word_vec_pkl = root + 'output/data/slogan/word_vec.pkl'
word_vocab_pkl = root + 'output/data/slogan/word_vocab.pkl'

tb_dir = data_dir + 'checkpoint/'
check_point = data_dir + 'model/{}/{}/'
result_path = data_dir + 'result/{}/{}/'
out_result_dir = root + 'log/{}/{}.txt'

MAX_DOC_LEN = 20