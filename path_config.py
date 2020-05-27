dataname = 'slogan'

root = '/Users/trongnguyen/Documents/Working/Projects/Previous/nlg/'
output_dir = root + 'outputs/'
data_pre = root + 'pre_data/'

vec_file = root + 'data/glove.6B.50d.txt'
text_file = root + 'data/slogan/TrainingData_Text.txt'
kwd_file = root + 'data/slogan/TrainingData_Keywords.txt'

train_data_pkl = data_pre + 'slogan/train_data.pkl'
kwd_pkl = data_pre + 'slogan/kwd_vocab.pkl'
word_vec_pkl = data_pre + 'slogan/word_vec.pkl'
word_vocab_pkl = data_pre + 'slogan/word_vocab.pkl'

tb_dir = output_dir + 'checkpoint/'
check_point = output_dir + 'model/{}/{}/'
result_path = output_dir + 'result/{}/{}/'
out_result_dir = output_dir + 'log/{}/{}.txt'

MAX_DOC_LEN = 20