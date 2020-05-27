import nltk
import re
from nltk.corpus import stopwords
import collections
import pickle
import numpy as np
import sys, os
sys.path.append('../')
import path_config as p

class Preprocessor(object):
    def __init__(self):
        self.spec_tok = ['<pad>', '<sos>', '<eos>', '<unk>']
        self.regex = re.compile(r"\\u[0-9]{1,}")

#     def _tokenize(self, s):
#         s = re.sub(r'(\(\d+\))*(\s)*(\d+)(\-\d+)+', ' fonenumber ', s) # fonenumber
#         s = re.sub(r'\$(\d)+(\.|\,)?(\d)*', ' moneynumber ', s) # money
#         s = re.sub(r'http://[^\s]*\s*', ' http ', s) # http
#         s = re.sub(r'www\.[^\s]*\.[^\s]*', ' www ', s)  # http
#         s = re.sub(r'(?:^| )\w(?:$| )', ' ', s) # single letter
#         s = re.sub(r'\w+@\w+\.\w+', ' emailname', s)  # single letter

#         pattern = r'''(?x)          # set flag to allow verbose regexps
#                 (?:[A-Z|a-z]\.)+        # abbreviations, e.g. U.S.A.
#                 | \w+    # words with optional internal hyphens
#             '''
#         # s = re.sub(r'[^\w\s]|(\_)+', ' ', s) # remove separate tokens
#         return nltk.regexp_tokenize(s, pattern)
#         # return nltk.word_tokenize(s)
        
    def read_wordvec(self):
        wordLS = []
        vec_ls =[]

        with open(p.vec_file, 'r') as fvec:
            for line in fvec:
                line = line.split()
                word = line[0]
                if word in self.spec_tok:
                    continue
                vec = np.array(line[1:]).astype(np.float)
                wordLS.append(word)
                vec_ls.append(vec)
            
            assert len(wordLS) == len(vec_ls)
            
            wordLS = self.spec_tok + wordLS
            word_vec = list(np.random.randn(len(self.spec_tok), len(vec_ls[0]))) + vec_ls
#             word_vec = np.array(word_vec, dtype=np.float32)
            
        return wordLS, word_vec

    
    def read_textfile(self, _file, _thres):
        kwd_ls = []
        with open(_file,'r') as fkwd:
            for line in fkwd:
                kwd = line.lower()
                kwd = re.sub(self.regex, "", kwd) 
                kwd_ls += nltk.tokenize.word_tokenize(kwd)

            #--- count word ---
            c = collections.Counter(kwd_ls)
            kwd_voc = []
            for word in c:
                if c[word] >= _thres:
                    kwd_voc.append(word)
#             pickle.dump(kwd_voc, open(p.kwd_pkl,'wb'), protocol=pickle.HIGHEST_PROTOCOL)
        
        return kwd_voc
    
    
    def gen_data(self):
        #--- prepare inputs ---
        kwd_voc = self.read_textfile(p.kwd_file, _thres=1)
        txt_voc = self.read_textfile(p.text_file, _thres=3)
        wordls, word_vec = pre.read_wordvec()
        vec_size = len(word_vec[0])
        #--- add new vocab ---
        new_voc = set(txt_voc).difference(set(wordls))
        for w in new_voc:
            print('Adding: ', w)
            wordls.append(w)
            word_vec.append(np.random.randn(vec_size))
        
        voc2i = {w:i for i, w in enumerate(wordls)}
        kwd2i = {w:i for i, w in enumerate(kwd_voc)}
        
        #--- gen data ---
        trainingdata = []
        with open(p.text_file,'r') as ftext, open(p.kwd_file,'r') as fkwd:
            for line1, line2 in zip(ftext, fkwd):
                #--- get value ---
                line1 = line1.lower()
                line1 = re.sub(self.regex, "", line1) 
                doc = nltk.tokenize.word_tokenize(line1)

                line2 = line2.lower()
                line2 = re.sub(self.regex, "", line2) 
                kwd = nltk.tokenize.word_tokenize(line2)
                
                #-- get index ---
                doc_idx = [1] + [voc2i[w] if w in voc2i else 3 for w in doc] + [2] # add SOS + DOC + EOS
                kwd_idx = [kwd2i[w] for w in kwd if w in kwd_voc]
                print(kwd_idx)
                #--- create vector ---
                doc_vec = doc_idx[:p.MAX_DOC_LEN] + [0]*(p.MAX_DOC_LEN - len(doc_idx))
                kwd_vec = np.zeros(len(kwd_voc))
                kwd_vec[kwd_idx] = 1
                print(doc_vec, np.sum(kwd_vec))
                trainingdata.append((doc_vec, kwd_vec))
                
        #--- dump data ---
        word_vec = np.array(word_vec, dtype=np.float32)
        pickle.dump(word_vec, open(p.word_vec_pkl,'wb'), protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(wordls, open(p.word_vocab_pkl,'wb'), protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(kwd_voc, open(p.kwd_pkl,'wb'), protocol=pickle.HIGHEST_PROTOCOL)
        print('- Finished dumping word vector')
        
        pickle.dump(trainingdata, open(p.train_data_pkl,'wb'), protocol=pickle.HIGHEST_PROTOCOL)
        print('- Finished dumping training data')
        print('- Size of vocab:', len(wordls))
        print('- Size of keyword:', len(kwd_voc))
        print('- Size of text file:', len(txt_voc))
          
        
if __name__ == '__main__':
    pre = Preprocessor()    
    pre.gen_data()
    
    
    
    

