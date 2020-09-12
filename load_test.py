import os
import gensim
import numpy as np
import re
import logging
from gensim import corpora, utils
from gensim.corpora import Dictionary


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
logging.root.setLevel(level=logging.DEBUG)

path = 'C:\\Users\Terolli\\Desktop\\LDA model\\'

the_dict = corpora.Dictionary.load(path + 'dictionary.dict')
for word in the_dict.token2id:
    print(word)

corpus = corpora.MmCorpus(path + 'mycorpus.mm')
