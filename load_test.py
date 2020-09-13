import os
import gensim
import numpy as np
import re
import logging
from gensim import corpora, utils
from gensim.corpora import Dictionary

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
logging.root.setLevel(level=logging.DEBUG)

target = 'Gutenberg3000'
path = 'C:\\Users\\Terolli\\Desktop\\LDA model\\'


the_dict = corpora.Dictionary.load(path + target + '_corpus+dictionary\\dictionary.dict')
print(len(the_dict))

file = open(path + "the_dict.txt", "w")
for word in the_dict.token2id:
    file.write(word + "\n")
file.close()

"""
corpus = corpora.MmCorpus(path + target + '_corpus+dictionary\\mycorpus.mm')
for doc in corpus:
    print(doc)
"""