import os
import logging
import gensim
from gensim.models import LdaMulticore
from gensim import corpora, utils
from gensim.corpora import Dictionary
from gensim.test.utils import datapath
from multiprocessing import Process, freeze_support

logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logging.root.setLevel(level=logging.DEBUG)

target = 'Gutenberg3000'
path = 'C:\\Users\\Terolli\\Desktop\\LDA model\\'

dct = corpora.Dictionary.load(
    path + target + '_corpus+dictionary\\dictionary.dict')

the_corpus = corpora.MmCorpus(
    path + target + '_corpus+dictionary\\mycorpus.mm')


num_topics = 100
workers = 5
chunksize = 3050
passes = 20
iterations = 400
eval_every = None

if __name__ == '__main__':
    freeze_support()
    lda_model = LdaMulticore(corpus=the_corpus, id2word=dct, num_topics=num_topics, workers=workers,
                             chunksize=chunksize, passes=passes, iterations=iterations, eval_every=eval_every)
    temp_file = datapath(path + target + '_corpus+dictionary\\lda_model')
    lda_model.save(temp_file)
