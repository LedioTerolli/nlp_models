import os
import gensim
import numpy as np
import re
import logging
import nltk
from gensim import corpora, utils
from gensim.models import LdaModel, Phrases
from gensim.corpora import Dictionary
from nltk.tokenize import WordPunctTokenizer, RegexpTokenizer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from collections import defaultdict
from smart_open import open

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
logging.root.setLevel(level=logging.DEBUG)

target = 'Gutenberg300'
path = 'C:\\Users\\Terolli\\Desktop\\LDA model\\'
dataset_path = 'C:\\Users\\Terolli\\Desktop\\Gutenberg_files\\'


def preprocess(directory):
    for dirpath, dirs, filenames in os.walk(directory):
        for file in filter(lambda file: file.endswith('.txt'), filenames):
            try:
                # convert txt file into continuous str
                document = open(os.path.join(dirpath, file), encoding="utf8").read() 

                # convert to lower case
                document = document.lower()

                # determine stop words
                stoplist = set(stopwords.words('english'))

                # tokenize document
                tk = RegexpTokenizer(r'\w+')
                tokens = [token for token in tk.tokenize(
                    document) if token not in stoplist or len(token) > 1]

                # remove words with frequency == 1
                frequency = defaultdict(int)
                for token in tokens:
                    frequency[token] += 1
                tokens = [token for token in tokens if frequency[token] > 1]

                # lemmatize doc
                lemma = WordNetLemmatizer()
                tokens = [lemma.lemmatize(token) for token in tokens]

                # TODO
                # compute bigrams            

                yield tokens

            except Exception as e:
                print(e)


class SmallCorpus(object):

    def __init__(self, dir):
        self.dir = dir
        self.dictionary = gensim.corpora.Dictionary(preprocess(dir))
        self.dictionary.filter_extremes(no_below=20, no_above=0.4)
        self.dictionary.compactify()
        self.dictionary.save(path + target + '_corpus+dictionary\\dictionary.dict')

    def __iter__(self):
        for tokens in preprocess(self.dir):
            yield self.dictionary.doc2bow(tokens)

    def __len__(self):
        return len(self.dictionary)


docs = SmallCorpus(dataset_path + target)
corpora.MmCorpus.serialize(path + target + '_corpus+dictionary\\mycorpus.mm', docs)


"""
the_dict = corpora.Dictionary.load(path + target + '_corpus+dictionary\\dictionary.dict')
print(len(the_dict))

for word in the_dict.token2id:
    print(word)

corpus = corpora.MmCorpus(path + target + '_corpus+dictionary\\mycorpus.mm')
for doc in corpus:
    print(doc)
"""