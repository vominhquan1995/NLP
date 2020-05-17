import pandas as pd
import re
import logging
from nltk import word_tokenize
from nltk.corpus import stopwords
from gensim.models import Word2Vec
import os

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)


# pre processing data
def cleanData(sentence):
    # convert to lowercase, ignore all special characters - keep only
    # alpha-numericals and spaces
    sentence = re.sub(r'[^A-Za-z0-9\s]', r'', str(sentence).lower())

    # remove stop words
    sentence = " ".join([word for word in sentence.split()
                        if word not in stopwords.words('english')])

    return sentence

# read data from file txt
with open('data/data_train.txt',encoding="utf8",errors='ignore') as f:
  lineList = f.readlines()
  print(len(lineList))
  # drop duplicate rows
  # lineList.drop_duplicates()
  # clean data
  # lineList.map(lambda x: cleanData(x))
  tok_titles = [word_tokenize(title) for title in lineList]
  # refer to here for all parameters:
  # https://radimrehurek.com/gensim/models/word2vec.html
  model = Word2Vec(tok_titles, sg=1, size=100, window=10, min_count=5, workers=5,
                    iter=10)
  # save model to file
  model.save('./data/data_train_full.model')




