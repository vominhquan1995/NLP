import pandas as pd
import re
import logging
# from nltk import word_tokenize
# from nltk.corpus import stopwords
from pyvi import ViTokenizer
from gensim.models import Word2Vec
import os

class TrainingModel:
    def train():
      # read data from file txt
      with open('input/corpus/data_train.txt',encoding="utf8",errors='ignore') as f:
          lineList = f.readlines()
          # tokenize data
          tok_titles = [ViTokenizer.word_tokenize(title) for title in lineList]
          # train word2vec
          model = Word2Vec(tok_titles, sg=1, size=100, window=5, min_count=5, workers=5,iter=10)
          # save model to file
          model.save('input/model/data_train_100v.model')

 





