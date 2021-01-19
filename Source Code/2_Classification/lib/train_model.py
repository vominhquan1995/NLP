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
      with open('input/data_topics.txt',encoding="utf8",errors='ignore') as f:
          lineList = f.readlines()
          # tokenize data
          tok_titles = [ViTokenizer.tokenize(title) for title in lineList]
          # train word2vec 
          # số chiều vector 200
          model = Word2Vec(tok_titles, sg=1, size=200, window=5, min_count=5, workers=5,iter=10)
          # save model to file
          model.save('input/data_train_200v.model')

 





