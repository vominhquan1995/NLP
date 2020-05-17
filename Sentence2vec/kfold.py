# import numpy as np
# from sklearn.model_selection import KFold
# X = np.array([[1, 2], [3, 4], [1, 2], [3, 4]])
# y = np.array([1, 2, 3, 4])
# kf = KFold(n_splits=2)
# kf.get_n_splits(X)
# 2
# print(kf)
# KFold(n_splits=2, random_state=None, shuffle=False)
# for train_index, test_index in kf.split(X):
#     print("TRAIN:", train_index, "TEST:", test_index)
#     X_train, X_test = X[train_index], X[test_index]
#     y_train, y_test = y[train_index], y[test_index]

import pandas as pd
import random
from sklearn.model_selection import KFold
import numpy as np
from lib.sentence2vec import Sentence2Vec
from model.svm_model import SVMModel
import numpy as p
from sklearn.svm import SVC
import os
import time
model = Sentence2Vec('./data/data_train.model')
clf = SVC(gamma='auto')

path_output = 'output/log_running.txt'
f_output = open(path_output, "a",encoding="utf8")
f_output.write("#########################START#########################"+ "\n")

df = pd.read_excel (r'input\data_train_fold.xlsx')
arr = df.values
# shuffle data
data_shuffle =random.sample(arr.tolist(), len(arr)) 
# print(data_shuffle)

kf = KFold(n_splits=11)
count =1
for train_index, test_index in kf.split(data_shuffle):
    # write data train to file 
    path_output_train = 'output/data_train_' + time.strftime("%d%m%Y_%H%M%S") + '.txt'
    path_output_test = 'output/data_test_' + time.strftime("%d%m%Y_%H%M%S") + '.txt'
    # f_data_train = open(path_output_train, "a",encoding="utf8")
    # f_data_test = open(path_output_test, "a",encoding="utf8")




    print('Loop is %s' % count)
    f_output.write('Loop is %s' % count + "\n")


    count = count +1
    # print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = np.array(data_shuffle)[train_index], np.array(data_shuffle)[test_index]
    # print(X_test)
    train_data = []

    for row in X_train:
        # f_data_train.write(row[0] + ',' + row[1] + "\n")
        train_data.append({"feature": model.get_vector(row[0]), "target": row[1]})

    df_train = pd.DataFrame(train_data)

    data_train = []
    label_train = []

    for i in df_train["feature"].values:
        data_train.append(i)
    count_train_pos= 0
    count_train_neg= 0
    for i in df_train["target"].values:
        if(i == 'tich_cuc'):
            count_train_pos= count_train_pos + 1
        else:
            count_train_neg = count_train_neg + 1
        label_train.append(i)
    f_output.write('Data train positive/negative is %s/%s' % (count_train_pos ,count_train_neg) + "\n")
    # print(data_train)
    # print(label_train)

    clf = clf.fit(data_train,label_train)
    print("Train data success")
    positive = []
    negative = []
    count_test_pos= 0
    count_test_neg= 0
    for row in X_test:
        if(row[1] == 'tich_cuc'):
            count_test_pos= count_test_pos + 1
        else:
            count_test_neg = count_test_neg + 1
        # f_data_test.write(row[0] + ',' + row[1] + "\n")
        predicted = clf.predict([model.get_vector(row[0])])
        if(predicted == 'tich_cuc'):
            positive.append(row[0])
        else:
            negative.append(row[0])
    f_output.write('Data test positive/negative is %s/%s' % (count_test_pos ,count_test_neg) + "\n")
    print("Predicted as Positive %s" % len(positive))
    print("Predicted  as Negative %s" % len(negative))
    f_output.write("Predicted as Positive %s" % len(positive) + "\n")
    f_output.write("Predicted  as Negative %s" % len(negative) + "\n")
f_output.write("#########################END#########################"+ "\n")



