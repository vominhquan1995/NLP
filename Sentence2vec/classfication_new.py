import pandas as pd
import random
from sklearn.model_selection import KFold
import numpy as np
from lib.sentence2vec import Sentence2Vec
import numpy as p
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
import os,time
model = Sentence2Vec('./data/data_train.model')
# clf = SVC(gamma='auto')
# svm
clf = Pipeline([
            ("vect", CountVectorizer()),
            ("tfidf", TfidfTransformer()),
            ("clf-svm", SGDClassifier(loss='log', penalty='l2', alpha=1e-3, max_iter=3000, random_state=None))
            ])
# naive_bayes
# clf = Pipeline([
#             ("vect", CountVectorizer()),#bag-of-words
#             ("tfidf", TfidfTransformer()),#tf-idf
#             ("clf", MultinomialNB())#model naive bayes
#         ])

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
        train_data.append({"feature": row[0], "target": row[1]})

    df_train = pd.DataFrame(train_data)


    count_train_pos= 0
    count_train_neg= 0
    for i in df_train["target"].values:
        if(i == 'tich_cuc'):
            count_train_pos= count_train_pos + 1
        else:
            count_train_neg = count_train_neg + 1
    f_output.write('Data train positive/negative is %s/%s' % (count_train_pos ,count_train_neg) + "\n")

    clf = clf.fit(df_train["feature"], df_train["target"])
    print("Train data success")
    # value save list label data
    y_test = []
    y_result =[]


    positive = []
    negative = []
    count_test_pos= 0
    count_test_neg= 0
    for row in X_test:
        y_test.append(row[1])
        if(row[1] == 'tich_cuc'):
            count_test_pos= count_test_pos + 1
        else:
            count_test_neg = count_test_neg + 1
        # f_data_test.write(row[0] + ',' + row[1] + "\n")
        predicted = clf.predict([row[0]])
        y_result.append(predicted)
        if(predicted == 'tich_cuc'):
            positive.append(row[0])
        else:
            negative.append(row[0])
    # show result 
    print(f1_score(y_test, y_result, average="macro"))
    print(precision_score(y_test, y_result, average="macro"))
    print(recall_score(y_test, y_result, average="macro")) 
    f_output.write('Data test positive/negative is %s/%s' % (count_test_pos ,count_test_neg) + "\n")
    f_output.write('F1_score is %s' % (f1_score(y_test, y_result, average="macro")) + "\n")
    f_output.write('precision_score is %s' % (precision_score(y_test, y_result, average="macro")) + "\n")
    f_output.write('recall_score is %s' % (recall_score(y_test, y_result, average="macro")) + "\n")
    print("Predicted as Positive %s" % len(positive))
    print("Predicted  as Negative %s" % len(negative))
    f_output.write("Predicted as Positive %s" % len(positive) + "\n")
    f_output.write("Predicted  as Negative %s" % len(negative) + "\n")
f_output.write("#########################END#########################"+ "\n")



