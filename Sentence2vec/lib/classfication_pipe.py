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
from sklearn.naive_bayes import MultinomialNB,GaussianNB
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
import os,time
import timeit
from sklearn.tree import DecisionTreeClassifier

class ClassificationPiPe:
    def Average(lst): 
        return sum(lst) / len(lst) 
    def __init__(self, option):
        if(option == 'svm'):
            self.option = 'SVM'
            self.clf = Pipeline([
            ("vect", CountVectorizer()),
            ("tfidf", TfidfTransformer()),
            ("clf", SGDClassifier(loss='log', penalty='l2', alpha=1e-3, max_iter=3000, random_state=None))
            ])
        if(option == 'naive'):
            self.option = 'NAVIE'
            self.clf = Pipeline([
                ("vect", CountVectorizer()),#bag-of-words
                ("tfidf", TfidfTransformer()),#tf-idf
                ("clf", MultinomialNB())#naive bayes
            ])
        if(option == 'tree'):
            self.option = 'tree'
            self.clf = Pipeline([
                ("vect", CountVectorizer()),#bag-of-words
                ("tfidf", TfidfTransformer()),#tf-idf
                ("clf", DecisionTreeClassifier())#tree
            ])
    def run(self):
        start = timeit.default_timer()
        path_output = 'output/log_running.txt'
        f_output = open(path_output, "a",encoding="utf8")
        f_output.writelines("######################### START WITH MODE %s #########################" %self.option)

        df = pd.read_excel (r'input\data_full.xlsx')
        arr = df.values
        # shuffle data
        data_shuffle =random.sample(arr.tolist(), len(arr)) 
        # print(data_shuffle)

        kf = KFold(n_splits=10)
        precision_avg = []
        recall_avg = []
        f1_avg = []
        for train_index, test_index in kf.split(data_shuffle):
            # write data train to file 
            path_output_train = 'output/data_train_' + time.strftime("%d%m%Y_%H%M%S") + '.txt'
            path_output_test = 'output/data_test_' + time.strftime("%d%m%Y_%H%M%S") + '.txt'
            # f_data_train = open(path_output_train, "a",encoding="utf8")
            # f_data_test = open(path_output_test, "a",encoding="utf8")

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

            clf = self.clf.fit(df_train["feature"], df_train["target"])
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
                y_result.append(predicted[0])
                if(predicted[0] == 'tich_cuc'):
                    positive.append(row[0])
                else:
                    negative.append(row[0])
            # show result 
            target_names = ['tich_cuc', 'tieu_cuc']
            print(classification_report(y_test, y_result,target_names=target_names))
            f_output.write('Data test positive/negative is %s/%s' % (count_test_pos ,count_test_neg) + "\n")
            f_output.writelines('Data predicted positive/negative is %s/%s \n' % (len(positive) ,len(negative)))
            f_output.writelines('Accuracy_score is %s \n' % (accuracy_score(y_test, y_result, normalize=True)))
            f_output.write('F1_score is %s' % (f1_score(y_test, y_result, average="macro")) + "\n")
            f_output.write('precision_score is %s' % (precision_score(y_test, y_result, average="macro")) + "\n")
            f_output.write('recall_score is %s' % (recall_score(y_test, y_result, average="macro")) + "\n")
            f_output.writelines(classification_report(y_test, y_result,target_names=target_names))
            precision_avg.append(precision_score(y_test, y_result,average="macro"))
            recall_avg.append(recall_score(y_test, y_result, average="macro"))
            f1_avg.append(accuracy_score(y_test, y_result, normalize=True))
        stop = timeit.default_timer()
        f_output.writelines("------------------------------------------------\n")
        f_output.writelines('Avg precision is %s \n' % ClassificationPiPe.Average(precision_avg))
        f_output.writelines('Avg recall is %s \n' % ClassificationPiPe.Average(recall_avg))
        f_output.writelines('Avg f1 is %s \n' % ClassificationPiPe.Average(f1_avg))
        f_output.writelines('Time is %s \n' % (stop - start))




