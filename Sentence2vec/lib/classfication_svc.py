import pandas as pd
from sklearn.svm import SVC
from lib.sentence2vec import Sentence2Vec
import numpy as p
import os,time,random
from sklearn.model_selection import KFold
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score,average_precision_score, recall_score, classification_report, confusion_matrix
class ClassificationSVC:
    def run():
        # define model classification
        model = Sentence2Vec('input/model/data_train_full.model')
        clf = SVC(kernel='linear', C = 1e3)
        # define file write log
        path_output = 'output/log_svc_running.txt'
        path_x_test = 'output/x_test' + time.strftime("%d%m%Y_%H%M%S") + '.txt'
        path_y_test = 'output/y_test_' + time.strftime("%d%m%Y_%H%M%S") + '.txt'
        f_x_test = open(path_x_test, "a",encoding="utf8")
        f_y_test = open(path_y_test, "a",encoding="utf8")
        f_output = open(path_output, "a",encoding="utf8")


        df = pd.read_excel ('input\data_full.xlsx')
        arr = df.values
        # shuffle data
        data_shuffle =random.sample(arr.tolist(), len(arr)) 
        # split list data
        kf = KFold(n_splits=10)
        for train_index, test_index in kf.split(data_shuffle):
            # path_output_predict_pos = 'output/data_pos_' + time.strftime("%d%m%Y_%H%M%S") + '.txt'
            # path_output_predict_neg = 'output/data_neg_' + time.strftime("%d%m%Y_%H%M%S") + '.txt'
            # f_output_pos = open(path_output_predict_pos, "a",encoding="utf8")
            # f_output_neg = open(path_output_predict_neg, "a",encoding="utf8")


            X_train, X_test = np.array(data_shuffle)[train_index], np.array(data_shuffle)[test_index]
            

            print('Begin run')
            f_output.writelines("##################################################\n")


           
            data_train = []
            label_train = []
            for row in X_train:
                data_train.append(model.get_vector(row[0]))
                label_train.append(row[1]) 

            clf = clf.fit(data_train,label_train)
            print("Train data success")


            y_test = []
            y_result =[]
            positive = []
            negative = []

            # run data test
            print("Find %s data test" % len(X_test))
            count_test_pos= 0
            count_test_neg= 0
            for row in X_test:
                f_x_test.write(row[0] + "\n")
                f_y_test.write(row[1] + "\n")
                y_test.append(row[1])
                if(row[1] == 'tich_cuc'):
                    count_test_pos= count_test_pos + 1
                else:
                    count_test_neg = count_test_neg + 1
                predicted = clf.predict([model.get_vector(row[0])])
                y_result.append(predicted[0])
                if(predicted[0] == 'tich_cuc'):
                    # f_output_pos.writelines(row[0]+ "\n")
                    positive.append(row[0])
                else:
                    # f_output_neg.writelines(row[0]+ "\n")
                    negative.append(row[0])
            # print(positive)
            target_names = ['tich_cuc', 'tieu_cuc']
            print(classification_report(y_test, y_result,target_names=target_names))
            f_output.writelines('Data test positive/negative is %s/%s \n' % (count_test_pos ,count_test_neg))
            f_output.writelines('Data predicted positive/negative is %s/%s \n' % (len(positive) ,len(negative)))
            # f_output.writelines("Predicted as Positive %s \n" % len(positive))
            # f_output.writelines("Predicted  as Negative %s \n" % len(negative))
            f_output.writelines('Accuracy_score is %s \n' % (accuracy_score(y_test, y_result, normalize=True)))
            # f_output.writelines('F1_score is %s \n' % (f1_score(y_test, y_result, average="weighted")))
            f_output.writelines('Precision_score is %s \n' % (precision_score(y_test, y_result,average="macro")))
            # f_output.writelines('average_precision_score is %s \n' % (average_precision_score(y_test, y_result)))
            f_output.writelines('Recall_score is %s \n' % (recall_score(y_test, y_result, average="macro")))
            f_output.writelines(classification_report(y_test, y_result,target_names=target_names))
            f_output.writelines("##################################################\n")
            break
         
          







