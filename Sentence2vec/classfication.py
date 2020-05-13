import pandas as pd
from sklearn.svm import SVC
from lib.sentence2vec import Sentence2Vec
from model.svm_model import SVMModel
import numpy as p
model = Sentence2Vec('./data/data_train.model')
clf = SVC(gamma='auto')
import os


# read data train 
train_data = []

df = pd.read_excel (r'input\data_train.xlsx')

for row in df.values:
    train_data.append({"feature": model.get_vector(row[0]), "target": row[1]})

df_train = pd.DataFrame(train_data)

data_train = []
label_train = []

for i in df_train["feature"].values:
    data_train.append(i)
for i in df_train["target"].values:
    label_train.append(i)


# for i in data_train:
#     print(i)
# print(data_train)
# print(label_train)


clf = clf.fit(data_train,label_train)

positive = []
negative = []

# run data test
df = pd.read_excel (r'input\test.xlsx')

f = open('output/negative.txt', "a",encoding="utf8")
try:
    os.rm('output/negative.txt')
except:
    pass

print("Finded %s row" % len(df.values))
for row in df.values:
    print(row)
    predicted = clf.predict([model.get_vector(row[0])])
    if(predicted == 'tich_cuc'):
        positive.append(row[0])
    else:
        f.write(row[0] + "\n")
        negative.append(row[0])
# print(positive)
print(len(positive))
print("####################")
print(negative)
print(len(negative))




