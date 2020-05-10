from lib.sentence2vec import Sentence2Vec

model = Sentence2Vec('./data/data_train.model')

# turn job title to vector
print(model.get_vector('cho đề_cương không vô cái gì hết'))
test = model.get_vector('cho đề_cương không vô cái gì hết')
test.save('./data/data_test.model')
# print(model.get_vector('sở sẽ cho in thêm khoảng bộ và phát_hành tiếp vào đầu tuần tới'))

# not similar job
# print(model.similarity('xin chao', 'xin chao chao'))
