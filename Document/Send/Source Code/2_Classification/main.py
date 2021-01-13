# @author by quanvo
# @des nen dieu huong cac phuong thuc 
import argparse
import sys
from lib.train_model import TrainingModel
from lib.classfication_sentence2vec import ClassificationSentence2vec
from lib.classfication_bow import ClassificationPow

# add help when run cmd
parser = argparse.ArgumentParser(description="Please add type run")
parser.add_argument('mode', type=str, help="Choice one mode to run [train, svc, pow-svm, pow-navie]")



if __name__ == '__main__':
    args = parser.parse_args(sys.argv[1:])
    print('Running with mode %s' %args.mode)
    if(args.mode == 'train'):
        TrainingModel.train()
    if(args.mode == 'svm'):
        ClassificationSentence2vec.run('svm')
    if(args.mode == 'naive'):
        ClassificationSentence2vec.run('naive')
    if(args.mode == 'pow-svm'):
        pipe =ClassificationPow('svm')
        pipe.run()
    if(args.mode == 'pow-navie'):
        pipe = ClassificationPow('navie')
        pipe.run()
    if(args.mode == 'pow-tree'):
        pipe = ClassificationPow('tree')
        pipe.run()
    
    

