# @author by quanvo
# @des


import argparse
import sys
from lib.train import TrainingModel
from lib.classfication_svc import ClassificationSVC
from lib.classfication_pipe import ClassificationPiPe

# add help when run cmd
parser = argparse.ArgumentParser(description="Please add type run")
parser.add_argument('mode', type=str, help="Choice one mode to run [train, svc, pipe-svm, pipe-navie]")



if __name__ == '__main__':
    args = parser.parse_args(sys.argv[1:])
    print('Running with mode %s' %args.mode)
    if(args.mode == 'train'):
        TrainingModel.train()
    if(args.mode == 'svm'):
        ClassificationSVC.run('svm')
    if(args.mode == 'naive'):
        ClassificationSVC.run('naive')
    if(args.mode == 'pipe-svm'):
        pipe =ClassificationPiPe('svm')
        pipe.run()
    if(args.mode == 'pipe-navie'):
        pipe = ClassificationPiPe('navie')
        pipe.run()
    if(args.mode == 'pipe-tree'):
        pipe = ClassificationPiPe('tree')
        pipe.run()
    
    

