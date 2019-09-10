import argparse
import os
from underthesea import word_tokenize
from underthesea import sent_tokenize
import re,string
from features import is_punct,PUNCTUATIONS
if __name__ == '__main__':
        file_in = 'tmp/input.txt'
        file_out ='tmp/output_final.txt'
        try:
            os.rm(file_out)
        except:
            pass
        f = open(file_out, "a",encoding="utf8")
        for text in open(file_in,encoding="utf8",errors='ignore') :
            text = text.strip()
            for childText in sent_tokenize(text):
                # word = childText.translate(str.maketrans(' ', ' ', string.punctuation))
                output =   word_tokenize(childText, format="text") + "\n"
                f.write(output)