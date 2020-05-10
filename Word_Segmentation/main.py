import argparse
import os
from underthesea import word_tokenize
from underthesea import sent_tokenize
import re,string
from features import is_punct,PUNCTUATIONS

import regex as re
import utils
from pyvi import ViTokenizer
EMAIL = re.compile(r"([\w0-9_\.-]+)(@)([\d\w\.-]+)(\.)([\w\.]{2,6})")
URL = re.compile(r"https?:\/\/(?!.*:\/\/)\S+")
PHONE = re.compile(r"(09|01[2|6|8|9])+([0-9]{8})\b")
MENTION = re.compile(r"@.+?:")
NUMBER = re.compile(r"\d+.?\d*")
DATETIME = '\d{1,2}\s?[/-]\s?\d{1,2}\s?[/-]\s?\d{4}'

RE_HTML_TAG = re.compile(r'<[^>]+>')
RE_CLEAR_1 = re.compile("[^_<>\s\p{Latin}]")
RE_CLEAR_2 = re.compile("__+")
RE_CLEAR_3 = re.compile("\s+")


class TextPreprocess:
    @staticmethod
    def replace_common_token(txt):
        txt = re.sub(EMAIL, ' ', txt)
        txt = re.sub(URL, ' ', txt)
        txt = re.sub(MENTION, ' ', txt)
        txt = re.sub(DATETIME, ' ', txt)
        txt = re.sub(NUMBER, ' ', txt)
        return txt

    @staticmethod
    def remove_emoji(txt):
        txt = re.sub(':v', '', txt)
        txt = re.sub(':D', '', txt)
        txt = re.sub(':3', '', txt)
        txt = re.sub(':\(', '', txt)
        txt = re.sub(':\)', '', txt)
        txt = re.sub('<3', '', txt)
        txt = re.sub('<', '', txt)
        return txt

    @staticmethod
    def remove_html_tag(txt):
        return re.sub(RE_HTML_TAG, ' ', txt)

    def preprocess(self, txt, tokenize=True):
        txt = re.sub('&.{3,4};', ' ', txt)
        txt = utils.convertwindown1525toutf8(txt)
        if tokenize:
            txt = ViTokenizer.tokenize(txt)
        txt = txt.lower()
        txt = self.replace_common_token(txt)
        txt = self.remove_emoji(txt)
        txt = re.sub(RE_CLEAR_1, ' ', txt)
        txt = re.sub(RE_CLEAR_2, ' ', txt)
        txt = re.sub(RE_CLEAR_3, ' ', txt)
        txt = utils.chuan_hoa_dau_cau_tieng_viet(txt)
        return txt.strip()

if __name__ == '__main__':
        file_in = 'input/input_1k.txt'
        file_out ='output/data_merge.txt'
        try:
            os.rm(file_out)
        except:
            pass
        f = open(file_out, "a",encoding="utf16")
        path = 'input/corpus'
        countF = 0
        countA =len(os.listdir(path))
        for filename in os.listdir(path):
            countF = countF+1
            print("Process train data: %s/%s" %(countF, countA))
            for text in open(os.path.join(path, filename),encoding="utf8",errors='ignore') :
                text = text.strip()
                # for childText in sent_tokenize(text):
                    # word = childText.translate(str.maketrans(' ', ' ', string.punctuation))
                output =   word_tokenize(text, format="text") 
                # print(len(output))
                print('word_tokenize: ', output)
                if(len(output) > 10):
                    output= TextPreprocess.preprocess(TextPreprocess,output) + "\n"
                    print('clean text: ' , output)
                    f.write(output)