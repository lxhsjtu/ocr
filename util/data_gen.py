import numpy as np
import tensorflow as tf
import sys
import os
import codecs
sys.path.append("..")
from util.bucketdata import BucketData

from PIL import Image
from six import BytesIO as IO
from collections import Counter
import pickle as cPickle
import random, math
from util.load_dict import loaddict
class DataGen(object):
    GO = 1
    EOS = 2
    IMAGE_HEIGHT = 32
    char_dict,char_list=loaddict()
    # CHARMAP = ['', '', ''] + list('0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ')
    CHARMAP=['','','']+list(char_list)

    def __init__(self,
                 annotation_fn,
                 buckets,
                 img_width_range=(12, 320),
                 word_len=30):
        img_height=32
        if os.path.exists(annotation_fn):
            self.annotation_path = annotation_fn

        self.bucket_specs = buckets
        self.bucket_min_width, self.bucket_max_width = img_width_range
        self.image_height = img_height

        self.bucket_data = BucketData()

    def clear(self):
        self.bucket_data = BucketData()

    def get_size(self):
        with open(self.annotation_path, 'r') as ann_file:
            return len(ann_file.readlines())

    def get_lex(self,imgpath):
        _,lex=os.path.split(imgpath)
        tmp=lex[9:-4]
        tmp = tmp.replace("~", "/")
        tmp = tmp.replace('?', '？')
        tmp = tmp.replace(',', '，')
        tmp = tmp.replace('×', '*')
        tmp = tmp.replace('！', '!')
        tmp = tmp.replace('＞', '>')
        tmp = tmp.replace('＜', '<')
        return tmp

    def gen(self, batch_size):
        with codecs.open(self.annotation_path, 'r',encoding='utf-8') as ann_file:
            lines = ann_file.readlines()
            for l in lines:
                img_path=l.strip()
                lex=self.get_lex(img_path)
                try:
                    img_,word=self.read_data(img_path,lex)
                    width=img_.shape[1]
                    bs = self.bucket_data.append(img_, word, lex)
                    if bs >= batch_size:
                        b = self.bucket_data.flush_out(
                            self.bucket_specs,
                            valid_target_length=float('inf'),
                            go_shift=1)
                        if b is not None:
                            yield b
                        else:
                            assert False, 'no valid bucket of width %d' % width
                except IOError:
                    raise ValueError
        self.clear()

    def read_data(self, img_path, lex):
        assert 0 < len(lex) < self.bucket_specs[-1][1]
        # L = R * 299/1000 + G * 587/1000 + B * 114/1000
        with open(os.path.join(img_path), 'rb') as img_file:
            img = Image.open(img_file)
            img=img.convert('RGB')
            w, h = img.size
            aspect_ratio = float(w) / float(h)
            if aspect_ratio < float(self.bucket_min_width) / self.image_height:
                img = img.resize(
                    (self.bucket_min_width, self.image_height),
                    Image.ANTIALIAS)
            elif aspect_ratio > float(
                    self.bucket_max_width) / self.image_height:
                img = img.resize(
                    (self.bucket_max_width, self.image_height),
                    Image.ANTIALIAS)
            elif h != self.image_height:
                img = img.resize(
                    (int(aspect_ratio * self.image_height), self.image_height),
                    Image.ANTIALIAS)
            img=img.resize((160,32),Image.ANTIALIAS)
            img_bw = np.asarray(img, dtype=np.float32)
            img_bw = img_bw / 255.0

        # TODO
        word = [self.GO]
        number = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        for c in lex:
            if DataGen.char_dict.get(c) != None:
                word.append(int(DataGen.char_dict[c]))
                continue
            if c in number:
                word.append(int(DataGen.char_dict[int(c)]))
        word.append(self.EOS)
        word = np.array(word, dtype=np.int32)

        return img_bw, word

if __name__ == '__main__':
    bucket=[(40,20)]
    s_gen = DataGen('..//dataset//imgpath.txt',bucket)
    count = 0
    for batch in s_gen.gen(1):
        count += 1
        print(batch['data'].shape)
        print(batch['labels'])
        print(batch['decoder_inputs'])
        # print(str(batch['bucket_id']) + ' ' + str(batch['data'].shape))
        assert batch['data'].shape[1] == 32
        break
    print(count)


