#! -*- coding:utf-8 -*-

'''
@Author:        ZM
@Date and Time: 2020/12/13 12:31
@File:          dataset.py
'''

import numpy as np

def read_data(file_path, max_length=7):
    with open(file_path, mode='r', encoding='utf-8') as f:
        lines = f.readlines()
    lines = [line.strip().replace(' ', '') for line in lines]
    lines = [line for line in lines if len(line) <= max_length]
    return lines

def get_dataset(dataset_dir='D:/datasets/couplet'):
    X_train = read_data(f'{dataset_dir}/train/in.txt')
    Y_train = read_data(f'{dataset_dir}/train/out.txt')
    X_val = read_data(f'{dataset_dir}/test/in.txt')
    Y_val = read_data(f'{dataset_dir}/test/in.txt')

    return (X_train, Y_train), (X_val, Y_val)

def is_chinese(c):
    if '\u4e00' <= c <= '\u9fff':
        return True
    return False

def is_chinese_sep(c):
    if c in ('\u3002', '\uff1b', '\uff0c', '\uff1a', '\u201c', '\u201d',
             '\uff08', '\uff09', '\u3001', '\uff1f', '\u300a', '\u300b'):
        return True
    return False

def str2id(s, token2id):
    out_s = []
    for c in s:
        if c not in token2id.keys():
            if is_chinese_sep(c):
                c = 'sep'
            else:
                c = 'unk'
        out_s.append(token2id[c])
    return out_s

def sequence_padding(text_list, max_length=None, padding=0):
    """Numpy函数，将序列padding到同一长度
    """
    if max_length is None:
        max_length = max([len(text) for text in text_list])

    outputs = []

    for text in text_list:
        text = text[:max_length]
        pad_width = (0, max_length - len(text))
        text = np.pad(text, pad_width, mode='constant', constant_values=padding)
        outputs.append(text)

    return np.array(outputs, dtype='int32')