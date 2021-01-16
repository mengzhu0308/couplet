#! -*- coding:utf-8 -*-

'''
@Author:        ZM
@Date and Time: 2020/12/15 20:27
@File:          ToOneHot.py
'''

import numpy as np

class ToOneHot:
    def __init__(self, num_classes):
        self.num_classes = num_classes

    def __call__(self, label):
        label_len = len(label)
        one_hot = np.zeros((label_len, self.num_classes), dtype='float32')
        one_hot[np.arange(label_len), label] = 1.
        return one_hot