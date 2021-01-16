#! -*- coding:utf-8 -*-

'''
@Author:        ZM
@Date and Time: 2020/12/16 6:52
@File:          generator.py
'''

import numpy as np

def generator(dataset, batch_size=64, shuffle=True, drop_last=False):
    x, y = dataset[0]

    true_examples = len(dataset)
    rd_index = np.arange(true_examples)
    false_examples = true_examples // batch_size * batch_size
    remain_examples = true_examples - false_examples

    i = 0
    while True:
        real_batch_size = batch_size
        if remain_examples != 0 and drop_last is False and i == false_examples:
            real_batch_size = remain_examples

        batch_X = np.empty((real_batch_size, *(x.shape)), dtype='float32')
        batch_Y = np.empty((real_batch_size, *(y.shape)), dtype='float32')

        for b in range(real_batch_size):
            if shuffle and i == 0:
                np.random.shuffle(rd_index)
                dataset.X = dataset.X[rd_index]
                dataset.Y = dataset.Y[rd_index]

            batch_X[b], batch_Y[b] = dataset[i]

            if remain_examples != 0 and drop_last is True:
                i = (i + 1) % false_examples
            else:
                i = (i + 1) % true_examples

        yield [batch_Y, batch_X], None