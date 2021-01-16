#! -*- coding:utf-8 -*-

'''
@Author:        ZM
@Date and Time: 2020/12/16 6:51
@File:          Dataset.py
'''

class Dataset:
    def __init__(self, X, Y, y_tf=None):
        self.X = X
        self.Y = Y
        self.y_tf = y_tf

    def __len__(self):
        return len(self.X)

    def __getitem__(self, item):
        x = self.X[item]
        y = self.Y[item]

        if self.y_tf is not None:
            y = self.y_tf(y)

        return x, y