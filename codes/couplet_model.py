#! -*- coding:utf-8 -*-

'''
@Author:        ZM
@Date and Time: 2021/1/7 4:24
@File:          couplet_model.py
'''

from keras.layers import *

from MyBidirectional import MyBidirectional

def Couplet_Model(x, voca_size, hidden_dim, max_length=7):
    x = Embedding(voca_size, hidden_dim, input_length=max_length, mask_zero=True)(x)

    x = MyBidirectional(LSTM(hidden_dim, dropout=0.1, recurrent_dropout=0.1, return_sequences=True))(x)
    x = MyBidirectional(LSTM(hidden_dim, dropout=0.1, recurrent_dropout=0.1, return_sequences=True))(x)
    x = MyBidirectional(LSTM(hidden_dim, dropout=0.1, recurrent_dropout=0.1, return_sequences=True))(x)
    x = MyBidirectional(LSTM(hidden_dim // 2, dropout=0.1, recurrent_dropout=0.1, return_sequences=True))(x)
    x = MyBidirectional(LSTM(hidden_dim // 2, dropout=0.1, recurrent_dropout=0.1, return_sequences=True))(x)
    x = MyBidirectional(LSTM(hidden_dim // 2, dropout=0.1, recurrent_dropout=0.1, return_sequences=True))(x)

    x = Dense(voca_size)(x)

    return x