#! -*- coding:utf-8 -*-

'''
@Author:        ZM
@Date and Time: 2020/12/13 16:28
@File:          train_CNN.py
'''

import numpy as np
import math
from keras.layers import Input
from keras import Model
from keras.callbacks import Callback
from keras.optimizers import Adam
from keras import backend as K

from dataset_utils import get_dataset, is_chinese, is_chinese_sep, str2id, sequence_padding
from Dataset import Dataset
from generator import generator
from Loss import Loss
from ToOneHot import ToOneHot
from couplet_model import Couplet_Model

class CrossEntropy(Loss):
    def compute_loss(self, inputs):
        y_true, y_pred = inputs
        loss = K.categorical_crossentropy(y_true, K.softmax(y_pred))
        return K.mean(loss)

if __name__ == '__main__':
    num_classes = 6676
    vocab_size = 6676
    max_length = 7
    hidden_dim = 128
    train_batch_size = 128
    val_batch_size = 500

    (X_train, Y_train), (X_val, Y_val) = get_dataset()
    tf = {}
    tf['unk'] = 0
    tf['sep'] = 0
    for s in X_train + Y_train + X_val + Y_val:
        for c in s:
            if is_chinese(c):
                tf[c] = tf.get(c, 0) + 1
            elif is_chinese_sep(c):
                tf['sep'] = tf.get('sep', 0) + 1
            else:
                tf['unk'] = tf.get('unk', 0) + 1
    tf = {i: j for i, j in tf.items() if j >= 2}
    token2id = {key: i + 1 for i, key in enumerate(tf.keys())}
    token2id['non'] = 0
    id2token = ['non'] + list(tf.keys())

    X_train = [str2id(x, token2id) for x in X_train]
    Y_train = [str2id(y, token2id) for y in Y_train]
    X_val = [str2id(x, token2id) for x in X_val]
    Y_val = [str2id(y, token2id) for y in Y_val]

    X_train = sequence_padding(X_train, max_length=max_length)
    Y_train = sequence_padding(Y_train, max_length=max_length)
    X_val = sequence_padding(X_val, max_length=max_length)
    Y_val = sequence_padding(Y_val, max_length=max_length)

    train_dataset = Dataset(X_train, Y_train, y_tf=ToOneHot(num_classes))
    val_dataset = Dataset(X_val, Y_val, y_tf=ToOneHot(num_classes))
    train_generator = generator(train_dataset, batch_size=train_batch_size, shuffle=True)
    val_generator = generator(val_dataset, batch_size=val_batch_size, shuffle=False)

    text_input = Input(shape=(max_length, ), name='text_input', dtype='int32')
    y_true = Input(shape=(max_length, num_classes), dtype='float32')
    out = Couplet_Model(text_input, vocab_size, hidden_dim, max_length=max_length)
    out = CrossEntropy(-1)([y_true, out])
    model = Model([y_true, text_input], out)
    model.compile(Adam())

    num_train_batches = math.ceil(len(Y_train) / train_batch_size)
    num_val_examples = len(Y_val)
    num_val_batches = math.ceil(num_val_examples / val_batch_size)

    def evaluate(model, in_s):
        s = str2id(in_s, token2id)
        x = sequence_padding([s], max_length=max_length)
        y = model.predict_on_batch([np.zeros((1, max_length, num_classes), dtype='float32'), x])[0]
        y = np.argmax(y, axis=-1)
        out_s = ''.join([id2token[i] for i in y])
        out_s = out_s.replace('sep', '，').replace('non', '')
        print(f'上联：{in_s}')
        print(f'下联：{out_s}')

    class Evaluator(Callback):
        def __init__(self):
            super(Evaluator, self).__init__()

        def on_epoch_end(self, epoch, logs=None):
            evaluate(self.model, '风弦未拨心先乱')
            evaluate(self.model, '薰风生殿阁')
            evaluate(self.model, '玉液')

    evaluator = Evaluator()

    model.fit_generator(
        train_generator,
        steps_per_epoch=num_train_batches,
        epochs=150,
        callbacks=[evaluator],
        validation_data=val_generator,
        validation_steps=num_val_batches,
        shuffle=False
    )