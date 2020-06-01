# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     BiLSTM
   Description :
   Author :       JingbiaoLi
   date：          2020/5/28
-------------------------------------------------
   Change Activity:
                   2020/5/28:
-------------------------------------------------
"""
__author__ = 'JingbiaoLi'

import os
import tensorflow as tf
from tensorflow.keras.layers import Dense, Embedding, LSTM, Bidirectional, Dropout, Input, BatchNormalization


class Config(object):
    def __init__(self, dataset):
        # 数据集参数
        self.dataset = dataset

        # 预训练参数
        self.embeding_size = 100  # embedding维度大小

        # 模型参数
        self.model_name = "BiLSTM"
        self.hidden_size = 128  # LSTM 隐藏层个数
        self.dropout = 0.5
        self.class_num = dataset.tags_num  # 类的数量
        self.epochs = 20  # epoch数
        self.batch_size = 512  # mini-batch大小
        self.max_len = 200  # 每句话处理的长度
        self.learn_rate = 1e-4  # 学习率

        # 模型训练结果
        self.save_path = os.path.join(self.dataset.data_dir, "saved_dict", self.model_name + ".h5")


class Model(tf.keras.Model):
    def __init__(self, config: Config):
        super(Model, self).__init__()
        self.config = config
        self.embedding = Embedding(input_dim=self.config.dataset.vocab_size,
                                   output_dim=self.config.dataset.embeding_size,
                                   input_length=self.config.max_len, weights=[self.config.dataset.embeding],
                                   trainable=True)
        self.biRNN = Bidirectional(LSTM(units=self.config.hidden_size, return_sequences=True, activation='relu'))
        self.dropout = Dropout(self.config.dropout)
        self.out_put = Dense(self.config.class_num, activation='softmax')

    def build(self, input_shape):
        super(Model, self).build(input_shape)

    def create_model(self, input_shape):
        model_input = Input(shape=input_shape, dtype='float64')
        x = self.embedding(model_input)
        x = self.biRNN(x)
        x = BatchNormalization()(x)
        x = self.dropout(x)
        model_output = self.out_put(x)
        model = tf.keras.Model(inputs=model_input, outputs=model_output)

        return model
