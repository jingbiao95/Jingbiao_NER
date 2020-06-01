# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     train
   Description :
   Author :       JingbiaoLi
   date：          2020/5/25
-------------------------------------------------
   Change Activity:
                   2020/5/25:
-------------------------------------------------
"""
__author__ = 'JingbiaoLi'

import argparse
from importlib import import_module
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint,TensorBoard
import test
import preprocessing.data_utils as utils

parse = argparse.ArgumentParser("NER model")
parse.add_argument("--dataset", default="BruceData", help="please chose datasets in 'datasets' folder ")
parse.add_argument("--model", default="BiLSTM", )
args = parse.parse_args()


def train(args):
    # 加载数据集
    x = import_module("preprocessing." + args.dataset)
    dataset = x.Dataset()

    # 加载模型
    x = import_module("models." + args.model)
    config = x.Config(dataset)
    model = x.Model(config)
    model = model.create_model(input_shape=(config.max_len,))
    # model.build(input_shape=(None, config.max_len))

    model.summary()

    model.compile(loss="categorical_crossentropy", optimizer='sgd', metrics=['accuracy'])

    # 准备训练数据
    train_x, train_y = utils.bulid_data(dataset.train_data, config.max_len, dataset.tags_num,
                                        config.dataset.word_to_id, config.dataset.tag_to_id)
    dev_x, dev_y = utils.bulid_data(dataset.dev_data, config.max_len, dataset.tags_num,
                                    config.dataset.word_to_id, config.dataset.tag_to_id)
    test_x, test_y = utils.bulid_data(dataset.test_data, config.max_len, dataset.tags_num,
                                    config.dataset.word_to_id, config.dataset.tag_to_id)

    # 开始训练
    callbacks = [
        ModelCheckpoint(filepath=config.save_path, save_best_only=True),
        EarlyStopping(patience=5, min_delta=1e-3),
        TensorBoard(r"datasets/BruceData/saved_dict/logs")
    ]

    history = model.fit(x=train_x, y=train_y, validation_data=(dev_x, dev_y), batch_size=config.batch_size,
                        epochs=config.epochs, callbacks=callbacks)

    model.evaluate()

if __name__ == '__main__':
    train(args)

