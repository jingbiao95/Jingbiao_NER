# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     test
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
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import preprocessing.data_utils as utils

parse = argparse.ArgumentParser("NER model")
parse.add_argument("--dataset", default="BruceData", help="please chose datasets in 'datasets' folder ")
parse.add_argument("--model", default="BiLSTM", )
args = parse.parse_args()


def test(args):
    # 加载数据集
    x = import_module("preprocessing." + args.dataset)
    dataset = x.Dataset()

    # 加载模型
    x = import_module("models." + args.model)
    config = x.Config(dataset)
    model = x.Model(config)
    model = model.create_model(input_shape=(config.max_len,))
    model.summary()

    model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=['accuracy'])
    model.load_weights(config.save_path)
    test_x, test_y = utils.bulid_data(dataset.test_data, config.max_len, dataset.tags_num,
                                      config.dataset.word_to_id, config.dataset.tag_to_id)

    history = model.evaluate(test_x, test_y, verbose=1)


if __name__ == '__main__':
    test(args)
