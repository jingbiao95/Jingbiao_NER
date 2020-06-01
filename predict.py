# -*- coding:utf-8 -*-
"""
-------------------------------------------------
   File Name：     predict
   Description :
   Author :       JingbiaoLi
   date：          2020/5/29
-------------------------------------------------
   Change Activity:
                   2020/5/29:
-------------------------------------------------
"""
__author__ = 'JingbiaoLi'

import argparse
import json
from importlib import import_module
import tensorflow as tf
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import preprocessing.data_utils as utils
from preprocessing.data_utils import format_result

parse = argparse.ArgumentParser("NER model")
parse.add_argument("--dataset", default="BruceData", help="please chose datasets in 'datasets' folder ")
parse.add_argument("--model", default="BiLSTM", )
args = parse.parse_args()


def predict(args):
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

    # 输入文本进行预测，并将结果打印出来
    # input_text = input("请输入文本")
    input_text = "李敬标就读于中国科学技术大学"
    predict_x = tf.keras.preprocessing.sequence.pad_sequences([[dataset.word_to_id.get(c, 1) for c in input_text]],
                                                              padding='post')
    predict_logisit = model.predict(predict_x)
    for index in predict_logisit:
        vertibi_path = np.argmax(index, axis=-1)
    predict_labels = [dataset.id_to_tag.get(idx) for idx in vertibi_path]
    print(predict_labels)
    entities_result = format_result(predict_x[0], predict_labels)
    print(json.dumps(entities_result, indent=4, ensure_ascii=False))


if __name__ == '__main__':
    predict(args)
