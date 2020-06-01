# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     BruceData
   Description :
   Author :       JingbiaoLi
   date：          2020/5/27
-------------------------------------------------
   Change Activity:
                   2020/5/27:
-------------------------------------------------
"""
__author__ = 'JingbiaoLi'

import os
import pickle
from preprocessing.data_utils import *
import numpy as np

UNK, PAD = "<UNK>", "<PAD>"  # 未知字，padding符号


class Dataset:
    def __init__(self):
        # 配置数据集参数
        self.data_name = "BruceData"
        self.data_dir = os.path.join("datasets", self.data_name)
        self.train_path = os.path.join(self.data_dir, 'data', "train.txt")
        self.test_path = os.path.join(self.data_dir, 'data', "test.txt")
        self.dev_path = os.path.join(self.data_dir, 'data', 'dev.txt')

        self.tag_schema = 'BIOES'  # 编码格式

        self.word_to_id, self.id_to_word, self.tag_to_id, self.id_to_tag = self.get_data_info()  # 词汇-id  # 标签-id

        self.vocab_size = len(self.word_to_id.keys())  # 词汇数量
        self.tags_num = len(self.tag_to_id.keys())  # 标签数量

        # 数据集超参数

        self.pre_embeding_path = os.path.join(self.data_dir, "data", "wiki_100.utf8")  # 外部词向量路径
        self.embeding_path = os.path.join(self.data_dir, "data", "prepared_embeding.npy")  # 处理好的词向量 路径
        self.embeding_size = 100  # 词向量维度
        self.embeding = self.load_embeding()  # 加载词向量

        # 处理好后的数据直接入模型
        self.model_datasets_path = os.path.join(self.data_dir, "data", "model_datasets.pkl")

        self.train_data = self.read_data(self.train_path)
        self.test_data = self.read_data(self.test_path)
        self.dev_data = self.read_data(self.dev_path)

    def get_data_info(self):
        """
        构建词汇表,获得word_to_id,id_to_word
        获取数据的标签（转化为BIOES） tag_to_id,id_to_tag
        :return:
        """

        datainfo_path = os.path.join(self.data_dir, "data", "data_info" + ".pkl")
        if os.path.exists(datainfo_path):
            datainfo = pickle.load(open(datainfo_path, 'rb'))
            return datainfo.get("word_to_id"), datainfo.get('id_to_word'), datainfo.get("tag_to_id"), datainfo.get(
                "id_to_tag")
        else:
            # 加载数据到内存中
            sentences = self.read(self.train_path)
            # 将编码进行转换
            update_tag_schema(sentences)

            # 在进行编码后应该统计编码类型的
            tag_count = dict()
            word_count = dict()
            for i, sentence in enumerate(sentences):
                tags = [w[-1] for w in sentence]
                words = [w[0] for w in sentence]
                for tag in tags:
                    tag_count[tag] = tag_count.get(tag, 0) + 1
                for word in words:
                    word_count[word] = word_count.get(tag, 0) + 1

            word_count = sorted(word_count, key=word_count.get, reverse=True)
            word_to_id = {word_count[0]: idx + 2 for idx, word_count in enumerate(word_count)}
            word_to_id.update({PAD: 0})
            word_to_id.update({UNK: 1})
            id_to_word = {idx: word for word, idx in word_to_id.items()}

            tag_count = sorted(tag_count, key=tag_count.get, reverse=True)
            tag_to_id = {tag: idx for idx, tag in enumerate(tag_count)}
            id_to_tag = {idx: tag for tag, idx in tag_to_id.items()}
            datainfo = {
                "word_to_id": word_to_id,
                "id_to_word": id_to_word,
                "tag_to_id": tag_to_id,
                "id_to_tag": id_to_tag
            }
            pickle.dump(datainfo, open(datainfo_path, "wb"))
            return word_to_id, id_to_word, tag_to_id, id_to_tag

    def read(self, file_path):
        # 将数据读入内存
        sentences = []  # 总句子列表
        sentence = []  # 一个句子
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, word_tag in enumerate(f.readlines()):
                word_tag = word_tag.strip()  # 去除后面回车符
                if not word_tag:  # 若为空则表示句子结束
                    sentences.append(sentence)
                    sentence = []
                else:
                    split_list = word_tag.split(" ")  # 以空格分隔
                    sentence.append(split_list)
        if len(sentence) > 0:  # 将最后一个读入
            sentences.append(sentence)
        return sentences

    def read_data(self, file_path ):
        """
        将数据读入内存并将标签转换为 BIOES, 并数字化
        :param file_path:
        :return:
        """
        save_path_name = file_path.split("/")[-1].split(".")[0]
        save_path = os.path.join(self.data_dir, "data", save_path_name + ".pkl")
        #   0.读取数据
        if os.path.exists(save_path):
            dataset = pickle.load(open(save_path, 'rb'))
        else:
            # 加载数据到内存中
            sentences = self.read(file_path)
            # 将tag编码转换BIOES
            sentences = update_tag_schema(sentences)
            # word_to_id, tag_to_id
            dataset = prepare_dataset(sentences, self.word_to_id, self.tag_to_id)
            pickle.dump(dataset, open(save_path, 'wb'))
        return dataset

    def load_embeding(self):
        if os.path.exists(self.embeding_path):
            embeding = np.load(self.embeding_path)
        else:
            # 声明全0
            embeding = np.zeros([self.vocab_size, self.embeding_size], dtype=np.float64)
            vocab = self.word_to_id.keys()
            # 读取
            with open(self.pre_embeding_path, "r", encoding="utf-8") as f:
                for line in f.readlines():
                    line_split = line.strip().split(" ")
                    if line_split[0] in vocab:
                        embeding[self.word_to_id.get(line_split[0])] = np.asarray(line_split[1:], dtype=np.float64)
            np.save(self.embeding_path, embeding)
        return embeding
