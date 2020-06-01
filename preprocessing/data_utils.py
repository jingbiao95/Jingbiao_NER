# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     load_data_utils
   Description :
   Author :       JingbiaoLi
   date：          2020/5/27
-------------------------------------------------
   Change Activity:
                   2020/5/27:
-------------------------------------------------
"""
__author__ = 'JingbiaoLi'

import tensorflow as tf

UNK, PAD = "<UNK>", "<PAD>"  # 未知字，padding符号


def prepare_dataset(sentences, word_to_id, tag_to_id):
    """
    数据预处理，返回list其实包含
    -word_list
    -word_id_list
    -tag_list
    -tag_id_list
    :return:
    """

    data = []
    for s in sentences:
        word_list = [w[0] for w in s]
        word_id_list = [word_to_id.get(w, word_to_id.get(UNK)) for w in word_list]
        tag_list = [w[1] for w in s]
        tag_id_list = [tag_to_id.get(tag, tag_to_id.get("O")) for tag in tag_list]
        data.append([word_list, word_id_list, tag_list, tag_id_list])
    return data


def update_tag_schema(sentences, schema='BIOES'):
    """
    将数据的tag转换为schema格式
    """
    for i, sentence in enumerate(sentences):
        tags = [w[-1] for w in sentence]
        if not check_tag_bio(tags):
            s_str = "\n".join(" ".join(w) for w in sentence)
            raise Exception("输入的句子不符合BIO编码，请检查句子为{}\n{}".format(i, s_str))

        if schema == "BIOES":
            bioes_tags = bio2bioes(tags)
            for word, bioes_tag in zip(sentence, bioes_tags):
                word[-1] = bioes_tag
        else:
            raise Exception("{}为非法编码，不在系统中".format(schema))

    return sentences


def check_tag_bio(tags):
    """
    错误的类型
    (1)编码不在BIO中
    (2)第一个编码是I
    (3)当前编码不是B,前一个编码不是O
    :param tags:
    :return:
    """
    for i, tag in enumerate(tags):
        if tag == "O":
            continue
        tag_list = tag.split("-")
        if len(tag_list) != 2 or tag_list[0] not in ["B", "I"]:
            return False
        if tag_list[0] == "B":
            continue
        # 下面判断基于标签以I开始
        elif i == 0 or tags[i - 1].startswith("O"):
            tags[i] = "B" + tag[1:]
            # return False
        elif tags[i - 1][1:] == tag[1:]:
            continue
        else:
            tags[i] = "B" + tag[1:]
    return True


def bio2bioes(tags):
    bioes_tags = []

    for i, tag in enumerate(tags):

        if tag == 'O':
            bioes_tags.append(tag)
            continue
        tag_list = tag.split("-")

        if tag_list[0] == "B":
            if i + 1 < len(tags) and tags[i + 1].startswith("I"):
                bioes_tags.append(tag)
            else:
                bioes_tags.append(tag.replace("B", "S"))
        elif tag_list[0] == "I":
            if i + 1 < len(tags) and tags[i + 1].startswith("I"):
                bioes_tags.append(tag)
            else:
                bioes_tags.append(tag.replace("I", "E"))

        else:
            raise Exception("非法编码；{}".format(tag_list[0]))
    return bioes_tags


def bulid_data(dataset, max_len, num_classes, word_to_id, tag_to_id):
    """
    将数据转换为Model指定max_seq_len长度的序列
    :param dataset: 数据集的配置信息
    :param data:
    :param config:
    :return:
    """
    x_data = [data[1] for data in dataset]
    x_data = tf.keras.preprocessing.sequence.pad_sequences(x_data, maxlen=max_len, dtype='int32', padding='post',
                                                           truncating='post', value=word_to_id.get("<PAD>"))
    y_data = [data[3] for data in dataset]
    y_data = tf.keras.preprocessing.sequence.pad_sequences(y_data, maxlen=max_len, dtype='int32', padding='post',
                                                           truncating='post', value=tag_to_id.get("O"))
    y_data = tf.keras.utils.to_categorical(y_data, num_classes=num_classes, )
    return x_data, y_data


def format_result(word_list, tag_list, ):
    """

    :param word_list:
    :param tag_list:
    :return:
    """
    entities = []
    entity = []

    for idx, (word, tag) in enumerate(zip(word_list, tag_list)):

        if not check_label_continue(tag_list[idx - 1] if idx > 0 else None, tag) and entity:
            entities.append(entity)
            entity = []
        entity.append([idx, word, tag])
    else:
        if entity:
            entities.append(entity)

    entity_results = []
    for entity in entities:
        if entity[0][2].startswith("B-"):
            entity_results.append({
                'begin': entity[0][0] + 1,
                "end": entity[-1][0] + 1,
                "words": "".join([word for idx, word, tag in entity]),
                "type": entity[0][2].split("-")[-1]
            })
    return entity_results


def check_label_continue(pre_tag, tag):
    if tag is None:
        raise Exception("current tag can't be None")
    if pre_tag is None or tag.startswith("B-") or tag.startswith('S-'):
        return False
    if (tag.startswith("I-") or tag.startswith("E-")) and (
            pre_tag.startswith("B-") or pre_tag.startswith("I-")) and tag.endwith(pre_tag.split("-")[-1]):
        return True
    return False
