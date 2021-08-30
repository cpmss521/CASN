# -*- coding: utf-8 -*-
# @Time    : 2021/7/20 下午9:37
# @Author  : cp
# @File    : ner_config.py.py



from transformers import BertConfig


class BertLabelNerConfig(BertConfig):
    def __init__(self, **kwargs):
        super(BertLabelNerConfig, self).__init__(**kwargs)
        self.ner_dropout = kwargs.get("ner_dropout", 0.1)
        self.max_entity_span = kwargs.get("max_entity_span",10)
        self.class_num = kwargs.get("class_num", 6)
        self.label_embed_url = kwargs.get("label_embed_url", None)
        self.label_freeze = kwargs.get("label_freeze", True)



