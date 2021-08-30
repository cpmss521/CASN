# -*- coding: utf-8 -*-
# @Time    : 2021/7/20 上午8:17
# @Author  : cp
# @File    : data_processor.py.py

import os
from DataLoader.utils import load_ner_examples

class NERProcessor(object):
    # processor for the query-based ner dataset
    def get_examples(self, data_dir,prefix,entity_type=None):
        ## get all ner examples
        file_path = os.path.join(data_dir, f"{prefix}.raw.pkl")
        entity_query = list(entity_type.keys())
        data = load_ner_examples(file_path,entity_query)
        return data



## labels order are consistent with entity types
class BC5CDRProcessor(NERProcessor):
    def get_labels2idx(self, ):
        return {"O":0, "Disease":1, "Chemical":2}

    def get_entity_type(self):
        return {"Disease":0,"Chemical":1}

class NCBIProcessor(NERProcessor):
    def get_labels2idx(self, ):
        return {"O":0, "Disease":1}

    def get_entity_type(self):
        return {"Disease":0}

class GeniaProcessor(NERProcessor):
    def get_labels2idx(self, ):
        return {"O" : 0, "DNA" :1, "RNA" :2, "protein" :3, "cell_line" :4, "cell_type" :5}

    def get_entity_type(self):
        return {"DNA":0,"RNA":1,"protein":2,"cell line":3,"cell type":4}


class JNLPBAProcessor(NERProcessor):
    def get_labels2idx(self, ):
        return {"O" : 0, "DNA" :1, "RNA" :2, "protein" :3, "cell_line" :4, "cell_type" :5}

    def get_entity_type(self):
        return {"DNA":0,"RNA":1,"protein":2,"cell line":3,"cell type":4}

class BC2GMProcessor(NERProcessor):
    def get_labels2idx(self, ):
        return {"O" : 0, "GENE" :1}

    def get_entity_type(self):
        return {"Gene":0}

