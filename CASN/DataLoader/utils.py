# -*- coding: utf-8 -*-
# @Time    : 2021/7/20 上午11:06
# @Author  : cp
# @File    : utils.py.py
import os
import numpy
import random
import torch
import joblib
from DataLoader.data_loader import load_raw_data
from torch.optim.lr_scheduler import LambdaLR
from os import makedirs
from os.path import dirname, join, normpath, exists


class InputExample(object):
    def __init__(self,
        context_item,
        query_item = None,
        span_position=None):

        self.context_item = context_item
        self.query_item = query_item
        self.span_position = span_position


def load_ner_examples(data_url, entity_query=None):
    """
    Desc:
    """
    sentences,records = joblib.load(data_url)

    examples = []
    for sentence,record in zip(sentences,records):

        example = InputExample(
                    context_item=sentence,
                    query_item = entity_query,
                    span_position=record
                )
        examples.append(example)
    return examples


def lr_linear_decay(optimizer, decay_rate=0.95):
    for param_group in optimizer.param_groups:
        param_group["lr"] = param_group["lr"]*decay_rate
        print("current learning rate", param_group["lr"])


def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    """ Create a schedule with a learning rate that decreases linearly after
    linearly increasing during a warmup period.
    from transformers
    """
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)




def from_project_root(rel_path, create=True):
    """ return system absolute path according to relative path, if path dirs not exists and create is True,
     required folders will be created
    Args:
        rel_path: relative path
        create: whether to create folds not exists

    Returns:
        str: absolute path

    """
    # to get the absolute path of current project
    project_root_url = normpath(join(dirname(__file__), '..'))

    abs_path = normpath(join(project_root_url, rel_path))
    if create and not exists(dirname(abs_path)):
        makedirs(dirname(abs_path))
    return abs_path




def set_random_seed(seed):
    """ set random seed for numpy and torch, more information here:
        https://pytorch.org/docs/stable/notes/randomness.html
    Args:
        seed: the random seed to set
    """
    random.seed(seed)
    numpy.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


