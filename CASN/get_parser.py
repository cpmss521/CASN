# -*- coding: utf-8 -*-
# @Time    : 2021/7/19 上午11:37
# @Author  : cp
# @File    : get_parser.py.py


import argparse


def args_parser():
    # start parser
    parser = argparse.ArgumentParser()

    # requires parameters
    parser.add_argument("--data_name", type=str, default="NCBI")
    parser.add_argument("--data_dir", type=str,default="/home/cp/CASN/data/NCBI/")
    parser.add_argument("--bert_model",type=str, default="/home/cp/Embedding/bert-base-cased/")
    parser.add_argument("--lowercase", type=bool, default=False,help='case or lowercase for bert')
    parser.add_argument("--hidden_size", type=int, default=768)

    parser.add_argument("--max_seq_length", type=int, default=150)
    parser.add_argument("--batch_size", type=int, default=32)#
    parser.add_argument("--learning_rate", type=float, default=5e-5)#5e-5
    parser.add_argument("--num_train_epochs", type=int, default=25)
    parser.add_argument("--label_freeze", type=bool, default=True)
    parser.add_argument("--label_embed_url", type=str, default="/home/cp/CASN/data/NCBI/label_NCBI_768.npy")
    parser.add_argument("--ner_dropout", type=float, default=0.2)
    parser.add_argument("--max_entity_span", type=int, default=20)####
    parser.add_argument("--early_stop", type=int, default=5)
    parser.add_argument("--clip_grad", type=float, default=1.0)#'1.0;5.0'
    parser.add_argument("--gamma", type=float, default=0.3)

    parser.add_argument("--optimizer", type=str, choices=["adamw", "BertAdam","Adam"], default="adamw",help="loss type")
    parser.add_argument("--LOG_PER_BATCH", type=int, default=10,help="log information")
    parser.add_argument("--bert_dropout", type=float, default=0.1, help="bert dropout rate")
    parser.add_argument("--n_gpu", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--warmup_proportion", type=float,default=0.1)
    parser.add_argument("--adam_epsilon",  type=float,default=1e-8,help="Epsilon for Adam optimizer")

    args = parser.parse_args()


    return args


