# -*- coding: utf-8 -*-
# @Time    : 2021/7/20 上午11:31
# @Author  : cp
# @File    : NER_DataSet.py.py



import torch
import numpy as np
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, segment_ids, head_tail_labels, span_label = None,ht_label_mask= None):
        self.input_ids = input_ids
        self.segment_ids = segment_ids
        self.head_tail_labels = head_tail_labels
        self.span_label = span_label
        self.ht_label_mask = ht_label_mask




class NERDataset(Dataset):
    def __init__(self,params,tokenizer,labels2idx,data_processor,device,prefix="train",
                 evaluating=False,entity_list=None):
        super().__init__()

        self.data_processor = data_processor
        self.entity_type2idx = entity_list
        self.labels2idx = labels2idx
        self.prefix = prefix
        self.device = device
        self.evaluating = evaluating
        self.tokenizer = tokenizer
        self.data_dir = params["data_dir"]
        self.max_seq_length = params["max_seq_length"]
        self.max_entity_span = params["max_entity_span"]

        self.examples = self.data_processor.get_examples(self.data_dir, self.prefix,self.entity_type2idx)#

    def __getitem__(self, index):

        return self.examples[index]

    def __len__(self):
        return len(self.examples)


    def collate_func(self,batch_example):


        batch_tokenizer_len = list()
        batch_region_label_id = list()# a batch region list
        batch_span_label = list()# a batch list of {span position}

        feature = []

        for ex_idx, example in enumerate(batch_example):
            context = example.context_item
            query = example.query_item
            record = example.span_position

            ##enttiy boundary
            start_position = []
            end_position = []
            for i in record:
                start_position.append(i[0])
                end_position.append(i[1]-1)

            ### offset position
            start_pos_offset = [x + sum([len(w) for w in context[:x]]) for x in start_position]
            end_pos_offset = [x + sum([len(w) for w in context[:x + 1]]) for x in end_position]

            # [CLS]+context+[SEP]
            query_str = " ".join(i for i in query)
            context_str = " ".join(j for j in context)
            context_tokens = self.tokenizer.encode(query_str, context_str, add_special_tokens=True)
            # context_tokens = self.tokenizer.encode(context_str, add_special_tokens=True)
            tokens_ids = context_tokens.ids
            type_ids = context_tokens.type_ids
            offsets = context_tokens.offsets

            # find new start_positions/end_positions, considering
            # 1. we add query tokens at the beginning
            # 2. word-piece tokenize
            origin_offset2token_idx_start = {}
            origin_offset2token_idx_end = {}

            sequence_len = self.max_seq_length if len(tokens_ids) >= self.max_seq_length else len(tokens_ids)
            for token_idx in range(sequence_len):
                # skip query tokens
                if type_ids[token_idx] == 0:
                    continue
                token_start, token_end = offsets[token_idx]
                # skip [CLS] or [SEP]
                if token_start == token_end == 0:
                    continue
                origin_offset2token_idx_start[token_start] = token_idx
                origin_offset2token_idx_end[token_end] = token_idx

            # get new start and end position after Bert tokenization
            new_start_positions = [origin_offset2token_idx_start.get(start) for start in start_pos_offset
                                   if origin_offset2token_idx_start.get(start) is not None]
            new_end_positions = [ origin_offset2token_idx_end.get(end)  for end in end_pos_offset
                                  if origin_offset2token_idx_end.get(end) is not None]

            # get new span position and label after Bert tokenization
            span_label = {}
            head_tail_labels = [0] * len(tokens_ids)
            for i, j, p, q in zip(new_start_positions, new_end_positions, start_position, end_position):
                span_label[(i, j + 1)] = record.get((p, q + 1))
                head_tail_labels[j] = 2
                head_tail_labels[i] = 1

            batch_span_label.append(span_label)

            ## label mask
            label_mask = [
                (0 if type_ids[token_idx] == 0 or offsets[token_idx] == (0, 0) else 1)
                for token_idx in range(len(tokens_ids))
            ]

            ht_label_mask = label_mask.copy()
            # the start/end position must be whole word
            for token_idx in range(len(tokens_ids)):
                current_word_idx = context_tokens.words[token_idx]
                next_word_idx = context_tokens.words[token_idx + 1] if token_idx + 1 < len(tokens_ids) else None
                prev_word_idx = context_tokens.words[token_idx - 1] if token_idx - 1 > 0 else None

                if (prev_word_idx is not None) and (next_word_idx is not None) and (current_word_idx == prev_word_idx == next_word_idx):
                    ht_label_mask[token_idx] = 0

            assert all(ht_label_mask[p] != 0 for p in new_start_positions)
            assert all(ht_label_mask[p] != 0 for p in new_end_positions)
            assert len(end_pos_offset) == len(start_pos_offset)
            assert len(label_mask) == len(tokens_ids)

            # truncate sentence
            if len(tokens_ids) >= self.max_seq_length:
                tokens_ids = tokens_ids[: self.max_seq_length]
                type_ids = type_ids[: self.max_seq_length]
                head_tail_labels = head_tail_labels[:self.max_seq_length]
                ht_label_mask = ht_label_mask[: self.max_seq_length]



            # make sure last token is [SEP]
            sep_token = self.tokenizer.token_to_id("[SEP]")
            if tokens_ids[-1] != sep_token:
                assert len(tokens_ids) == self.max_seq_length
                tokens_ids = tokens_ids[: -1] + [sep_token]
                head_tail_labels[-1] = 0
                ht_label_mask[-1] = 0


            ## save batch tokenizer len:
            batch_tokenizer_len.append(len(tokens_ids))

            # get a sent of region label
            region_label_ids = []
            length =  len(tokens_ids)
            for start in range(0, length):
                if head_tail_labels[start] == 1:
                    region_label_ids.append(self.labels2idx[span_label[(start, start + 1)]] if (start, start + 1) in span_label else 0)
                    for end in range(start + 1, length):
                        if head_tail_labels[end] == 2 and (end-start)<=self.max_entity_span:
                            ## set max span len no more than max_entity_span
                            region_label_ids.append(
                                self.labels2idx[span_label[(start, end + 1)]] if (start, end + 1) in span_label else 0)

            batch_region_label_id.extend(region_label_ids)

            feature.append(
                InputFeatures(input_ids=torch.LongTensor(tokens_ids),
                              segment_ids=torch.LongTensor(type_ids),
                              head_tail_labels=torch.LongTensor(head_tail_labels),
                              ht_label_mask= torch.LongTensor(ht_label_mask)
                              )
            )

            if ex_idx < 0:
                print("********** Example : %s **********" % (ex_idx))
                print("context_item:\n",context)
                print("query_item:\n", query)
                print("record_item:\n",record)
                print("start_position:\n",start_position)
                print("end_position:\n",end_position)
                print("tokens_ids:\n",tokens_ids)
                print("type_ids:\n",type_ids)
                print("offsets:\n",offsets)
                print("context_tokens tokens:\n",context_tokens.tokens)
                print("new_start_positions:\n",new_start_positions)
                print("new_end_positions:\n",new_end_positions)
                print("ht_label_mask:\n", ht_label_mask)
                print("span_label:\n",span_label)
                print("head_tail_labels:\n",head_tail_labels)
                print("region_label:\n",region_label_ids)

        # print("batch_tokenizer_len:\n", batch_tokenizer_len)
        # print("batch_span_label:\n",batch_span_label)
        # print("batch_region_label_id:\n",batch_region_label_id)
        # print("*****"*20)

        ## padding to max batch len;
        input_ids = pad_sequence([f.input_ids for f in feature],batch_first=True).to(self.device)
        segment_ids = pad_sequence([f.segment_ids for f in feature],batch_first=True).to(self.device)
        head_tail_labels = pad_sequence([f.head_tail_labels for f in feature],batch_first=True).to(self.device)
        ht_label_mask = pad_sequence([f.ht_label_mask for f in feature], batch_first=True).to(self.device)


        batch_region_label_id = torch.LongTensor(batch_region_label_id).to(self.device)
        batch_Entity_type = np.array([list(self.entity_type2idx.values()) for _ in range(len(batch_tokenizer_len))])
        batch_Entity_type = torch.from_numpy(batch_Entity_type).to(self.device)


        if self.evaluating:
            return input_ids,segment_ids,head_tail_labels,ht_label_mask,batch_region_label_id,batch_tokenizer_len,\
                   batch_span_label,batch_Entity_type

        return input_ids,segment_ids,head_tail_labels,ht_label_mask,batch_region_label_id,batch_tokenizer_len,batch_Entity_type













