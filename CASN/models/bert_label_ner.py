# -*- coding: utf-8 -*-
# @Time    : 2021/7/20 下午9:30
# @Author  : cp
# @File    : bert_label_ner.py.py

import torch
import torch.nn.init as init
import numpy as np
import torch.nn as nn
from transformers import BertModel, BertPreTrainedModel


class BertLabelNER(BertPreTrainedModel):
    def __init__(self, config):
        super(BertLabelNER, self).__init__(config)
        self.bert = BertModel(config)

        self.hidden_size = config.hidden_size
        self.max_entity_span = config.max_entity_span
        self.class_num = config.class_num
        self.label_embed_url = config.label_embed_url
        print("label_embed_url:", self.label_embed_url)

        self.label_embedding = nn.Embedding.from_pretrained(
            embeddings=torch.Tensor(np.load(self.label_embed_url)),
            freeze=config.label_freeze
        )

        self.label_liner = torch.nn.Linear(2*self.hidden_size, self.hidden_size)#200+ for BC2GM

        self.Co_Attention = COAttention(self.hidden_size, config.ner_dropout)

        # task2: entity type classifier
        self.region_clf = SpanCLF(
            repr_dim=self.hidden_size,
            n_classes=self.class_num,
            drop_rate=config.ner_dropout

        )
        # task1: head and tail classifier
        self.ht_labeler = nn.Sequential(
            nn.GELU(),
            nn.Linear(self.hidden_size, 3),
        )

        self.init_weights()

    def forward(self, input_ids, token_type_ids, attention_mask, label_tensor, sentence_lengths,
                head_tail_positions=None):


        bert_outputs = self.bert(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)

        start_index = input_ids[0].tolist().index(102)

        sequence_heatmap = bert_outputs[0]
        label_heatmap = bert_outputs[0][:,1:start_index,:]#output label represent

        label_repr = self.label_embedding(label_tensor)
        label_repr = torch.cat([label_repr,label_heatmap],dim=-1)#b,l,2h
        label_repr = self.label_liner(label_repr)#b,l,h

        sequence_out,M_atten = self.Co_Attention(sequence_heatmap, attention_mask, label_repr)


        ###### task1: head and tail sequence labeler ######
        sentence_outputs = self.ht_labeler(sequence_out)#(B,S, n_classes)

        ###### task2: region classification ######
        # if not self.training, just for evaluate or test
        if head_tail_positions is None:
            head_tail_positions = torch.argmax(sentence_outputs, dim=-1)

        regions = list()
        for hidden, sentence_label, length in zip(sequence_out, head_tail_positions, sentence_lengths):
            for start in range(0, length):
                if sentence_label[start] == 1:
                    regions.append(hidden[start:start + 1])
                    for end in range(start + 1, length):
                        if sentence_label[end] == 2 and (end - start) <= self.max_entity_span:
                            regions.append(hidden[start:end + 1])
        region_outputs = self.region_clf(regions)
        # shape of region_labels: (n_regions, n_classes)

        return sentence_outputs, region_outputs,M_atten


class SpanCLF(nn.Module):
    def __init__(self, repr_dim, n_classes,drop_rate=0.0):
        super().__init__()

        self.dropout = nn.Dropout(p=drop_rate)
        self.fc = nn.Sequential(
            nn.Linear(repr_dim *2, repr_dim),
            nn.GELU(),
            nn.Linear(repr_dim, n_classes),
        )

    def forward(self, data_list):
        data_repr = self.span_repr(data_list)
        return self.fc(data_repr)

    def span_repr(self,data_list):
        """
        data_list: [n_regions,1,768]
        """
        # represent 1: cat[hi+hj;hi-hj]:2*dim
        cat_regions = [torch.cat([hidden[0] + hidden[-1], hidden[0] - hidden[-1]], dim=-1).view(1, -1)
                       for hidden in data_list]
        # represent 2: cat[left,mean,right]: 3*dim
        
        # cat_regions = [torch.cat([hidden[0], torch.mean(hidden, dim=0), hidden[-1]], dim=-1).view(1, -1)
        #        for hidden in data_list]

        cat_out = torch.cat(cat_regions, dim=0)
        cat_out = self.dropout(cat_out)

        return cat_out



def mask_logits(inputs, mask, mask_value=-1e30):
    mask = mask.type(torch.float32)
    return inputs + (1.0 - mask) * mask_value


class Conv1D(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size=1, stride=1, padding=0, bias=True):
        super(Conv1D, self).__init__()
        self.conv1d = nn.Conv1d(in_channels=in_dim, out_channels=out_dim,
                                kernel_size=(kernel_size,),
                                padding=padding, stride=(stride,), bias=bias)
    def forward(self, x):
        # suppose all the input with shape (batch_size, seq_len, dim)
        x = x.transpose(1, 2)  # (B,h,S)
        x = self.conv1d(x)
        return x.transpose(1, 2)  #(B,S,h)


class COAttention(nn.Module):
    def __init__(self, dim, drop_rate=0.0):
        super(COAttention, self).__init__()

        w4C = torch.empty(dim, 1)
        w4Q = torch.empty(dim, 1)
        w4mlu = torch.empty(1, 1, dim)
        init.xavier_uniform_(w4C)
        init.xavier_uniform_(w4Q)
        init.xavier_uniform_(w4mlu)

        self.w4C = nn.Parameter(w4C, requires_grad=True)
        self.w4Q = nn.Parameter(w4Q, requires_grad=True)
        self.w4mlu = nn.Parameter(w4mlu, requires_grad=True)
        self.dropout = nn.Dropout(p=drop_rate)
        self.cqa_linear = Conv1D(in_dim=4 * dim, out_dim=dim, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, context, c_mask, query):
        score = self.trilinear_attention(context, c_mask, query)  # B,S,L
        score_ = nn.Softmax(dim=2)(score)  # B,S,L
        score_t = nn.Softmax(dim=1)(mask_logits(score, c_mask.unsqueeze(2)))  # B,S,L
        score_t = score_t.transpose(1, 2)  # B,L,S
        c2q = torch.matmul(score_, query)  # B,S,h
        q2c = torch.matmul(torch.matmul(score_, score_t), context)  # B,S,h
        output = torch.cat([context, c2q, torch.mul(context, c2q), torch.mul(context, q2c)], dim=2)
        out = self.cqa_linear(output)  # B,S,h
        out += context
        return out,score_

    def trilinear_attention(self, context, c_mask, query):
        batch_size, c_seq_len, dim = context.shape
        batch_size, q_seq_len, dim = query.shape
        context = self.dropout(context)
        query = self.dropout(query)
        subres0 = torch.matmul(context, self.w4C).expand([-1, -1, q_seq_len])  # B,S,L
        subres1 = torch.matmul(query, self.w4Q).transpose(1, 2).expand([-1, c_seq_len, -1])
        subres2 = torch.matmul(context * self.w4mlu, query.transpose(1, 2))
        mask = (1.0 - c_mask.float()) * -10000.0
        res = subres0 + subres1 + subres2 + mask.unsqueeze(2)  # B,S,L
        return res
