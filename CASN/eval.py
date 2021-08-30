# -*- coding: utf-8 -*-
# @Time    : 2021/7/20 下午9:49
# @Author  : cp
# @File    : eval.py.py

import torch
import torch.nn.functional as F
from torch.nn.modules import CrossEntropyLoss
from sklearn.metrics import classification_report


def evaluate_e2e(model, eval_dataloader, params,labels2idx):

    # eval mode
    model.eval()
    with torch.no_grad():
        sentence_true_list, sentence_pred_list = list(), list()
        region_true_list, region_pred_list = list(), list()
        region_true_count, region_pred_count = 0, 0

        for input_ids, segment_ids, head_tail_labels, ht_mask, region_labels, sent_len, span_pos,entity_type in eval_dataloader:

            attention_mask = (input_ids != 0).long()
            try:
                pred_sentence_output,pred_region_output,M_atten = model.forward(input_ids,segment_ids,attention_mask,entity_type,sent_len)
                pred_sentence_labels = torch.argmax(pred_sentence_output, dim=-1)
            except RuntimeError:
                print("all 0 tags, no evaluating this epoch")
                continue

            # pred_region_output (n_regions, n_tags)
            pred_region_labels = torch.argmax(pred_region_output, dim=1).view(-1).cpu()
            # (n_regions)

            ind = 0
            for sent_labels, length, true_records in zip(pred_sentence_labels, sent_len, span_pos):
                pred_records = dict()
                for start in range(0, length):
                    if sent_labels[start] == 1:
                        if pred_region_labels[ind]>0:
                            pred_records[(start,start+1)] = pred_region_labels[ind]
                        ind += 1
                        for end in range(start + 1, length):
                            if sent_labels[end] == 2 and (end-start)<=params['max_entity_span']:
                                if pred_region_labels[ind]:
                                    pred_records[(start,end+1)] = pred_region_labels[ind]
                                ind += 1

                for region in true_records:
                    true_label = labels2idx[true_records[region]]
                    pred_label = pred_records[region] if region in pred_records else 0
                    region_true_list.append(true_label)
                    region_pred_list.append(pred_label)
                for region in pred_records:
                    if region not in true_records:
                        region_pred_list.append(pred_records[region])
                        region_true_list.append(0)

            region_labels = region_labels.view(-1).cpu()
            region_true_count += int((region_labels > 0).sum())
            region_pred_count += int((pred_region_labels > 0).sum())

            pred_sentence_labels = pred_sentence_labels.view(-1).cpu()
            sentence_labels = head_tail_labels.view(-1).cpu()
            label_mask = ht_mask.view(-1).cpu()
            pred_sentence_labels = pred_sentence_labels*label_mask

            for tv, pv, in zip(sentence_labels, pred_sentence_labels):
                sentence_true_list.append(tv)
                sentence_pred_list.append(pv)


        print("sentence head and tail labeling result:")
        print(classification_report(sentence_true_list, sentence_pred_list,
                                    target_names=['out-entity', 'head-entity','tail-entity'], digits=6))

        print("entity span Classification result:")
        print(classification_report(region_true_list, region_pred_list,
                                    target_names=list(labels2idx), digits=6))
        ret = dict()
        tp = 0
        for pv, tv in zip(region_pred_list, region_true_list):
            if pv == tv == 0:
                continue
            if pv == tv:
                tp += 1
        fp = region_pred_count - tp
        fn = region_true_count - tp
        ret['precision'], ret['recall'], ret['f1'] = calc_f1(tp, fp, fn)

        return ret


def calc_f1(tp, fp, fn, print_result=True):
    """ calculating f1

    Args:
        tp: true positive
        fp: false positive
        fn: false negative
        print_result: whether to print result

    Returns:
        precision, recall, f1

    """
    precision = 0 if tp + fp == 0 else tp / (tp + fp)
    recall = 0 if tp + fn == 0 else tp / (tp + fn)
    f1 = 0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    if print_result:
        print(" precision = %f, recall = %f, micro_f1 = %f\n" % (precision, recall, f1))
    return precision, recall, f1

def compute_loss(pred_sentence_labels,head_tail_labels,
                 pred_region_labels, region_labels,ht_mask):


    loss_fct = CrossEntropyLoss(reduction="sum")
    if ht_mask is not None:

        active_logits = pred_sentence_labels.view(-1, 3)# head/tail/other
        active_loss = ht_mask.view(-1) == 1
        active_labels = torch.where(
            active_loss,
            head_tail_labels.view(-1),
            torch.tensor(loss_fct.ignore_index).type_as(head_tail_labels),
        )

        ht_loss = loss_fct(active_logits, active_labels)
    else:
        pred_sentence_labels = pred_sentence_labels.permute(0,2,1)#B,3,S
        ht_loss = loss_fct(pred_sentence_labels, head_tail_labels)

    span_loss = loss_fct(pred_region_labels, region_labels)

    return ht_loss, span_loss



