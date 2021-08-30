# -*- coding: utf-8 -*-
# @Time    : 2021/7/20 上午11:41
# @Author  : cp
# @File    : LASN.py

from DataLoader.utils import set_random_seed
set_random_seed(3006)
import warnings
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore")

import torch
import nni
from torch import nn
from nni.utils import merge_parameter
from datetime import datetime
from get_parser import args_parser
from DataLoader.data_processor import *
from tokenizers import BertWordPieceTokenizer
from DataLoader.NER_DataSet import NERDataset
from torch.utils.data import DataLoader
from transformers import AdamW
from pytorch_pretrained_bert import BertAdam
from models.bert_label_ner import BertLabelNER
from models.ner_config import BertLabelNerConfig
from DataLoader.utils import get_linear_schedule_with_warmup,lr_linear_decay
from DataLoader.utils import from_project_root,dirname
from eval import evaluate_e2e,compute_loss





class LoadData(object):

    def __init__(self,config,):
        self.config = config
        vocab_path = os.path.join(config['bert_model'], "vocab.txt")
        self.tokenizer = BertWordPieceTokenizer(vocab_path, lowercase=config['lowercase'])
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.load_data()

    def load_data(self):
        print("-*-" * 10)
        print("current data_name: {}".format(self.config['data_name']))

        if self.config['data_name'] == "BC5CDR":
            self.data_processor = BC5CDRProcessor()
        elif self.config['data_name'] == "JNLPBA":
            self.data_processor = JNLPBAProcessor()
        elif self.config['data_name'] == "NCBI":
            self.data_processor = NCBIProcessor()
        elif self.config['data_name'] == "BC2GM":
            self.data_processor = BC2GMProcessor()
        else:
            raise ValueError("Please Notice that your data_sign DO NOT exits !!!!!")

        self.labels2idx = self.data_processor.get_labels2idx()
        self.entity_type2idx = self.data_processor.get_entity_type()

    def train_dataloader(self):
        return self.get_dataloader(prefix="train")

    def dev_dataloader(self):
        return self.get_dataloader(prefix="dev")

    def test_dataloader(self):
        return self.get_dataloader(prefix="test")

    def get_dataloader(self,prefix):
        dataset = NERDataset(self.config, self.tokenizer, self.labels2idx,self.data_processor,self.device,
                             prefix, evaluating=False if prefix== "train" else True,entity_list=self.entity_type2idx)

        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.config['batch_size'],
            shuffle=True if prefix == "train" else False,
            drop_last=False,
            collate_fn=dataset.collate_func
        )
        return dataloader



def load_model(configs,train_dataloader,class_num,device):
    n_gpu = configs['n_gpu']

    bert_config = BertLabelNerConfig.from_pretrained(configs['bert_model'],
                                                     hidden_dropout_prob=configs['bert_dropout'],
                                                     attention_probs_dropout_prob=configs['bert_dropout'],
                                                     ner_dropout=configs['ner_dropout'],
                                                     max_entity_span = configs['max_entity_span'],
                                                     class_num = class_num,
                                                     use_entity_label = configs['use_entity_label'],
                                                     label_embed_url=configs['label_embed_url'],
                                                     label_freeze=configs['label_freeze'],
                                                     )

    model = BertLabelNER.from_pretrained(configs['bert_model'],
                                        config=bert_config)
    model.to(device)

    if configs['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)

    # prepare optimzier
    param_optimizer = list(model.named_parameters())

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], "weight_decay": 0.01},
        {"params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], "weight_decay": 0.0}]

    t_total = (len(train_dataloader) // (configs['gradient_accumulation_steps']*configs['n_gpu']) ) * configs['num_train_epochs']
    warmup_steps = int(t_total * configs['warmup_proportion'])

    if configs['optimizer'] == "adamw":
        optimizer = AdamW(optimizer_grouped_parameters,
                          betas=(0.9, 0.98),
                          lr=configs['learning_rate'],
                          eps=configs['adam_epsilon'])
    elif configs['optimizer'] == "BertAdam":
        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=configs["learning_rate"],
                             warmup=configs["warmup_proportion"],
                             t_total=t_total)
    else:
        optimizer = torch.optim.Adam(optimizer_grouped_parameters,
                                     lr=configs['learning_rate'],
                                     betas=(0.9, 0.99))

    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                num_training_steps=t_total)

    return model, optimizer, n_gpu, scheduler



def train(model, optimizer, scheduler, train_dataloader, dev_dataloader, test_dataloader, config,
          n_gpu, labels2idx,test_url=True,save_only_best=True):

    start_time = datetime.now()

    cnt = 0
    max_f1, max_f1_epoch = 0, 0
    best_model_url = None

    for epoch in range(int(config['num_train_epochs'])):

        print("#######"*10)
        print("EPOCH: ", str(epoch))
        model.train()

        batch_id = 0
        for step, batch in enumerate(train_dataloader):

            input_ids, segment_ids, head_tail_labels,ht_mask,region_label_id,sent_len,entity_type = batch

            attention_mask = (input_ids != 0).long()

            pred_sentence_labels,pred_region_labels,_ = model(input_ids=input_ids,
                                                            token_type_ids=segment_ids,
                                                            attention_mask=attention_mask,
                                                            label_tensor =entity_type,
                                                            sentence_lengths = sent_len,
                                                            head_tail_positions=head_tail_labels,
                                                            )

            ht_loss, span_loss = compute_loss(pred_sentence_labels=pred_sentence_labels,
                                              head_tail_labels =head_tail_labels,
                                              pred_region_labels=pred_region_labels,
                                              region_labels=region_label_id,
                                              ht_mask = ht_mask
                                              )
            loss =  ht_loss +  span_loss

            if n_gpu > 1:
                loss = loss.mean()

            loss.backward()
            nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=config['clip_grad'])
            optimizer.step()
            scheduler.step()  # Update learning rate schedule
            model.zero_grad()

            if batch_id % config['LOG_PER_BATCH'] == 0:
                print("epoch #%d, batch #%d, loss: %.12f, %s" %
                      (epoch, batch_id, loss.item(), datetime.now().strftime("%X")))
            batch_id += 1

        cnt += 1
        # evaluating model use development dataset and predict on test dataset
        precision, recall, f1 = evaluate_e2e(model, dev_dataloader, config, labels2idx).values()
        if f1 > max_f1:
            max_f1, max_f1_epoch = f1, epoch
            if save_only_best and best_model_url:
                os.remove(best_model_url)
            best_model_url = from_project_root(dirname(config['data_dir']) + "/model_epoch%d_%f.pt" % (epoch, f1))
            torch.save(model, best_model_url)
            cnt = 0

        print("maximum of f1 value: %.6f, in epoch #%d" % (max_f1, max_f1_epoch))
        print("training time:", str(datetime.now() - start_time).split('.')[0])
        print(datetime.now().strftime("%c\n"))

        if cnt >= config["early_stop"]  > 0:
            break

    if test_url:
        best_model = torch.load(best_model_url)
        print("best model url:", best_model_url)
        print("evaluating on test dataset ..........")
        test_precision, test_recall, test_f1 = evaluate_e2e(best_model, test_dataloader, config,labels2idx)
        final_ret = {"test_precision":test_precision,"test_recall":test_recall,"test_f1":test_f1}
        nni.report_intermediate_result(final_ret)




def main():
    tuner_params = nni.get_next_parameter()
    config = vars(merge_parameter(args_parser(), tuner_params))
    load_data = LoadData(config)
    train_loader = load_data.train_dataloader()
    dev_loader = load_data.dev_dataloader()
    test_loader = load_data.test_dataloader()

    model, optimizer, n_gpu,scheduler = load_model(config,train_loader,len(load_data.labels2idx),load_data.device)

    train(model, optimizer,scheduler, train_loader, dev_loader, test_loader, config, n_gpu, load_data.labels2idx)


if __name__ == '__main__':
    main()
