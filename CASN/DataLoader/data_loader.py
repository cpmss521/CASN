# -*- coding: utf-8 -*-
# @Time    : 2021/7/20 上午11:11
# @Author  : cp
# @File    : DataLoader.py

import re
import joblib
import os


def load_raw_data(data_url, update=False):
    """ load data into sentences and records
    Args:
        data_url: url to data file
        update: whether force to update
    Returns:
        sentences(raw), records
        format:
        sentences: [['-DOCSTART-'], ['EU', 'rejects', 'German',...... '.'], ['Peter', 'Blackburn']]
        records: [{}, {(0, 1): 'ORG', (2, 3): 'MISC', (6, 7): 'MISC'}, {(0, 2): 'PER'}]
    """
    # load from pickle
    save_url = data_url.replace('.tsv', '.raw.pkl').replace('.iob2', '.raw.pkl')

    if not update and os.path.exists(save_url):
        return joblib.load(save_url)

    sentences = list()
    records = list()
    with open(data_url, 'r', encoding='utf-8') as iob_file:
        first_line = iob_file.readline()
        n_columns = len(first_line.split())-1 if 'genia' in data_url else len(first_line.split())# nested genia dataset format
        columns = [[x] for x in first_line.split()]

        for line in iob_file:
            if len(line) == 0 or line[0]=='\n' or line.startswith("-DOCSTART-"):
                if len(columns[0])>0:
                    sentence = columns[0]
                    sentences.append(sentence)
                    row = columns[1:] if 'genia' in data_url else [columns[-1]]  # a sent of label list
                    records.append(infer_records(row))
                    columns = [list() for i in range(n_columns)]
                continue

            line_values = line.split()
            columns[0].append(normalize_word(line_values[0]) if line_values[-1] == "O" else line_values[0])
            # other feature and label information
            for i in range(1, n_columns):
                columns[i].append(line_values[i])

        ## last sentence
        if len(columns[0]) > 0:
            sentence = columns[0]
            sentences.append(sentence)
            row = columns[1:] if 'genia' in data_url else [columns[-1]]  # label
            # print('row:',row)
            records.append(infer_records(row))
            columns = [list() for i in range(n_columns)]


    joblib.dump((sentences, records), save_url)
    return sentences, records


def normalize_word(token):
    """replace number with 0"""
    # token :string
    pattern = re.compile(r'^[-+]?[-0-9]\d*\.\d*|[-+]?\.?[0-9]\d*$')
    result = pattern.match(token)
    word = '0' if result else token
    new_word = ""
    for char in word:
        if char.isdigit():
            new_word += '0'
        else:
            new_word += char
    return new_word

def infer_records(columns):
    """ inferring all entity records of a sentence
    Args:
        columns: columns of a sentence in iob2 format
    Returns:
        entity record in gave sentence
        {(0, 1): 'ORG', (2, 3): 'MISC', (6, 7): 'MISC'}
    """
    records = dict()
    for col in columns:
        start = 0
        while start < len(col):
            end = start + 1
            if col[start][0] == 'B':
                while end < len(col) and col[end][0] == 'I':
                    end += 1
                records[(start, end)] = col[start][2:]
            start = end
    return records


if __name__ == '__main__':
    data_file = '/home/cp/data/BC5CDR/train.tsv'
    sentences,records = load_raw_data(data_file,True)
    print(sentences[0])
    print(records[0])
    print(sentences[-1])
    print(records[-1])