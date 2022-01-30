!pip install pytorch_pretrained_bert

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM, BertForQuestionAnswering, BertForPreTraining

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

import pandas as pd
# import sentencepiece as spm
import urllib.request
import csv

# urllib.request.urlretrieve("http://www.manythings.org/anki/fra-eng.zip", filename="fra-eng.zip")
! wget http://www.manythings.org/anki/fra-eng.zip

! unzip fra-eng.zip

total_df = pd.read_csv('fra.txt', sep="\t", header=None)

# total_df = total_df.sample(frac=1)  # row 전체 shuffle
# total_df = total_df[:20000]
total_df[:5]

total_df.rename(columns={0: 'english', 1: 'french', 2: 'speaker'}, inplace=True)

total_df[:5]

print('Translation Pair :',len(total_df)) # 리뷰 개수 출력

total_df["eng_len"] = ""
total_df["fra_len"] = ""
total_df.head()

import sys
for idx in range(len(total_df['english'])):
    # initialize string
    text_eng = str(total_df.iloc[idx]['english'])

    # default separator: space
    result_eng = len(text_eng.split())
    total_df.at[idx, 'eng_len'] = int(result_eng)

    text_fra = str(total_df.iloc[idx]['french'])
    # default separator: space
    result_fra = len(text_fra.split())
    total_df.at[idx, 'fra_len'] = int(result_fra)

# country 컬럼을 선택합니다.
# 컬럼의 값과 조건을 비교합니다.
# 그 결과를 새로운 변수에 할당합니다.
is_within_len = ( 7 < total_df['eng_len']) & ( total_df['eng_len']<17)

# 조건를 충족하는 데이터를 필터링하여 새로운 변수에 저장합니다.
total_df = total_df[is_within_len]

# 결과를 출력합니다.
total_df.head()

# n_samples = 43693
n_samples = 256
total_df = total_df.sample(n=n_samples, # number of items from axis to return.
          random_state=1234) # seed for random number generator for reproducibility
len(total_df)

with open('corpus_src.txt', 'w', encoding='utf8') as f:
    f.write('\n'.join(total_df['english']))

with open('corpus_trg.txt', 'w', encoding='utf8') as f:
    f.write('\n'.join(total_df['french']))

raw_encoder_input = total_df['english'].tolist()
raw_data_fr = total_df['french'].tolist()

print(raw_encoder_input)

import unicodedata
import re

from tensorflow.keras.preprocessing.text import Tokenizer

def unicode_to_ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s)
            if unicodedata.category(c) != 'Mn')

def preprocess(sent):
    # 위에서 구현한 함수를 내부적으로 호출
    sent = unicode_to_ascii(sent.lower())

    # 단어와 구두점 사이에 공백을 만듭니다.
    # Ex) "he is a boy." => "he is a boy ."
    sent = re.sub(r"([?.!,¿])", r" \1", sent)

    # (a-z, A-Z, ".", "?", "!", ",") 이들을 제외하고는 전부 공백으로 변환합니다.
    sent = re.sub(r"[^a-zA-Z!.?]+", r" ", sent)

    sent = re.sub(r"\s+", " ", sent)
    return sent

# 인코딩 테스트
en_sent = u"Have you had dinner?"
fr_sent = u"Avez-vous déjà diné?"

print(preprocess(en_sent))
print(preprocess(fr_sent).encode('utf-8'))


input_text = ['[CLS] ' + preprocess(data) + ' [SEP]' for data in raw_encoder_input]
target_text = [preprocess(data) for data in raw_data_fr]

print(input_text[:5])
print(target_text[:5])

# Load pre-trained model tokenizer (vocabulary)
modelpath = "bert-base-uncased"

# Load pre-trained model tokenizer (vocabulary)
model = BertForMaskedLM.from_pretrained(modelpath)
model = model.to(device)

n_seq_length = 80

tokenizer = BertTokenizer.from_pretrained(modelpath)

for idx in range(len(input_text)):

    tokenized_inp_text = tokenizer.tokenize(input_text[idx])
    
    tokenized_trg_text = tokenizer.tokenize(target_text[idx])
    len_input_text = len(tokenized_inp_text)
    
    # Processing for model
    for _ in range(n_seq_length-len(tokenized_inp_text)):
        tokenized_inp_text.append('[MASK]')

    indexed_inp_tokens = tokenizer.convert_tokens_to_ids(tokenized_inp_text)

    pad_idx = -1
    converted_trg_inds = []
    converted_trg_inds = [pad_idx] * len_input_text
    
    indexed_trg_tokens = tokenizer.convert_tokens_to_ids(tokenized_trg_text)
    tmp_trg_tensors   = torch.tensor([indexed_trg_tokens])
    converted_trg_inds += tmp_trg_tensors[0].tolist()
    
    converted_trg_inds.append(tokenizer.convert_tokens_to_ids(['[SEP]'])[0])

    for _ in range(n_seq_length-len(converted_trg_inds)):
        converted_trg_inds.append(pad_idx)

    src_tensor = torch.tensor([indexed_inp_tokens]).to(device)
    trg_tensor = torch.tensor([converted_trg_inds]).to(device)

    if idx == 0:
        tensors_src = src_tensor
    else :
        tensors_src = torch.cat((tensors_src, src_tensor), 0)

    if idx == 0:
        tensors_trg = trg_tensor
    else :
        tensors_trg = torch.cat((tensors_trg, trg_tensor), 0)


from torch.utils.data import TensorDataset   # 텐서데이터셋
from torch.utils.data import DataLoader      # 데이터로더

batch_size = 64
dataset = TensorDataset(tensors_src, tensors_trg)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# optimizer = torch.optim.Adam(model.parameters(), lr=5e-7)
# optimizer = torch.optim.SGD(model.parameters(), lr = 5e-5, momentum=0.9)
optimizer = torch.optim.Adamax(model.parameters(), lr = 5e-5)

num_epochs = 300

model.train()
n_batches = len(dataset)/ batch_size

for i in range(num_epochs):
    
    epoch_loss = 0
    for batch_idx, samples in enumerate(dataloader):
        x_train, y_train = samples
        loss = model(x_train, masked_lm_labels=y_train)
        eveloss = loss.mean().item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += eveloss / n_batches

    if (i+1)%10 == 0:
        print("step "+ str(i+1) + " : " + str(eveloss))

print(tensors_src[6])
test_list = tensors_src[6].tolist()
test_tokens_tensor = torch.tensor([test_list]).to(device)
print(test_tokens_tensor)

result = []
result_ids = []
model.eval()
with torch.no_grad():
    predictions = model(test_tokens_tensor)

    start = len(tokenizer.tokenize(input_text[6]))
    count = 0
    while start < len(predictions[0]):
        predicted_index = torch.argmax(predictions[0,start]).item()
        
        predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])
        if '[SEP]' in predicted_token:
            break
        if count == 0:
            result = predicted_token
            result_ids = [predicted_index]
        else:
            result+= predicted_token
            result_ids+= [predicted_index]

        count += 1
        start += 1
print("input_text       :", input_text[6])
print("target_text      :", target_text[6])
print("tokenized target :", tokenizer.tokenize(target_text[6]))
print("result_ids       :",result_ids)
print("result           :",result)

