!pip install pytorch_pretrained_bert

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
from pytorch_pretrained_bert import BertModel, BertForMaskedLM, BertForQuestionAnswering, BertForPreTraining

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

input_text  = "I am a student"
target_text = "Je suis étudiant"

modelpath = "bert-base-uncased"

# Load pre-trained model tokenizer (vocabulary)
model = BertForMaskedLM.from_pretrained(modelpath)
model = model.to(device)

n_seq_length = 12

print("input_text       :", input_text)
print("target_text      :", target_text)

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


special_tokens = ['[CLS] [SEP] [MASK]']

# ----------------------------------------------------------
input_text  = [preprocess(input_text)]
target_text = [preprocess(target_text)]

# Encoder Input Define
tokenizer = Tokenizer(filters="", lower=False)
tokenizer.fit_on_texts(special_tokens + input_text+ target_text)

print(tokenizer.word_index)
print(tokenizer.index_word)

# ----------------------------------------------------------

input_text  = [("[CLS] " + input_text[0]  + " [SEP]").split()]
target_text = [(target_text[0] + " [SEP]").split()]

len_input_text = len(input_text[0])

print("input_text       :", input_text)
print("target_text      :", target_text)
print("len_input_text   :", len_input_text)

tokenized_inp_text = tokenizer.texts_to_sequences(input_text)
tokenized_trg_text = tokenizer.texts_to_sequences(target_text)

print("tokenized input  :", tokenized_inp_text)
print("tokenized target :", tokenized_trg_text)

input_text = tokenizer.texts_to_sequences(input_text)
mask_idx   = tokenizer.texts_to_sequences(['[MASK]'])
indexed_inp_tokens = input_text[0] + mask_idx[0] * (n_seq_length - len_input_text)
# print(indexed_inp_tokens)

# use -1 or 0 only for pytorch_pretrained_bert
pad_idx = -1  
converted_trg_inds = []
converted_trg_inds = [pad_idx] * len_input_text
indexed_trg_tokens = tokenizer.texts_to_sequences(target_text)[0]
tmp_trg_tensors    = torch.tensor([indexed_trg_tokens])
converted_trg_inds += tmp_trg_tensors[0].tolist()

for _ in range(n_seq_length-len(converted_trg_inds)):
    converted_trg_inds.append(pad_idx)
    
tensors_src = torch.tensor([indexed_inp_tokens]).to(device)
tensors_trg = torch.tensor([converted_trg_inds]).to(device)

# optimizer = torch.optim.Adam(model.parameters(), lr=5e-7)
# optimizer = torch.optim.SGD(model.parameters(), lr = 5e-5, momentum=0.9)
optimizer = torch.optim.Adamax(model.parameters(), lr = 5e-5)

num_epochs = 1500

model.train()
for i in range(num_epochs):
    loss = model(tensors_src, masked_lm_labels=tensors_trg)
    eveloss = loss.mean().item()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (i+1)%10 == 0:
        print("step "+ str(i+1) + " : " + str(eveloss))
