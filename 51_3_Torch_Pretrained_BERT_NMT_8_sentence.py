!pip install pytorch_pretrained_bert

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM, BertForQuestionAnswering, BertForPreTraining

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

raw_data = (
    ('What a ridiculous concept!', 'Quel concept ridicule !'),
    ('Your idea is not entirely crazy.', "Votre idée n'est pas complètement folle."),
    ("A man's worth lies in what he is.", "La valeur d'un homme réside dans ce qu'il est."),
    ('What he did is very wrong.', "Ce qu'il a fait est très mal."),
    ("All three of you need to do that.", "Vous avez besoin de faire cela, tous les trois."),
    ("Are you giving me another chance?", "Me donnez-vous une autre chance ?"),
    ("Both Tom and Mary work as models.", "Tom et Mary travaillent tous les deux comme mannequins."),
    ("Can I have a few minutes, please?", "Puis-je avoir quelques minutes, je vous prie ?"))

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

raw_encoder_input, raw_data_fr = list(zip(*raw_data))
raw_encoder_input, raw_data_fr = list(raw_encoder_input), list(raw_data_fr)

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

# optimizer = torch.optim.Adam(model.parameters(), lr=5e-7)
# optimizer = torch.optim.SGD(model.parameters(), lr = 5e-5, momentum=0.9)
optimizer = torch.optim.Adamax(model.parameters(), lr = 5e-5)

num_epochs = 300

model.train()
for i in range(num_epochs):
    loss = model(tensors_src, masked_lm_labels=tensors_trg)
    eveloss = loss.mean().item()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
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

