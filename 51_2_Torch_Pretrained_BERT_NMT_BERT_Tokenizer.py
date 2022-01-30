!pip install pytorch_pretrained_bert

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM, BertForQuestionAnswering, BertForPreTraining

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

input_text  = "[CLS] I want to buy the new Apple M1 Pro MacBook pro [SEP] "
target_text = "Je veux acheter le nouveau MacBook Pro Apple M1 Pro"

modelpath = "bert-base-uncased"

# Load pre-trained model tokenizer (vocabulary)
model = BertForMaskedLM.from_pretrained(modelpath)
model = model.to(device)

n_seq_length = 30

print("input_text       :", input_text)
print("target_text      :", target_text)

tokenizer = BertTokenizer.from_pretrained(modelpath)
tokenized_inp_text = tokenizer.tokenize(input_text)
tokenized_trg_text = tokenizer.tokenize(target_text)

len_input_text = len(tokenized_inp_text)
print("len_input_text   :", len_input_text)

print("tokenized input  :", tokenized_inp_text)
print("tokenized target :", tokenized_trg_text)

# Processing for model
for _ in range(n_seq_length-len(tokenized_inp_text)):
    tokenized_inp_text.append('[MASK]')
    
indexed_inp_tokens = tokenizer.convert_tokens_to_ids(tokenized_inp_text)

# use -1 or 0 only for pytorch_pretrained_bert
pad_idx = -1
converted_trg_inds = []
converted_trg_inds = [pad_idx] * len_input_text
indexed_trg_tokens = tokenizer.convert_tokens_to_ids(tokenized_trg_text)
tmp_trg_tensors    = torch.tensor([indexed_trg_tokens])
converted_trg_inds += tmp_trg_tensors[0].tolist()
converted_trg_inds.append(tokenizer.convert_tokens_to_ids(['[SEP]'])[0])

for _ in range(n_seq_length-len(converted_trg_inds)):
    converted_trg_inds.append(pad_idx)
    
tensors_src = torch.tensor([indexed_inp_tokens]).to(device)
tensors_trg = torch.tensor([converted_trg_inds]).to(device)

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

result = []
result_ids = []
model.eval()
with torch.no_grad():
    predictions = model(tensors_src)

    start = len(tokenizer.tokenize(input_text))
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
        
print("tokenized target :", tokenized_trg_text)
print("result_ids       :",result_ids)
print("result           :",result)
