pip install soynlp emoji

pip install transformers

from google.colab import drive
drive.mount('/content/drive')

import pandas as pd
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel, AdamW 
import numpy as np
from tqdm.notebook import tqdm

device = torch.device("cuda")

#데이터 경로 바꿔야 함~!
test_data = pd.read_excel('/content/drive/MyDrive/한국어_단발성_대화_데이터셋.xlsx')

test_datae.loc[(test_data['Emotion'] == "행복"), 'Emotion'] = 0 
test_data.loc[(test_data['Emotion'] == "슬픔"), 'Emotion'] = 1  
test_data.loc[(test_data['Emotion'] == "분노"), 'Emotion'] = 2  
test_data.loc[(test_data['Emotion'] == "불안"), 'Emotion'] = 3  
test_data.loc[(test_data['Emotion'] == "중립"), 'Emotion'] = 4  

train_dataset = []
for sen, label in zip(test_data['Sentence'], test_data['Emotion']):
  data_train = []
  data_train.append(sen)
  data_train.append(str(label))

  train_dataset.append(data_train)

class TrainDataset(Dataset):
  
  def __init__(self, dataset):
    self.tokenizer = AutoTokenizer.from_pretrained("beomi/KcELECTRA-base")

    self.sentences = [str([i[0]]) for i in dataset]
    self.labels = [np.int32(i[1]) for i in dataset]

  def __len__(self):
    return (len(self.labels))
  
  def __getitem__(self, i):
    text = self.sentences[i]
    y = self.labels[i]

    inputs = self.tokenizer(
        text, 
        return_tensors='pt',
        truncation=True,
        max_length=64,
        pad_to_max_length=True,
        add_special_tokens=True
        )
    
    input_ids = inputs['input_ids'][0]
    attention_mask = inputs['attention_mask'][0]

    return input_ids, attention_mask, y

train_dataset = TrainDataset(train_dataset)

from torch import nn
model = AutoModel.from_pretrained("beomi/KcELECTRA-base", num_labels=363)
model = model.to(device)

pip install emoji==1.7

import re
import emoji
from soynlp.normalizer import repeat_normalize

emojis = ''.join(emoji.UNICODE_EMOJI.keys())
pattern = re.compile(f'[^ .,?!/@$%~％·∼()\x00-\x7Fㄱ-ㅣ가-힣{emojis}]+')
url_pattern = re.compile(
    r'https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)')

def clean(x):
    x = pattern.sub(' ', x)
    x = url_pattern.sub('', x)
    x = x.strip()
    x = repeat_normalize(x, num_repeats=2)
    return x

#epochs 수정
batch_size = 32
epochs = 

optimizer = AdamW(model.parameters(), lr=3e-5)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=5, shuffle=True)

losses = []
accuracies = []

#정확도 측정을 위한 함수 정의
def calc_accuracy(X,Y):
    max_vals, max_indices = torch.max(X, 1)
    train_acc = (max_indices == Y).sum().data.cpu().numpy()/max_indices.size()[0]
    return train_acc

loss_fn = nn.CrossEntropyLoss()

for i in range(epochs):
  train_acc = 0.0
  total_loss = 0.0
  correct = 0
  total = 0
  batches = 0

  model.train()

  for input_ids_batch, attention_masks_batch, y_batch in tqdm(train_dataloader):
    optimizer.zero_grad()
    y_batch = y_batch.long().to(device)
    y_pred = model(input_ids_batch.to(device), attention_mask=attention_masks_batch.to(device))[0]
    y_pred = y_pred[:, -1, :]
    loss = loss_fn(y_pred, y_batch)
    loss.backward()
    optimizer.step()

    total_loss += loss.item()

    train_acc += calc_accuracy(y_pred, y_batch)
    total += len(y_batch)

    batches += 1
    if batches % 50 == 0:
      print("epoch {} loss {} train acc {}".format(i+1, loss.data.cpu().numpy(), train_acc / (batches+1)))
  print("epoch {} loss {} train acc {}".format(i+1, loss.data.cpu().numpy(), train_acc / (batches+1)))
  model.eval()

torch.save(model.state_dict(), "/content/drive/MyDrive/Colab Notebooks/model.pt")

def predict(sentence):
    data = [sentence, '0']
    dataset_another = [data]
    logits = 0
    another_test = TrainDataset(dataset_another)
    test_dataloader = torch.utils.data.DataLoader(another_test)

    model.eval()

    for input_ids_batch, attention_masks_batch, y_batch in test_dataloader:
        y_batch = y_batch.long().to(device)
        out = model(input_ids_batch.to(device), attention_mask=attention_masks_batch.to(device))[0]
        out = out[:, -1, :]

        for i in out:
            logits = i
            logits = logits.detach().cpu().numpy()
            logits = np.argmax(logits)
    return logits

predict("일기 내용")
