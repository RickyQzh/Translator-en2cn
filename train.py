
import torch
import torch.optim as optim
import torch.nn as nn
from model import TransformerMT
from dataloader import TranslationDataset, collate_fn
from torch.utils.data import DataLoader
import json

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('./cmn-eng-simple/word2int_en.json') as f:
    word2int_en = json.load(f)
with open('./cmn-eng-simple/word2int_cn.json') as f:
    word2int_cn = json.load(f)

SRC_VOCAB_SIZE = len(word2int_en)
TRG_VOCAB_SIZE = len(word2int_cn)

train_loader = DataLoader(
    TranslationDataset('./cmn-eng-simple/training.txt', word2int_en, word2int_cn),
    batch_size=64, shuffle=True, collate_fn=collate_fn)

model = TransformerMT(SRC_VOCAB_SIZE, TRG_VOCAB_SIZE).to(DEVICE)
criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = optim.Adam(model.parameters(), lr=0.0005)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.7)

for epoch in range(10):
    model.train()
    total_loss = 0
    for src, trg in train_loader:
        src, trg = src.to(DEVICE), trg.to(DEVICE)
        optimizer.zero_grad()
        output = model(src, trg[:, :-1])
        loss = criterion(output.reshape(-1, TRG_VOCAB_SIZE), trg[:, 1:].reshape(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    scheduler.step()
    print(f'Epoch {epoch+1}, Loss: {total_loss / len(train_loader):.4f}, LR: {scheduler.get_last_lr()[0]}')

torch.save(model.state_dict(), "transformer_mt.pth")
