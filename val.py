
import torch
from torch.utils.data import DataLoader
from model import TransformerMT
from dataloader import TranslationDataset, collate_fn
import json
import torch.nn as nn

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('./cmn-eng-simple/word2int_en.json') as f:
    word2int_en = json.load(f)
with open('./cmn-eng-simple/word2int_cn.json') as f:
    word2int_cn = json.load(f)

SRC_VOCAB_SIZE = len(word2int_en)
TRG_VOCAB_SIZE = len(word2int_cn)

val_loader = DataLoader(
    TranslationDataset('./cmn-eng-simple/validation.txt', word2int_en, word2int_cn),
    batch_size=64, shuffle=False, collate_fn=collate_fn)

model = TransformerMT(SRC_VOCAB_SIZE, TRG_VOCAB_SIZE).to(DEVICE)
model.load_state_dict(torch.load("transformer_mt.pth"))
model.eval()

criterion = nn.CrossEntropyLoss(ignore_index=0)

total_loss = 0
with torch.no_grad():
    for src, trg in val_loader:
        src, trg = src.to(DEVICE), trg.to(DEVICE)
        output = model(src, trg[:, :-1])
        loss = criterion(output.reshape(-1, TRG_VOCAB_SIZE), trg[:, 1:].reshape(-1))
        total_loss += loss.item()

print(f'Validation Loss: {total_loss / len(val_loader):.4f}')
