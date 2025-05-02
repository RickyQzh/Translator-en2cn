
import torch
from torch.utils.data import Dataset, DataLoader
import json

class TranslationDataset(Dataset):
    def __init__(self, path, word2int_en, word2int_cn, max_len=50):
        self.data = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                en, cn = line.strip().split('\t')
                src = [1] + [word2int_en.get(w, 3) for w in en.split()] + [2]
                trg = [1] + [word2int_cn.get(w, 3) for w in cn.split()] + [2]
                self.data.append((src[:max_len], trg[:max_len]))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def collate_fn(batch):
    src, trg = zip(*batch)
    src_lens, trg_lens = [len(s) for s in src], [len(t) for t in trg]
    src_padded = torch.zeros(len(src), max(src_lens)).long()
    trg_padded = torch.zeros(len(trg), max(trg_lens)).long()
    for i in range(len(src)):
        src_padded[i, :src_lens[i]] = torch.tensor(src[i])
        trg_padded[i, :trg_lens[i]] = torch.tensor(trg[i])
    return src_padded, trg_padded
