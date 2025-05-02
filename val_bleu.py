
import torch
from model import TransformerMT
from dataloader import TranslationDataset, collate_fn
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import json
from torch.utils.data import DataLoader

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('./cmn-eng-simple/word2int_en.json') as f:
    word2int_en = json.load(f)
with open('./cmn-eng-simple/int2word_cn.json') as f:
    int2word_cn = json.load(f)
with open('./cmn-eng-simple/word2int_cn.json') as f:
    word2int_cn = json.load(f)

SRC_VOCAB_SIZE = len(word2int_en)
TRG_VOCAB_SIZE = len(word2int_cn)

PAD_IDX = 0

# 加载模型
model = TransformerMT(SRC_VOCAB_SIZE, TRG_VOCAB_SIZE, pad_idx=PAD_IDX).to(DEVICE)
model.load_state_dict(torch.load("transformer_mt.pth")) 
model.eval()

# Beam search
def beam_search(src_sentence, beam_width=3, max_len=50):
    src_tensor = torch.tensor([[1] + [word2int_en.get(w, 3) for w in src_sentence.split()] + [2]], device=DEVICE)
    memory = model.transformer.encoder(model.src_emb(src_tensor).transpose(0, 1))
    sequences = [([1], 0.0)]
    for _ in range(max_len):
        all_candidates = []
        for seq, score in sequences:
            trg_tensor = torch.tensor([seq], device=DEVICE)
            trg_emb = model.trg_emb(trg_tensor).transpose(0, 1)
            output = model.transformer.decoder(trg_emb, memory)
            output = model.fc_out(output.transpose(0, 1))
            log_probs = torch.log_softmax(output[0, -1], dim=-1)
            top_probs, top_ids = torch.topk(log_probs, beam_width)
            for prob, idx in zip(top_probs, top_ids):
                candidate = (seq + [idx.item()], score + prob.item())
                all_candidates.append(candidate)
        sequences = sorted(all_candidates, key=lambda x: x[1], reverse=True)[:beam_width]
        if sequences[0][0][-1] == 2:
            break
    return [int2word_cn[str(i)] for i in sequences[0][0][1:-1] if str(i) in int2word_cn]

# BLEU评估
from nltk.translate.bleu_score import SmoothingFunction

def compute_bleu(candidate_tokens, reference_tokens):
    smoothie = SmoothingFunction().method4
    return sentence_bleu([reference_tokens], candidate_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothie)

# 验证数据评估BLEU
val_data = TranslationDataset('./cmn-eng-simple/validation.txt', word2int_en, word2int_cn)
total_bleu = 0

for en, cn in val_data:
    en_sentence = ' '.join([k for k, v in word2int_en.items() if v in en and v > 3])
    cn_sentence = [k for k, v in word2int_cn.items() if v in cn and v > 3]
    pred_tokens = beam_search(en_sentence)
    score = compute_bleu(pred_tokens, cn_sentence)
    total_bleu += score

print(f'Validation BLEU Score: {total_bleu / len(val_data):.4f}')
