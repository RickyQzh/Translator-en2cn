
import torch
from model import TransformerMT
import json

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('./cmn-eng-simple/word2int_en.json') as f:
    word2int_en = json.load(f)
with open('./cmn-eng-simple/int2word_cn.json') as f:
    int2word_cn = json.load(f)

SRC_VOCAB_SIZE = len(word2int_en)
TRG_VOCAB_SIZE = len(int2word_cn)

model = TransformerMT(SRC_VOCAB_SIZE, TRG_VOCAB_SIZE).to(DEVICE)
model.load_state_dict(torch.load("transformer_mt.pth"))
model.eval()

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

    return ''.join([int2word_cn[str(idx)] for idx in sequences[0][0][1:-1]])

sentence = "tom is a student ."
print("翻译结果：", beam_search(sentence))
