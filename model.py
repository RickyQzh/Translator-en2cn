
import torch
import torch.nn as nn

class TransformerMT(nn.Module):
    def __init__(self, src_vocab_size, trg_vocab_size, emb_size=256, nhead=8,
                 num_encoder_layers=3, num_decoder_layers=3, dim_feedforward=512, pad_idx=0):
        super().__init__()
        self.src_emb = nn.Embedding(src_vocab_size, emb_size, padding_idx=pad_idx)
        self.trg_emb = nn.Embedding(trg_vocab_size, emb_size, padding_idx=pad_idx)
        self.transformer = nn.Transformer(
            d_model=emb_size, nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward
        )
        self.fc_out = nn.Linear(emb_size, trg_vocab_size)
        self.pad_idx = pad_idx

    def make_pad_mask(self, seq):
        # (batch_size, seq_len) => (batch_size, seq_len)
        return (seq == self.pad_idx)

    def make_trg_subsequent_mask(self, size):
        # Mask out subsequent positions (1s in upper triangle)
        return torch.triu(torch.ones((size, size)) * float('-inf'), diagonal=1)

    def forward(self, src, trg):
        src_pad_mask = self.make_pad_mask(src)  # (batch, src_len)
        trg_pad_mask = self.make_pad_mask(trg)  # (batch, trg_len)
        trg_seq_mask = self.make_trg_subsequent_mask(trg.size(1)).to(trg.device)  # (trg_len, trg_len)

        src_emb = self.src_emb(src).transpose(0, 1)  # (src_len, batch, d_model)
        trg_emb = self.trg_emb(trg).transpose(0, 1)  # (trg_len, batch, d_model)

        output = self.transformer(
            src_emb, trg_emb,
            tgt_mask=trg_seq_mask,
            src_key_padding_mask=src_pad_mask,
            tgt_key_padding_mask=trg_pad_mask
        )
        return self.fc_out(output.transpose(0, 1))  # (batch, trg_len, vocab_size)
