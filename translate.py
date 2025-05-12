import torch
import json
import argparse
from model import TransformerMT
import heapq
import numpy as np

def load_vocab(vocab_path):
    with open(vocab_path, 'r', encoding='utf-8') as f:
        return json.load(f)

class BeamHypothesis:
    def __init__(self, tokens, score):
        self.tokens = tokens  # List of token indices
        self.score = score    # Log probability score
    
    def __lt__(self, other):
        # For heap operations, lower score is better
        # But we want higher scores, so negate
        return -self.score < -other.score
    
    def extend(self, token, log_prob):
        """Create a new hypothesis by extending with a token and score"""
        return BeamHypothesis(self.tokens + [token], self.score + log_prob)

def translate_sentence_greedy(model, sentence, word2int_en, int2word_cn, device, max_len=50):
    """Translate a sentence using greedy decoding"""
    # Tokenize and convert to indices
    tokens = sentence.lower().split()
    indices = [1]  # BOS token
    for token in tokens:
        indices.append(word2int_en.get(token, 3))  # 3 is UNK token
    indices.append(2)  # EOS token
    
    # Convert to tensor
    src_tensor = torch.LongTensor(indices).unsqueeze(0).to(device)
    
    # Initialize target with BOS token
    trg_indices = [1]
    
    # Set model to evaluation mode
    model.eval()
    
    with torch.no_grad():
        # Encode the source sentence
        enc_output, src_mask = model.encode(src_tensor)
        
        for _ in range(max_len):
            # Convert current target indices to tensor
            trg_tensor = torch.LongTensor(trg_indices).unsqueeze(0).to(device)
            
            # Get next token prediction
            output = model.decode(trg_tensor, enc_output, src_mask)
            pred_token = output[:, -1, :].argmax(dim=1).item()
            
            # Add prediction to sequence
            trg_indices.append(pred_token)
            
            # Stop if EOS token
            if pred_token == 2:
                break
    
    # Convert indices to words
    translated_tokens = []
    for idx in trg_indices[1:-1]:  # Skip BOS and EOS
        translated_tokens.append(int2word_cn.get(str(idx), "UNK"))
    
    return " ".join(translated_tokens)

def translate_sentence_beam_search(model, sentence, word2int_en, int2word_cn, device, beam_size=5, max_len=50, alpha=0.7):
    """Translate a sentence using beam search decoding"""
    # Tokenize and convert to indices
    tokens = sentence.lower().split()
    indices = [1]  # BOS token
    for token in tokens:
        indices.append(word2int_en.get(token, 3))  # 3 is UNK token
    indices.append(2)  # EOS token
    
    # Convert to tensor
    src_tensor = torch.LongTensor(indices).unsqueeze(0).to(device)
    
    # Set model to evaluation mode
    model.eval()
    
    with torch.no_grad():
        # Encode the source sentence
        enc_output, src_mask = model.encode(src_tensor)
        
        # Initialize with BOS token
        initial_trg_tensor = torch.LongTensor([[1]]).to(device)  # BOS token
        
        # Get first set of predictions
        initial_output = model.decode(initial_trg_tensor, enc_output, src_mask)
        initial_probs = torch.log_softmax(initial_output[:, -1, :], dim=-1)
        
        # Get top beam_size predictions
        topk_probs, topk_indices = torch.topk(initial_probs, beam_size, dim=-1)
        
        # Initialize beam hypotheses
        beams = []
        for i in range(beam_size):
            token = topk_indices[0, i].item()
            log_prob = topk_probs[0, i].item()
            beams.append(BeamHypothesis([1, token], log_prob))  # Start with BOS token
        
        # Store completed hypotheses
        completed_hypotheses = []
        
        # Beam search loop
        for _ in range(2, max_len):  # Start from 2 because we already added 1 token
            # If all beams are complete, stop
            if len(beams) == 0:
                break
                
            # Extend each beam
            all_candidates = []
            
            # For each existing hypothesis
            for hyp in beams:
                # If last token is EOS, add to completed
                if hyp.tokens[-1] == 2:  # EOS token
                    completed_hypotheses.append(hyp)
                    continue
                    
                # Otherwise, get next token predictions
                trg_tensor = torch.LongTensor([hyp.tokens]).to(device)
                output = model.decode(trg_tensor, enc_output, src_mask)
                probs = torch.log_softmax(output[:, -1, :], dim=-1)
                
                # Get top candidates for this beam
                topk_probs, topk_indices = torch.topk(probs, beam_size, dim=-1)
                
                # Add candidates to all_candidates
                for i in range(beam_size):
                    token = topk_indices[0, i].item()
                    log_prob = topk_probs[0, i].item()
                    all_candidates.append(hyp.extend(token, log_prob))
            
            # No candidates, all beams complete
            if len(all_candidates) == 0:
                break
                
            # Sort candidates and keep top beam_size
            beams = heapq.nsmallest(beam_size, all_candidates)
        
        # Add any remaining beams to completed
        for hyp in beams:
            if hyp.tokens[-1] != 2:  # Add EOS if not present
                hyp = hyp.extend(2, 0.0)
            completed_hypotheses.append(hyp)
        
        # If no complete hypotheses, use the best incomplete one
        if len(completed_hypotheses) == 0:
            completed_hypotheses = beams
            
        # Length normalization: (score) / (length)^alpha
        for hyp in completed_hypotheses:
            hyp.score = hyp.score / (len(hyp.tokens) ** alpha)
            
        # Get the best hypothesis
        best_hyp = heapq.nsmallest(1, completed_hypotheses)[0]
        
        # Convert indices to words (skip BOS and EOS)
        translated_tokens = []
        for idx in best_hyp.tokens[1:]:
            if idx == 2:  # EOS token
                break
            translated_tokens.append(int2word_cn.get(str(idx), "UNK"))
    
    return " ".join(translated_tokens)

def translate_sentence(model, sentence, word2int_en, int2word_cn, device, beam_size=5, max_len=50):
    """Translate using beam search if beam_size > 1, otherwise greedy search"""
    if beam_size <= 1:
        return translate_sentence_greedy(model, sentence, word2int_en, int2word_cn, device, max_len)
    else:
        return translate_sentence_beam_search(model, sentence, word2int_en, int2word_cn, device, beam_size, max_len)

def main():
    parser = argparse.ArgumentParser(description='Translate English to Chinese')
    parser.add_argument('--model', type=str, default='best_bleu_model.pth', help='Path to the model')
    parser.add_argument('--sentence', type=str, required=True, help='English sentence to translate')
    parser.add_argument('--beam_size', type=int, default=5, help='Beam size for beam search (1 for greedy search)')
    parser.add_argument('--max_len', type=int, default=50, help='Maximum length of translation')
    args = parser.parse_args()
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load vocabulary
    print("Loading vocabulary...")
    word2int_en = load_vocab('./cmn-eng-simple/word2int_en.json')
    word2int_cn = load_vocab('./cmn-eng-simple/word2int_cn.json')
    int2word_cn = load_vocab('./cmn-eng-simple/int2word_cn.json')
    
    # Create the model
    print("Creating model...")
    model = TransformerMT(
        src_vocab_size=len(word2int_en),
        trg_vocab_size=len(word2int_cn),
        d_model=256,
        num_heads=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        d_ff=1024,
        dropout=0.1
    ).to(device)
    
    # Load model weights
    print(f"Loading model weights from {args.model}...")
    checkpoint = torch.load(args.model, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Translate
    print("\nTranslating...")
    search_method = "beam search" if args.beam_size > 1 else "greedy search"
    print(f"Using {search_method} with beam size {args.beam_size}")
    
    translation = translate_sentence(
        model, args.sentence, word2int_en, int2word_cn, device, 
        beam_size=args.beam_size, max_len=args.max_len
    )
    
    print(f"\nEnglish: {args.sentence}")
    print(f"Chinese: {translation}")

if __name__ == '__main__':
    main() 