import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import _LRScheduler
from model import TransformerMT
from dataloader import TranslationDataset, collate_fn
from torch.utils.data import DataLoader
import json
import os
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# 自定义学习率调度器，实现Warmup + 余弦退火
class WarmupCosineScheduler(_LRScheduler):
    def __init__(self, optimizer, warmup_steps, max_steps, min_lr=1e-6, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.min_lr = min_lr
        super(WarmupCosineScheduler, self).__init__(optimizer, last_epoch)
        
    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            # 线性warmup
            warmup_factor = float(self.last_epoch) / float(max(1.0, self.warmup_steps))
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        else:
            # 余弦退火
            progress = float(self.last_epoch - self.warmup_steps) / float(max(1, self.max_steps - self.warmup_steps))
            cosine_factor = 0.5 * (1.0 + np.cos(np.pi * progress))
            return [self.min_lr + (base_lr - self.min_lr) * cosine_factor for base_lr in self.base_lrs]

def load_vocab(vocab_path):
    with open(vocab_path, 'r', encoding='utf-8') as f:
        return json.load(f)

# 标签平滑损失函数
class LabelSmoothingLoss(nn.Module):
    def __init__(self, smoothing=0.1, ignore_index=0):
        super(LabelSmoothingLoss, self).__init__()
        self.smoothing = smoothing
        self.ignore_index = ignore_index
        self.confidence = 1.0 - smoothing
        
    def forward(self, pred, target):
        # pred: (batch_size * seq_len, vocab_size)
        # target: (batch_size * seq_len)
        
        # 创建有效位置的掩码 (排除pad位置)
        mask = (target != self.ignore_index).float()
        
        # 计算实际词汇表大小 (去除ignore_index)
        vocab_size = pred.size(-1)
        
        # 将预测转换为对数概率
        log_probs = F.log_softmax(pred, dim=-1)
        
        # 在正确的类别上分配confidence的概率，其余类别平均分配smoothing概率
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (vocab_size - 1))
            true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        
        # 计算KL散度损失，并只计算非pad位置的损失
        loss = -torch.sum(true_dist * log_probs, dim=-1) * mask
        return loss.sum() / mask.sum()

def train_epoch(model, train_loader, optimizer, criterion, device, int2word_cn=None, calculate_bleu=False):
    model.train()
    total_loss = 0
    bleu_scores = []
    smooth = SmoothingFunction().method5  # Better smoothing method
    weights = (0.25, 0.25, 0.25, 0.25)  # Equal weights for 1-gram to 4-gram
    
    for src, trg in train_loader:
        src, trg = src.to(device), trg.to(device)
        
        optimizer.zero_grad()
        output = model(src, trg[:, :-1])
        
        output_flat = output.contiguous().view(-1, output.shape[-1])
        trg_flat = trg[:, 1:].contiguous().view(-1)
        
        # 使用标准loss或KLDivLoss（取决于criterion类型）
        loss = criterion(output_flat, trg_flat)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        
        total_loss += loss.item()
        
        # Calculate BLEU if needed
        if calculate_bleu and int2word_cn:
            # Get predicted translations
            _, predicted = torch.max(output, dim=-1)
            
            for i in range(trg.size(0)):
                # Convert to words
                pred_sentence = []
                for idx in predicted[i]:
                    if idx.item() == 2:  # EOS token
                        break
                    if idx.item() not in [0, 1, 2, 3]:  # Skip PAD, BOS, EOS, UNK
                        pred_sentence.append(int2word_cn.get(str(idx.item()), "UNK"))
                
                # Get reference translation
                ref_sentence = []
                for idx in trg[i, 1:]:  # Skip BOS token
                    if idx.item() == 2:  # EOS token
                        break
                    if idx.item() not in [0, 1, 2, 3]:  # Skip PAD, BOS, EOS, UNK
                        ref_sentence.append(int2word_cn.get(str(idx.item()), "UNK"))
                
                # Calculate BLEU score with equal weights for 1-gram to 4-gram
                if pred_sentence and ref_sentence:
                    bleu = sentence_bleu([ref_sentence], pred_sentence, 
                                        weights=weights,
                                        smoothing_function=smooth)
                    bleu_scores.append(bleu)
    
    avg_loss = total_loss / len(train_loader)
    avg_bleu = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0
    
    return avg_loss, avg_bleu

def evaluate(model, data_loader, criterion, device, int2word_cn=None, calculate_bleu=False):
    model.eval()
    total_loss = 0
    bleu_scores = []
    smooth = SmoothingFunction().method5  # Better smoothing method
    weights = (0.25, 0.25, 0.25, 0.25)  # Equal weights for 1-gram to 4-gram
    
    with torch.no_grad():
        for src, trg in data_loader:
            src, trg = src.to(device), trg.to(device)
            
            output = model(src, trg[:, :-1])
            
            # Calculate loss
            output_flat = output.contiguous().view(-1, output.shape[-1])
            trg_flat = trg[:, 1:].contiguous().view(-1)
            loss = criterion(output_flat, trg_flat)
            total_loss += loss.item()
            
            # Calculate BLEU if needed
            if calculate_bleu and int2word_cn:
                # Get predicted translations
                _, predicted = torch.max(output, dim=-1)
                
                for i in range(trg.size(0)):
                    # Convert to words
                    pred_sentence = []
                    for idx in predicted[i]:
                        if idx.item() == 2:  # EOS token
                            break
                        if idx.item() not in [0, 1, 2, 3]:  # Skip PAD, BOS, EOS, UNK
                            pred_sentence.append(int2word_cn.get(str(idx.item()), "UNK"))
                    
                    # Get reference translation
                    ref_sentence = []
                    for idx in trg[i, 1:]:  # Skip BOS token
                        if idx.item() == 2:  # EOS token
                            break
                        if idx.item() not in [0, 1, 2, 3]:  # Skip PAD, BOS, EOS, UNK
                            ref_sentence.append(int2word_cn.get(str(idx.item()), "UNK"))
                    
                    # Calculate BLEU score with equal weights for 1-gram to 4-gram
                    if pred_sentence and ref_sentence:
                        bleu = sentence_bleu([ref_sentence], pred_sentence, 
                                            weights=weights,
                                            smoothing_function=smooth)
                        bleu_scores.append(bleu)
    
    avg_loss = total_loss / len(data_loader)
    avg_bleu = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0
    
    return avg_loss, avg_bleu

def plot_metrics(metrics_df, save_path='training_metrics.png'):
    # Convert epoch to integer for better display
    metrics_df['epoch'] = metrics_df['epoch'].astype(int)
    
    # Create a figure with 2 side-by-side subplots (square shape)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Plot loss
    line1, = ax1.plot(metrics_df['epoch'], metrics_df['train_loss'], 'bo-', label='Training Loss')
    line2, = ax1.plot(metrics_df['epoch'], metrics_df['val_loss'], 'ro-', label='Validation Loss')
    line3, = ax1.plot(metrics_df['epoch'], metrics_df['test_loss'], 'go-', label='Test Loss')
    
    # Set integer ticks for x-axis
    ax1.set_xticks(metrics_df['epoch'])
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Loss vs. Epoch', fontsize=14)
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Plot BLEU
    line4, = ax2.plot(metrics_df['epoch'], metrics_df['train_bleu'], 'bo-', label='Training BLEU')
    line5, = ax2.plot(metrics_df['epoch'], metrics_df['val_bleu'], 'ro-', label='Validation BLEU')
    line6, = ax2.plot(metrics_df['epoch'], metrics_df['test_bleu'], 'go-', label='Test BLEU')
    
    # Set integer ticks for x-axis
    ax2.set_xticks(metrics_df['epoch'])
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('BLEU Score', fontsize=12)
    ax2.set_title('BLEU Score vs. Epoch', fontsize=14)
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    # Ensure the aspect ratio is more square-like
    ax1.set_box_aspect(1)
    ax2.set_box_aspect(1)
    
    plt.tight_layout(pad=3.0)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def calculate_ngram_bleus(model, data_loader, device, int2word_cn):
    """
    计算1-gram到4-gram的BLEU分数
    """
    model.eval()
    bleu_scores_by_ngram = [[] for _ in range(4)]  # 存储1-gram到4-gram的BLEU分数
    smooth = SmoothingFunction().method5
    
    with torch.no_grad():
        for src, trg in data_loader:
            src, trg = src.to(device), trg.to(device)
            
            output = model(src, trg[:, :-1])
            
            # Get predicted translations
            _, predicted = torch.max(output, dim=-1)
            
            for i in range(trg.size(0)):
                # Convert to words
                pred_sentence = []
                for idx in predicted[i]:
                    if idx.item() == 2:  # EOS token
                        break
                    if idx.item() not in [0, 1, 2, 3]:  # Skip PAD, BOS, EOS, UNK
                        pred_sentence.append(int2word_cn.get(str(idx.item()), "UNK"))
                
                # Get reference translation
                ref_sentence = []
                for idx in trg[i, 1:]:  # Skip BOS token
                    if idx.item() == 2:  # EOS token
                        break
                    if idx.item() not in [0, 1, 2, 3]:  # Skip PAD, BOS, EOS, UNK
                        ref_sentence.append(int2word_cn.get(str(idx.item()), "UNK"))
                
                # Calculate BLEU scores for each n-gram order
                if pred_sentence and ref_sentence:
                    for n in range(1, 5):  # 1-gram to 4-gram
                        # Create weights for current n-gram order
                        weights = tuple([1.0/n] * n)
                        bleu = sentence_bleu([ref_sentence], pred_sentence, 
                                            weights=weights,
                                            smoothing_function=smooth)
                        bleu_scores_by_ngram[n-1].append(bleu)
    
    # Calculate average BLEU for each n-gram order
    avg_bleus = []
    for n in range(4):
        if bleu_scores_by_ngram[n]:
            avg_bleus.append(sum(bleu_scores_by_ngram[n]) / len(bleu_scores_by_ngram[n]))
        else:
            avg_bleus.append(0.0)
    
    return avg_bleus

def calculate_warmup_steps(total_steps, warmup_ratio=0.1):
    """计算合适的warmup步数，默认为总步数的10%"""
    return max(1, int(total_steps * warmup_ratio))

def train_model(config, output_dir=None):
    """
    使用给定配置训练模型，可选择指定输出目录
    """
    # 解包配置
    BATCH_SIZE = config.get('batch_size', 64)
    EPOCHS = config.get('epochs', 50)
    LEARNING_RATE = config.get('learning_rate', 1e-3)
    D_MODEL = config.get('d_model', 256)
    NUM_HEADS = config.get('num_heads', 8)
    NUM_ENCODER_LAYERS = config.get('num_encoder_layers', 6)
    NUM_DECODER_LAYERS = config.get('num_decoder_layers', 6)
    D_FF = config.get('d_ff', 1024)
    DROPOUT = config.get('dropout', 0.2)
    LABEL_SMOOTHING = config.get('label_smoothing', 0.1)
    PATIENCE = config.get('patience', 5)
    
    # 如果没有指定输出目录，使用默认目录
    if output_dir is None:
        output_dir = './model_output'
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 为config文件创建JSON
    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load vocabulary
    word2int_en = load_vocab('./cmn-eng-simple/word2int_en.json')
    word2int_cn = load_vocab('./cmn-eng-simple/word2int_cn.json')
    int2word_cn = load_vocab('./cmn-eng-simple/int2word_cn.json')
    
    print(f"English vocabulary size: {len(word2int_en)}")
    print(f"Chinese vocabulary size: {len(word2int_cn)}")
    
    # Create datasets
    train_dataset = TranslationDataset('./cmn-eng-simple/training.txt', word2int_en, word2int_cn)
    val_dataset = TranslationDataset('./cmn-eng-simple/validation.txt', word2int_en, word2int_cn)
    test_dataset = TranslationDataset('./cmn-eng-simple/testing.txt', word2int_en, word2int_cn)
    
    print(f"Training set size: {len(train_dataset)}")
    print(f"Validation set size: {len(val_dataset)}")
    print(f"Test set size: {len(test_dataset)}")
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    
    # Initialize model
    model = TransformerMT(
        src_vocab_size=len(word2int_en),
        trg_vocab_size=len(word2int_cn),
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        num_encoder_layers=NUM_ENCODER_LAYERS,
        num_decoder_layers=NUM_DECODER_LAYERS,
        d_ff=D_FF,
        dropout=DROPOUT
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    
    # Initialize optimizer
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.98), eps=1e-9)
    
    # 计算总步数
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * EPOCHS
    
    # 如果config中有warmup_steps，使用指定值，否则自动计算
    if 'warmup_steps' in config:
        warmup_steps = config['warmup_steps']
    else:
        # 默认使用总步数的10%作为warmup步数，约为1-2个epoch
        warmup_steps = calculate_warmup_steps(total_steps, warmup_ratio=0.1)
    
    print(f"Steps per epoch: {steps_per_epoch}")
    print(f"Total steps: {total_steps}")
    print(f"Warmup steps: {warmup_steps} (约 {warmup_steps/steps_per_epoch:.1f} 个epoch)")
    
    # 使用Warmup + 余弦退火学习率调度器
    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_steps=warmup_steps,
        max_steps=total_steps,
        min_lr=1e-6
    )
    
    # 使用标签平滑损失函数
    criterion = LabelSmoothingLoss(smoothing=LABEL_SMOOTHING, ignore_index=0)
    
    # Training loop
    best_val_loss = float('inf')
    best_bleu = 0.0
    early_stop_counter = 0
    
    # Create dataframe to store metrics
    metrics = {
        'epoch': [], 'step': [],
        'train_loss': [], 'train_bleu': [],
        'val_loss': [], 'val_bleu': [],
        'test_loss': [], 'test_bleu': [],
        'learning_rate': []
    }
    
    # Create metrics table
    print("\nTraining metrics:")
    print("-" * 90)
    print(f"{'Epoch':^6} | {'Step':^8} | {'Train Loss':^10} | {'Train BLEU':^10} | {'Val Loss':^10} | {'Val BLEU':^10} | {'Test Loss':^10} | {'Test BLEU':^10} | {'LR':^10}")
    print("-" * 90)
    
    global_step = 0
    for epoch in range(EPOCHS):
        epoch_train_loss = 0
        epoch_train_bleu = 0
        batch_count = 0
        
        # 获取当前学习率
        current_lr = optimizer.param_groups[0]['lr']
        
        # 训练一个epoch
        for src, trg in train_loader:
            model.train()
            src, trg = src.to(device), trg.to(device)
            
            optimizer.zero_grad()
            output = model(src, trg[:, :-1])
            
            output_flat = output.contiguous().view(-1, output.shape[-1])
            trg_flat = trg[:, 1:].contiguous().view(-1)
            
            loss = criterion(output_flat, trg_flat)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            scheduler.step()  # 更新学习率
            
            global_step += 1
            batch_count += 1
            epoch_train_loss += loss.item()
            
            # 每100步评估一次
            if global_step % 100 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                # print(f"Step {global_step}, LR: {current_lr:.6f}, Loss: {loss.item():.4f}")
        
        # 计算epoch平均损失
        epoch_train_loss /= batch_count
        
        # Validation with BLEU
        val_loss, val_bleu = evaluate(model, val_loader, criterion, device, int2word_cn, calculate_bleu=True)
        
        # Testing with BLEU
        test_loss, test_bleu = evaluate(model, test_loader, criterion, device, int2word_cn, calculate_bleu=True)
        
        # 计算单独的训练集BLEU分数（只在epoch结束时计算，节省时间）
        _, train_bleu = evaluate(model, train_loader, criterion, device, int2word_cn, calculate_bleu=True)
        
        # Save metrics
        metrics['epoch'].append(epoch + 1)
        metrics['step'].append(global_step)
        metrics['train_loss'].append(epoch_train_loss)
        metrics['train_bleu'].append(train_bleu)
        metrics['val_loss'].append(val_loss)
        metrics['val_bleu'].append(val_bleu)
        metrics['test_loss'].append(test_loss)
        metrics['test_bleu'].append(test_bleu)
        metrics['learning_rate'].append(current_lr)
        
        # Print metrics table row
        print(f"{epoch+1:^6} | {global_step:^8} | {epoch_train_loss:^10.4f} | {train_bleu:^10.4f} | {val_loss:^10.4f} | {val_bleu:^10.4f} | {test_loss:^10.4f} | {test_bleu:^10.4f} | {current_lr:^10.6f}")
        
        # 早停判断：验证集损失
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'global_step': global_step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_bleu': val_bleu,
            }, os.path.join(output_dir, 'best_loss_model.pth'))
            # print('Best loss model saved!')
            early_stop_counter = 0  # 重置早停计数器
        else:
            early_stop_counter += 1
            
        # 早停判断：BLEU分数
        if val_bleu > best_bleu:
            best_bleu = val_bleu
            torch.save({
                'epoch': epoch,
                'global_step': global_step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_bleu': val_bleu,
            }, os.path.join(output_dir, 'best_bleu_model.pth'))
            # print('Best BLEU model saved!')
        
        # 检查是否需要早停
        if early_stop_counter >= PATIENCE:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
        
        # Plot and save metrics every 5 epochs
        if (epoch + 1) % 5 == 0 or (epoch + 1) == EPOCHS or early_stop_counter >= PATIENCE:
            metrics_df = pd.DataFrame(metrics)
            plot_metrics(metrics_df, save_path=os.path.join(output_dir, 'training_metrics.png'))
            metrics_df.to_csv(os.path.join(output_dir, 'training_metrics.csv'), index=False)
    
    # Final evaluation on test set with the best BLEU model
    print("\nFinal evaluation on test set...")
    checkpoint = torch.load(os.path.join(output_dir, 'best_bleu_model.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])
    test_loss, test_bleu = evaluate(model, test_loader, criterion, device, int2word_cn, calculate_bleu=True)
    print(f'Final Test Loss: {test_loss:.4f}, BLEU: {test_bleu:.4f}')
    
    # Calculate and display n-gram BLEU scores
    print("\nDetailed n-gram BLEU scores on test set:")
    ngram_bleus = calculate_ngram_bleus(model, test_loader, device, int2word_cn)
    for n, bleu_score in enumerate(ngram_bleus, 1):
        print(f'BLEU-{n} (n-gram={n}): {bleu_score:.4f}')
    print(f'BLEU-4 (weighted): {test_bleu:.4f}')
    
    # 额外绘制学习率变化曲线
    plt.figure(figsize=(10, 6))
    plt.plot(metrics['step'], metrics['learning_rate'], 'b-')
    plt.xlabel('Training Step')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedule')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'learning_rate.png'))
    plt.close()
    
    # Save final metrics and plot
    metrics_df = pd.DataFrame(metrics)
    plot_metrics(metrics_df, save_path=os.path.join(output_dir, 'training_metrics.png'))
    metrics_df.to_csv(os.path.join(output_dir, 'training_metrics.csv'), index=False)
    
    return {
        'best_val_loss': best_val_loss,
        'best_val_bleu': best_bleu,
        'final_test_loss': test_loss,
        'final_test_bleu': test_bleu,
        'final_test_bleu_1': ngram_bleus[0],
        'final_test_bleu_2': ngram_bleus[1], 
        'final_test_bleu_3': ngram_bleus[2],
        'final_test_bleu_4': ngram_bleus[3]
    }

def main():
    # 默认配置
    config = {
        'batch_size': 64,
        'epochs': 30,
        'learning_rate': 1e-3,  # 初始学习率
        'd_model': 256,
        'num_heads': 8,
        'num_encoder_layers': 6,
        'num_decoder_layers': 6,
        'd_ff': 1024,
        'dropout': 0.2,
        'label_smoothing': 0.1,
        'patience': 5,
        # 不再硬编码warmup_steps，而是由函数自动计算
    }
    
    # 训练模型
    results = train_model(config)
    print("Training completed!")
    print(f"Best validation loss: {results['best_val_loss']:.4f}")
    print(f"Best validation BLEU: {results['best_val_bleu']:.4f}")
    print(f"Final test loss: {results['final_test_loss']:.4f}")
    print(f"Final test BLEU: {results['final_test_bleu']:.4f}")
    
    # 输出详细的n-gram BLEU分数
    print("\nSummary of n-gram BLEU scores:")
    print(f"BLEU-1: {results['final_test_bleu_1']:.4f}")
    print(f"BLEU-2: {results['final_test_bleu_2']:.4f}")
    print(f"BLEU-3: {results['final_test_bleu_3']:.4f}")
    print(f"BLEU-4: {results['final_test_bleu_4']:.4f}")
    print(f"BLEU-4 (weighted): {results['final_test_bleu']:.4f}")

if __name__ == '__main__':
    main()
