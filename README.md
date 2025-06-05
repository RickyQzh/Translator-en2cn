# English-to-Chinese Neural Machine Translation with Comprehensive Ablation Studies

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1.2-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive Transformer-based English-to-Chinese translation system with extensive ablation study capabilities for systematic analysis of model components.

## 🚀 Quick Start

### Direct Experiment Execution
```bash
# Run different experiment types directly with Python:
python3 ablation_study.py --max_concurrent 4                    # Basic experiments
python3 ablation_study.py --advanced --max_concurrent 4         # Advanced experiments  
python3 ablation_study.py --warmup --max_concurrent 4           # Warmup experiments
python3 ablation_study.py --modules --max_concurrent 4          # Module experiments
python3 ablation_study.py --all --max_concurrent 4              # All experiments

# View help and options
python3 ablation_study.py --help
```

### Translation
```bash
# Interactive translation
python translate.py --sentence "Hello, how are you?" --beam_size 5

# Greedy decoding
python translate.py --sentence "Good morning!" --beam_size 1
```

### Single Model Training
```bash
python train.py
```

## 📊 Model Architecture

### Core Transformer Components
- **Architecture**: Encoder-Decoder Transformer
- **Default Configuration**:
  - Model Dimension (`d_model`): 256
  - Attention Heads: 8
  - Encoder/Decoder Layers: 6 each
  - Feed-forward Dimension (`d_ff`): 1024
  - Dropout: 0.2

### Enhanced Features
- **Flexible Positional Encoding**: Sinusoidal, learned, relative, or none
- **Multiple Activation Functions**: ReLU, GELU, Swish
- **Configurable Layer Normalization**: Pre-norm, post-norm, or disabled
- **Multiple Optimizers**: Adam, Adagrad, RMSprop
- **Beam Search Translation**: Configurable beam size with length normalization

### Vocabulary
- **English Vocabulary**: ~3,900 tokens
- **Chinese Vocabulary**: ~3,800 tokens
- **Special Tokens**: PAD (0), BOS (1), EOS (2), UNK (3)

## 📚 Dataset

### CMN-ENG Simple Dataset
```
cmn-eng-simple/
├── training.txt        # 18,000 sentence pairs
├── validation.txt      # 500 sentence pairs  
├── testing.txt         # 2,636 sentence pairs
├── word2int_en.json    # English vocabulary mapping
├── word2int_cn.json    # Chinese vocabulary mapping
├── int2word_en.json    # Reverse English mapping
└── int2word_cn.json    # Reverse Chinese mapping
```

### Data Format
Each line contains tab-separated English and Chinese sentences:
```
it 's none of your concern .    这不关 你 的 事 。
she has a habit of biting her nails .   她 有 咬 指甲 的 习惯 。
he is a teacher .       他 是 老师 。
```

### Preprocessing Features
- **Subword Tokenization**: BPE-style tokenization with `@@` markers
- **Case Normalization**: Lowercase English text
- **Chinese Segmentation**: Character and word-level tokenization

## 🔬 Comprehensive Ablation Studies

Our system provides the most extensive ablation study framework for Transformer translation models, with **4 experiment categories** and **50+ individual experiments**.

### 📊 Basic Ablation Experiments (~25 experiments)
**Traditional hyperparameter optimization studies**

| Category | Variants | Purpose |
|----------|----------|---------|
| **Model Size** | Small (d_model=128), Medium (256), Large (512) | Capacity vs efficiency |
| **Network Depth** | 3, 6, 9 encoder/decoder layers | Depth vs performance |
| **Regularization** | Dropout: 0.1, 0.2, 0.3 | Overfitting prevention |
| **Learning Rate** | 2e-4, 1e-3, 5e-3 | Convergence optimization |
| **Label Smoothing** | 0.0, 0.1, 0.2 | Confidence calibration |
| **Attention Heads** | 4, 8, 16 heads | Multi-head attention analysis |

### 🔬 Advanced Ablation Experiments (~13 experiments) - **Core Innovation**
**Deep architectural component analysis**

#### 1. Optimizer Comparison
- **Adam** (baseline): Adaptive moment estimation, standard for Transformers
- **Adagrad**: Adaptive gradient algorithm, good for sparse data
- **RMSprop**: Root mean square propagation, improved adaptive method

#### 2. Activation Function Analysis
- **ReLU** (baseline): Rectified Linear Unit, standard non-linearity
- **GELU**: Gaussian Error Linear Unit, used in BERT and modern models
- **Swish**: x * sigmoid(x), Google's smooth activation function

#### 3. Hidden Dimension Systematic Study
- **d_model=128**: Lightweight model, fast training
- **d_model=256** (baseline): Balanced performance and efficiency
- **d_model=512**: High-capacity model, strong representation

#### 4. Positional Encoding Deep Analysis
- **Sinusoidal** (baseline): sin/cos fixed encoding, original Transformer
- **Learned**: Trainable positional parameters
- **Relative**: Relative position relationships, modern approach
- **None**: Complete removal of positional information

#### 5. Layer Normalization Strategy Research
- **Post-Norm** (baseline): Original Transformer, normalize after residual
- **Pre-Norm**: Normalize before sublayer, more stable training
- **No Normalization**: Remove all LayerNorm, study necessity

### ⚡ Warmup Experiments (~6 experiments)
**Learning rate warmup strategy analysis**
- Different warmup ratios: 2%, 5%, 10%, 15%, 20%
- Impact on convergence speed and final performance

### 🔧 Module Experiments (~7 experiments)
**Core Transformer component importance analysis**
- Positional encoding necessity
- Multi-head attention effectiveness
- Different attention head configurations

## 🎯 Experiment Framework

### Core Experiment Program (`ablation_study.py`)
**The main experiment engine**

All experiments are run directly through `ablation_study.py`:

```bash
# Basic ablation experiments (default)
python3 ablation_study.py --max_concurrent 4

# Advanced ablation experiments (optimizers, activations, etc.)
python3 ablation_study.py --advanced --max_concurrent 4

# Warmup ratio experiments
python3 ablation_study.py --warmup --max_concurrent 4

# Transformer module experiments
python3 ablation_study.py --modules --max_concurrent 4

# Run all experiment types sequentially
python3 ablation_study.py --all --max_concurrent 4
```

### Concurrency Control (GPU Memory Based)

| GPU Memory | Recommended Concurrency | Command |
|------------|-------------------------|---------|
| < 8GB | 1 process | `python3 ablation_study.py --advanced --max_concurrent 1` |
| 8-16GB | 2 processes | `python3 ablation_study.py --advanced --max_concurrent 2` |
| > 16GB | 3-4 processes | `python3 ablation_study.py --advanced --max_concurrent 4` |

### How It Works
The experiment system is centered around `ablation_study.py`:

**🔬 Core Python Program** (`ablation_study.py`):
- The main experiment engine that runs all ablation studies
- Handles parallel execution, experiment configuration, and result collection
- Supports different experiment types through command-line arguments
- Includes GPU detection and memory management

### Smart Features
- **Automatic GPU Detection**: Recommends optimal concurrency
- **Colored Output**: Clear status indicators and progress tracking
- **Dry-run Mode**: Preview commands before execution
- **Fault Tolerance**: Individual experiment failures don't affect others
- **Silent Mode**: Reduced output during ablation studies

## 🏃 Training Process

### Optimization Features
- **Early Stopping**: Monitors validation loss (patience: 5 epochs)
- **Label Smoothing**: Factor 0.1 with KL divergence loss
- **Learning Rate Scheduling**: Warmup + cosine decay
- **Mixed Precision**: Automatic if available
- **Gradient Clipping**: Prevents gradient explosion

### Training Configuration
```python
# Default training parameters
{
    "epochs": 30,
    "batch_size": 256,
    "learning_rate": 1e-3,
    "warmup_ratio": 0.1,
    "dropout": 0.2,
    "label_smoothing": 0.1,
    "early_stopping_patience": 5
}
```

### Loss Function
- **Primary**: Label-smoothed cross-entropy
- **Evaluation**: Standard cross-entropy + BLEU score
- **Metrics**: Real-time BLEU-1, BLEU-2, BLEU-3, BLEU-4 tracking

## 🌐 Translation Features

### Decoding Strategies
```bash
# Greedy decoding (fastest)
python translate.py --sentence "Hello world" --beam_size 1

# Beam search with different beam sizes
python translate.py --sentence "Hello world" --beam_size 3
python translate.py --sentence "Hello world" --beam_size 5
python translate.py --sentence "Hello world" --beam_size 10
```

### Advanced Translation Features
- **Length Normalization**: Adjustable alpha parameter
- **Coverage Penalty**: Prevents repetition
- **Early Stopping**: Beam search optimization
- **Batch Translation**: Support for multiple sentences

## 📈 Results and Evaluation

### Output Structure
```
./ablation/                                    # Experiment results root
├── experiment_20231201_143022/                # Timestamped experiment directory
│   ├── baseline/                              # Baseline experiment
│   │   ├── config.json                        # Experiment configuration
│   │   ├── training_metrics.csv               # Training progress metrics
│   │   ├── best_bleu_model.pth                # Best BLEU score model
│   │   ├── best_loss_model.pth                # Best loss model
│   │   └── training_metrics.png               # Training curves
│   ├── optimizer_adagrad/                     # Adagrad optimizer experiment
│   ├── activation_gelu/                       # GELU activation experiment
│   ├── position_encoding_learned/             # Learned positional encoding
│   ├── layer_norm_pre/                        # Pre-normalization experiment
│   ├── ...                                    # Other individual experiments
│   ├── all_results.csv                        # All experiment results summary
│   ├── advanced_ablation_results.csv          # Advanced experiment results
│   ├── ablation_comparison.png                # Basic experiment comparison
│   └── advanced_ablation_comparison.png       # Advanced experiment comparison
./logs/                                        # Detailed logs
└── experiment_advanced_20231201.log           # Execution log
```

### Key Result Files

- **`*_results.csv`**: Experiment results summary table
  - Contains BLEU scores, loss values, training time for each experiment
  - Easy to import for statistical analysis

- **`*_comparison.png`**: Visualization comparison charts
  - Grouped display of different experiment types
  - Shows which configuration performs best

- **`config.json`**: Detailed configuration for each experiment
  - Can be used to reproduce best results
  - Records all hyperparameter settings

### Evaluation Metrics
- **BLEU Scores**: BLEU-1 through BLEU-4 with detailed analysis
- **Training Metrics**: Loss, accuracy, convergence speed
- **Computational Efficiency**: Training time, memory usage
- **Statistical Significance**: Multiple runs for reliable conclusions

## 🛠️ Installation and Requirements

### Dependencies
```bash
pip install torch torchvision torchaudio
pip install pandas matplotlib nltk tqdm
pip install sacrebleu  # For BLEU evaluation
```

### NLTK Data
```python
import nltk
nltk.download('punkt')
```

### System Requirements
- **Python**: 3.10
- **PyTorch**: 2.1.2
- **CUDA**: 11.8+ (implemented on single NVIDIA GeForce RTX 4090)

## 🎮 Usage Workflows

### Production Usage

```bash
# Train best configuration (based on ablation results)
python train.py --d_model 256 --num_heads 8 --activation gelu

# Translate with trained model
python translate.py --model best_bleu_model.pth \
                   --sentence "Machine learning is fascinating" \
                   --beam_size 5
```


## 📄 Project Structure

```
Translator-en2cn/
├── 📜 README.md                    # This comprehensive documentation
├── 🔬 ablation_study.py            # Main ablation experiment program (CORE)
├── 🤖 model.py                     # Enhanced Transformer model
├── 🏃 train.py                     # Training script with optimizations
├── 🌐 translate.py                 # Translation with beam search
├── 📊 dataloader.py                # Data loading utilities
├── 📚 cmn-eng-simple/              # Dataset directory
├── 📈 ablation/                    # Experiment results
└── 📋 logs/                        # Detailed logs
```

## 📚 Citation

If you use this work in your research, please cite:

```bibtex
@misc{transformer-en2cn-ablation,
  title={English-to-Chinese Neural Machine Translation with Comprehensive Ablation Studies},
  author={Zihan Qian},
  year={2025},
  publisher={GitHub},
  url={https://github.com/RickyQzh/Translator-en2cn}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**🎉 Ready to Explore!** This comprehensive framework provides everything needed for systematic Transformer analysis and high-quality English-to-Chinese translation.
