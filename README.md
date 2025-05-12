# English-to-Chinese Transformer Translation

A neural machine translation system that translates English sentences to Chinese using a custom Transformer architecture.

## Dataset

The model is trained on English-Chinese parallel corpus located in `./cmn-eng-simple/`:
- Training set: 18,000 sentences
- Validation set: 500 sentences
- Test set: 2,636 sentences

## Files

- `model.py`: Implementation of the transformer model from scratch
- `dataloader.py`: Data loading utilities
- `train.py`: Training script with integrated validation and testing
- `translate.py`: Script for translating new English sentences to Chinese with beam search
- `ablation_study.py`: Script for running ablation studies on model parameters

## Model Architecture

- Custom Transformer architecture implemented from scratch
- 6 encoder and 6 decoder layers
- 8 attention heads
- 256 model dimensions
- 1024 feed-forward dimensions
- Dropout rate of 0.2 (increased to prevent overfitting)

## Training Improvements

### Overfitting Prevention

The training process now includes several techniques to prevent overfitting:

1. **Early Stopping**
   - Monitors validation loss over epochs
   - Stops training when validation loss doesn't improve for 5 consecutive epochs
   - Prevents the model from overfitting on the training data

2. **Label Smoothing (KL Divergence Loss)**
   - Implements label smoothing with a factor of 0.1
   - Prevents the model from becoming overconfident
   - Improves generalization by distributing small probability to non-target classes

3. **Warmup + Cosine Learning Rate Schedule**
   - Implements a warmup phase followed by cosine decay
   - Gradually increases learning rate during warmup (helps stabilize early training)
   - After warmup, follows a cosine curve to decrease learning rate to a minimum value
   - Warmup steps configurable (default: 4000 steps)
   - Better convergence and generalization than standard schedules

4. **Increased Dropout**
   - Dropout rate increased from 0.1 to 0.2
   - Helps prevent overfitting by randomly dropping connections during training

## Training

To train the model, run:

```bash
python train.py
```

This will:
- Train the model for up to 50 epochs (usually stops earlier due to early stopping)
- Calculate loss and BLEU scores for training, validation, and test sets after each epoch
- Display metrics in a tabular format, including current learning rate
- Save the best models based on validation loss and BLEU score
- Generate plots for loss, BLEU, and learning rate trends
- Automatically stop training when validation performance plateaus

### Metrics Tracking

The training script tracks and saves the following metrics for each epoch:
- Training loss and BLEU score
- Validation loss and BLEU score
- Test loss and BLEU score
- Current learning rate

These metrics are:
- Displayed in a table format during training
- Saved to `training_metrics.csv`
- Plotted and saved as `training_metrics.png` every 5 epochs and at the end of training
- Learning rate changes plotted in `learning_rate.png`

## Ablation Studies

To run ablation studies on various model parameters, use:

```bash
python ablation_study.py
```

This script automatically:
1. Creates a timestamped experiment directory in `./ablation/`
2. Runs multiple training experiments with different parameter configurations
3. Tests variations of:
   - Model size (d_model, num_heads, feed-forward dimension)
   - Layer count (encoder and decoder layers)
   - Dropout rate
   - Learning rate
   - Label smoothing factor
   - Warmup steps
4. Collects and compares results across all experiments
5. Generates comparison visualizations and an HTML report

### Ablation Study Results

Results are stored in organized directories with:
- Training metrics for each experiment
- Loss and BLEU plots for each configuration
- CSV file with all experiment results
- Comparison bar charts for validation BLEU and test loss
- Interactive HTML report highlighting the best configurations

## Translation

To translate an English sentence to Chinese, run:

```bash
python translate.py --sentence "Your English sentence here" --beam_size 5
```

### Beam Search

The translation script uses beam search for better translation quality:

- `--beam_size`: Number of hypotheses to keep during beam search (default: 5)
  - Set to 1 for greedy search
  - Higher values can provide better translations but are slower
- `--max_len`: Maximum length of translation (default: 50)

Example with custom beam size:

```bash
python translate.py --model best_bleu_model.pth --sentence "Hello, how are you today?" --beam_size 10
```

For greedy search:

```bash
python translate.py --sentence "Hello, how are you today?" --beam_size 1
```

## Requirements

- PyTorch
- tqdm
- matplotlib
- pandas
- nltk

Before running, install the required NLTK data:

```python
import nltk
nltk.download('punkt')
```

## Results

The model is evaluated using both loss and BLEU score:
- Training loss shows how well the model fits the training data
- Validation loss shows the model's generalization capability
- BLEU score measures the quality of the translations

The training process generates plots showing:
- Loss trends for training, validation, and test sets
- BLEU score trends for training, validation, and test sets
- Learning rate schedule changes

These plots help visualize the model's performance and identify potential overfitting or areas for improvement.
