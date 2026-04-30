import torch
import torch.nn as nn
import pandas as pd
import pickle
import json
import copy
import editdistance
import random
import numpy as np
from torch.utils.data import DataLoader
from torch.optim import Optimizer, AdamW
from pathlib import Path
from itertools import product
from model import Seq2Seq

from char_tokeniser import CharTokeniser

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_data():
    train_df = pd.read_csv('data/spa_train.tsv', sep='\t', names=['word', 'ipa'])
    val_df = pd.read_csv('data/spa_val.tsv', sep='\t', names=['word', 'ipa'])
    test_df = pd.read_csv('data/spa_test.tsv', sep='\t', names=['word', 'ipa'])

    # Filter out train examples that are in the test set
    train_df = train_df[~train_df['word'].isin(test_df['word'])]

    assert len(set(train_df['word']) & set(train_df['word'])) == 0

    return train_df, val_df, test_df

def load_best_model(best_model=None, best_config: dict=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if best_config is None:
        with open('config.json', 'r') as f:
            best_config = json.load(f)

    # Load model
    model_keys = {'lr', 'weight_decay'} # Filter out optimiser hyperparameters
    model_config = {k: v for k, v in best_config.items() if k not in model_keys}
    model = Seq2Seq(**model_config).to(device)

    if best_model is None:
        model.load_state_dict(torch.load('model.pt', map_location=device, weights_only=True))
    else:
        model.load_state_dict(best_model)

    return model

def train(model: nn.Module, train_loader, optimiser: Optimizer, loss_fn: nn.modules.loss._Loss, tokeniser: CharTokeniser, device):
    model.train()
    vocab_size = model.lm_head.projection.out_features

    epoch_loss = 0.0

    for batch in train_loader:
        optimiser.zero_grad()

        # Tokenise (B, L)
        word_tokens = tokeniser.encode(batch['word']).to(device)
        ipa_tokens = tokeniser.encode(batch['ipa'], is_ipa=True).to(device)

        # Shift targets (teacher forcing)
        dec_input = ipa_tokens[:, :-1] # Drop last token (EOS)
        target_y = ipa_tokens[:, 1:]   # Drop first token (BOS)

        # Compute loss
        logits = model(word_tokens, dec_input)
        loss = loss_fn(logits.reshape(-1, vocab_size), target_y.flatten())
        epoch_loss += loss.item()

        # Update parameters
        loss.backward()
        optimiser.step()

    return epoch_loss / len(train_loader)

@torch.no_grad()
def eval(model: nn.Module, val_loader, loss_fn: nn.modules.loss._Loss, tokeniser: CharTokeniser, device):
    model.eval()
    vocab_size = model.lm_head.projection.out_features

    epoch_loss = 0.0
    for batch in val_loader:
        
        # Tokenise
        word_tokens = tokeniser.encode(batch['word']).to(device)
        ipa_tokens = tokeniser.encode(batch['ipa'], is_ipa=True).to(device)

        # Shift targets (teacher forcing)
        dec_input = ipa_tokens[:, :-1]
        target_y = ipa_tokens[:, 1:]

        # Compute loss
        logits = model(word_tokens, dec_input)
        loss = loss_fn(logits.reshape(-1, vocab_size), target_y.flatten())
        epoch_loss += loss.item()

    return epoch_loss / len(val_loader)

class EarlyStopping:
    """Class for early stopping.
    
    Attributes:
        patience (int): number of epochs to wait for improvement.
        delta (float): small margin by which the condition must improve by.
        counter (int): counter for number of epochs since improvement.
        condition (str): the metric on which to condition early stopping (loss, PER, WAcc).
        best_weights: the state dict of the best model.
        best_epoch (int): the model's best epoch in terms of the given condition.
        best_loss (float): the model's loss on the best epoch (unless loss is the condition, this might not be the best loss overall)
        best_per (float): the model's PER on the best epoch (same thing applies here).
        best_wacc (float): the model's WAcc on the best epoch (same thing applies here).
    """
    def __init__(self, patience: int, condition: str, delta=1e-4):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.condition = condition

        self.best_weights = None
        self.best_epoch = None
        self.best_loss = float('inf')
        self.best_per = float('inf')
        self.best_wacc = -float('inf')

    def step(self, model, epoch, loss, per, wacc):
        """
        Args:
            model: the model.
            metric (float): loss at current epoch.
            epoch (int): the current epoch.
            per (float): PER at current epoch.
            wacc (float): WAcc at current epoch.

        Returns:
            True if training should terminate, else False. 
        """
        if self.condition == 'loss':
            is_better = loss < self.best_loss - self.delta
        elif self.condition == 'per':
            is_better = per < self.best_per - self.delta
        else:
            is_better = wacc > self.best_wacc + self.delta

        if is_better:
            self.counter = 0
            self.best_loss = loss
            self.best_epoch = epoch
            self.best_per = per
            self.best_wacc = wacc
            self.best_weights = copy.deepcopy(model.state_dict())
        else:
            self.counter += 1

        return self.counter >= self.patience
    
def calculate_wacc(preds: list[tuple]):
    return sum(gold == pred for gold, pred in preds) / len(preds)

def calculate_per(preds: list[tuple], tokeniser: CharTokeniser):
    total_dist = 0.0
    for gold, pred in preds:
        gold_phonemes = tokeniser.ipa_to_units(gold)
        pred_phonemes = tokeniser.ipa_to_units(pred)
        total_dist += editdistance.eval(gold_phonemes, pred_phonemes) / len(gold_phonemes)
    return total_dist / len(preds)

def train_model(train_loader, val_loader, tokeniser: CharTokeniser, stopping_metric, lr, weight_decay, dropout_p, d_model, num_layers, num_heads, mlp_mode):
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Seq2Seq(
        vocab_size=tokeniser.vocab_size,
        max_len=tokeniser.max_len,
        d_model=d_model,
        num_heads=num_heads, 
        mlp_mode=mlp_mode, 
        num_layers=num_layers, 
        dropout_p=dropout_p
        ).to(device)
    
    optim = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss(ignore_index=0) # Ignore padding index
    early_stopping = EarlyStopping(patience=30, condition=stopping_metric, delta=0.0001)

    train_log = []
    for epoch in range(1000000): # Effectively infinite training

        # Compute loss on train and val
        train_loss = train(model, train_loader, optim, criterion, tokeniser, device)
        val_loss = eval(model, val_loader, criterion, tokeniser, device)
        
        # Make predictions and compute PER and WAcc on val 
        preds = []
        for batch in val_loader:
            batch_preds = greedy_generate(model, batch['word'], device, tokeniser)
            preds.extend(zip(batch['ipa'], batch_preds))
        val_per = calculate_per(preds, tokeniser)
        val_wacc = calculate_wacc(preds)

        train_log.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_per': val_per,
            'val_wacc': val_wacc
        })

        if early_stopping.step(model, epoch, val_loss, val_per, val_wacc):
            print(f"Early stopping triggered. Best model saved: Epoch {early_stopping.best_epoch} | Val loss: {early_stopping.best_loss:.4f} | Val PER: {early_stopping.best_per * 100:.2f}% | Val WAcc: {early_stopping.best_wacc * 100:.2f}%")
            break

    return early_stopping, train_log

def hparam_search(train_df: pd.DataFrame, val_df: pd.DataFrame, tokeniser: CharTokeniser, params: dict, stopping_metric: str):

    # Init metics to determine best model
    best_loss = float('inf')
    best_per = float('inf')
    best_wacc = -float('inf')
    best_epoch = 0
    best_config = None
    best_model_weights = None
    best_train_log = None

    # Fixed batch size of 128
    train_loader = DataLoader(train_df.to_dict('records'), batch_size=128, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_df.to_dict('records'), batch_size=128, pin_memory=True)

    param_keys = params.keys()
    combos = list(product(*params.values())) # List of all combos of parameters

    # Loop over params
    for i, values in enumerate(combos):
        current_config = dict(zip(param_keys, values))

        print(f"\nTesting [{i+1}/{len(combos)}]: {current_config}", flush=True)

        # Train model
        early_stopping, train_log = train_model(train_loader, val_loader, tokeniser, stopping_metric, **current_config)

        if stopping_metric == 'loss':
            model_is_better = early_stopping.best_loss < best_loss
        elif stopping_metric == 'per':
            model_is_better = early_stopping.best_per < best_per
        else:
            model_is_better = early_stopping.best_wacc > best_wacc

        if model_is_better:
            best_loss = early_stopping.best_loss
            best_per = early_stopping.best_per
            best_wacc = early_stopping.best_wacc
            best_epoch = early_stopping.best_epoch
            best_model_weights = early_stopping.best_weights
            best_config = current_config
            best_train_log = train_log

    print("\nHyperparameter search complete")
    print(f"Best config: {best_config}")
    print(f"Epoch: {best_epoch} | Val loss: {best_loss:.4f} | Val PER: {best_per * 100:.2f}% | Val WAcc: {best_wacc * 100:.2f}%")

    return best_config, best_model_weights, best_train_log

@torch.no_grad()
def greedy_generate(model: nn.Module, words: list, device, tokeniser:CharTokeniser):
    model.eval()

    input_ids = tokeniser.encode(words).to(device) # (B, max_length_in_batch)
    enc_out = model.encoder(input_ids)         # (B, L, d_model)
    batch_size = enc_out.shape[0]

    bos_token = 1
    eos_token = 2

    # Create tensor of shape (B, 1) containing the BOS token (1)
    generated_seqs = torch.full((batch_size, 1), bos_token, dtype=torch.long, device=device)

    # Each element set to True if the sequence has finished (hit EOS)
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

    max_len = tokeniser.max_len
    for _ in range(max_len - 1):
        dec_out = model.decoder(generated_seqs, enc_out)
        logits = model.lm_head(dec_out) # (B, current_step, V)
        next_tokens = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True) # (B, 1)

        # Add the generated tokens to the sequences
        generated_seqs = torch.cat([generated_seqs, next_tokens], dim=1)

        # Check if the last token is EOS and change finished to True if it is 
        finished = torch.logical_or(finished, next_tokens.squeeze(-1) == eos_token)

        # If every word has finished, break
        if finished.all():
            break

    results = []
    for row in generated_seqs:
        # Sequences that finished early will still have tokens after the EOS token, so we select up to the first EOS token
        seq = row.tolist()
        if eos_token in seq:
            seq = seq[:seq.index(eos_token) + 1]
        results.append(tokeniser.decode(seq))

    return results

if __name__ == "__main__":

    PARAMS = {
        'lr': [1e-4],
        'weight_decay': [1e-3, 1e-4],
        'dropout_p': [0.2, 0.3],
        'd_model': [128, 256],
        'num_layers': [4],
        'num_heads': [4, 8],
        'mlp_mode': ['relu', 'swiglu']
    } 

    STOPPING_METRIC = 'per'

    # Load data
    train_df, val_df, test_df = load_data()

    # Create tokeniser and save it (it will always be the same)
    tokeniser_path = Path('tokeniser.pkl')
    if tokeniser_path.exists():
        with open('tokeniser.pkl', 'rb') as f:
            tokeniser = pickle.load(f)
    else:
        tokeniser = CharTokeniser(train_df)
        with open(tokeniser_path, 'wb') as f:
            pickle.dump(tokeniser, f)

    # Get best model
    best_config, best_model, train_log = hparam_search(train_df, val_df, tokeniser, PARAMS, STOPPING_METRIC)

    # Save everything
    torch.save(best_model, 'model.pt')
    pd.DataFrame(train_log).to_csv('train_log.csv', index=False)
    best_config['vocab_size'] = int(tokeniser.vocab_size)
    best_config['max_len'] = int(tokeniser.max_len)
    Path('config.json').write_text(json.dumps(best_config, indent=2))