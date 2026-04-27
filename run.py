import torch
import torch.nn as nn
import pandas as pd
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

from decoding_funcs import greedy_generate
from model import byte_tokenise

def set_seed(seed: int):
    random.seed(seed)
    np.random(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_data():
    train_df = pd.read_csv('data/spa_train.tsv', sep='\t', names=['word', 'ipa'])
    val_df = pd.read_csv('data/spa_val.tsv', sep='\t', names=['word', 'ipa'])
    test_df = pd.read_csv('data/spa_test.tsv', sep='\t', names=['word', 'ipa'])

    return train_df, val_df, test_df

def load_best_model(best_model, best_config: dict):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_keys = ['d_model', 'num_heads', 'mlp_mode', 'num_layers', 'dropout_p'] # Filter out optimiser config

    if best_config is None:
        with open('config.json', 'r') as f:
            best_config = json.load(f)


    model_config = {k: best_config[k] for k in model_keys if k in best_config}
    model = Seq2Seq(**model_config).to(device)

    if best_model is None:
        model.load_state_dict(torch.load('model.pt', map_location=device, weights_only=True))
    else:
        model.load_state_dict(best_model)

    return model

def train(model: nn.Module, train_loader, optimiser: Optimizer, loss_fn: nn.modules.loss._Loss, device):
    model.train()
    vocab_size = model.lm_head.projection.out_features

    epoch_loss = 0.0

    for batch in train_loader:
        optimiser.zero_grad()

        # Tokenise (B, L)
        word_tokens = byte_tokenise(batch['word']).to(device)
        ipa_tokens = byte_tokenise(batch['ipa']).to(device)

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
def eval(model: nn.Module, val_loader, loss_fn: nn.modules.loss._Loss, device):
    model.eval()
    vocab_size = model.lm_head.projection.out_features

    epoch_loss = 0.0
    for batch in val_loader:
        
        # Tokenise
        word_tokens = byte_tokenise(batch['word']).to(device)
        ipa_tokens = byte_tokenise(batch['ipa']).to(device)

        # Shift targets (teacher forcing)
        dec_input = ipa_tokens[:, :-1]
        target_y = ipa_tokens[:, 1:]

        # Compute loss
        logits = model(word_tokens, dec_input)
        loss = loss_fn(logits.reshape(-1, vocab_size), target_y.flatten())
        epoch_loss += loss.item()

    return epoch_loss / len(val_loader)

class EarlyStopping:
    def __init__(self, patience: int, metric: str, delta=1e-4):
        self.patience = patience
        self.delta = delta
        self.metric = metric
        self.counter = 0
        self.best_epoch = None
        self.best_weights = None

        if self.metric == 'loss':
            self.best_metric = float('inf')
        elif metric == 'wacc' or metric == 'per':
            self.best_metric = -float('inf')

    def step(self, model, metric, epoch):
        if self.metric == 'loss':
            is_better = metric < self.best_metric - self.delta
        else:
            is_better = metric > self.best_metric + self.delta

        if is_better:
            self.counter = 0
            self.best_metric = metric
            self.best_epoch = epoch
            self.best_weights = copy.deepcopy(model.state_dict())
        else:
            self.counter += 1

        return self.counter >= self.patience
    
def calculate_wacc(model: nn.Module, val_loader, device):
    """Calculate word accuracy (WAcc) instead of loss during validation step."""
    correct = 0
    total = 0

    for batch in val_loader:
        preds = greedy_generate(model, batch['word'], device)
        for pred, gold in zip(preds, batch['ipa']):
            if pred == gold:
                correct += 1
            total += 1

    return correct / total

def calculate_per(model: nn.Module, val_loader, device):

    total = 0.0
    for batch in val_loader:
        preds = greedy_generate(model, batch['word'], device)

        for pred, gold in zip(preds, batch['ipa']):
            dist = editdistance.eval(pred, gold)
            norm_dist = dist / len(gold)
            total += norm_dist

def train_model(train_loader, val_loader, stopping_metric, lr, weight_decay, dropout_p, d_model, num_layers, num_heads, mlp_mode):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Seq2Seq(
        d_model=d_model,
        num_heads=num_heads, 
        mlp_mode=mlp_mode, 
        num_layers=num_layers, 
        dropout_p=dropout_p
        ).to(device)
    
    optim = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss(ignore_index=0) # Ignore padding index
    early_stopping = EarlyStopping(patience=30, metric=stopping_metric, delta=0.0001)

    train_log = []
    for epoch in range(1000000): # Effectively infinite training

        train_loss = train(model, train_loader, optim, criterion, device=device)
        val_loss = eval(model, val_loader, criterion, device)
        
        if stopping_metric == 'loss':
            metric = val_loss
        elif stopping_metric == 'wacc':
            metric = calculate_wacc(model, val_loader, device)
        elif stopping_metric == 'per':
            metric = calculate_per(model, val_loader, device)

        train_log.append({
            'epoch': epoch+1,
            'train_loss': train_loss,
            'val_loss': val_loss,
            f'val_{stopping_metric}': metric
        })

        stop = early_stopping.step(model, metric, epoch=epoch+1)
        if stop:
            print(f"Early stopping triggered. Best model saved at epoch {early_stopping.best_epoch} with val {stopping_metric} {early_stopping.best_metric:.4f}")
            break

    return early_stopping.best_weights, early_stopping.best_metric, train_log

def hparam_search(train_df: pd.DataFrame, val_df: pd.DataFrame, params: dict, stopping_metric: str):

    # Init best outcomes
    best_metric = 0.0
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

        print(f"\nTesting [{i}/{len(combos)}]: {current_config}", flush=True)

        # Train model
        model_weights, model_metric, train_log = train_model(train_loader, val_loader, stopping_metric, **current_config)
        print(f"Val {stopping_metric}: {model_metric:.4f}", flush=True)

        # Determine if model is best
        if stopping_metric == 'loss':
            model_is_better = model_metric < best_metric
        elif stopping_metric == 'wacc' or stopping_metric == 'per':
            model_is_better = model_metric > best_metric

        if model_is_better:
            best_metric = model_metric
            best_config = current_config
            best_model_weights = model_weights
            best_train_log = train_log

    print("\nHyperparameter search complete")
    print(f"Best config: {best_config}")
    print(f"Best val wacc: {best_wacc:.4f}")

    return best_metric, best_config, best_model_weights, best_train_log

if __name__ == "__main__":
    set_seed(42)

    PARAMS = {
        'lr': [1e-4, 5e-4],
        'weight_decay': [1e-3],
        'dropout_p': [0.0, 0.1, 0.2],
        'd_model': [128, 256],
        'num_layers': [4, 6],
        'num_heads': [16],
        'mlp_mode': ['relu', 'swiglu']
    } 

    STOPPING_METRIC = 'per'

    # Load data
    train_df, val_df, test_df = load_data()

    # Get best model
    best_wacc, best_config, best_model, train_log = hparam_search(train_df, val_df, PARAMS, STOPPING_METRIC)

    # Save everything
    torch.save(best_model, 'model.pt')
    pd.DataFrame(train_log).to_csv('train_log.csv', index=False)
    Path('config.json').write_text(json.dumps(best_config, indent=2))

    # Reinit model for test generation
    model = load_best_model(best_model, best_config)

    # Generate test set predictions
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_words = test_df['word'].tolist()
    results_df = test_df.copy()
    results_df['prediction'] = greedy_generate(model, test_words, device)
    results_df.to_csv('test_results.csv', index=False)