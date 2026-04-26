import torch
import torch.nn as nn
import copy
from torch.optim import AdamW, Optimizer
from model import Seq2Seq, byte_tokenise
from decoding_funcs import greedy_generate

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
    def __init__(self, patience, delta=1e-4):
        self.patience = patience
        self.delta = delta
        self.best_loss = float('inf')
        self.counter = 0
        self.best_epoch = None
        self.best_weights = None

    def step(self, model, val_loss, epoch):
        if val_loss < self.best_loss - self.delta:
            self.counter = 0
            self.best_loss = val_loss
            self.best_epoch = epoch
            self.best_weights = copy.deepcopy(model.state_dict())
        else:
            self.counter += 1

        return self.counter >= self.patience
    
def evaluate_wacc(model: nn.Module, val_loader, device):
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

class EarlyStoppingWAcc:
    def __init__(self, patience, delta=1e-3):
        self.patience = patience
        self.delta = delta
        self.best_wacc = -float('inf')
        self.counter = 0
        self.best_epoch = 0
        self.best_weights = None

    def step(self, model, val_wacc, epoch):
        if val_wacc > self.best_wacc + self.delta:
            self.counter = 0
            self.best_wacc = val_wacc
            self.best_epoch = epoch
            self.best_weights = copy.deepcopy(model.state_dict())
        else:
            self.counter += 1

        return self.counter >= self.patience

def train_model(train_loader, val_loader, lr, weight_decay, dropout_p, d_model, num_layers, num_heads, mlp_mode):
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
    early_stopping = EarlyStoppingWAcc(patience=30)

    train_log = []
    for epoch in range(1000000): # Effectively infinite training

        train_loss = train(model, train_loader, optim, criterion, device=device)
        val_loss = eval(model, val_loader, criterion, device)
        val_wacc = evaluate_wacc(model, val_loader, device)

        train_log.append({
            'epoch': epoch+1,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_wacc': val_wacc
        })

        stop = early_stopping.step(model, val_wacc, epoch=epoch+1)
        if stop:
            print(f"Early stopping triggered. Best model saved at epoch {early_stopping.best_epoch+1} with val wacc {early_stopping.best_wacc:.4f}")
            break

    return early_stopping.best_weights, early_stopping.best_wacc, train_log