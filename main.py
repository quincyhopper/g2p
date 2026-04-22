import torch
import pandas as pd
import torch.nn as nn
import json
from pathlib import Path
from itertools import product
from training import train_model, byte_tokenise
from model import Seq2Seq
from torch.utils.data import DataLoader

def load_data():
    train_df = pd.read_csv('data/spa_train.tsv', sep='\t', names=['word', 'ipa'])
    val_df = pd.read_csv('data/spa_val.tsv', sep='\t', names=['word', 'ipa'])
    test_df = pd.read_csv('data/spa_test.tsv', sep='\t', names=['word', 'ipa'])

    return train_df, val_df, test_df

def load_best_model(best_model, best_config: dict, from_save: bool=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_keys = ['d_model', 'num_heads', 'mlp_mode', 'num_layers', 'dropout_p'] # Filter out optimiser config
    model_config = {k: best_config[k] for k in model_keys if k in best_config}
    model = Seq2Seq(**model_config).to(device)

    if from_save and best_model is None:
        model.load_state_dict(torch.load('model.pt', map_location=device, weights_only=True))
    else:
        model.load_state_dict(best_model)

    return model

def hparam_search(train_df: pd.DataFrame, val_df: pd.DataFrame, params: dict):
    print("Starting hyperparameter search", flush=True)

    # Init best outcomes
    best_loss = float('inf')
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
        model_weights, model_loss, train_log = train_model(train_loader, val_loader, **current_config)
        print(f"Val loss: {model_loss:.4f}", flush=True)

        # Determine if model is best
        if model_loss < best_loss:
            best_loss = model_loss
            best_config = current_config
            best_model_weights = model_weights
            best_train_log = train_log

    return best_loss, best_config, best_model_weights, best_train_log

@torch.no_grad()
def greedy_generate(model: nn.Module, word: str, device, max_len: int=50):
    model.eval()

    input_ids = byte_tokenise(word).to(device) # (1, L)
    enc_out = model.encoder(input_ids)         # (1, L, d_model)
    generated_indices = [256]

    for _ in range(max_len):
        current_input = torch.tensor([generated_indices], dtype=torch.long).to(device) # (1, step)
        dec_out = model.decoder(current_input, enc_out)
        logits = model.lm_head(dec_out)
        next_token = torch.argmax(logits[:, -1, :], dim=-1).item()
        generated_indices.append(next_token)

        if next_token == 257:
            break

    filtered_bytes = bytes([i for i in generated_indices if 0 < i < 256])
    return filtered_bytes.decode(encoding='utf-8', errors='ignore')

def generate_test_preds(model, test_df: pd.DataFrame, generation_func, device):
    df = test_df.copy()
    df['prediction'] = df['word'].apply(lambda word: generation_func(model, word, device))
    return df

if __name__ == "__main__":

    # Load data
    print("Loading data")
    train_df, val_df, test_df = load_data()

    # Define search space
    params = {
        'lr': [1e-4, 5e-4],
        'weight_decay': [1e-3],
        'dropout_p': [0.1, 0.5],
        'd_model': [128, 256],
        'num_layers': [4, 6],
        'num_heads': [16],
        'mlp_mode': ['relu', 'swiglu']
    }   

    # Get best model
    best_loss, best_config, best_model, train_log = hparam_search(train_df, val_df, params)
    print("\nHyperparameter search complete")
    print(f"Best config: {best_config}")
    print(f"Best val loss: {best_loss:.4f}")

    # Save everything
    torch.save(best_model, 'model.pt')
    pd.DataFrame(train_log).to_csv('train_log.csv', index=False)
    Path('config.json').write_text(json.dumps(best_config, indent=2))

    # Reinit model for test generation
    model = load_best_model(best_model, best_config)

    # Generate test set predictions
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results_df = generate_test_preds(model, test_df, greedy_generate, device)
    results_df.to_csv('test_results.csv', index=False)