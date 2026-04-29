import pandas as pd
import torch
import pickle
from pathlib import Path
from run import load_best_model, load_data, set_seed, greedy_generate

if __name__ == "__main__":
    Path("results").mkdir(exist_ok=True)

    with open('tokeniser.pkl', 'rb') as f:
        tokeniser = pickle.load(f)

    set_seed(42)
    train, val, test = load_data()
    
    model = load_best_model(best_model=None, best_config=None)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train = train.copy()
    train_words = train['word'].tolist()
    train['prediction'] = greedy_generate(model, train_words, device, tokeniser)
    train.to_csv('results/train_results.csv', index=False)

    val = val.copy()
    val_words = val['word'].tolist()
    val['prediction'] = greedy_generate(model, val_words, device, tokeniser)
    val.to_csv('results/val_results.csv', index=False)

    test = test.copy()
    test_words = test['word'].tolist()
    test['prediction'] = greedy_generate(model, test_words, device, tokeniser)
    test.to_csv('results/test_results.csv', index=False)