# Grapheme-to-sequence transformer model
This repository contains the code for implementing a grapheme-to-sequence transformer model for converting Spanish grapheme sequences into their respective IPA transcriptions. The final model achieved a PER of 3.73% and a WER of 16% on the test set. 

## Project structure
```
.
├── analysis.ipynb    # Statistics, plots, etc      
├── char_tokeniser.py # Tokeniser class
├── config.json       # Final model config
├── load_and_test.py  # Script for loading best model and producing predictions
├── model.py          # Model code
├── pyproject.toml    # Dependencies
├── README.md         # readme file
├── results           # CSV files of results
├── run.py            # Main code
├── train_log.csv     # CSV of training log of best model
```

## Setup and installation
This project uses uv for dependency management and reproducibility. If you don't have uv installed, install it via:
```
curl -LsSf https://astral.sh/uv/install.sh | sh
```
1. **Clone this repo.**
```
git clone https://github.com/quincyhopper/g2p.git
cd g2p
```

2. **Install dependencies**
```
uv sync
```

## Run
To perform a hyperparameter search and save the best model, along with the training log and model config, run the following command:
```
uv run run.py
```