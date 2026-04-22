import editdistance
import pandas as pd

def evaluate_results(df: pd.DataFrame):
    df = df.copy()
    df['word_length'] = df['word'].apply(len) # Length of input word
    df['ipa_length'] = df['ipa'].apply(lambda x: len(x.split())) # Length of gold IPA (ignores spaces)
    df['edit_distance'] = df.apply(lambda row: editdistance.eval(row['ipa'], row['prediction']), axis=1)
    df['per'] = df['edit_distance'] / df['ipa_length']
    df['exact_match'] = df['ipa'] == df['prediction']

    print(f"Mean Phoneme Error-Rate (PER): {df['per'].mean():.3f}")
    print(f"Word Error Rate (WER): {(1 - df['exact_match'].mean()) * 100:.2f}%")
    print(f"Word Accuracy (WAcc): {df['exact_match'].mean() * 100:.2f}%")

    return df