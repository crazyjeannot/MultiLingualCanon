import os
import glob
import bz2
import pandas as pd
import textstat
from tqdm import tqdm
import re
import random
import numpy as np


def compute_msttr(tokens, segment_size=100):
    """
    Compute Mean Segmental Type-Token Ratio (MSTTR) over non-overlapping
    segments of `segment_size` tokens.
    """
    n = len(tokens)
    if n < segment_size:
        return len(set(tokens)) / n if n > 0 else 0.0
    ttrs = []
    for start in range(0, n, segment_size):
        segment = tokens[start:start+segment_size]
        if len(segment) < segment_size:
            break
        ttrs.append(len(set(segment)) / segment_size)
    return sum(ttrs) / len(ttrs) if ttrs else 0.0


def compute_ttr_sliding(tokens, window_size=100):
    """
    Compute sliding window Type-Token Ratio (TTR) with window of `window_size` tokens.
    """
    n = len(tokens)
    if n < window_size:
        return len(set(tokens)) / n if n > 0 else 0.0
    ttrs = []
    for start in range(0, n - window_size + 1):
        window = tokens[start:start + window_size]
        ttrs.append(len(set(window)) / window_size)
    return sum(ttrs) / len(ttrs) if ttrs else 0.0


def compute_compressibility(text):
    """
    Compute compressibility as original_size / compressed_size using bz2.
    """
    data = text.encode('utf-8')
    compressed = bz2.compress(data)
    return len(data) / len(compressed) if len(compressed) > 0 else 0.0


def compute_normalized_compressibility(text, sample_size=500, n_samples=3):
    """
    Compute compressibility as the average of n_samples taken from non-overlapping
    500-word chunks. Returns the mean compressibility of these chunks.
    """
    words = text.split()
    total_words = len(words)
    chunk_size = sample_size

    if total_words < chunk_size * n_samples:
        return np.nan  # or 0.0 if you prefer a default fallback

    # Pick non-overlapping random start indices
    max_start = total_words - chunk_size
    starts = random.sample(range(0, max_start, chunk_size), n_samples)

    compressibilities = []
    for start in starts:
        chunk = " ".join(words[start:start + chunk_size])
        data = chunk.encode('utf-8')
        compressed = bz2.compress(data)
        ratio = len(data) / len(compressed) if len(compressed) > 0 else 0.0
        compressibilities.append(ratio)

    return np.mean(compressibilities)


def compute_passive_active_ratio(df):
    """
    Approximate passive/active verb ratio via dependency tags.
    passive = count of aux:pass relations
    active = total AUX+VERB - passive
    """
    passive = (df['dependency_relation'] == 'aux:pass').sum()
    total_verbs = df['POS_tag'].isin(['AUX', 'VERB']).sum()
    active = total_verbs - passive
    return (passive / active) if active > 0 else 0.0


def compute_dk_passive_active_ratio(df):
    """
    Approximate passive/active verb ratio via morphological features in Danish.
    Passive = verbs with 'Voice=Pass' in token_morph
    Active = other verbs (excluding passive ones)
    """
    morph = df['token_morph'].astype(str)
    is_verb = df['POS_tag'].isin(['AUX', 'VERB'])

    passive = morph[is_verb].str.contains('Voice=Pass').sum()
    total = is_verb.sum()
    active = total - passive

    return (passive / active) if active > 0 else 0.0


def compute_relative_frequency(df, lemma):
    """
    Relative frequency of a given lemma in the document.
    """
    total = len(df)
    count = (df['lemma'] == lemma).sum()
    return (count / total) if total > 0 else 0.0


def compute_readability(text, lang="fr"):
    """
    Compute Flesch Reading Ease as a French-adapted readability score.
    """
    # Using English formula as proxy; for French consider Douma variant manually
    textstat.set_lang(lang)
    return textstat.flesch_reading_ease(text)


# --- danish readability function ---
def compute_dk_readability(text):
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    n_sentences = len(sentences) if sentences else 1

    words = re.findall(r'\b\w+\b', text)
    n_words = len(words) if words else 1

    n_long_words = sum(1 for w in words if len(w) > 6)

    lix_score = (n_words / n_sentences) + (n_long_words * 100) / n_words
    return lix_score


def compute_mean_sentence_length(df):
    """
    Compute mean sentence length in tokens (excluding punctuation).
    """
    # grouping by paragraph_ID and sentence_ID
    group = df[df['POS_tag'] != 'PUNCT'].groupby(['paragraph_ID', 'sentence_ID'])
    lengths = group.size().values
    return lengths.mean() if len(lengths) > 0 else 0.0


def compute_mean_word_length(df):
    """
    Compute mean word length (in characters), excluding punctuation.
    """
    words = df[df['POS_tag'] != 'PUNCT']['word']
    lengths = words.str.len().values
    return lengths.mean() if len(lengths) > 0 else 0.0


def compute_noun_word_ratio(df):
    """
    Ratio of noun tokens to total word tokens (excluding punctuation).
    """
    nouns = df['POS_tag'] == 'NOUN'
    total = df['POS_tag'] != 'PUNCT'
    return nouns.sum() / total.sum() if total.sum() > 0 else 0.0


def extract_features_for_file(filepath):
    df = pd.read_csv(filepath, sep='\t', quoting=3, dtype=str)
    df['token_ID_within_document'] = df['token_ID_within_document'].astype(int)
    df = df.sort_values('token_ID_within_document')

    tokens = df['word'].tolist()

    # reconstruct raw text for readability & compression
    text = ''
    for _, row in df.iterrows():
        if row['POS_tag'] == 'PUNCT':
            text += row['word']
        else:
            text += ' ' + row['word']
    text = text.strip()

    feats = {
        'doc_name': os.path.splitext(os.path.basename(filepath))[0],
        'MSTTR-100': compute_msttr(tokens),
        'TTR-100-sliding': compute_ttr_sliding(tokens),
        'Compressibility': compute_compressibility(text),
        'Passive/Active ratio': compute_passive_active_ratio(df),
        '"que" relative frequency': compute_relative_frequency(df, 'que'),
        '"de" relative frequency': compute_relative_frequency(df, 'de'),
        'Flesch Reading Ease': compute_readability(text),
        'Mean sentence length': compute_mean_sentence_length(df),
        'Mean word length': compute_mean_word_length(df),
        'Noun/Word ratio': compute_noun_word_ratio(df),
        'dk_readability': compute_dk_readability(text)
    }

    return feats


def main(input_folder, output_csv='features.csv'):
    files = glob.glob(os.path.join(input_folder, '*.tokens'))
    results = []
    for fp in tqdm(files, desc='Processing files'):
        feats = extract_features_for_file(fp)
        results.append(feats)

    df_out = pd.DataFrame(results)
    df_out.to_csv(output_csv, index=False)
    return df_out


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Extract stylistic features from French .tokens novels')
    parser.add_argument('--input_folder', required=True,
                        help='Path to folder containing .tokens files')
    parser.add_argument('--output_csv', default='features.csv',
                        help='Where to save the output CSV')
    args = parser.parse_args()
    print(f"Processing files in {args.input_folder}...")
    df_results = main(args.input_folder, args.output_csv)
    print(f"Done! Features saved to {args.output_csv}.")
