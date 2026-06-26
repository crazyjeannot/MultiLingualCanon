# MultiLingualCanon

Code and data for a multilingual, stylometric approach to literary canonicity. Rather than asking whether canonised and non-canonised novels differ semantically, this project asks whether they differ *stylistically* — in readability, lexical diversity, sentence length, compressibility and similar measures — and whether canonisation status leaves a trace in the similarity networks built from those measures, across four languages: German (DE), Danish (DK), English (EN) and French (FR). 

--> [full publication](https://doi.org/10.63744/Y0dv4ooACREY )

## Repository structure

```
MultiLingualCanon/
├── scripts/                              # Feature extraction, normalisation, network analysis
├── data_de/                               # German feature matrices, canon scores, distance matrix
├── data_dk/                               # Danish feature matrices, canon score bins/distributions
├── data_en_br/                            # English (British) feature matrices, canon scores
├── data_fr/                               # French chapter-level features, canon tags
├── nearest3NeighboursAdjacency/           # Per-language 3-nearest-neighbour adjacency matrices
├── networkMetrics_time_sensitive_3nn/     # Per-language network metrics (+ by_cluster breakdowns)
├── clusterCorrelationResults_time_sensitive_3nn/  # Cluster vs. canonisation correlations, figures, temporal trends
├── .gitignore
└── README.md
```

### `scripts/`

| Script | What it does |
|---|---|
| `extract_features.py` | Extracts stylometric features (MSTTR-100, sliding TTR, bz2 compressibility, passive/active ratio, readability, mean sentence/word length, noun/word ratio) from tokenised `.tokens` files via CLI (`--input_folder`, `--output_csv`). Includes language-specific readability functions (Flesch Reading Ease for EN/FR, Wiener Sachtextformel for DE, LIX-style score for DK). |
| `normalize_df.py` | Row-normalises (L2) a fixed set of feature columns from a hard-coded input file (`CANON_FEATURES_CHAPITRES_LAST.csv`). |
| `normalization_functions.py` | Reusable normalisation functions (`normalize_dataframe`: pure L2 row-normalisation; `standardize_features`: MinMax scaling followed by L2 row-normalisation) — used by, but not called from, `normalize_df.py`. |
| `compute_network.py` | Builds a cosine-similarity graph with a fixed similarity threshold (0.9), detects Leiden communities, computes betweenness/degree/PageRank centrality and a custom "mean path to top PageRank nodes" metric, correlates each against canon score, and writes an interactive Plotly graph. |
| `compute_network_top10.py` | Same pipeline as above, but builds the graph from each node's top-10 most similar neighbours instead of a fixed threshold (Leiden step is present but commented out). |

Both network scripts read and write to the working directory using hard-coded filenames (`CANON_FEATURES_CHAPITRES.csv` / `CANON_FEATURES_CHAPITRES_LAST_NORMALIZED.csv`, `Canon_tag.json`, `distance_matrix.csv`, `correlations_with_canon.txt`, `graph_plot.html`) rather than CLI arguments, so they need to be run from inside the relevant language folder with those exact files present.

### Data and results folders

- `data_de/`, `data_dk/`, `data_en_br/`, `data_fr/` — per-language feature matrices (raw and normalised), canon tags/scores, and (for DE/EN) precomputed adjacency and distance matrices. Column and file-naming conventions differ slightly between languages (e.g. `Canon_tag.json` for FR vs. `titles_with_scores_*.csv` for DE/EN), reflecting separate per-language pipelines rather than one shared script run four times.
- `nearest3NeighboursAdjacency/` — adjacency matrices built from each text's 3 nearest neighbours by feature similarity, one CSV per language.
- `networkMetrics_time_sensitive_3nn/` — indegree, PageRank, betweenness and closeness centrality per text, with Louvain/Leiden cluster assignments and canonisation score; `by_cluster/` splits these by cluster ID.
- `clusterCorrelationResults_time_sensitive_3nn/` — correlation of cluster membership with canonisation score (`*_cluster_correlation_*.csv`), summary correlation statistics (`centrality_correlation_summary.csv`), correlation plots per language and centrality measure, and a `temporalTrend/` subfolder tracking how these network metrics evolve over publication time.

## Requirements

Python ≥ 3.9, with: `pandas`, `numpy`, `scipy`, `scikit-learn`, `networkx`, `python-igraph`, `leidenalg`, `plotly`, `textstat`, `tqdm`.
