import pandas as pd
import numpy as np
import networkx as nx
from sklearn.metrics.pairwise import cosine_distances
from scipy.stats import pearsonr
import json
import igraph as ig
from leidenalg import ModularityVertexPartition, find_partition
import plotly.graph_objects as go
from tqdm import tqdm

# === Load and preprocess data ===
df = pd.read_csv("CANON_FEATURES_CHAPITRES_LAST_NORMALIZED.csv", index_col=0)
df = df[['MSTTR-100', 'Compressibility', 'Flesch Reading Ease', 'Mean sentence length', 'Mean word length']]

# === Compute cosine distance matrix ===
features = df.values
distance_matrix = cosine_distances(features)
distance_df = pd.DataFrame(distance_matrix, index=df.index, columns=df.index)
distance_df.to_csv("distance_matrix.csv")

# === Add Canon scores ===
with open("Canon_tag.json") as f:
    canon_dict = json.load(f)
df["Canon"] = df.index.map(lambda x: canon_dict.get(x, 0))

# === Build graph with per-node edge threshold ===
edges_per_node = 10
G = nx.Graph()
for i, doc_i in tqdm(enumerate(df.index), total=len(df.index), desc="Building Graph"):
    G.add_node(doc_i)
    similarities = [(1 - distance_matrix[i][j], df.index[j]) for j in range(len(df.index)) if j != i]
    top_similar = sorted(similarities, reverse=True)[:edges_per_node]
    for sim, doc_j in top_similar:
        G.add_edge(doc_i, doc_j, weight=sim)

# === Compute Leiden Communities (optional/commented) ===
# g_ig = ig.Graph.TupleList(G.edges(), directed=False)
# partition = find_partition(g_ig, partition_type=ModularityVertexPartition)
# leiden_communities = {v["name"]: int(partition.membership[i]) for i, v in enumerate(g_ig.vs)}
# df["Community"] = df.index.map(leiden_communities.get)

# === Graph metrics ===
betweenness = nx.betweenness_centrality(G)
degree_centrality = nx.degree_centrality(G)
pagerank = nx.pagerank(G)

df["Betweenness"] = df.index.map(betweenness)
df["DegreeCentrality"] = df.index.map(degree_centrality)
df["PageRank"] = df.index.map(pagerank)

# === Compute Mean Path to Top nodes (global) ===
pr_threshold = np.percentile(list(pagerank.values()), 95)
top_nodes = [n for n, pr in pagerank.items() if pr >= pr_threshold]
shortest_paths = {}
for source in tqdm(G.nodes(), desc="Computing all-pairs shortest paths"):
    shortest_paths[source] = nx.single_source_shortest_path_length(G, source)

def mean_path_to_top(node):
    lengths = [shortest_paths[node].get(n) for n in top_nodes if n in shortest_paths[node]]
    return np.mean(lengths) if lengths else np.nan

df["MeanPathToTop"] = [mean_path_to_top(n) for n in tqdm(df.index, desc="Computing Mean Path to Top")]

# === Correlations with Canon ===
metrics = ["Betweenness", "DegreeCentrality", "PageRank", "MeanPathToTop"]
correlations = {}
for metric in metrics:
    valid = df[["Canon", metric]].dropna()
    if valid[metric].nunique() <= 1:
        correlations[metric] = np.nan
    else:
        corr, _ = pearsonr(valid["Canon"], valid[metric])
        correlations[metric] = corr

print("Correlation with Canon:")
with open("correlations_with_canon.txt", "w") as f:
    for k, v in correlations.items():
        line = f"{k}: {v if not np.isnan(v) else 'undefined'}\n"
        print(line, end="")
        f.write(line)

# === Plot with Plotly ===
pos = nx.spring_layout(G, seed=42)
x_vals = [pos[n][0] for n in G.nodes()]
y_vals = [pos[n][1] for n in G.nodes()]

x_edges, y_edges = [], []
for edge in G.edges():
    x0, y0 = pos[edge[0]]
    x1, y1 = pos[edge[1]]
    x_edges += [x0, x1, None]
    y_edges += [y0, y1, None]

edge_trace = go.Scatter(
    x=x_edges, y=y_edges,
    line=dict(width=0.5, color="#888"),
    hoverinfo='none',
    mode='lines')

node_color = df.loc[list(G.nodes()), "Canon"].values
node_text = [f"{n}\nCanon Score: {df.loc[n, 'Canon']:.2f}" for n in G.nodes()]

node_trace = go.Scatter(
    x=x_vals, y=y_vals,
    mode='markers',
    marker=dict(
        showscale=True,
        colorscale='Viridis',
        color=node_color,
        size=10,
        colorbar=dict(
            thickness=15,
            title='Canon Score'
        )
    ),
    text=node_text,
    hoverinfo='text')

fig = go.Figure(data=[edge_trace, node_trace],
                layout=go.Layout(
                    title='Graph of Novel Similarity',
                    showlegend=False,
                    hovermode='closest',
                    margin=dict(b=20, l=5, r=5, t=40)))

fig.write_html("graph_plot.html")
