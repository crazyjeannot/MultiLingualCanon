import pandas as pd
import numpy as np
import networkx as nx
from sklearn.metrics.pairwise import cosine_distances
from scipy.stats import pearsonr
import json
from leidenalg import ModularityVertexPartition, find_partition
import igraph as ig
import plotly.graph_objects as go

# === Load data ===
df = pd.read_csv("CANON_FEATURES_CHAPITRES.csv", index_col=0)
df = df[['MSTTR-100', 'TTR-100-sliding', 'Compressibility', 'Flesch Reading Ease', 'Mean sentence length', 'Mean word length']]

# === Step 1: Cosine Distance Matrix ===
features = df.values
distance_matrix = cosine_distances(features)
distance_df = pd.DataFrame(distance_matrix, index=df.index, columns=df.index)
distance_df.to_csv("distance_matrix.csv")

# === Step 2: Add Canon scores ===
with open("Canon_tag.json") as f:
    canon_dict = json.load(f)
df["Canon"] = df.index.map(lambda x: canon_dict.get(x, 0))

# === Step 3: Build Graph ===
threshold = 0.9  # similarity threshold (i.e., 1 - cosine distance < 0.1)
G = nx.Graph()
for i, doc_i in enumerate(df.index):
    G.add_node(doc_i)
    for j in range(i + 1, len(df.index)):
        doc_j = df.index[j]
        similarity = 1 - distance_matrix[i][j]
        if similarity > threshold:
            G.add_edge(doc_i, doc_j, weight=similarity)

# === Step 4: Leiden Communities ===
g_ig = ig.Graph.TupleList(G.edges(), directed=False)
partition = find_partition(g_ig, partition_type=ModularityVertexPartition)
#leiden_communities = {df.index[i]: int(partition.membership[i]) for i in range(len(df.index))}
leiden_communities = {v["name"]: int(partition.membership[i]) for i, v in enumerate(g_ig.vs)}
df["Community"] = df.index.map(leiden_communities.get)

# === Step 5: Graph Metrics ===
betweenness = nx.betweenness_centrality(G)
degree_centrality = nx.degree_centrality(G)
pagerank = nx.pagerank(G)

df["Betweenness"] = df.index.map(betweenness)
df["DegreeCentrality"] = df.index.map(degree_centrality)
df["PageRank"] = df.index.map(pagerank)

# === Step 6: Innovative Graph Path Metric ===
# For each node: mean shortest path length to top-5% PageRank nodes in its community
pr_threshold = np.percentile(list(pagerank.values()), 95)
top_nodes = [n for n, pr in pagerank.items() if pr >= pr_threshold]
shortest_paths = dict(nx.all_pairs_shortest_path_length(G))

def mean_path_to_top(node):
    top_in_community = [n for n in top_nodes if leiden_communities[n] == leiden_communities[node]]
    lengths = [shortest_paths[node][n] for n in top_in_community if n in shortest_paths[node]]
    return np.mean(lengths) if lengths else np.nan

df["MeanPathToTop"] = df.index.map(mean_path_to_top)

# === Step 7: Correlations with Canon ===
metrics = ["Betweenness", "DegreeCentrality", "PageRank", "MeanPathToTop"]
correlations = {}
for metric in metrics:
    valid = df[["Canon", metric]].dropna()
    corr, _ = pearsonr(valid["Canon"], valid[metric])
    correlations[metric] = corr

print("Correlation with Canon:")
with open("correlations_with_canon.txt", "w") as f:
    for k, v in correlations.items():
        line = f"{k}: {v:.3f}\n"
        print(line, end="")
        f.write(line)

# === Optional: Plotting with Plotly ===
pos = nx.spring_layout(G, seed=42)
x_vals = [pos[n][0] for n in G.nodes()]
y_vals = [pos[n][1] for n in G.nodes()]

edge_trace = go.Scatter(
    x=[], y=[],
    line=dict(width=0.5, color="#888"),
    hoverinfo='none',
    mode='lines')

for edge in G.edges():
    x0, y0 = pos[edge[0]]
    x1, y1 = pos[edge[1]]
    edge_trace["x"] += [x0, x1, None]
    edge_trace["y"] += [y0, y1, None]

node_trace = go.Scatter(
    x=x_vals, y=y_vals,
    mode='markers',
    marker=dict(
        showscale=True,
        colorscale='Viridis',
        color=list(df.loc[G.nodes(), "Canon"]),
        size=10,
        colorbar=dict(
            thickness=15,
            title='Canon Score',
            xanchor='left',
            titleside='right'
        )
    ),
    text=[f"{n}" for n in G.nodes()],
    hoverinfo='text')

fig = go.Figure(data=[edge_trace, node_trace],
                layout=go.Layout(
                    title='Graph of Novel Similarity',
                    showlegend=False,
                    hovermode='closest',
                    margin=dict(b=20, l=5, r=5, t=40)))

fig.write_html("graph_plot.html")
