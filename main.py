import pandas as pd, numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import kneighbors_graph
from sklearn.cluster import AgglomerativeClustering

def run(file, n_clusters, k=10):
    df = pd.read_csv(file)
    X  = StandardScaler().fit_transform(np.log1p(df.drop(columns=['id']).values))
    conn = kneighbors_graph(X, n_neighbors=k, include_self=True, n_jobs=-1)
    labels = AgglomerativeClustering(
                n_clusters=n_clusters, linkage="ward",
                connectivity=conn).fit_predict(X)
    return pd.DataFrame({'id': df.id, 'label': labels})

run("public_data.csv", 15).to_csv("public_submission.csv",  index=False)
run("private_data.csv", 23).to_csv("private_submission.csv", index=False)

