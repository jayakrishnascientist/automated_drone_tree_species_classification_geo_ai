
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

def run_clustering(feature_path, csv_path, k_range):

    features = np.load(feature_path)
    df_names = pd.read_csv(csv_path)

    scaler = StandardScaler()
    X = scaler.fit_transform(features)

    X = PCA(n_components=50, random_state=42).fit_transform(X)

    scores = []

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X)
        score = silhouette_score(X, labels)
        scores.append(score)
        print("k:", k, "score:", score)

    best_k = k_range[np.argmax(scores)]
    print("Best k:", best_k)

    return best_k
