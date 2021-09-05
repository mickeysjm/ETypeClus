import pickle as pk
import numpy as np
from sklearn.cluster import SpectralClustering
from tqdm import tqdm
from itertools import product
from sklearn.metrics.pairwise import cosine_similarity as cos
from sklearn.metrics.pairwise import euclidean_distances as euc
import json
import argparse
from collections import defaultdict
from sklearn.cluster import KMeans


if __name__ == '__main__':
    
    # python baseline-kmeans.py --input pandemic/po_tuple_features_all_svos.pk --output kmeans_test.json --k 30

    parser = argparse.ArgumentParser()

    parser.add_argument("--input", help='Path to input data dictionary')
    parser.add_argument("--output", help='Path to output file')
    parser.add_argument("--k", help='Number of clusters', type=int)
    
    args = parser.parse_args()
    
    data_dict = pk.load(open(args.input, 'rb'))
    
    X = np.concatenate((data_dict['vs_emb'], data_dict['oh_emb']), axis=1)
    
    kmeans = KMeans(n_clusters=args.k).fit(X)
    centers = kmeans.cluster_centers_
    predicted = kmeans.labels_
    
    mention_clusters = [set() for _ in range(100)]
    mention_clusters_tup_id = [[] for _ in range(100)]
    for i, clus_num in enumerate(predicted):
        tup = data_dict['inv_vocab'][i]
        if tup not in mention_clusters[clus_num]:
            mention_clusters[clus_num].add(tup)
            mention_clusters_tup_id[clus_num].append(i)
    kmeans_results = []
    for tup_id, center in zip(mention_clusters_tup_id, centers):
        dist = euc(center[np.newaxis, :], X[tup_id])[0]
        ranked = [data_dict['inv_vocab'][tup_id[i]] for i in np.argsort(dist)]
        kmeans_results.append(ranked[:10])
    
    
    json.dump(kmeans_results, open(args.output, 'w'), indent=4)