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
from spherecluster import SphericalKMeans


def count_tup(data_dict):
    tup2count = defaultdict(int)
    for t, i in data_dict['vocab'].items():
        vs, oh = t
        tup2count[(vs, oh)] = data_dict['tuple_freq'][i]
    return tup2count


if __name__ == '__main__':
    
    # python baseline-spkmeans.py --input pandemic/po_tuple_features_all_svos.pk --output spkmeans_test.json --k 30

    parser = argparse.ArgumentParser()

    parser.add_argument("--input", help='Path to input data dictionary')
    parser.add_argument("--output", help='Path to output file')
    parser.add_argument("--k", help='Number of clusters', type=int)
    
    args = parser.parse_args()
    
    data_dict = pk.load(open(args.input, 'rb'))
    
    tup2count = count_tup(data_dict)
    
    X = np.concatenate((data_dict['vs_emb'], data_dict['oh_emb']), axis=1)
    
    predicted = SphericalKMeans(n_clusters=args.k).fit_predict(X)
    
    tuple_clusters = [{} for _ in range(args.k)]
    for i, clus_num in enumerate(predicted):
        tup = data_dict['inv_vocab'][i]
        tuple_clusters[clus_num][tup] = tup2count[tup]

    results = [sorted(tuple_cluster.keys(), key=lambda x: tuple_cluster[x], reverse=True)[:10] for tuple_cluster in tuple_clusters]
    
    
    json.dump(results, open(args.output, 'w'), indent=4)