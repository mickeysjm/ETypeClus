import pickle as pk
import numpy as np
from tqdm import tqdm
from itertools import product
import json
import argparse
from collections import defaultdict
import os

SUBJ = '[SUBJ]'

def save_data_for_triframes(data_dict):
    sense_emb = []
    obj_head_emb = []
    vs_vocab = {}
    oh_vocab = {}
    vs_inv_vocab = []
    oh_inv_vocab = []
    tup2count = defaultdict(int)
    for t, i in data_dict['vocab'].items():
        vs, oh = t
        if vs not in vs_vocab:
            vs_vocab[vs] = len(vs_vocab)
            vs_inv_vocab.append(vs)
            sense_emb.append(data_dict['vs_emb'][i])
        if oh not in oh_vocab:
            oh_vocab[oh] = len(oh_vocab)
            oh_inv_vocab.append(oh)
            obj_head_emb.append(data_dict['oh_emb'][i])
        tup2count[(vs, oh)] = data_dict['tuple_freq'][i]
    sense_emb = np.vstack(sense_emb)
    obj_head_emb = np.vstack(obj_head_emb)
    
    with open('triframes/triplets.tsv', 'w') as f:
        for t, i in data_dict['vocab'].items():
            vs, oh = t
            print(f'{vs}\t{SUBJ}\t{oh}\t1.0', file=f)

    from gensim.models.keyedvectors import KeyedVectors

    dim = sense_emb.shape[1]
    kv = KeyedVectors(dim)
    kv.add(vs_inv_vocab, sense_emb)
    kv.add(oh_inv_vocab, obj_head_emb)
    kv.add([SUBJ], [np.ones(dim)*1e-8])
    kv.save_word2vec_format('triframes/w2v.bin', binary=True)
    
    return tup2count

def triframes_call(N):
    os.chdir("./triframes")
    os.system(f'N={N} WEIGHT=0 W2V=w2v.bin VSO=triplets.tsv make triw2v-watset.txt')
    os.chdir('..')
        
def load_triframes(data_dict):
    p2o2c = defaultdict(lambda:defaultdict(int))
    with open('triframes/triw2v-watset.txt') as f:
        curr_c = None
        preds = None
        objs = None
        for line in f:
            if line.startswith('#'):
                curr_c = int(line.strip().split(' ')[-1]) - 1
                preds = None
                objs = None
            elif line.startswith('Predicates'):
                preds = line[len('Predicates: '):].strip().split(', ')
            elif line.startswith('Objects'):
                objs = line[len('Objects: '):].strip().split(', ')
                for p, o in product(preds, objs):
                    p2o2c[p][o] = curr_c
    triframe_predicted = []
    for t, i in data_dict['vocab'].items():
        vs, oh = t
        pred = p2o2c[vs][oh]
        triframe_predicted.append(pred)
    return triframe_predicted, curr_c+1



if __name__ == '__main__':
    
    # python baseline-triframes.py --input pandemic/po_tuple_features_all_svos.pk --output triframes_test.json --N 100 --min_size 100

    parser = argparse.ArgumentParser()

    parser.add_argument("--input", help='Path to input data dictionary')
    parser.add_argument("--output", help='Path to output file')
    parser.add_argument("--N", help='N for Triframes', type=int, default=100)
    parser.add_argument("--min_size", help='Minimum size of clusters', type=int, default=0)
    
    args = parser.parse_args()
    
    data_dict = pk.load(open(args.input, 'rb'))
    
    tup2count = save_data_for_triframes(data_dict)
    
    triframes_call(args.N)
    
    predicted, num_clus = load_triframes(data_dict)
#     print(len(predicted))
    print(f'Total number of clusters: {num_clus}')
    
    tuple_clusters = [{} for _ in range(num_clus)]
    for i, clus_num in enumerate(predicted):
        tup = data_dict['inv_vocab'][i]
        tuple_clusters[clus_num][tup] = tup2count[tup]

    results = [sorted(tuple_cluster.keys(), key=lambda x: tuple_cluster[x], reverse=True)[:10] for tuple_cluster in tuple_clusters if len(tuple_cluster) >= args.min_size]
    
    print(f'Number of clusters: {len(results)}')
    
    json.dump(results, open(args.output, 'w'), indent=4)