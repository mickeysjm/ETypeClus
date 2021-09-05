import argparse
import pickle as pk
from collections import defaultdict
from tqdm import tqdm
import math
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline
from sklearn.cluster import KMeans
from nltk.stem import WordNetLemmatizer
import torch

lemmatizer = WordNetLemmatizer()


def tfidf(item_list, item2features, item2freq):
    item_doc = []
    for item in item_list:
        features = item2features[item]
        doc = ""
        for k, v in features.items():
            cnt = math.ceil(v * item2freq[item])
            doc += ((k+" ")*cnt)
        item_doc.append(doc)
    
    vectorizer = TfidfVectorizer(sublinear_tf=True)
    X = vectorizer.fit_transform(item_doc)
    return X


def pca(X, pca_dim=50):
    pca_dim = min(pca_dim, X.shape[1]-1)
    svd = TruncatedSVD(pca_dim)
    normalizer = Normalizer(copy=False)
    lsa = make_pipeline(svd, normalizer)
    X = lsa.fit_transform(X)
    explained_variance = svd.explained_variance_ratio_.sum()
    print("Explained variance of the SVD step: {}%".format(int(explained_variance * 100)))
    return X


def tfidf_pca(item_list, item2features, item2freq, pca_dim=50):
    X_init = tfidf(item_list, item2features, item2freq)
    X = pca(X_init, pca_dim=pca_dim)
    return X


def kmean_clustering_w_init_feature(item_list, X, n_clusters=10):
    km = KMeans(n_clusters=n_clusters, init='k-means++', max_iter=100, n_init=5, random_state=42)
    km.fit(X)
    cluster_id2items = {}
    item2center_distance = km.transform(X)
    for row_id, item in enumerate(item_list):
        cluster_id = km.labels_[row_id]
        if cluster_id not in cluster_id2items:
            cluster_id2items[cluster_id] = {}
        distance = item2center_distance[row_id, cluster_id]
        cluster_id2items[cluster_id][item] = distance 
    return km, cluster_id2items


def main(mention_file, sense_mapping, save_file, use_all_svos, pca_dim):

    print("=== Loading Data ===") 
    with open(mention_file, "rb") as f:
        po_mentions = pk.load(f)
    with open(sense_mapping, "rb") as f:
        svo_id2sense_id = pk.load(f)

    print("=== Collecting P-O Mention Features ===")
    sense2embed_list = defaultdict(list)
    sense2expd_lemma_w_weight = {}
    sense2freq = defaultdict(int)
    obj_head2embed_list = defaultdict(list)
    obj_head2expd_lemma_w_weight = {}
    obj_head2freq = defaultdict(int)
    processed_svo_cnt = 0
    for svo_id, svo_info in tqdm(po_mentions.items()):
        if not use_all_svos and (not svo_info['all_salient_flag']):
            continue
        verb = svo_info['verb']
        verb_embed = svo_info['verb_embed']
        verb_expd_results = svo_info['verb_expansion_results']
        obj_head = svo_info['obj_head']
        obj_head_embed = svo_info['obj_head_embed']
        obj_head_expd_results = svo_info['obj_head_expansion_results']

        # process verb     
        if verb_embed is not None:  # not salient verb
            expd_lemma2weight = defaultdict(float)
            for ele in verb_expd_results:
                expd_lemma = lemmatizer.lemmatize(ele[0], 'v')
                expd_lemma2weight[expd_lemma] += ele[1]
            expd_lemma2weight = dict(expd_lemma2weight)

            sense_id = svo_id2sense_id[svo_id]
            sense = f"{verb}_{sense_id}"
            sense2embed_list[sense].append(verb_embed)
            sense2freq[sense] += 1
            if sense not in sense2expd_lemma_w_weight:
                sense2expd_lemma_w_weight[sense] = defaultdict(float)
            for expd_lemma, weight in expd_lemma2weight.items():
                sense2expd_lemma_w_weight[sense][expd_lemma] += weight
                
        # process obj (head)
        if obj_head_embed is not None:  # not salient object head
            expd_lemma2weight = defaultdict(float)
            for ele in obj_head_expd_results:
                expd_lemma = lemmatizer.lemmatize(ele[0])
                expd_lemma2weight[expd_lemma] += ele[1]
            expd_lemma2weight = dict(expd_lemma2weight)
            
            obj_head2embed_list[obj_head].append(obj_head_embed)
            obj_head2freq[obj_head] += 1
            if obj_head not in obj_head2expd_lemma_w_weight:
                obj_head2expd_lemma_w_weight[obj_head] = defaultdict(float)
            for expd_lemma, weight in expd_lemma2weight.items():
                obj_head2expd_lemma_w_weight[obj_head][expd_lemma] += weight

        processed_svo_cnt += 1  
    
    print(f"Processed {processed_svo_cnt} SVO triplets")
    sense2freq = dict(sense2freq)
    sense2embed = {k:np.average(v, axis=0) for k, v in sense2embed_list.items() if k in sense2embed_list}
    sense2expd_lemma_w_weight = {k: dict(v) for k, v in sense2expd_lemma_w_weight.items() if k in sense2expd_lemma_w_weight}
    obj_head2freq = dict(obj_head2freq)
    obj_head2embed = {k:np.average(v, axis=0) for k, v in obj_head2embed_list.items() if k in obj_head2freq}
    obj_head2expd_lemma_w_weight = {k: dict(v) for k, v in obj_head2expd_lemma_w_weight.items() if k in obj_head2expd_lemma_w_weight}

    print("=== Getting Seperate Features for Predicate Senses and Object Heads ===")
    selected_vs_list = list(sense2freq.keys())
    inv_vs_vocab = {vs:i for i, vs in enumerate(selected_vs_list)}
    selected_oh_list = list(obj_head2freq.keys())
    inv_oh_vocab = {oh:i for i, oh in enumerate(selected_oh_list)}

    selected_vs_expd_lemma_w_weight = {vs:sense2expd_lemma_w_weight[vs] for vs in selected_vs_list}
    print("Get verb sense context expansion embedding")
    vs_context_embed = tfidf_pca(selected_vs_list, selected_vs_expd_lemma_w_weight, sense2freq, pca_dim=pca_dim)
    print("Get verb sense BERT embedding")
    selected_vs_bert_embed = np.array([sense2embed[vs] for vs in selected_vs_list])
    vs_bert_embed = pca(selected_vs_bert_embed, pca_dim=pca_dim)
    print("Merge and get verb sense final embedding")
    # vs_final_embed = pca(np.concatenate([vs_bert_embed, vs_context_embed], axis=1), math.ceil(pca_dim*1.5))   
    vs_final_embed = np.concatenate([vs_bert_embed, vs_context_embed], axis=1)

    selected_oh_expd_lemma_w_weight = {oh:obj_head2expd_lemma_w_weight[oh] for oh in selected_oh_list}
    print("Get object head context expansion embedding")
    oh_context_embed = tfidf_pca(selected_oh_list, selected_oh_expd_lemma_w_weight, obj_head2freq, pca_dim=pca_dim)
    print("Get object head BERT embedding")
    selected_oh_bert_embed = np.array([obj_head2embed[oh] for oh in selected_oh_list])
    oh_bert_embed = pca(selected_oh_bert_embed, pca_dim=pca_dim)
    print("Merge and get verb sense final embedding")
    # oh_final_embed = pca(np.concatenate([oh_bert_embed, oh_context_embed], axis=1), math.ceil(pca_dim*1.5))   
    oh_final_embed = np.concatenate([oh_bert_embed, oh_context_embed], axis=1)

    print("=== Getting P-O Tuple Features ===")
    vocab = {}
    inv_vocab = {}
    vs_emb = []
    oh_emb = []
    vs_w_oh2freq = defaultdict(int)
    for svo_id, svo_info in tqdm(po_mentions.items()):
        sense_id = svo_id2sense_id[svo_id]
        vs = f"{svo_info['verb']}_{sense_id}"
        oh = svo_info['obj_head']
        if vs not in selected_vs_list or oh not in selected_oh_list:
            continue
        vs_oh_tuple = (vs, oh)
        if vs_oh_tuple not in vocab:
            index = len(vocab)
            vocab[vs_oh_tuple] = index
            inv_vocab[index] = vs_oh_tuple
            vs_emb.append(vs_final_embed[inv_vs_vocab[vs],:])
            oh_emb.append(oh_final_embed[inv_oh_vocab[oh],:])

        vs_w_oh2freq[vs_oh_tuple] += 1
        
    vs_w_oh2freq = dict(vs_w_oh2freq)
    vs_emb = np.array(vs_emb).astype(np.float32)
    oh_emb = np.array(oh_emb).astype(np.float32)
    tuple_freq = []
    for i in range(len(inv_vocab)):
        tuple_freq.append(vs_w_oh2freq[inv_vocab[i]])
    print(f"Total number of distinct (verb sense, object head) pairs: {len(vocab)}")

    with open(save_file, "wb") as f:
        pk.dump({
            "vocab": vocab,
            "inv_vocab": inv_vocab,
            "vs_emb": vs_emb,
            "oh_emb": oh_emb,
            "tuple_freq": tuple_freq,
        }, f)
 
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--mention_file", help="input <predicate, object head> mentions pickle file")
    parser.add_argument("--sense_mapping", help="input predicate sense disambiguation result file")
    parser.add_argument("--save_file", help="output file name")
    parser.add_argument('--use_all_svos', default=False, action='store_true', \
        help="""whether to use all po mentions (i.e., those contain either a salient predicate or an objct head 
                or a strict set of po mentions (i.e., those contain both a salient predicate and an object head)"""
    )
    parser.add_argument('--pca_dim', default=500, help="reduced dimensionality")

    args = parser.parse_args()
    print(vars(args))
    main(args.mention_file, args.sense_mapping, args.save_file, args.use_all_svos, args.pca_dim)
