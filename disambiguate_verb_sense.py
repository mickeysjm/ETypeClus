import argparse
import pickle as pk
import json
import rbo
import numpy as np
from nltk.stem import WordNetLemmatizer
from collections import defaultdict
import math
from tqdm import tqdm

lemmatizer = WordNetLemmatizer()


def cosine_similarity_embedding(emb_a, emb_b):
    return np.dot(emb_a, emb_b) / np.linalg.norm(emb_a) / np.linalg.norm(emb_b)


def aggregate_sense_feature_by_ranking(bert_expanded_verbnet):
    """ Return lemma -> sense with features
    {
        lemma 1: {
            sense1: [{'expd_lemma1': score, 'expd_lemma2': score, ...}, sense_embed],
            sense2: [{'expd_lemma1': score, 'expd_lemma2': score, ...}, sense_embed],
        },
        lemma 2: {
            sense1: [{'expd_lemma1': score, 'expd_lemma2': score, ...}, sense_embed],
            sense2: [{'expd_lemma1': score, 'expd_lemma2': score, ...}, sense_embed],
        }
    }
    """
    lemmatizer = WordNetLemmatizer()
    lemma2sense_features = {}
    for lemma, info in tqdm(bert_expanded_verbnet.items(), desc="generate verb sense features"):
        lemma2sense_features[lemma] = {}
        for sense_id, sense in enumerate(info['senses']):
            sense_feature = defaultdict(float)
            sense_embed = sense['sense_embed']
            for example_w_expansion in sense['examples_w_expansion']:
                for ele in example_w_expansion[1]:
                    expd_results = ele['expansion_results']
                    expd_lemma2rank = {}
                    for expd_lemma_rank, term in enumerate(expd_results):
                        expd_lemma = lemmatizer.lemmatize(term['token_str'], 'v')
                        if expd_lemma not in expd_lemma2rank:
                            expd_lemma2rank[expd_lemma] = 1+expd_lemma_rank
                        
                    for expd_lemma, expd_lemma_rank in expd_lemma2rank.items():
                        sense_feature[expd_lemma] += 1.0/math.log(1+expd_lemma_rank)
            sense_feature = dict(sense_feature)
            sorted_features = {ele[0]:ele[1] for ele in sorted(sense_feature.items(), key=lambda x:-x[1])}
            lemma2sense_features[lemma][sense_id] = [sorted_features, sense_embed]
    return lemma2sense_features


def disambiguate_word_sense(verb_info, lemma2sense_features, options):
    """
    verb_info corresponds to one verb mention and is represented as a dict {
        'verb': str,
        'verb_embed': str,
        'verb_expansion_results': List[{'token_str': str, 'score': float}]
    }
    options is a dictionary of potential wsd parameters
    """
    sense_top_k = options.get("sense_top_k", -1)

    lemma = verb_info['verb']
    expd_results = verb_info['verb_expansion_results']
    lemma_embed = verb_info['verb_embed']
    if lemma_embed is None:  # not salient predicate mention thus no embedding feature
        return -2
    if lemma not in lemma2sense_features:  # lemma does not appear in the dictionary
        return -1
    if len(lemma2sense_features[lemma]) == 1:  # only one sense, nothing to disambiguate
        return 0
    
    # get verb mention features
    expd_lemma2score = defaultdict(float)
    for rank, ele in enumerate(expd_results):
        expd_lemma = lemmatizer.lemmatize(ele[0], 'v')
        expd_lemma2score[expd_lemma] += (1.0/math.log(2+rank))
    expd_lemma2score = dict(expd_lemma2score)
    sorted_expd_lemma = [ele[0] for ele in sorted(expd_lemma2score.items(), key=lambda x:-x[1])]
    
    # start disambiguation
    sense_id2rbo_score = {}
    sense_id2embed_score = {}
    for sense_id, sense_feature in lemma2sense_features[lemma].items():
        ranked_sense_feature = list(sense_feature[0].keys())[:sense_top_k]
        rbo_score = rbo.RankingSimilarity(sorted_expd_lemma, ranked_sense_feature).rbo()
        sense_id2rbo_score[sense_id] = rbo_score

        sense_embed = sense_feature[1]
        if sense_embed is None or lemma_embed is None:
            embed_score = 0.0
        else:
            embed_score = cosine_similarity_embedding(sense_embed, lemma_embed)
        sense_id2embed_score[sense_id] = embed_score

    sense_id2final_score = {
        sense_id: sense_id2rbo_score[sense_id]*embed_score 
        for sense_id, embed_score in sense_id2embed_score.items()
    }
    sorted_senses = sorted(sense_id2final_score.items(), key=lambda x:-x[1])
    return sorted_senses[0][0]


def main(mention_file, save_path, dict_file):
    
    with open(dict_file, "r") as read_file:
        verb_sense_dict = json.load(read_file)
    
    with open(mention_file, "rb") as f:
        parsed_data = pk.load(f)

    lemma2sense_features = aggregate_sense_feature_by_ranking(verb_sense_dict)

    parsed_data_wsd = {}
    options = {}
    options['sense_top_k'] = 10
    for svo_id, svo in tqdm(parsed_data.items(), desc="disambiguate svo verbs"):
        sense_id = disambiguate_word_sense(svo, lemma2sense_features, options)
        parsed_data_wsd[svo_id] = sense_id

    with open(save_path, "wb") as f:
        pk.dump(parsed_data_wsd, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--mention_file", help="input <predicate, object head> mentions pickle file")
    parser.add_argument("--save_path", help="output file path")
    parser.add_argument("--dict_file", default="./resources/verb_sense_dict_w_features.json", \
        help="input file for verb sense dictionary")
    args = parser.parse_args()
    print(vars(args))
    main(args.mention_file, args.save_path, args.dict_file)
