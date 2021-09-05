"""
Please run parse_corpus_and_extract_svo.py first to obtain the .pk sentence files before running this code.
"""
import argparse
import json
import math
import os
import pickle as pk
import random
import re

import torch
from transformers import BertForMaskedLM, BertTokenizer, pipeline
from tqdm import tqdm
import numpy as np


MODELS = {
    'blu': (BertForMaskedLM, BertTokenizer, 'bert-large-uncased-whole-word-masking'),
}

def tensor_to_numpy(tensor):
    return tensor.clone().detach().cpu().numpy()


def prepare_sentence(token_list, tokenizer):
    """
    Inputs:
        token_list: a list of tokens
        tokenizer: a TokenizerClass object from HuggingFace
    Return:
        tokenized_text: A list of tokens obtained from basic tokenizer (e.g., WhiteSpaceTokenizer)
        tokenized_to_id_indicies: A list of (tokenids_chunks_index, token_id_start_index, token_id_end_index)
        tokenids_chunks: A list of token_id_end_index
    """
    # setting for BERT
    model_max_tokens = 512
    has_sos_eos = True
    ############## ########
    max_tokens = model_max_tokens
    if has_sos_eos:
        max_tokens -= 2
    sliding_window_size = max_tokens // 2

    if not hasattr(prepare_sentence, "sos_id"):
        prepare_sentence.sos_id, prepare_sentence.eos_id = tokenizer.encode("", add_special_tokens=True)

    # tokenized_text = tokenizer.basic_tokenizer.tokenize(text, never_split=tokenizer.all_special_tokens)
    tokenized_text = token_list
    tokenized_to_id_indicies = []

    tokenids_chunks = []  # useful only if the sentence is longer than max_tokens
    tokenids_chunk = []

    for index, token in enumerate(tokenized_text + [None]):
        if token is not None:
            tokens = tokenizer.wordpiece_tokenizer.tokenize(token)
        if token is None or len(tokenids_chunk) + len(tokens) > max_tokens:
            tokenids_chunks.append([prepare_sentence.sos_id] + tokenids_chunk + [prepare_sentence.eos_id])
            if sliding_window_size > 0:
                tokenids_chunk = tokenids_chunk[-sliding_window_size:]
            else:
                tokenids_chunk = []
        if token is not None:
            tokenized_to_id_indicies.append((len(tokenids_chunks),
                                             len(tokenids_chunk),
                                             len(tokenids_chunk) + len(tokens)))
            tokenids_chunk.extend(tokenizer.convert_tokens_to_ids(tokens))

    return tokenized_text, tokenized_to_id_indicies, tokenids_chunks


def sentence_encode(tokens_id, model, layer):
    input_ids = torch.tensor([tokens_id], device=model.device)

    with torch.no_grad():
        last_hidden_states = model(input_ids).last_hidden_state
        
    layer_embedding = tensor_to_numpy(last_hidden_states.squeeze(0))[1: -1]
    return layer_embedding


def sentence_to_wordtoken_embeddings(layer_embeddings, tokenized_text, tokenized_to_id_indicies):
    word_embeddings = []
    for text, (chunk_index, start_index, end_index) in zip(tokenized_text, tokenized_to_id_indicies):
        word_embeddings.append(np.average(layer_embeddings[chunk_index][start_index: end_index], axis=0))
    assert len(word_embeddings) == len(tokenized_text)
    return np.array(word_embeddings)


def handle_sentence(model, layer, tokenized_text, tokenized_to_id_indicies, tokenids_chunks):
    layer_embeddings = [
        sentence_encode(tokenids_chunk, model, layer) for tokenids_chunk in tokenids_chunks
    ]
    word_embeddings = sentence_to_wordtoken_embeddings(layer_embeddings,
                                                       tokenized_text,
                                                       tokenized_to_id_indicies)
    return word_embeddings

def process_sentence(token_list, tokenizer, model, layer):
    tokenized_text, tokenized_to_id_indicies, tokenids_chunks = prepare_sentence(token_list, tokenizer)
    contextualized_word_representations = handle_sentence(
        model, layer, tokenized_text, tokenized_to_id_indicies, tokenids_chunks)
    return contextualized_word_representations
    
def predict_masked_words(sentence_w_mask, model, tokenizer, top_k=50):
    """
    sentence_w_mask: `str`, a single sentence with [MASK] token
    model: a BertForMaskedLM model
    """
    inputs = tokenizer(sentence_w_mask, return_tensors="pt").to(model.device)
    masked_position_indice = torch.where(inputs.input_ids[0] == tokenizer.mask_token_id)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits[0], axis=1)

    word_prob, word_index = torch.topk(probs[masked_position_indice], top_k, dim=-1)
    word_prob = tensor_to_numpy(word_prob)
    word_index = tensor_to_numpy(word_index)

    predicted_words = []
    for i in range(len(word_prob)):    
        pred_word = []
        for r, j in enumerate(word_index[i]):
            pred_word.append([tokenizer.ids_to_tokens[j], float(word_prob[i][r])])
        predicted_words.append(pred_word)
    return predicted_words

def main(input_file,
         lm_type,
         layer,
         top_k,
         gpu_id):

    print("loading Transformer model")
    model_class, tokenizer_class, pretrained_weights = MODELS[lm_type]
    need_lower = (lm_type.endswith("u"))

    tokenizer = tokenizer_class.from_pretrained(pretrained_weights)

    # get model for mlm prediction
    mlm_model = model_class.from_pretrained(pretrained_weights)
    mlm_model.eval()
    mlm_model = mlm_model.to(torch.device(f'cuda:{gpu_id}'))

    # get model for embedding extraction
    model = mlm_model.bert

    print("loading corpus")
    with open(input_file, "rb") as f:
        parsed_data = pk.load(f)

    save_dict_data = {}
    for sent_id in tqdm(parsed_data):
        sent_info = parsed_data[sent_id]
        if len(sent_info['svos']) == 0:
            continue

        token_list = sent_info['token_list']
        if need_lower:
            token_list = [token.lower() for token in token_list]
        token_embeds = process_sentence(token_list, tokenizer, model, layer)
        for svo_id, svo in enumerate(sent_info['svos']):
            svo_index = f"{sent_id}_{svo_id}"
            if svo[0] is not None:
                subj = svo[0][0]
                subj_range = svo[0][1]
                subj_embed = np.average(token_embeds[subj_range,:], axis=0)
                subj_masked_sent = " ".join(token_list[:subj_range[0]] + ["[MASK]"] + token_list[subj_range[-1]+1:])
                subj_expand_res = predict_masked_words(subj_masked_sent, mlm_model, tokenizer, top_k)[0]
            else:
                subj = None
                subj_embed = None
                subj_expand_res = []
            
            verb = svo[1][0]
            verb_index = svo[1][1]
            verb_embed = token_embeds[verb_index]
            verb_masked_sent = " ".join(token_list[:verb_index] + ["[MASK]"] + token_list[verb_index+1:])
            verb_expand_res = predict_masked_words(verb_masked_sent, mlm_model, tokenizer, top_k)[0]
            
            if svo[2] is not None:
                obj = svo[2][0]
                obj_range = svo[2][1]
                obj_embed = np.average(token_embeds[obj_range,:], axis=0)
                obj_masked_sent = " ".join(token_list[:obj_range[0]] + ["[MASK]"] + token_list[obj_range[-1]+1:])
                obj_expand_res = predict_masked_words(obj_masked_sent, mlm_model, tokenizer, top_k)[0]
            else:
                obj = None
                obj_embed = None
                obj_expand_res = []

            save_dict_data[svo_index] = {
                "subj": subj,
                "subj_embed": subj_embed,
                "subj_expansion_results": subj_expand_res,
                "verb": verb,
                "verb_embed": verb_embed,
                "verb_expansion_results": verb_expand_res,
                "obj": obj,
                "obj_embed": obj_embed,
                "obj_expansion_results": obj_expand_res,
            }

    save_path = f"{input_file[:-3]}_{lm_type}_expanded_l{layer}_topk_{top_k}_embeded.pk"
    with open(save_path, "wb") as f:
        pk.dump(save_dict_data, f)


if __name__ == '__main__':
    # python get_bert_features.py --gpu_id 7
    # python get_bert_features.py --input_file ./2659docs_cleaned_parsed_svo_0415.pk --gpu_id 2
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_file", default="./20docs_cleaned_parsed_svo_0415.pk")
    parser.add_argument("--lm_type", default="blu", help="language model name")
    parser.add_argument("--lm_layer", default=-1, help="language model layer for features")
    parser.add_argument("--top_k", default=10, type=int, help="top_k expansion results")
    parser.add_argument("--gpu_id", default=0, type=int, help="gpu id for bert model")
    args = parser.parse_args()
    print(vars(args))
    main(args.input_file, args.lm_type, args.lm_layer, args.top_k, args.gpu_id)
