import argparse
import json

import spacy
import numpy as np
import torch
from transformers import BertForMaskedLM, BertTokenizer
from tqdm import tqdm
from get_bert_features import process_sentence, predict_masked_words

MODELS = {
    'blu': (BertForMaskedLM, BertTokenizer, 'bert-large-uncased-whole-word-masking'),
}

def main(input_file,
         spacy_model,
         lm_type,
         top_k,
         gpu_id,
         save_path):

    print("loading Spacy model")
    nlp = spacy.load(spacy_model)

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
    
    with open(input_file, "r") as read_file:
        ontonotes = json.load(read_file)

    bert_expanded_ontonotes = {}
    not_matched_example_cnt = 0
    not_matched_sentences = []
    example_cnt = 0
    for ele in tqdm(ontonotes):
        lemma = ele['lemma'][:-2]
        senses = ele['senses']
        new_senses = []
        for sense_info in senses:
            sense_def = sense_info['definition']
            sense_examples = sense_info['examples']
            sentence_w_expd = []
            embed_array = []
            for sent in sense_examples:
                doc = nlp(sent)
                token_list = [token.text for token in doc]
                if need_lower:
                    token_list = [tok.lower() for tok in token_list]
                token_embeds = process_sentence(token_list, tokenizer, model, -1)
                verb_info = []
                for token in doc:
                    if token.lemma_ == lemma:
                        verb_masked_sent = " ".join(token_list[:token.i]+["[MASK]"]+token_list[token.i+1:])
                        res = predict_masked_words(verb_masked_sent, mlm_model, tokenizer, top_k)[0]
                        expand_res = [{
                            "token_str": ele[0],
                            "score": ele[1],
                        } for ele in res]
                        verb_info.append({
                            "text": token.text,
                            "lemma": token.lemma_,
                            "index": token.i,
                            "expansion_results": expand_res,
                        })
                        embed_array.append(token_embeds[token.i])
                example_cnt += 1
                if len(verb_info) == 0:
                    not_matched_example_cnt += 1
                    not_matched_sentences.append([lemma, sent])
                sentence_w_expd.append([sent, verb_info])
            if len(embed_array) > 0:
                sense_embed = np.average(np.array(embed_array), axis=0).tolist()
            else:
                sense_embed = None

            new_senses.append({
                'definition': sense_def,
                'examples_w_expansion': sentence_w_expd,
                'sense_embed': sense_embed,
            })
        bert_expanded_ontonotes[lemma] = {
            'lemma': lemma,
            'senses': new_senses
        }

    print(f"Total example sentences: {example_cnt}")
    print(f"Unmatched example sentences: {not_matched_example_cnt}")
    print(f"Ratio: {1.0*not_matched_example_cnt/example_cnt}")

    with open(save_path, "w") as write_file:
        json.dump(bert_expanded_ontonotes, write_file, indent=4, sort_keys=True, separators=(',', ': '))

    with open(save_path[:-5]+".not_matched.json", "w") as write_file:
        json.dump(not_matched_sentences, write_file, indent=4, sort_keys=True, separators=(',', ': '))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", default="./resources/verb_sense_dict.json")
    parser.add_argument("--spacy_model", default="en_core_web_lg", help="Spacy model for POS tagging")
    parser.add_argument("--lm_type", default="blu", help="language model name")
    parser.add_argument("--top_k", type=int, default=10, help="top_k expansion results")
    parser.add_argument("--gpu_id", default=3, type=int, help="gpu id for bert embed")
    parser.add_argument("--save_path", default="./resources/verb_sense_dict_w_features.json")
    args = parser.parse_args()
    print(vars(args))
    main(args.input_file, args.spacy_model, args.lm_type, args.top_k, args.gpu_id, args.save_path)
