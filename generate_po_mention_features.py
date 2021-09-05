import argparse
import pickle as pk
from collections import defaultdict
import torch
from transformers import BertForMaskedLM, BertTokenizer
from tqdm import tqdm
from get_bert_features import predict_masked_words, process_sentence

MODELS = {
    'blu': (BertForMaskedLM, BertTokenizer, 'bert-large-uncased-whole-word-masking'),
}

def main(corpus_w_svo_pickle,
         lm_type,
         top_k,
         gpu_id,
         ):

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

    print("loading corpus and salient term files")
    with open(corpus_w_svo_pickle, "rb") as f:
        corpus = pk.load(f)
    with open(f"{corpus_w_svo_pickle[:-3]}_salient_verbs.pk", "rb") as f:
        salient_verbs = pk.load(f)
    with open(f"{corpus_w_svo_pickle[:-3]}_salient_obj_heads.pk", "rb") as f:
        salient_obj_heads = pk.load(f) 
    with open(f"{corpus_w_svo_pickle[:-3]}_obj2obj_head_info.pk", "rb") as f:
        obj2obj_head_infos = pk.load(f)     

    print("start feature extraction ... ")
    svo_id2features = {}
    all_salient_cnt = 0
    for sent_id in tqdm(corpus):
        sent_info = corpus[sent_id]
        if len(sent_info['svos']) == 0:
            continue
        
        # collect all svo_ids in this sentence that needs to be processed
        to_processed_svo_id_list = []
        for svo_id, svo in enumerate(sent_info['svos']):
            all_salient_flag = True  # indicate if both verb or object head are salient
            any_salient_flag = False  # indicate if ether verb or object head is salient
            verb_lemma = svo[1][0]
            if verb_lemma.startswith("!"):
                verb_lemma = verb_lemma[1:]
            if verb_lemma in salient_verbs:
                any_salient_flag = True
            else:
                all_salient_flag = False
            if svo[2] is not None:
                obj = svo[2][0]
                obj_head_info = obj2obj_head_infos.get(obj, {})
                if len(obj_head_info) != 0:
                    obj_head_lemma = obj_head_info['obj_head_lemma']
                    if obj_head_lemma in salient_obj_heads:
                        any_salient_flag = True
                    else:
                        all_salient_flag = False
                else:
                    all_salient_flag = False
            else:
                all_salient_flag = False
            
            if any_salient_flag:
                to_processed_svo_id_list.append([svo_id, all_salient_flag])

            all_salient_cnt += int(all_salient_flag)

        if len(to_processed_svo_id_list) > 0:
            token_list = sent_info['token_list']
            if need_lower:
                token_list = [token.lower() for token in token_list]
            token_embeds = process_sentence(token_list, tokenizer, model, -1)

            for ele in to_processed_svo_id_list:
                svo_id = ele[0]
                all_salient_flag = ele[1]
                svo_index = f"{sent_id}_{svo_id}"
                svo = sent_info['svos'][svo_id]
                verb_lemma = svo[1][0]
                verb_index = svo[1][1]
                if verb_lemma.startswith("!"):  
                    verb_lemma = verb_lemma[1:]
                if verb_lemma in salient_verbs:  # only save salient verb lemma features
                    verb_embed = token_embeds[verb_index]
                    verb_masked_sent = " ".join(token_list[:verb_index] + ["[MASK]"] + token_list[verb_index+1:])
                    verb_expand_res = predict_masked_words(verb_masked_sent, mlm_model, tokenizer, top_k)[0]
                else:
                    verb_embed = None
                    verb_expand_res = None

                obj = svo[2]
                if obj is None:
                    obj = None
                    obj_head = None
                    obj_head_index = -1
                    obj_head_embed = None
                    obj_head_expand_res = None
                else:
                    obj = svo[2][0]
                    obj_range = svo[2][1]
                    obj_head_info = obj2obj_head_infos.get(obj, {})
                    if len(obj_head_info) == 0:
                        obj_head = None
                        obj_head_index = None
                        obj_head_embed = None
                        obj_head_expand_res = None
                    else:
                        obj_head = obj_head_info["obj_head_lemma"]
                        if obj_head not in salient_obj_heads:
                            obj_head_index = None
                            obj_head_embed = None
                            obj_head_expand_res = None
                        else:
                            obj_head_index = obj_range[obj_head_info['obj_head_relative_index']]
                            obj_head_embed = token_embeds[obj_head_index]
                            obj_head_masked_sent = " ".join(token_list[:obj_head_index] + ["[MASK]"] + token_list[obj_head_index+1:])
                            obj_head_expand_res = predict_masked_words(obj_head_masked_sent, mlm_model, tokenizer, top_k)[0]

                svo_id2features[svo_index] = {
                    "verb": verb_lemma,
                    "verb_index": verb_index,
                    "verb_embed": verb_embed,
                    "verb_expansion_results": verb_expand_res,
                    "obj": obj,
                    "obj_head": obj_head,
                    "obj_head_index": obj_head_index,
                    "obj_head_embed": obj_head_embed,
                    "obj_head_expansion_results": obj_head_expand_res,
                    "all_salient_flag": all_salient_flag,
                }

    print(f"select {len(svo_id2features)} mentions with EITHER salient verb OR object heads")
    print(f"select {all_salient_cnt} mentions with BOTH salient verb AND object heads")
    save_path = f"{corpus_w_svo_pickle[:-3]}_salient_po_mention_features.pk"
    print(f"save path: {save_path}")
    with open(save_path, "wb") as f:
        pk.dump(svo_id2features, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus_w_svo_pickle", help="input svo triplet extracted corpus file")
    parser.add_argument("--lm_type", default="blu", help="language model name")
    parser.add_argument("--top_k", default=10, type=int, help="top_k expansion results")
    parser.add_argument("--gpu_id", default=0, type=int, help="gpu id for bert model")
    args = parser.parse_args()
    print(vars(args))
    main(args.corpus_w_svo_pickle, args.lm_type, args.top_k, args.gpu_id)
