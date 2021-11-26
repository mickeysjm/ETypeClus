# ETypeClus

This repository contains the code and data for EMNLP 2021 paper "[Corpus-based Open-Domain Event Type Induction](https://arxiv.org/pdf/2109.03322.pdf)".

## Datasets and Resources

Please download the datasets and related resources at: [https://drive.google.com/drive/folders/1_QVv9XwN6PjZGdeMJWqW5D75NJmaD6F1?usp=sharing](https://drive.google.com/drive/folders/1-oqDLwlt94dFqAYDh6hhIbjt852_qCm7?usp=sharing)

* Each dataset has its own subfolder, e.g., **./covid19/** and **./pandemic/**.
* The verb sense dictionary and background corpus statistics are placed under **./resources/** subfolder.

Please put the downloaded folders under the root directory.

## Running ETypeClus

### Parse Corpus and Extract Subject-Verb-Object Triplets

```Bash
python3 parse_corpus_and_extract_svo.py \
    --is_sentence 1 \
    --input_file ./covid19/corpus.txt \
    --save_path ./covid19/corpus_parsed_svo.pk
```

### Select Salient Verb Lemmas and Object Heads

```Bash
python3 select_salient_terms.py \
    --corpus_w_svo_pickle ./covid19/corpus_parsed_svo.pk \
    --min_verb_freq 3 \
    --min_obj_freq 3
```

### Generate Features for Each Salient <Predicate Lemma, Object Head> Mention

```Bash
python3 generate_po_mention_features.py \
    --corpus_w_svo_pickle ./covid19/corpus_parsed_svo.pk \
    --top_k 50 \
    --gpu_id 5
```

### Disambiguate Predicate Senses 

```Bash
python3 disambiguate_verb_sense.py \
    --mention_file ./covid19/corpus_parsed_svo_salient_po_mention_features.pk \
    --save_path ./covid19/po_mention_disambiguated.pk
```

### Generate Features for Each Salient <Predicate Sense, Object Head> Tuples

```Bash
python3 generate_po_tuple_features.py \
    --mention_file ./covid19/corpus_parsed_svo_salient_po_mention_features.pk \
    --sense_mapping ./covid19/po_mention_disambiguated.pk \
    --save_file ./covid19/po_tuple_features_all_svos.pk \
    --use_all_svos
```

### Latent Space Clustering

```Bash
CUDA_VISIBLE_DEVICES=0 python3 latent_space_clustering.py \
	--dataset_path ./pandemic \
	--input_emb_name po_tuple_features_all_svos.pk
```


## Running Baselines

First follow previous section to generate the features for each salient <Predicate Sense, Object Head> tuples. 

Then, Use the following command (with the corresponding baseline code file) to run **Kmeans**, **sp-Kmeans**, **AggClus**, and **JCSC**. Note that the [spherecluster](https://github.com/jasonlaska/spherecluster) package requires an older version of scikit-learn, and we recommend using version 0.20.0.

```Bash
python ./baselines/baseline-{agglo/kmeans/spkmeans/jcsc}.py \
    --input ./covid19/po_tuple_features_all_svos.pk \
    --output kmeans_result.json \
    --k 30
```

For **Triframes**, first follow the instructions in this [link](https://github.com/uhh-lt/triframes) to install and set up the environment, and put it under the root directory. Then, run the following command

```Bash
python ./baselines/baseline-triframes.py \
    --input ./covid19/po_tuple_features_all_svos.pk \
    --output triframes_result.json \
    --N 100 \
    --min_size 100
```

## Reference

If you find this repository is useful, please consider citing our paper with the below bibliography. Thanks.

```
@inproceedings{Shen2021ETypeClus,
  title={Corpus-based Open-Domain Event Type Induction},
  author={Jiaming Shen and Yunyi Zhang and Heng Ji and Jiawei Han},
  booktitle={EMNLP},
  year={2021}
}
```





