import argparse
import pickle as pk
import spacy
from tqdm import tqdm

from extract_svo import findSVOs

def main(input_file,
         is_sentence,
         spacy_model,
         save_path):

    print("loading Spacy model")
    nlp = spacy.load(spacy_model)

    lines = []
    with open(input_file, "r") as fin:
        for line in fin:
            line = line.strip()
            if line:
                lines.append(line)

    save_dict_data = {}

    if is_sentence == 1:
        for sent_id, sent in tqdm(enumerate(lines)):
            doc = nlp(sent)
            token_list = [token.text for token in doc]
            raw_sent = " ".join(token_list)
            svos = findSVOs(doc)

            save_dict_data[sent_id] = {
                "sent_id": sent_id,
                "raw_sentence": raw_sent,
                "token_list": token_list,
                "svos": svos,
            }
    else:
        sent_id = 0
        for line in tqdm(lines):
            doc = nlp(line)
            for sent in doc.sents:
                new_doc = nlp(sent.text)

                token_list = [token.text for token in new_doc]
                raw_sent = " ".join(token_list)
                svos = findSVOs(new_doc)

                save_dict_data[sent_id] = {
                    "sent_id": sent_id,
                    "raw_sentence": raw_sent,
                    "token_list": token_list,
                    "svos": svos,
                }
                sent_id += 1

    with open(save_path, "wb") as f:
        pk.dump(save_dict_data, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_file", help="input corpus file")
    parser.add_argument("--save_path", help="output file with subject-verb-object triplets extracted")
    parser.add_argument("--is_sentence", default=1, help="1 if each line in input_file is a sentence, 0 if each line is a document")
    parser.add_argument("--spacy_model", default="en_core_web_lg", help="Spacy model for POS tagging")
    args = parser.parse_args()
    print(vars(args))
    main(args.input_file, args.is_sentence, args.spacy_model, args.save_path)
