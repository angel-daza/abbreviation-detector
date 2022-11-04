from email.mime import base
from typing import List, Dict, Iterable, Set
from utils.abbr_utils import clean_token
import json
from seqeval.metrics import classification_report
import classla

HN_DICT = "resources/emnlp2022_data/hunspell_sl_SI.dic.txt"
GFIDA_DICT = "resources/emnlp2022_data/LEXICON_GFidaWords-entire.tsv"
CORP_DICT = "resources/emnlp2022_data/SloBio_AbbrCandidates.vocab" # This file is generated by text_to_abbr_candidates.py using the best baseline: Corp+Dict
INPUT_DATASET = "data/sbl-51abbr.sentences.test.json"


def load_tsv_lexicon(filepath:str, lowercase: bool) -> Set[str]:
    lexicon = set()
    with open(filepath) as f:
        for line in f.readlines():
            if "/" in line:
                w = line.strip().split("/")[0]
            else:
                w = line.strip()
            lexicon.add(w.lower()) if lowercase else lexicon.add(w)
    return lexicon


def load_sentence_dataset(filepath:str) -> List[Dict]:
    data = []
    with open(filepath) as f:
        for line in f.readlines():
            row = json.loads(line)
            data.append(row)
    return data


def load_corpus_candidates(filepath:str, lowercase: bool, threshold: int = 0.8) -> Set[str]:
    lexicon = []
    with open(filepath) as f:
        for line in f.readlines():
            row = json.loads(line)
            if row['probability'] >= threshold:
                if lowercase:
                    tok = row['token'].lower()
                else:
                    tok = row['token']
                lexicon.append(tok)
    return set(lexicon)


def _abbr_set_to_file(abbrs: set, filepath: str):
    with open(filepath, "w") as f:
        for elem in abbrs:
            f.write(elem + "\n")


def baseline_naive_predictions(lexicon:Set, dataset:Iterable):
    all_predictions, all_gold = [], []
    abbreviations_found = set()
    for instance in dataset:
        tokens = instance['abbreviated_text'].split()
        labels = []
        for token in tokens:
            clean_tok = clean_token(token).lower()
            contains_number = any([ch.isdigit() for ch in clean_tok])
            if '.' in token and not contains_number and clean_tok not in lexicon:
                labels.append('B-ABBR')
                abbreviations_found.add(token)
            else:
                labels.append('O')
        assert len(labels) == len(instance['token_labels_naive'])
        all_predictions.append(labels)
        all_gold.append(instance['token_labels_naive'])
    print(abbreviations_found)
    return all_gold, all_predictions, abbreviations_found


def baseline_lemmatized_predictions(lexicon:Set, dataset:Iterable):
    nlp = classla.Pipeline('sl', processors='tokenize,pos,lemma', tokenize_pretokenized=True)
    all_predictions, all_gold = [], []
    abbreviations_found = set()
    for instance in dataset:
        doc = nlp(" ".join(instance['abbreviated_tokenized']))
        tokens_lemmas = [(word.text,word.lemma) for sent in doc.sentences for word in sent.words]
        labels = []
        for token, lemma in tokens_lemmas:
            contains_number = any([ch.isdigit() for ch in token])
            if '.' in token and not contains_number and lemma not in lexicon:
                labels.append('B-ABBR')
                abbreviations_found.add(token)
            else:
                labels.append('O')
        assert len(labels) == len(instance['token_labels'])
        all_predictions.append(labels)
        all_gold.append(instance['token_labels'])
    print(abbreviations_found)
    return all_gold, all_predictions, abbreviations_found


def baseline_dict_lookup_predictions(lexicon:Set, dataset:Iterable):
    all_predictions, all_gold = [], []
    abbreviations_found = set()
    for instance in dataset:
        tokens = instance['abbreviated_tokenized']
        labels = []
        for token in tokens:
            clean_tok = clean_token(token).lower() + '.' # Artificially add '.' at the end to o=look it up in the dictionary of abbreviations
            contains_number = any([ch.isdigit() for ch in clean_tok])
            if '.' in token and not contains_number and clean_tok in lexicon:
                labels.append('B-ABBR')
                abbreviations_found.add(token)
            else:
                labels.append('O')
        assert len(labels) == len(instance['token_labels'])
        all_predictions.append(labels)
        all_gold.append(instance['token_labels'])
    print(abbreviations_found)
    return all_gold, all_predictions, abbreviations_found


def baseline_dict_hybrid_predictions(abbr_lexicon:Set, words_lexicon:Set, dataset:Iterable):
    all_predictions, all_gold = [], []
    abbreviations_found = set()
    for instance in dataset:
        tokens = instance['abbreviated_tokenized']
        labels = []
        for token in tokens:
            clean_tok = clean_token(token).lower() + '.' # Artificially add '.' at the end to o=look it up in the dictionary of abbreviations
            contains_number = any([ch.isdigit() for ch in clean_tok])
            if '.' in token and not contains_number and (clean_tok in abbr_lexicon or clean_tok not in words_lexicon):
                labels.append('B-ABBR')
                abbreviations_found.add(token)
            else:
                labels.append('O')
        assert len(labels) == len(instance['token_labels'])
        all_predictions.append(labels)
        all_gold.append(instance['token_labels'])
    print(abbreviations_found)
    return all_gold, all_predictions, abbreviations_found


if __name__ == '__main__':
    lex_hun = load_tsv_lexicon(HN_DICT, lowercase=True)
    lex_gfida = load_tsv_lexicon(GFIDA_DICT, lowercase=True)
    data = load_sentence_dataset(INPUT_DATASET)

    ngold, npreds, abbrs = baseline_naive_predictions(lex_hun, data)
    _abbr_set_to_file(abbrs, "data/abbrev_found_naive.txt")
    print(classification_report(ngold, npreds, digits=4))

    ngold, npreds, abbrs = baseline_lemmatized_predictions(lex_gfida, data)
    print(classification_report(ngold, npreds, digits=4))

    lex_corpus = load_corpus_candidates(CORP_DICT, lowercase=True)
    ngold, npreds, abbrs = baseline_dict_lookup_predictions(lex_corpus, data)
    _abbr_set_to_file(abbrs, "data/abbrev_found_bigrams.txt")
    print(classification_report(ngold, npreds, digits=4))

    lex_hybrid = lex_corpus.union(lex_hun)
    ngold, npreds, abbrs = baseline_dict_hybrid_predictions(lex_corpus, lex_hun, data)
    _abbr_set_to_file(abbrs, "data/abbrev_found_big_dict_hybrid.txt")
    print(classification_report(ngold, npreds, digits=4))
    

