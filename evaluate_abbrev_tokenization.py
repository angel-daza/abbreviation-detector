from numpy import mat
import pandas as pd
from typing import List, NamedTuple, Dict, Optional, TypeVar
import classla
ClasslaToken = TypeVar("ClasslaToken")
from utils.abbr_utils import read_slovene_conll, AnnotatedSentence, get_dataset_partition
import json
from seqeval.metrics import classification_report
import copy, os
import Levenshtein as lev
from collections import Counter

class NERprediction(NamedTuple):
    entity_type: str
    text: str
    token_indices: List
    token_texts: List

class SentTokMapping(NamedTuple):
    gold_sentence: List[str]
    pred_sentence: List[str]


def clean_abbr_token(token: str) -> str:
    """
    Args:
        token (str): Dirty token from naive tokenization (i.e. sentence.split())
    Returns:
        str: Returns the same token without any weird symbols (keeps the fullstop)
    """
    return "".join([t for t in token if t not in '!"#$%&\'()*+,-[]/:;<=>?@\\^_`{|}~'])


def evaluate_tokenization(gold_data: List[AnnotatedSentence], nlp_tokenized: List[List[List[ClasslaToken]]]):
    assert len(gold_data) == len(nlp_tokenized)
    total_gold_sentences = len(gold_data) # Here each element of dataset is a sentence
    total_tokens = 0 # Total tokens in the GOLD dataset
    total_predicted_sentences = 0 # Measure how many sentences have less or more number of tokens
    token_global_errors = 0 # Measure how many token_errors per total_tokens happened
    sent_errors, sent_matches = [], []
    for gold_sent, nlp_sents in zip(gold_data, nlp_tokenized):
        total_predicted_sentences += len(nlp_sents) # Each gold sentence can correspond to 1 or more predicted sentences (these are the sentence split errors!)
        gold_tokens = [t.text for t in gold_sent.tokens]
        total_tokens += len(gold_tokens)
        nlp_tokens = [t.text for sent in nlp_sents for t in sent.tokens]
        if len(gold_tokens) < len(nlp_tokens):
            token_global_errors += abs(len(gold_tokens) - len(nlp_tokens))
            sent_errors.append(SentTokMapping(gold_tokens, nlp_tokens))
        elif len(gold_tokens) > len(nlp_tokens):
            token_global_errors += abs(len(gold_tokens) - len(nlp_tokens))
            sent_errors.append(SentTokMapping(gold_tokens, nlp_tokens))
        else:
            sent_matches.append(SentTokMapping(gold_tokens, nlp_tokens))

    print(f"Sentencewise Errors: Gold Sentences = {total_gold_sentences} || Predicted Sentences = {total_predicted_sentences} || Correct = {len(sent_matches)} || Precision = {len(sent_matches)*100/total_predicted_sentences:.2f} || Recall = {len(sent_matches)*100/total_gold_sentences:.2f}")
    print(f"Tokenwise Errors: Total Tokens = {total_tokens} || Correct = {total_tokens-token_global_errors} || Errors = {token_global_errors} || Acc. = {(total_tokens-token_global_errors)*100/total_tokens:.2f}")


def evaluate_entities(gold_data: List[AnnotatedSentence], nlp_tokenized: List[List[List[ClasslaToken]]]):
    total_sentences = len(gold_data) # Here each element of dataset is a sentence
    sent_correct, ner_sent_err, ner_mismatch = 0, 0, 0 # Measure how many sentences have errors
    Y_true, Y_pred = [], []
    for gold_sent, nlp_sents in zip(gold_data, nlp_tokenized):
        gold_ner = [t.ner for t in gold_sent.tokens if '*' not in t.ner]
        nlp_ner = [t.ner for sent in nlp_sents for t in sent.tokens if '*' not in t.ner]
        if len(gold_ner) == len(nlp_ner):
            Y_true.append(gold_ner)
            Y_pred.append(nlp_ner)
        else:
            ner_mismatch += sum([1 for n in gold_ner if n != 'O'])
            Y_true.append(gold_ner)
            Y_pred.append(['O' for n in gold_ner])
        
        if gold_ner == nlp_ner:
            sent_correct += 1
        else:
            ner_sent_err += 1
    
    print(f"\nConsidered as All False because of sentence length mismatch = {ner_mismatch}\n") # However this is a subset of the ner_sent_err errors!
    print(classification_report(Y_true, Y_pred, digits=4))
    print(f"\nSentencewise Total = {total_sentences} || Corr. = {sent_correct} || Err. = {ner_sent_err} || Acc. = {sent_correct*100/total_sentences:.2f}")


def add_nlp_annotations(data: List[AnnotatedSentence], nlp: classla.Pipeline, use_gold_tokens: bool) -> List[List[List[ClasslaToken]]]:
    model_predictions = []
    for annotated in data:
        if use_gold_tokens:
            nlp_doc = nlp(" ".join([tok.text for tok in annotated.tokens]))
        else:
            nlp_doc = nlp(annotated.source_text)
        model_predictions.append(nlp_doc.sentences) # Each gold sentence can be mapped to 1 or more predicted sentences!
    return model_predictions

def full_dataset_evaluation(filepath: str, dict_dataset_ids: Dict, nlp: classla.Pipeline, use_gold_tokens:bool, title_eval: str) -> None:
    print(f"\n --- Evaluating {title_eval} Data ---")
    annotated_sents = read_slovene_conll(filepath, mode="exp2abbr")
    gold_data = get_dataset_partition(annotated_sents, dict_dataset_ids)
    nlp_tokens = add_nlp_annotations(gold_data, nlp, use_gold_tokens)
    evaluate_tokenization(gold_data, nlp_tokens)
    evaluate_entities(gold_data, nlp_tokens)


def bert_predictions_evaluation(predictions: pd.DataFrame, gold_sentences: List[AnnotatedSentence], nlp: classla.Pipeline):
    print(f"\n --- Evaluating BERT Predictions Data ---")
    Y_pred, Y_true = [], []
    total_sentences = len(predictions.index) # Here each element of dataset is a sentence
    sent_correct, ner_sent_err, ner_mismatch = 0, 0, 0 # Measure how many sentences have errors
    gold_sent_dict = {s.sent_id:[t.ner for t in s.tokens] for s in gold_sentences}
    for ix, row in predictions.iterrows():
        sentence = copy.deepcopy(row['predicted_sentence'])
        # fully_expanded = copy.deepcopy(row['predicted_sentence'])
        for ix, gold, pred, abbr in zip(row['indices'], row['gold'], row['pred'], row['abbrs']):
            # fully_expanded[ix] = gold
            if pred == '<NO-FIT>':
                sentence[ix] = abbr
            else:
                sentence[ix] = pred
        # Get NER Predictions --> Gold vs BERT 
        # print(f"{sentence}\n{fully_expanded}\n----")
        # doc_gold = nlp(" ".join(fully_expanded))
        doc_pred = nlp(" ".join(sentence))
        # gold_ner = [t.ner for sent in doc_gold.sentences for t in sent.tokens]
        gold_ner = [n for n in gold_sent_dict[row['sent_id']] if '*' not in n]
        nlp_ner = [t.ner for sent in doc_pred.sentences for t in sent.tokens]
        if len(gold_ner) == len(nlp_ner):
            Y_true.append(gold_ner)
            Y_pred.append(nlp_ner)
        else:
            ner_mismatch += 1
            Y_true.append(gold_ner)
            Y_pred.append(['O' for _ in gold_ner])
        
        if gold_ner == nlp_ner:
            sent_correct += 1
        else:
            ner_sent_err += 1
    
    print(f"\nConsidered as errors in report because of length mismatch = {ner_mismatch}\n") # However this is a subset of the ner_sent_err errors!
    print(classification_report(Y_true, Y_pred, digits=4))
    print(f"\nSentencewise Total = {total_sentences} || Corr. = {sent_correct} || Err. = {ner_sent_err} || Acc. = {sent_correct*100/total_sentences:.2f}")



def load_abbreviations_dataset(filepath: str) -> List[Dict]:
    data = []
    with open(filepath) as f:
        for line in f.readlines():
            data.append(json.loads(line))

    return data


def get_token_vocabulary(filepath: str, abbreviations_only: bool = False, train_vocab: set = set()):
    token_vocab = set()
    with open(filepath) as f:
        for line in f.readlines():
            obj = json.loads(line)
            if abbreviations_only:
                if 'ABBR' in obj['gold_label']:
                    token_vocab.add(clean_abbr_token(obj['token']).lower())
                else:
                    continue
            else:
                token_vocab.add(clean_abbr_token(obj['token']).lower())
    unique_token_vocab = None
    if len(train_vocab) > 0:
        unique_token_vocab = token_vocab.difference(train_vocab)

    return token_vocab, unique_token_vocab


def evaluate_abbr_candidates_task(inputs_path: str, outputs_path: str, all_abbreviations: set, only_this_abbreviations: set):
    """This evaluates the results of the 1_bert_token_classifier_predict.py outputs!
    """
    token_counter, unk_abbrev_counter = 0, 0
    fp_all, fn_all, tp_all, tn_all = [], [], [], []
    miss_unk_all, match_unk_all = [], []
    all_inputs = [clean_abbr_token(line.strip().replace(" ", "").replace("▁", "")) for line in open(inputs_path).readlines()]
    all_outputs = ["ABBR" if "ABBR" in line.strip() else "O" for line in open(outputs_path).readlines()]
    assert len(all_inputs) == len(all_outputs)
    for tok, label in zip(all_inputs, all_outputs):
        # General Evaluation
        token_counter += 1
        if tok.lower() in all_abbreviations:
            if label == "ABBR":
                tp_all.append(tok)
            else:
                fn_all.append(tok)
        else:
            if label == "ABBR":
                fp_all.append(tok)
            else:
                tn_all.append(tok)
        # Only Consider Abbreviation Previously Unkwnown
        if tok.lower() in only_this_abbreviations:
            unk_abbrev_counter += 1
            if label == "ABBR":
                match_unk_all.append(tok)
            else:
                miss_unk_all.append(tok)

    tp, tn, fp, fn = len(tp_all), len(tn_all), len(fp_all), len(fn_all)
    prec = tp*100/(tp+fp) or 0
    rec = tp*100/(tp+fn) or 0
    print(tp, tn, fp, fn)
    print(f"Accuracy_ALL = {(tp+tn)*100/token_counter}")
    print(f"Precision = {prec}")
    print(f"Recall = {rec}")
    print(f"F1 = {2*prec*rec/(prec+rec)}")
    miss, match = len(miss_unk_all), len(match_unk_all)
    print(f"Unknown Token Recall = {match*100/unk_abbrev_counter} ({match} out of {unk_abbrev_counter} | Errors = {miss})")
    print(miss_unk_all)




if __name__ == '__main__':

    PATH_TO_MODEL = "resources/emnlp2022_data"# Normally would be model trained with the bert_token_classifier.py script, such as: "saved_models/BERT_ABBR_876972"

    # Evaluate Abbreviation Candidate Identification as Token Prediction Task
    train_all_abbr, _ = get_token_vocabulary("data/sbl-51abbr.tok.train.json", abbreviations_only=True)
    dev_all_abbr, dev_unique_abbr = get_token_vocabulary("data/sbl-51abbr.tok.dev.json", abbreviations_only=True, train_vocab=train_all_abbr)
    evaluate_abbr_candidates_task(f"{PATH_TO_MODEL}/bert_inputs.dev.txt", 
                                  f"{PATH_TO_MODEL}/bert_outputs.dev.txt",
                                  dev_all_abbr, dev_unique_abbr)
    test_all_abbr, test_unique_abbr = get_token_vocabulary("data/sbl-51abbr.tok.test.json", abbreviations_only=True, train_vocab=train_all_abbr)
    evaluate_abbr_candidates_task(f"{PATH_TO_MODEL}/bert_inputs.test.txt", 
                                  f"{PATH_TO_MODEL}/bert_outputs.test.txt",
                                  test_all_abbr, test_unique_abbr)
    print(dev_unique_abbr)
    print(len(train_all_abbr),len(dev_all_abbr),len(dev_unique_abbr))
    with open("data/abbrev_found_bert_classifier.txt", "w") as f:
        for elem in test_all_abbr:
            f.write(elem+"\n")
    

    # First of all: MAX UPPERBOUND - Evaluate the difference of Tokenization and NER with abbreviations vs no abbreviations
    # This is Table 2 in the Paper
    USE_GOLD_TOKENS=False
    dict_train_ids = json.load(open("data/sbl-51abbr.ids.train"))
    dict_test_ids = json.load(open("data/sbl-51abbr.ids.test"))
    if USE_GOLD_TOKENS:
        nlp = classla.Pipeline('sl', processors='tokenize,ner,pos,lemma', tokenize_pretokenized=True)
    else:
        nlp = classla.Pipeline('sl', processors='tokenize,ner,pos,lemma')
    full_dataset_evaluation("resources/emnlp2022_data/sbl-51abbr-expan.conll", dict_test_ids, nlp, USE_GOLD_TOKENS, title_eval="Test Set Expan")
    full_dataset_evaluation("resources/emnlp2022_data/sbl-51abbr-abbr.conll", dict_test_ids, nlp, USE_GOLD_TOKENS, title_eval="Test Set Abbr")


    ## NOW, Evaluate NLP performance over the Expansion Predictions 
    # This is Table 5 in the Paper
    nlp = classla.Pipeline('sl', processors='tokenize,ner,pos,lemma', tokenize_pretokenized=True)
    predictions = pd.read_json('data/model_unified_predictions.test.jsonl')
    annotated_sents = read_slovene_conll("resources/emnlp2022_data/sbl-51abbr-expan.conll", mode="exp2abbr")
    gold_data = get_dataset_partition(annotated_sents, dict_test_ids)
    bert_predictions_evaluation(predictions, gold_data, nlp)
    print(predictions.head())
    print(predictions.columns)
    
