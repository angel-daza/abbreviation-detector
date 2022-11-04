from typing import List, Dict
from collections import defaultdict, Counter

from utils.bert_utils import get_torch_device
import logging, sys, json

import pandas as pd
from utils.abbr_utils import ExpansionInstance, BestFitPolicy, choose_best_fit, get_lang_model_predictions



def unify_predictions(predictions: List[ExpansionInstance]) -> List[Dict]:
    sentences_unified_substitutions = defaultdict(list)
    indexed_predictions: Dict[str, ExpansionInstance] = {}
    for pred in predictions:
        doc_id, parag_id, s_id, sent_mask_part = pred.sent_id.split('.') # sent_id = sbl00756.p1.s147.0
        sent_id = f"{doc_id}.{parag_id}.{s_id}"
        sentences_unified_substitutions[sent_id].append((pred.mask_index, pred.prediction, pred.gold, pred.abbrev))
        indexed_predictions[sent_id] = pred # Just save one per sent_id, the masks will be overriden anyway on the next step...
    
    unified_predictions: List[ExpansionInstance] = []
    for sent_id, substitutes in sentences_unified_substitutions.items():
        template_obj = indexed_predictions[sent_id]
        template = template_obj.query.split()
        prd_lst, gld_lst, ix_lst, abbr_lst = [], [], [], []
        for s_ix, s_pred_tok, s_gold_tok, s_abbrev in substitutes:
            template[s_ix] = s_pred_tok
            ix_lst.append(s_ix)
            prd_lst.append(s_pred_tok)
            gld_lst.append(s_gold_tok)
            abbr_lst.append(s_abbrev)
        unified_predictions.append({'doc_id': template_obj.doc_id, "sent_id": sent_id, 'predicted_sentence': template, 'pred':prd_lst, 'gold': gld_lst, 'indices': ix_lst, 'abbrs': abbr_lst})
    return unified_predictions


def mask_token_replacer(dataset: List, current_mask: str, new_mask: str) -> List:
    for example in dataset:
        sentence = example['masked_tokens']
        mask_ix = sentence.index(current_mask)
        sentence[mask_ix] = new_mask
        example['masked_tokens'] = sentence
    return dataset



def load_abbreviations_dataset(filepath: str) -> List[Dict]:
    data = []
    with open(filepath) as f:
        for line in f.readlines():
            data.append(json.loads(line))

    return data


def load_abbreviations_prompt_dataset(filepath: str, mask_token: str = '<mask>', is_word: str = 'je') -> List[Dict]:
    data = []
    window_size = 20
    with open(filepath) as f:
        for line in f.readlines():
            example = json.loads(line)
            sentence = example['masked_tokens']
            mask_ix = sentence.index(mask_token)
            sentence[mask_ix] = example['candidate']
            prev_ctx_ix = mask_ix - window_size if mask_ix - window_size > 0 else 0
            post_ctx_ix = mask_ix - window_size if mask_ix + window_size < len(sentence) else len(sentence)
            sentence[mask_ix] = example['candidate']
            full_prompt = sentence[prev_ctx_ix:post_ctx_ix] +  ['[SEP]'] + [example['candidate'], is_word, mask_token]
            assert full_prompt.count(mask_token) == 1, 'The full prompt is not properly masked!'
            example['masked_tokens'] = full_prompt
            data.append(example)
    return data


def evaluate_expansion_predictions(predictions: List[ExpansionInstance]):
    matches, errors = 0, 0
    soft_matches, soft_errors = 0, 0
    error_data, soft_error_list = [], []
    word_not_found = 0
    for pred in predictions:
        if pred.prediction.lower() == pred.gold.lower():
            matches += 1
        else:
            errors += 1
            if pred.prediction == '<NO-FIT>': 
                word_not_found += 1
            error_data.append(pred._asdict())

        if pred.gold.lower() in [p.text.lower() for p in pred.pred_candidates]:
            soft_matches += 1
        else:
            soft_errors += 1
            top_prob_candidates = [p.text for p in pred.pred_candidates[:5]]
            levenshtein_candidates = choose_best_fit(pred.pred_candidates, BestFitPolicy.LEVENSHTEIN, topk=5)
            jaccard_n1_candidates = choose_best_fit(pred.pred_candidates, BestFitPolicy.JACCARD_N1, topk=5)
            jaccard_n2_candidates = choose_best_fit(pred.pred_candidates, BestFitPolicy.JACCARD_N2, topk=5)
            soft_error_list.append({'sent_id': pred.sent_id, 'masked': pred.abbrev, 'gold_expansion': pred.gold, 
                                    'model_candidates': top_prob_candidates,
                                    'lev_candidates': levenshtein_candidates,
                                    'jacc_n1': jaccard_n1_candidates,
                                    'jacc_n2': jaccard_n2_candidates
                                    })
        
    logging.info(f"Correct = {matches} || Errors: {errors} (from which {word_not_found} where <NO-FIT>) || Accuracy = {matches*100/(errors+matches):.2f}%")
    logging.info(f"Possibly Correct = {soft_matches} || Definitely Errors (not found in the Top-N): {soft_errors} || Accuracy = {soft_matches*100/(soft_errors+soft_matches):.2f}%")
    err_ignore_no_found = errors - word_not_found
    logging.info(f"Correct = {matches} || Errors (ignoring <NO-FIT>): {err_ignore_no_found}|| Accuracy (ignoring <NO-FIT>) = {matches*100/(err_ignore_no_found+matches):.2f}%")
    pd.DataFrame(error_data).to_csv('data/ERRORS.csv', columns=['gold', 'prediction', 'query'])
    pd.DataFrame(soft_error_list).to_csv('data/ERRORS_NO-TOPN.csv')


if __name__ == '__main__':
    TOP_K=5 # Move this hyperparam to 'choose from' the top-k predictions of the BERT model for a given Masked Token
    DATA_PARTITION='test'
    BERT_MODEL_NAME= "EMBEDDIA/sloberta"
    BERT_TOKENIZER_NAME='EMBEDDIA/sloberta'
    # Get GPU (if available)
    gpu_wanted = 0
    GPU_DEV, USE_CUDA = get_torch_device(verbose=True, gpu_ix=gpu_wanted)
    if USE_CUDA:
        GPU_IX = gpu_wanted
    else:
        GPU_IX = -1
    
    # Initialize Logger
    console_hdlr = logging.StreamHandler(sys.stdout)
    file_hdlr = logging.FileHandler(filename=f"data/bert_abbrev_expansion.{DATA_PARTITION}.log")
    logging.basicConfig(level=logging.INFO, handlers=[console_hdlr, file_hdlr])

    # Load Dataset
    test_data = load_abbreviations_dataset(f"data/sbl-51abbr.masked.upperbound.{DATA_PARTITION}.json")

    #### Fill-in Masked abbreviation with BERT predictions ...
    predictions, pred_dict = get_lang_model_predictions(test_data, BERT_MODEL_NAME, BERT_TOKENIZER_NAME, TOP_K, GPU_IX)
    pd.DataFrame([p._asdict() for p in predictions]).to_json(f'data/model_raw_predictions.{DATA_PARTITION}.jsonl', orient='records')
    # Save Predictions
    unified_predictions = unify_predictions(predictions)
    pd.DataFrame(unified_predictions).to_json(f'data/model_unified_predictions.{DATA_PARTITION}.jsonl')

    # Save Accumulated Per-Candidate Predictions 
    unique_vals_preds = {k: Counter(v).most_common(10) for k,v in pred_dict.items()}
    json.dump(unique_vals_preds, open(f"data/bert_prediction_mapping.{DATA_PARTITION}.json", "w"), indent=2)
    evaluate_expansion_predictions(predictions)