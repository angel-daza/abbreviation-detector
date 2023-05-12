from dataclasses import dataclass
from typing import List, Dict
import json
from collections import defaultdict
from sklearn.model_selection import train_test_split
from utils.abbr_utils import AnnotatedSentence, read_slovene_conll, get_masked_text



def _get_multitok_abbrs(tokens_abbr, tokens_exp, labels):
    """
        This works better when the mode is exp2abbr, otherwise the mapping is just repetitions og the same expansion given a fragmented abbreviation
        Example: abbr2exp: B. - jev ---> Burgarjev || exp2abbr: Burgarjev ---> B.-jev
    """
    multitok = defaultdict(list)
    open_exp = []
    curr_abbr = ""
    for i, l in enumerate(labels):
        if len(open_exp) > 0:
            # if l == 'I-ABBR' is the STRICTEST way to catch multi-token abbreviations, however annotations are inconsistent: some are with consecutive [B-ABBR, B-ABBR, ...] instead of [[B-ABBR, I-ABBR, ...]].
            # l != 'O' catches ALL consecutive abbreviations, however this has a lot of FalsePositives and introduces more entropy, is better to treat each abbreviation individually
            if l == 'I-ABBR': 
                open_exp.append(tokens_exp[i])
            elif l == 'O':
                if len(open_exp) > 1:
                    multitok[curr_abbr].append(" ".join(open_exp))
                open_exp = []
        elif l == 'B-ABBR':
            open_exp.append(tokens_exp[i])
            curr_abbr = tokens_abbr[i]
        else:
            continue
    return multitok


def get_dataset_stats(dataset: List[AnnotatedSentence], verbose: bool, use_naive_tokens: bool) -> Dict:
    """Displays the structured data. For debug purposes (put verbose=True), it also returns global statistics information from the whole corpus

    Args:
        dataset (List[AnnotatedSentence]): The list of clean structured corpus examples
        verbose (bool): Decide if print the entire corpus examples or not

    Returns:
        Dict: It includes some relevant corpus stats that might be useful for "global" methods...
    """    
    all_mappings = defaultdict(set)
    mappings_counter = defaultdict(int)
    
    # General Stats Counters
    number_of_sentences, number_of_tokens, number_of_types = 0, 0, 0
    all_tokens = []
    all_abbr_candidates, all_abbr_expansions = [], []
    all_multitok_abbrs = []
    
    for sent in dataset:
        number_of_sentences += 1
        number_of_tokens += len(sent.tokens)
        all_tokens += [tok.text for tok in sent.tokens]
        for item in sent.mapping:
            if sent.mode == 'abbr2exp':
                index, abbrev, expan = item
                all_mappings[abbrev].add(expan)
            else:
                index, expan, abbrev = item
                all_mappings[expan].add(abbrev)
            all_abbr_expansions.append(expan)
            all_abbr_candidates.append(abbrev)
            mappings_counter[(abbrev, expan)] += 1
        
        abbr_toks, exp_toks, lbls = sent.get_labeled_sequences(use_naive_tokens=use_naive_tokens)

        if verbose:
            print(" ".join(abbr_toks))
            print(" ".join(exp_toks))
            print(" ".join(lbls))
            print(sent.mapping)
            print('-------')
    
        multitok = _get_multitok_abbrs(abbr_toks, exp_toks, lbls)
        if len(multitok) > 0:
            all_multitok_abbrs.append(multitok)

    if verbose:
        for k,v in sorted(all_mappings.items(), key= lambda x: len(x[1]), reverse=True):
            print(f"{k} --> {v}")
        print("\n\n")
        for k,v in sorted(mappings_counter.items(), key= lambda x: x[1], reverse=True):
            print(f"{k} --> {v}")
        
    number_of_types = len(set(all_tokens))
    number_abbr_candidates = len(all_abbr_candidates)
    number_unique_abbr_candidates = len(set(all_abbr_candidates))
    number_abbr_expansions = len(all_abbr_expansions)
    number_unique_expansions = len(set(all_abbr_expansions))
    print(f"\tSentences = {number_of_sentences}\n\tTokens = {number_of_tokens}\n\tTypes = {number_of_types}\n\tAbbreviations = {number_abbr_candidates} (unique = {number_unique_abbr_candidates})\n\tExpansions = {number_abbr_expansions} (unique = {number_unique_expansions})")
    print(f"\tAbbr->Expansion Pairs = {len(mappings_counter)}")
    [print(x) for x in all_multitok_abbrs]

    return {
        "exp2abbr_mapping": all_mappings,
        "exp_abbr_pairwise_mapping_counts": mappings_counter,
        "all_abbr_candidates": all_abbr_candidates
    }


def compare_data_partitions(train_stats: Dict, test_stats: Dict, label: str):
    unique_test_abbrs = set(test_stats['all_abbr_candidates']).difference(train_stats['all_abbr_candidates'])
    unique_candidates = set([abbr.lower() for abbr in unique_test_abbrs])
    print(f"\nUnique Abbreviation Candidates in {label} (UNSEEN) = {len(unique_candidates)}")


def create_token_classification_data(data: List[AnnotatedSentence], use_naive_tokens: bool, output_path: str):
    """This function builds the dataset for STEP 1: Token Classification [ABBR, NO-ABBR] for each Token in the dataset
        We save it in a file so it can be later loaded as a HuggingFace Dataset and make batched experiments with it!

    Args:
        data (List[AnnotatedSentence]): The list of clean structured corpus examples
        use_naive_tokens (bool): If we can use the conll tokens or to do 'naive' space-based splitting
    """
    with open(output_path, "w", encoding='utf-8') as fout:
        for sent in data:
            abbr_toks, exp_toks, lbls = sent.get_labeled_sequences(use_naive_tokens=use_naive_tokens)
            for tok, lbl in zip(abbr_toks, lbls):
                fout.write(json.dumps({"token": tok, "gold_label": lbl, "document_id": sent.doc_id, "sent_id": sent.sent_id}) + "\n")


def create_abbreviation_expansion_data(data: List[AnnotatedSentence], output_path: str, pre_expand_others: bool):
    with open(output_path, "w", encoding='utf-8') as fout:
        for doc in data:
            masked_doc = get_masked_text(doc, mask_token='<mask>', pre_expand_others=pre_expand_others)
            for sent_id, masked_sent in masked_doc.items():
                data_obj = {'doc_id': doc.doc_id, 'sent_id':sent_id, 'masked_tokens': masked_sent['sentence'], 'candidate': masked_sent['candidate'], 
                            'gold_expansion': masked_sent['gold_expansion'], 'mask_index': masked_sent['mask_index']}
                fout.write(json.dumps(data_obj) + "\n")


def save_document_data(data: List[AnnotatedSentence], output_path: str):
    with open(output_path, "w", encoding='utf-8') as fout:
        for doc in data:
            abbr_toks, exp_toks, labels = doc.get_labeled_sequences(use_naive_tokens=False)
            naive_abbr_toks, naive_exp_toks, naive_labels = doc.get_labeled_sequences(use_naive_tokens=True)
            data_obj = {
                'document_id': doc.doc_id,
                'original_text': doc.source_text,
                'abbreviated_text': " ".join(naive_abbr_toks),
                'expanded_text': " ".join(naive_exp_toks), 
                'mapping': doc.mapping,
                'token_objects': [tok.asdict() for tok in doc.tokens],
                'abbreviated_tokenized': abbr_toks,
                'expanded_tokenized': exp_toks,
                'token_labels': labels,
                'token_labels_naive': naive_labels
            }
            fout.write(json.dumps(data_obj) + "\n")


def save_text_data(data: List[AnnotatedSentence], output_path: str):
    with open(output_path, "w", encoding='utf-8') as fout:
        for doc in data:
            naive_abbr_toks, _, _ = doc.get_labeled_sequences(use_naive_tokens=True)
            fout.write(" ".join(naive_abbr_toks) + "\n")


def save_dataset_split_ids(filepath: str, dataset: List[AnnotatedSentence]) -> None:
    doc_dict = defaultdict(list)
    for doc in dataset: 
        doc_dict[doc.doc_id].append(doc.sent_id)
    with open(filepath, "w") as fout:
        json.dump(doc_dict, fout, indent=4)


if __name__ == '__main__':

    all_sentences = read_slovene_conll("resources/emnlp2022_data/sbl-51abbr-expan.conll", mode='exp2abbr')

    X_train, X_test = train_test_split(all_sentences, test_size=0.2, random_state=4239)
    X_train, X_dev = train_test_split(X_train, test_size=0.125, random_state=4239)
    print(len(X_train), len(X_dev), len(X_test))
    save_dataset_split_ids("data/sbl-51abbr.ids.train", X_train)
    save_dataset_split_ids("data/sbl-51abbr.ids.dev", X_dev)
    save_dataset_split_ids("data/sbl-51abbr.ids.test", X_test)
    
    # Compute Statistics for each given portion of the dataset
    use_naive_tokens=False # We want to report the GOLD statistics as they are manually curated on CoNLL Format
    print(f"\nComputing stats for {len(X_train)} examples in TRAIN")
    train_stats = get_dataset_stats(X_train, verbose=False, use_naive_tokens=use_naive_tokens) 
    print(f"\nComputing stats for {len(X_dev)} examples in DEVELOPMENT")
    dev_stats = get_dataset_stats(X_dev, verbose=False, use_naive_tokens=use_naive_tokens)
    print(f"\nComputing stats for {len(X_test)} examples in TEST")
    test_stats = get_dataset_stats(X_test, verbose=False, use_naive_tokens=use_naive_tokens)
    compare_data_partitions(train_stats, test_stats, label="TRAIN vs TEST")
    compare_data_partitions(train_stats, dev_stats, label="TRAIN vs DEV")
    compare_data_partitions(dev_stats, test_stats, label="DEV vs TEST")


    # DATA FOR EXPERIMENT 1: A SINGLE TOKEN BINARY CLASSIFIER (BERT) for Abbreviation or No_Abbreviation
    use_naive_tokens=True
    # Train Set for STEP 1
    create_token_classification_data(X_train, use_naive_tokens, output_path='data/sbl-51abbr.tok.train.json')
    save_document_data(X_train, output_path='data/sbl-51abbr.sentences.train.json')
    # Development Set for STEP 1
    create_token_classification_data(X_dev, use_naive_tokens, output_path='data/sbl-51abbr.tok.dev.json')
    save_document_data(X_dev, output_path='data/sbl-51abbr.sentences.dev.json')
    save_text_data(X_dev, output_path='data/sbl-51abbr.sentences.dev.txt')
    # Test Set for STEP 1
    create_token_classification_data(X_test, use_naive_tokens, output_path='data/sbl-51abbr.tok.test.json')
    save_document_data(X_test, output_path='data/sbl-51abbr.sentences.test.json')
    save_text_data(X_test, output_path='data/sbl-51abbr.sentences.test.txt')


    # EXPERIMENT 2: Save the Train/Dev/Test partitions with the documents. This will be used later to [MASK] them and prepare data (from STEP 1) for expansion prediction
    # The files say 'upperbound' because we know all abbreviation candidates inside the sentences are GOLD (which we won't know in the real-world scenario).
    # PreExpanded Dataset means that only one abbreviation at a time is treated. Assuming everything else has been correctly expanded already (meaning that the abbreviation expanded has more "explicit" context)
    create_abbreviation_expansion_data(X_train, output_path='data/sbl-51abbr.masked.upperbound.preexp.train.json', pre_expand_others=True)
    create_abbreviation_expansion_data(X_dev, output_path='data/sbl-51abbr.masked.upperbound.preexp.dev.json', pre_expand_others=True)
    create_abbreviation_expansion_data(X_test, output_path='data/sbl-51abbr.masked.upperbound.preexp.test.json', pre_expand_others=True)
    # Again, this is the realistic scenario, where each sentence has more than one abbreviation that needs to be identified
    create_abbreviation_expansion_data(X_train, output_path='data/sbl-51abbr.masked.upperbound.train.json', pre_expand_others=False)
    create_abbreviation_expansion_data(X_dev, output_path='data/sbl-51abbr.masked.upperbound.dev.json', pre_expand_others=False)
    create_abbreviation_expansion_data(X_test, output_path='data/sbl-51abbr.masked.upperbound.test.json', pre_expand_others=False)