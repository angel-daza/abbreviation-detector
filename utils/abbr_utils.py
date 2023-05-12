import copy, re
from dataclasses import dataclass
from collections import defaultdict, deque
import json, logging
from enum import Enum
from typing import Callable, List, Dict, Tuple, NamedTuple
from transformers import pipeline
from transformers import AutoModelForMaskedLM

import Levenshtein as lev


class ExpansionInstance(NamedTuple):
    doc_id: str
    sent_id: str 
    query: str
    gold: str
    prediction: str
    pred_candidates: List 
    abbrev: str
    mask_index: int

class CandidateLM(NamedTuple):
    text: str
    gold_abbr: str
    score_lm: float
    levenshtein: float
    jaccard_n1: float
    jaccard_n2: float

class BestFitPolicy(Enum):
    HIGHEST_PROB = 1
    FIRST_LETTER = 2
    LEVENSHTEIN = 3
    JACCARD_N1 = 4
    JACCARD_N2 = 5


#### Classes for the Slovenian Corpus

class ConllToken:
    def __init__(self, row: str, sent_id: str = "[UNK]", tok_id_offset: int = 0, splitter='\t', misc_tags_processor: Callable = lambda x: x) -> None:
        self.splitter = splitter
        self.raw = row
        fields = row.split(splitter)
        self.sent_id = sent_id # This is valid for gold/silver data. Set to UNK otherwise
        self.index = int(fields[0]) - 1 + tok_id_offset # The offset is to adapt it to a Global Document Indexing. Still NOT Implemented properly
        self.text = fields[1]
        self.lemma = fields[2]
        self.pos = fields[3]
        self.case = fields[4]
        self.morph = fields[5]
        if misc_tags_processor is not None:
            self.ner, self.abbr, self.space_after = misc_tags_processor(fields[9])
        self.starts_with_uppercase = self.text[0].isupper()
    
    def asconll(self):
        return self.raw

    def asdict(self):
        return {'sent_id': self.sent_id, 'index': self.index, 'text': self.text, 'lemma': self.lemma, 'pos': self.pos,
                'case': self.case, 'morph': self.morph, 'ner': self.ner, 'abbr': self.abbr, 'space_after': self.space_after}

@dataclass
class AnnotatedSentence:
    doc_id: str
    sent_id: str
    source_text: str
    mode: str # 'abbr2exp' | 'exp2abbr'
    tokens: List[ConllToken] = None
    mapping: List = None

    def get_mapping(self) -> None:
        """ 
            Gets the Abbr -> Exp or Exp -> Abbr mapping, depending on the mode...
        """
        mapping = []
        for tok in self.tokens:
            if tok.abbr.startswith('B-'):
                my_abbr = tok.abbr[2:]
                mapping.append((tok.index, tok.text, my_abbr))
        self.mapping = mapping


    def get_labeled_sequences(self, use_naive_tokens: bool) -> Tuple[List, List, List]:
        """ Compute the appropriate label sequence for the abbreviated and expanded sequences.
            It can process both directions abbr2exp and exp2abbr) depending on the sentence.mode

        Args:
            sentence (AnnotatedSentence): The sentence object containing annotated information
            use_naive_tokens (bool): If True, then we ignore the conll tokens and split the text by space (as in real world scenario)

        Returns:
            Tuple[List, List, List]: the abbreviated sentence, the expanded sentence, the sequence of labels always in this order. 
        """
        tokens, labels = [], []
        # The 'easy case' is when we have tokens available from the Conll file
        if use_naive_tokens == False:
            for tok in self.tokens:
                if tok.abbr == 'O':
                    labels.append('O')
                    tokens.append(tok.text)
                elif tok.abbr.startswith('B-'):
                    labels.append('B-ABBR')
                    tokens.append(tok.abbr[2:])
                elif tok.abbr.startswith('I-'):
                    labels.append('I-ABBR')
                    tokens.append(tok.abbr[2:])
            assert len(labels) == len(self.tokens) == len(tokens)
            source_tokens = [tok.text for tok in self.tokens]
            target_tokens = tokens
        # The 'common case' is when we have only raw text available and tokens are obtained using text.split() on the raw text
        elif use_naive_tokens == True and self.mapping is not None:
            naive_source_tokens = self.source_text.split()
            naive_target_tokens = []
            if len(self.mapping) == 0: return naive_source_tokens, naive_source_tokens, ['O' for t in naive_source_tokens]
            if self.mode == 'abbr2exp':
                keywords = deque([abbr for ix,abbr,exp in self.mapping])
                targets = [exp for ix,abbr,exp in self.mapping]
            else:
                keywords = deque([exp for ix,exp,abbr in self.mapping])
                targets = [abbr for ix,exp,abbr in self.mapping]
            
            if len(keywords) == 0: return naive_source_tokens, naive_source_tokens, ['O' for t in naive_source_tokens]

            substitute_ix = 0
            for tok in naive_source_tokens:
                clean_tok = re.sub(r'[^\w\s]',"",tok)
                try:
                    if clean_tok == keywords[0].strip('.'):
                        labels.append('B-ABBR')
                        keywords.popleft()
                        naive_target_tokens.append(targets[substitute_ix])
                        substitute_ix += 1
                    else:
                        labels.append('O')
                        naive_target_tokens.append(tok)
                except:
                    labels.append('O') # There are no more keywords so the rest is 'O' by default...
                    naive_target_tokens.append(tok)
                
            assert len(labels) == len(naive_source_tokens) == len(naive_target_tokens)
            source_tokens = naive_source_tokens
            target_tokens = naive_target_tokens
        else:
            raise Exception("There is no valid token <--> label mapping assigned!")
        
        # Always Return Sentence_with_abbreviations, Sentence_with_expansions, Labels in this order
        if self.mode == 'abbr2exp':
            return source_tokens, target_tokens, labels
        else:
            return target_tokens, source_tokens, labels


    def get_as_conll(self):
        final_conll = []
        prev_sent = self.tokens[0].sent_id
        for tok in self.tokens:
            if tok.sent_id == prev_sent:
                final_conll.append(tok.asconll())
            else:
                prev_sent = tok.sent_id
                final_conll.append(["\n"])
                final_conll.append(tok.asconll())
        return "\n".join(final_conll)


    def __post_init__(self):
        self.get_mapping()


#### Classes for the German Corpus

class ApisToken:
    def __init__(self, row: str, doc_id: str = "[UNK]", sent_id: str = "[UNK]", tok_id_offset: int = 0, splitter='\t') -> None:
        self.raw = row.strip()
        fields = self.raw.split(splitter)
        self.doc_id = doc_id
        self.sent_id = sent_id
        self.token_id = int(fields[0]) + tok_id_offset
        self.text = fields[1]
        misc_tags = self.misc_tags_processor(fields[2])
        self.is_abbr = misc_tags[0]
        self.expansion = misc_tags[1]
        self.abbr_is_person_name = misc_tags[2]
        self.space_after = misc_tags[3]
    
    def misc_tags_processor(self, misc_tag: str):
        # EXPAN=B-studierte|PersonName=No|SpaceAfter=No
        is_expan, expan, is_person_name, space_after = False, None, False, True
        for elem in misc_tag.split('|'):
            key, val = elem.split('=')
            if key == 'EXPAN':
                expan = val
                if val != 'O':
                    is_expan = True
            elif key == 'PersonName' and val == 'Yes':
                is_person_name = True
            elif key == 'Spaceafter' and val == 'No':
                space_after = False
        return is_expan, expan, is_person_name, space_after

    def asdict(self):
        return {'doc_id': self.doc_id,'sent_id': self.sent_id, 'index': self.token_id, 'text': self.text,
                'abbr': self.text, 'expan': self.expansion, 'space_after': self.space_after}

@dataclass
class AnnotatedApisSentence:
    doc_id: str
    sent_id: str
    source_text: str
    mode: str = 'abbr2exp'
    tokens: List[ApisToken] = None
    mapping: List = None

    def get_mapping(self) -> None:
        """ 
            Gets the Abbr -> Exp mapping, depending on the mode...
        """
        mapping = []
        for tok in self.tokens:
            if tok.expansion.startswith('B-'):
                my_exp = tok.expansion[2:]
                mapping.append((tok.token_id, tok.text, my_exp))
        self.mapping = mapping


    def get_labeled_sequences(self, use_naive_tokens: bool) -> Tuple[List, List, List]:
        """ Compute the appropriate label sequence for the abbreviated and expanded sequences.
            It can process both directions abbr2exp and exp2abbr) depending on the sentence.mode

        Args:
            sentence (AnnotatedSentence): The sentence object containing annotated information
            use_naive_tokens (bool): If True, then we ignore the conll tokens and split the text by space (as in real world scenario)

        Returns:
            Tuple[List, List, List]: the abbreviated sentence, the expanded sentence, the sequence of labels always in this order. 
        """
        tokens, labels = [], []
        # The 'easy case' is when we have tokens available from the Conll file
        if use_naive_tokens == False:
            for tok in self.tokens:
                if tok.expansion == 'O':
                    labels.append('O')
                    tokens.append(tok.text)
                elif tok.expansion.startswith('B-'):
                    labels.append('B-ABBR')
                    tokens.append(tok.expansion[2:])
                elif tok.expansion.startswith('I-'):
                    labels.append('I-ABBR')
                    tokens.append(tok.expansion[2:])
            assert len(labels) == len(self.tokens) == len(tokens)
            source_tokens = [tok.text for tok in self.tokens]
            target_tokens = tokens
        # The 'common case' is when we have only raw text available and tokens are obtained using text.split() on the raw text
        elif use_naive_tokens == True and self.mapping is not None:
            naive_source_tokens = self.source_text.split()
            naive_target_tokens = []
            if len(self.mapping) == 0: return naive_source_tokens, naive_source_tokens, ['O' for t in naive_source_tokens]
            if self.mode == 'abbr2exp':
                keywords = deque([abbr for ix,abbr,exp in self.mapping])
                targets = [exp for ix,abbr,exp in self.mapping]
            else:
                keywords = deque([exp for ix,exp,abbr in self.mapping])
                targets = [abbr for ix,exp,abbr in self.mapping]
            
            if len(keywords) == 0: return naive_source_tokens, naive_source_tokens, ['O' for t in naive_source_tokens]

            substitute_ix = 0
            for tok in naive_source_tokens:
                clean_tok = re.sub(r'[^\w\s]',"",tok)
                try:
                    if clean_tok == keywords[0].strip('.'):
                        labels.append('B-ABBR')
                        keywords.popleft()
                        naive_target_tokens.append(targets[substitute_ix])
                        substitute_ix += 1
                    else:
                        labels.append('O')
                        naive_target_tokens.append(tok)
                except:
                    labels.append('O') # There are no more keywords so the rest is 'O' by default...
                    naive_target_tokens.append(tok)
                
            assert len(labels) == len(naive_source_tokens) == len(naive_target_tokens)
            source_tokens = naive_source_tokens
            target_tokens = naive_target_tokens
        else:
            raise Exception("There is no valid token <--> label mapping assigned!")
        
        # Always Return Sentence_with_abbreviations, Sentence_with_expansions, Labels in this order
        return source_tokens, target_tokens, labels




def clean_token(token:str)->str:
    return re.sub(r"[^\w\s]", "", token)


def _decompose_abbr_bio_tag(complex_tag: str) -> str:
        # ABBR=O|NER=O|SpaceAfter=No
        # EXPAN=B-julija|NER=O
        x = complex_tag.split('|')
        abbr = x[0].split('=')[1].strip()
        ner = x[1].split('=')[1].strip()
        if len(x) > 2:
            space_after = x[2].split('=')[1]
        else:
            space_after = 'Yes'
        return ner, abbr, space_after


def read_slovene_conll(filepath:str, mode:str) -> List[AnnotatedSentence]:
    """[summary]

    Args:
        filepath (str): Filepath to the CoNLL File
        mode (str): 'abbr2exp' | 'exp2abbr' indicates what the status of the "original" text is

    Returns:
        List[AnnotatedSentence]: List of Annotated Documents
    """    
    sentences = []
    buffer = []
    mode_opts = ['abbr2exp' , 'exp2abbr']
    assert mode in mode_opts, f"MODE needs to be one of these options: {mode_opts}"
    with open(filepath) as f:
        for line in f.readlines():
            if line.startswith('# sent_id = '):
                sent_id = line[12:].rstrip('\n')
            elif line.startswith('# newdoc id ='):
                doc_id = line[14:].rstrip('\n')
            elif line.startswith('# text = '):
                text = line[9:].rstrip('\n')
            elif line.startswith('# '):
                pass
            elif len(line) > 1:
                buffer.append(ConllToken(line, sent_id, misc_tags_processor=_decompose_abbr_bio_tag))
            else:
                sentences.append(AnnotatedSentence(doc_id.lstrip(), sent_id.lstrip(), text.lstrip(), tokens=buffer, mode=mode))
                buffer = []

    return sentences


def read_apis_conll(filepath:str) -> List[AnnotatedApisSentence]:
    """[summary]
    """    
    tokens, sentences, apis_docs_ids = [], [], set()
    buffer = []
    with open(filepath) as f:
        for line in f.readlines():
            if line.startswith('# doc_id ='):
                doc_id = line[10:].rstrip('\n').lstrip()
                apis_docs_ids.add(doc_id)
            elif line.startswith('# sent_id = '):
                sent_id = line[12:].rstrip('\n').lstrip()
            elif line.startswith('# text = '):
                text = line[9:].rstrip('\n').lstrip()
            elif line.startswith('# '):
                pass
            elif len(line) > 1:
                apis_tok = ApisToken(line, doc_id, sent_id)
                buffer.append(apis_tok)
                clean_expansion = 'O' if not apis_tok.is_abbr else apis_tok.expansion[2:]
                tokens.append({'doc_id': apis_tok.doc_id, 'sent_id': apis_tok.sent_id, 'tok_id': apis_tok.token_id, 'text': apis_tok.text, 'expansion': clean_expansion, 'is_person': apis_tok.abbr_is_person_name})
            else:
                sentences.append(AnnotatedApisSentence(doc_id, sent_id, text, tokens=buffer))
                buffer = []

    return tokens, sentences, apis_docs_ids



def get_masked_text(sentence: AnnotatedSentence, mask_token:str='[MASK]', pre_expand_others:bool=False):
    masked_sentences = defaultdict(list)
    masked_correspondence = defaultdict(list) # Abbreviations that where masked IN ORDER i.e. masked_correspondence[n] is the nth [MASK] token in the sentence
    for token in sentence.tokens:
        if token.abbr == 'O':
            masked_sentences[token.sent_id].append(token.text)
        else:
            masked_sentences[token.sent_id].append(mask_token)
            masked_correspondence[token.sent_id].append((token.index, token.abbr, token.text))

    # Expand to 1-Example-Per-Mask (note that this filters away the sentences that had ZERO abbreviations)
    expanded_masked_sentences = dict()
    for id_key, mask_list in masked_correspondence.items():
        to_be_expanded = masked_sentences[id_key]
        for mask_ix in range(len(mask_list)):
            freeze_ix, freeze_abbr, freeze_expan = mask_list[mask_ix]
            substitutes = [mask_list[x] for x in range(len(mask_list)) if x != mask_ix]
            new = copy.deepcopy(to_be_expanded)
            for sub_ix, sub_abbr, sub_exp in substitutes:
                if pre_expand_others:
                    new[sub_ix] = sub_exp
                else:
                    new[sub_ix] = sub_abbr
            assert new.count(mask_token) == 1
            expanded_masked_sentences[f"{id_key}.{mask_ix}"] = {'sentence': new, 'candidate': freeze_abbr, 'gold_expansion': freeze_expan, 'mask_index': freeze_ix}

    return expanded_masked_sentences

def choose_best_fit(possible_fits: List[CandidateLM], policy: BestFitPolicy, topk: int) -> List[str]:
    assert topk > 0
    if topk > len(possible_fits): topk = len(possible_fits)
    top_fits = []
    if policy == BestFitPolicy.FIRST_LETTER:
        for p in possible_fits:
            if len(p.text) > 0 and p.text[0].lower() == p.gold_abbr[0].lower():
                top_fits.append(p.text)
        if len(top_fits) == 0: return ["<NO-FIT>"]
        return top_fits[:topk]
    elif policy == BestFitPolicy.HIGHEST_PROB:
        top_fits = [p.text for p in possible_fits[:topk]]
        return top_fits
    elif policy == BestFitPolicy.LEVENSHTEIN:
        top_fits = [p.text for p in sorted(possible_fits, key=lambda x: x.levenshtein)[:topk]]
        return top_fits
    elif policy == BestFitPolicy.JACCARD_N1:
        top_fits = [p.text for p in sorted(possible_fits, key=lambda x: x.jaccard_n1)[:topk]]
        return top_fits
    elif policy == BestFitPolicy.JACCARD_N2:
        # top_fits = [p.text for p in sorted(possible_fits, key=lambda x: x.jaccard_n2)[:topk]]
        top_fits = []
        for p in possible_fits:
            if len(p.text) > 0 and p.text[0].lower() == p.gold_abbr[0].lower():
                top_fits.append(p)
        if len(top_fits) == 0: return ["<NO-FIT>"]
        return [p.text for p in sorted(top_fits, key=lambda x: x.jaccard_n2)[:topk]]
    else:
        raise NotImplementedError



def jaccard_distance(str1, str2, ngram=1):
    if ngram == 1:
        A, B = set(str1), set(str2)
    else:
        A = set([str1[i:i+ngram] for i in range(len(str1)+1-ngram)])
        B = set([str2[i:i+ngram] for i in range(len(str2)+1-ngram)])
    union_len = len(A.union(B))
    jacc = len(A.intersection(B)) / len(A.union(B)) if union_len > 0 else 0
    return float(1 - jacc)



def get_dataset_partition(dataset: List[AnnotatedSentence], partition_dict: Dict[str, str]) -> List[AnnotatedSentence]:
    partition_docs: List[AnnotatedSentence] = []
    for doc in dataset:
        if doc.doc_id in partition_dict:
            if doc.sent_id in partition_dict[doc.doc_id]:
                partition_docs.append(doc)
    return partition_docs


def get_lang_model_predictions(dataset: List, model_name: str, tokenizer_name: str, top_k: int = 5, gpu_ix: int = -1) -> List:
    prediction_list = []
    prediction_dict = defaultdict(list)
    total_examples = len(dataset)
    skept_examples = []
    if gpu_ix >= 0:
        pipe = pipeline('fill-mask', model=model_name, tokenizer=model_name, device=gpu_ix)
    else:
        model = AutoModelForMaskedLM.from_pretrained(model_name)
        pipe = pipeline('fill-mask', model=model, tokenizer=tokenizer_name)
    for k, row in enumerate(dataset):
        masked_sent = " ".join(row['masked_tokens'])
        logging.info(f"\n------- {k+1}/{total_examples} ---------\nLooking for: {row['candidate']} --> {row['gold_expansion']}\n{masked_sent}\n\n")
        best_fit = "<NO-FIT>"
        try:
            model_predictions = pipe(masked_sent, top_k=top_k) # ADDING PARAM targets= ['milk'] RETURNS [{'sequence': 'I ate bacon and milk for breakfast', 'score': 0.010439875535666943, 'token': 6831, 'token_str': 'milk'}]
        except:
            skept_examples.append(row['sent_id'])
            continue
        possible_fits = []
        for res in model_predictions:
            logging.info(res['sequence'])
            logging.info(f"TOK: {res['token_str']} | SCORE: {res['score']}")
            # For now save all predictions, and later we choose a best_fit based on some kind of policy...
            possible_fits.append(CandidateLM(res['token_str'], row['candidate'], res['score'], 
                                            lev.distance(row['candidate'], res['token_str']), 
                                            jaccard_distance(row['candidate'], res['token_str'], ngram=1),
                                            jaccard_distance(row['candidate'], res['token_str'], ngram=2)
                                            ))
        better_fits = choose_best_fit(possible_fits, policy=BestFitPolicy.FIRST_LETTER, topk=5)
        better_fits_jaccard = choose_best_fit(possible_fits, policy=BestFitPolicy.JACCARD_N2, topk=10)
        best_fit = better_fits[0]
        # prediction_dict[row['candidate']].append(best_fit)
        prediction_dict[row['candidate']] += better_fits_jaccard
        prediction_list.append(ExpansionInstance(row['doc_id'], row['sent_id'], masked_sent, row['gold_expansion'], best_fit, possible_fits, row['candidate'], row['mask_index']))
    print(skept_examples)
    return prediction_list, prediction_dict
