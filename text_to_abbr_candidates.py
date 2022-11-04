import json, re
from typing import Counter, Set, Generator, List, TypeVar, Tuple, Dict, NamedTuple
import classla
NLP_Pipeline = TypeVar("NLP_Pipeline")


class AbbreviationEntry(NamedTuple):
    token: str
    token_clean: str
    abbrev_seed_count: int
    abbrev_cand_count: int
    corpus_count: int
    probability: float


def has_no_numbers(token):
    has_numbers = any([ch.isdigit() for ch in token])
    return not has_numbers

def clean_token(token, lowercase=False):
    clean = re.sub(r'[^\w\s]',"",token)
    if lowercase: 
        return clean.lower()
    else:
        return clean


def load_vocab_file(filepath: str, keep_threshold: int = -1, lowercase: bool = False) -> Dict:
    vocab = {}
    with open(filepath) as f:
        for line in f.readlines():
            try:
                token, count = line.strip('\n').split('\t')
                tok_cnt = int(count)
            except:
                continue
            if tok_cnt >= keep_threshold:
                if lowercase:
                    tok = token.lower()
                    if tok in vocab:
                        vocab[tok] = vocab[tok] + tok_cnt
                    else:
                        vocab[tok] = tok_cnt
                else:
                    vocab[token] = tok_cnt
    return vocab


def load_vocab_entry_file(filepath: str, keep_threshold: int = -1, lowercase: bool = False) -> Dict:
    vocab = {}
    with open(filepath) as f:
        for line in f.readlines():
            entry = AbbreviationEntry(**json.loads(line))
            if entry.probability >= keep_threshold:
                if lowercase:
                    vocab[entry.token_clean.lower()] = entry
                else:
                    vocab[entry.token_clean] = entry
    return vocab

def _dict_to_vocab_file(outpath, vocab_dict, threshold=0, order_by='count'):
    with open(outpath, 'w', encoding='utf-8') as vocab_file:
        if order_by == 'name':
            ordered_items = sorted(vocab_dict.items(), key= lambda x: x[0], reverse=False)
        elif order_by == 'vocab_entry':
            ordered_items = sorted(vocab_dict.items(), key= lambda x: x[1].probability, reverse=True)
        else: # 'count'
            ordered_items = sorted(vocab_dict.items(), key= lambda x: x[1], reverse=True)
        for item in ordered_items:
            if isinstance(item[1], int):
                if item[1] == threshold: break
                vocab_file.write(f"{item[0]}\t{item[1]}\n")
            elif isinstance(item[1], AbbreviationEntry):
                vocab_file.write(json.dumps(item[1]._asdict()) + "\n")


def load_hunspell_file(vocab_path:str) -> Set[str]:
    full_words = set() 
    with open(vocab_path) as f:
        for line in f.readlines():
            if '/' in line:
                word, _ = line.strip('\n').split('/')
            else:
                word = line.strip('\n')
            full_words.add(word.lower())
    return full_words


def build_dictionaries(bios: List, nlp: NLP_Pipeline, language_vocab: Set, file_prefix: str):
    count_abbr, count_before_period, count_before_any = Counter(), Counter(), Counter()
    all_seen = Counter()
    for i, (bio_id, bio_text) in enumerate(bios):
        doc = nlp(bio_text)
        tokens = [tok.text for sentence in doc.sentences for tok in sentence.words]

        for tok, next_tok in zip(tokens, tokens[1:]):
            my_tok_txt = tok #.lower() # Case sensitivity can actually be valuable information to know if it is an abbreviation!!
            if len(my_tok_txt) > 1 and my_tok_txt[-1] == '.':
                    my_tok_txt = my_tok_txt[:-1] # TODO: How to gracefully handle H.T.V type of acronyms? Maybe this is fine already?
                    count_abbr[my_tok_txt] += 1
                    all_seen[my_tok_txt] += 1
            elif has_no_numbers(tok):
                all_seen[my_tok_txt] += 1
                if my_tok_txt.lower() not in language_vocab and next_tok == '.':
                    count_before_period[my_tok_txt] += 1
                else:
                    count_before_any[tok] += 1
    
    abbreviation_candidates = Counter()
    for token, tot_count in all_seen.items():
        a = count_abbr.get(token, 0) # Number of times the tokenizer identified it as an abbreviation
        b = count_before_period.get(token, 0) # Number of times the word appears before a '.'
        abbr_probability = (a+b)/tot_count
        # Heuristic: We will keep record of anything which the tokenizer already knows (a > 0)
        # we also add all of a+b higher than threshold i.e. it appears consistently in several places and is not only a tokenization error
        # Higher threshold means higher precision and lower recall
        threshold = 5
        if a > 0: 
            abbreviation_candidates[token] = AbbreviationEntry(f"{token}.", clean_token(token, lowercase=False), a, b, tot_count, 1.0)
        elif (a+b) >= threshold:
            abbreviation_candidates[token] = AbbreviationEntry(f"{token}.", clean_token(token, lowercase=False), a, b, tot_count, abbr_probability)


    # print(corpus_words.most_common(10))
    print(count_abbr.most_common(10))
    print(abbreviation_candidates.most_common(10))
    _dict_to_vocab_file(f'data/{file_prefix}_AbbrSeed.vocab', count_abbr, threshold=0, order_by='count')
    _dict_to_vocab_file(f'data/{file_prefix}_AbbrCandidates.vocab', abbreviation_candidates, threshold=0, order_by='vocab_entry')


def create_token_classification_data(data: List[Tuple[str, str]], candidates_filename: str, output_path: str):
    """This function builds the dataset for STEP 1: Token Classification [ABBR, NO-ABBR] for each Token in the dataset
        We save it in a file so it can be later loaded as a HuggingFace Dataset and make batched experiments with it!

    Args:
        data (List[Tuple[doc_id, doc_text]]): The list of raw_text examples with their document ids
    """
    # Load Abbreviation Dictionaries
    abbr_candidates = load_vocab_entry_file(candidates_filename, keep_threshold=0.8, lowercase=False)
    # Save the training file for the Token Classification Task
    with open(output_path, "w", encoding='utf-8') as fout:
        for doc_id, doc_text in data:
            doc_tokens = doc_text.split() # Naive Tokens
            for tok in doc_tokens:
                clean_tok = clean_token(tok, lowercase=True)
                if clean_tok in abbr_candidates and tok[-1] == '.':
                    lbl = 'B-ABBR'
                else:
                    lbl = 'O'
                fout.write(json.dumps({"token": tok, "clean": clean_tok, "gold_label": lbl, "document_id": doc_id}) + "\n")




def get_slovene_texts(filepath: str) -> Generator:
    with open(filepath) as f:
        for line in f.readlines():
            obj = json.loads(line)
            yield (obj['document_id'], obj['abbreviated_text'])


def run_slovene_pipeline():
    BIO_FILE = "data/sbl-51abbr.sentences.train.json"
    slovene_bios = get_slovene_texts(BIO_FILE)
    vocab = load_hunspell_file("resources/emnlp2022_data/hunspell_sl_SI.dic.txt")
    nlp = classla.Pipeline('sl', processors='tokenize,pos,lemma')
    build_dictionaries(slovene_bios, nlp, vocab, file_prefix="SloBio")
    # Build Abbreviation TokenClassification Dataset using the lexicons
    slovene_bios = get_slovene_texts(BIO_FILE)
    create_token_classification_data(slovene_bios, candidates_filename="data/SloBio_AbbrCandidates.vocab", output_path="data/slovene_abbr_corpus.jsonl")


if __name__ == '__main__':
    run_slovene_pipeline()


    
