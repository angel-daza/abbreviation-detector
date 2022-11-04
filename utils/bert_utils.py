from typing import List, Dict, Tuple
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from keras.preprocessing.sequence import pad_sequences
import torch
import numpy as np
import json, datetime, os
from transformers.utils import logging
import logging, re
from seqeval.metrics import f1_score, precision_score, recall_score, classification_report
from transformers.utils.dummy_pt_objects import AutoModel

logger = logging.getLogger(__name__)

def get_torch_device(verbose: bool = True, gpu_ix: int = 0) -> Tuple[torch.device, bool]:
    use_cuda = False
    if torch.cuda.is_available():
        # Tell PyTorch to use the GPU.
        device = torch.device("cuda")
        use_cuda = True
        if verbose:
            logger.info('There are %d GPU(s) available.' % torch.cuda.device_count())
            logger.info('We will use the GPU:', torch.cuda.get_device_name(gpu_ix))
    else:
        if verbose: logger.info('No GPU available, using the CPU instead.')
        device = torch.device("cpu")
    return device, use_cuda


device, USE_CUDA = get_torch_device(verbose=False)
LongTensor = torch.LongTensor

##### Data Loading Functions ##### 

def wordpieces_to_tokens(wordpieces: List, labelpieces: List = None, wordpiece_symbol: str = '##') -> Tuple[List, List]:
    textpieces = " ".join(wordpieces)
    full_words = re.sub(rf'\s{wordpiece_symbol}', '', textpieces).split()
    full_labels = []
    if labelpieces:
        for ix, wp in enumerate(wordpieces):
            if not wp.startswith(wordpiece_symbol):
                full_labels.append(labelpieces[ix])
        assert len(full_words) == len(full_labels)
    return full_words, full_labels


def expand_abbr_to_wordpieces(original_sentence: List, tokenizer: AutoTokenizer, wordpiece_symbol: str = '##', original_labels: List=None) -> Tuple[List, List]:
    """
    Also Expands BIO, but assigns the original label ONLY to the Head of the WordPiece (First WP)
    :param original_sentence: List of Full-Words
    :param original_labels: List of Labels corresponding to each Full-Word
    :param tokenizer: To convert it into BERT-model WordPieces
    :return:
    """
    txt_sentence = " ".join(original_sentence)
    txt_sentence = txt_sentence.replace(wordpiece_symbol, "")
    word_pieces = tokenizer.tokenize(txt_sentence)
    expanded_labels = ['O'] + original_labels * len(word_pieces) + ['O']
    word_pieces = ["[CLS]"] + word_pieces + ["[SEP]"]

    return word_pieces, expanded_labels
        

def abbr_data_to_tensors(dataset: List, tokenizer: AutoTokenizer, max_len: int, labels: List=None, label2index: Dict=None, pad_token_label_id: int=-100) -> Tuple:
    tokenized_sentences, label_indices = [], []
    for i, sentence in enumerate(dataset):
        # Get WordPiece Indices
        if labels and label2index:
            wordpieces, labelset = expand_abbr_to_wordpieces(sentence, tokenizer, wordpiece_symbol="▁", original_labels=labels[i])
            label_indices.append([label2index.get(lbl, pad_token_label_id) for lbl in labelset])
        else:
             wordpieces, labelset = expand_abbr_to_wordpieces(sentence, tokenizer, None)
        input_ids = tokenizer.convert_tokens_to_ids(wordpieces)
        tokenized_sentences.append(input_ids)

    seq_lengths = [len(s) for s in tokenized_sentences]
    logger.info(f"MAX TOKENIZED SEQ LENGTH IN DATASET IS {max(seq_lengths)}")
    # PAD ALL SEQUENCES
    input_ids = pad_sequences(tokenized_sentences, maxlen=max_len, dtype="long", value=0, truncating="post", padding="post")
    if label_indices:
        label_ids = pad_sequences(label_indices, maxlen=max_len, dtype="long", value=pad_token_label_id, truncating="post", padding="post")
        label_ids = LongTensor(label_ids)
    else:
        label_ids = None
    # Create attention masks
    attention_masks = []
    # For each sentence...
    for i, sent in enumerate(input_ids):
        # Create the attention mask.
        #   - If a token ID is 0, then it's padding, set the mask to 0.
        #   - If a token ID is > 0, then it's a real token, set the mask to 1.
        att_mask = [int(token_id > 0) for token_id in sent]
        # Store the attention mask for this sentence.
        attention_masks.append(att_mask)
    return LongTensor(input_ids), LongTensor(attention_masks), label_ids,  LongTensor(seq_lengths)



def add_to_label_dict(labels:List, label_dict: Dict) -> Dict:
    for l in labels:
        if l not in label_dict:
            label_dict[l] = len(label_dict)
    return label_dict


def read_abbr_tokens(filepath: str, has_labels: bool, sample_limit: int = -1) -> Tuple[List, List, Dict]:
    tokens, labels = [], []
    label2index = {}
    with open(filepath) as f:
        for i, line in enumerate(f.readlines()):
            if sample_limit > 0 and i > sample_limit: break
            row = json.loads(line.strip())
            tokens.append([row['token']])
            # DIRTY TRICK! For now we are ignoring BIO tags and treating as Binary classification (ABBR | NO_ABBR)
            if row['gold_label'] in ['B-ABBR']: # no: 'I-ABBR'
                labels.append(['ABBR'])
                add_to_label_dict(['ABBR'], label2index)
            else:
                labels.append([row['gold_label']])
                add_to_label_dict([row['gold_label']], label2index)
    assert len(labels) == len(tokens)
    return tokens, labels, label2index



def prepare_abbreviations_as_prompting(example: Dict, mask_token: str, sep_token: str, is_word: str) -> Dict:
    window_size = 20
    sentence = example['masked_tokens']
    mask_ix = sentence.index(mask_token)
    prev_ctx_ix = mask_ix - window_size if mask_ix - window_size > 0 else 0
    post_ctx_ix = mask_ix - window_size if mask_ix + window_size < len(sentence) else len(sentence)
    sentence[mask_ix] = example['candidate']
    full_prompt = sentence[prev_ctx_ix:post_ctx_ix] +  ['[SEP]'] + [example['candidate'], is_word, example['gold_expansion']]
    logger.info(full_prompt)
    return {'text': " ".join(full_prompt)}


def tokenize_function(examples, tokenizer):
    return tokenizer(examples["text"])


##### Evaluation Functions ##### 

def evaluate_bert_model(eval_dataloader: DataLoader, eval_batch_size: int, model: AutoModel, tokenizer:AutoTokenizer, label_map: dict, 
                        pad_token_label_id:int, wordpiece_symbol: str = '##', full_report:bool=False, prefix: str="") -> Tuple[Dict, List]:
    logger.info("***** Running evaluation %s *****", prefix)
    logger.info("  Batch size = %d", eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    input_ids, gold_label_ids = None, None
    # Put model on Evaluation Mode!
    model.eval()
    for batch in eval_dataloader:
        # Add batch to GPU
        batch = tuple(t.to(device) for t in batch)

        # Unpack the inputs from our dataloader
        b_input_ids, b_input_mask, b_labels, b_len = batch

        with torch.no_grad():
            outputs = model(b_input_ids, attention_mask=b_input_mask, labels=b_labels)
            tmp_eval_loss, logits = outputs[:2]
            eval_loss += tmp_eval_loss.item()
        nb_eval_steps += 1
        if preds is None:
            preds = logits.detach().cpu().numpy()
            gold_label_ids = b_labels.detach().cpu().numpy()
            input_ids = b_input_ids.detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            gold_label_ids = np.append(gold_label_ids, b_labels.detach().cpu().numpy(), axis=0)
            input_ids = np.append(input_ids, b_input_ids.detach().cpu().numpy(), axis=0)

    eval_loss = eval_loss / nb_eval_steps
    preds = np.argmax(preds, axis=2)

    gold_label_list = [[] for _ in range(gold_label_ids.shape[0])]
    pred_label_list = [[] for _ in range(gold_label_ids.shape[0])]
    full_word_preds = []

    logger.info(label_map)
    for seq_ix in range(gold_label_ids.shape[0]):
        for j in range(gold_label_ids.shape[1]):
            if gold_label_ids[seq_ix, j] != pad_token_label_id:
                gold_label_list[seq_ix].append(label_map[gold_label_ids[seq_ix][j]])
                pred_label_list[seq_ix].append(label_map[preds[seq_ix][j]])


        if full_report:
            wordpieces = tokenizer.convert_ids_to_tokens(input_ids[seq_ix], skip_special_tokens=True) 
            full_words, _ = wordpieces_to_tokens(wordpieces, labelpieces=None, wordpiece_symbol=wordpiece_symbol)
            full_preds = pred_label_list[seq_ix]
            full_gold = gold_label_list[seq_ix]
            full_word_preds.append((full_words, full_preds))
            logger.info(f"\n----- {seq_ix+1} -----\n{full_words}\n\nGOLD: {full_gold}\nPRED:{full_preds}\n")
           

    results = {
        "loss": eval_loss,
        "precision": precision_score(gold_label_list, pred_label_list),
        "recall": recall_score(gold_label_list, pred_label_list),
        "f1": f1_score(gold_label_list, pred_label_list),
    }

    if full_report:
        logger.info("\n\n"+classification_report(gold_label_list, pred_label_list))
    return results, full_word_preds


##### Input/Output Functions ##### 

def save_losses(losses: Dict, filename: str):
    out = open(filename, "w")
    out.write(json.dumps({"losses": losses})+"\n")


def save_label_dict(label2index: Dict, filename: str):
    out = open(filename, "w")
    out.write(json.dumps(label2index))


def save_label_dict(label2index: Dict, filename: str) -> None:
    out = open(filename, "w")
    out.write(json.dumps(label2index))


def load_label_dict(modelpath: str) -> Dict:
    fp = open(modelpath)
    label_dict = json.load(fp)
    return label_dict


# Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
def save_model(output_dir:str, arg_dict: Dict, model: AutoModel, tokenizer: AutoTokenizer):
    # Create output directory if needed
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    logger.info("Saving model to %s" % output_dir)
    # Save a trained model, configuration and tokenizer using `save_pretrained()`.
    # They can then be reloaded using `from_pretrained()`
    model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
    model_to_save.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    # Good practice: save your training arguments together with the trained model
    torch.save(arg_dict, os.path.join(output_dir, 'training_args.bin'))


def load_model(model_class, tokenizer_class, model_dir):
    # Load a trained model and vocabulary that you have fine-tuned
    model = model_class.from_pretrained(model_dir)
    tokenizer = tokenizer_class.from_pretrained(model_dir)
    # Copy the model to the GPU.
    model.to(device)
    return model, tokenizer


##### Misc Functions ##### 

def format_time(elapsed: float) -> str:
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))

    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


def get_bool_value(str_bool: str) -> bool:
    if str_bool.upper() == "TRUE" or str_bool.upper() == "T":
        return True
    else:
        return False