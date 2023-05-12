from collections import defaultdict
import torch
from torch.nn import CrossEntropyLoss
import numpy as np
import argparse
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
import logging, sys
# Our code behind the scenes!
import utils.bert_utils as utils


if __name__ == "__main__":
    """
    RUN EXAMPLE:
        python3 bert_token_classifier_predict.py -m saved_models/BERT_ABBR_SL_876972 --bert_model 'EMBEDDIA/sloberta' \
            --epoch 1 --test_path data/outputs/slovenian/sbl-51abbr.tok.test.json --gold_labels True
        
        python3 bert_token_classifier_predict.py -m saved_models/BERT_ABBR_DE_876972 --bert_model 'bert-base-german-cased' \
            --epoch 1 --test_path data/outputs/german/apis-de-abbr.tok.test.json --gold_labels True
 
    """

    # =====================================================================================
    #                    GET PARAMETERS
    # =====================================================================================
    # Read arguments from command line
    parser = argparse.ArgumentParser()

    # GENERAL SYSTEM PARAMS
    parser.add_argument('-t', '--test_path', help='Filepath containing the JSON File to Predict', required=True)
    parser.add_argument('-m', '--model_dir', required=True)
    parser.add_argument('-l', '--lang', default="EN")
    parser.add_argument('-e', '--epoch', help="Epoch to Load model from", required=True)
    parser.add_argument('-g', '--gold_labels', default="False")
    parser.add_argument('-bm', '--bert_model', default="bert-base-cased")
    parser.add_argument('-b', '--batch_size', default=1, help="For BEST results: same value as when training the Model")
    parser.add_argument('-mx', '--seq_max_len', default=256, help="BEST results: same value as when training the Model")
    parser.add_argument('--gpu', type=int, default=0)

    args = parser.parse_args()

    confusion_dict = defaultdict(list)
    console_hdlr = logging.StreamHandler(sys.stdout)
    file_hdlr = logging.FileHandler(filename=f"{args.model_dir}/BERT_TokenClassifier_{args.epoch}_predictions.log")
    logging.basicConfig(level=logging.INFO, handlers=[console_hdlr, file_hdlr])

    device, USE_CUDA = utils.get_torch_device(args.gpu)
    file_has_gold = utils.get_bool_value(args.gold_labels)
    SEQ_MAX_LEN = int(args.seq_max_len)
    BATCH_SIZE = int(args.batch_size)
    TEST_DATA_PATH = args.test_path
    MODEL_DIR = args.model_dir
    OUTPUTS_PATH=f"{MODEL_DIR}/outputs.txt"
    INPUTS_PATH=f"{MODEL_DIR}/inputs.txt"
    PAD_TOKEN_LABEL_ID = CrossEntropyLoss().ignore_index # -100

    # Load Saved Model
    model, tokenizer = utils.load_model(AutoModelForTokenClassification, AutoTokenizer, f"{MODEL_DIR}/EPOCH_{args.epoch}")
    label2index = utils.load_label_dict(f"{MODEL_DIR}/label2index.json")
    index2label = {v:k for k,v in label2index.items()}

    # Load File for Predictions
    test_data, test_labels, _ = utils.read_abbr_tokens(TEST_DATA_PATH, has_labels=True)
    prediction_inputs, prediction_masks, gold_labels, seq_lens = utils.abbr_data_to_tensors(test_data, 
                                                                                 tokenizer, 
                                                                                 max_len=SEQ_MAX_LEN, 
                                                                                 labels=test_labels, 
                                                                                 label2index=label2index,
                                                                                 pad_token_label_id=PAD_TOKEN_LABEL_ID)


    # Create the DataLoader.
    if file_has_gold:
        prediction_data = TensorDataset(prediction_inputs, prediction_masks, gold_labels, seq_lens)
        prediction_sampler = SequentialSampler(prediction_data)
        prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=BATCH_SIZE)

        logging.info('Predicting labels for {:,} test sentences...'.format(len(prediction_inputs)))
        
        results, preds_list = utils.evaluate_bert_model(prediction_dataloader, BATCH_SIZE, model, tokenizer, index2label, 
                                                         PAD_TOKEN_LABEL_ID, full_report=True, prefix="Test Set")
        logging.info("  Test Loss: {0:.2f}".format(results['loss']))
        logging.info("  Precision: {0:.2f} || Recall: {1:.2f} || F1: {2:.2f}".format(results['precision']*100, results['recall']*100, results['f1']*100))

        with open(OUTPUTS_PATH, "w") as fout:
            with open(INPUTS_PATH, "w") as fin:
                for sent, pred in preds_list:
                    fin.write(" ".join(sent)+"\n") # Avoid the [CLS] and the [SEP]
                    fout.write(" ".join(pred[1:-1])+"\n")

    else:
        # https://huggingface.co/transformers/main_classes/pipelines.html
        # https://huggingface.co/transformers/main_classes/pipelines.html#transformers.TokenClassificationPipeline
        logging.info('Predicting labels for {:,} test sentences...'.format(len(test_data)))
        nlp = pipeline('token-classification', model=model, tokenizer=tokenizer, device=args.gpu)
        nlp.ignore_labels = []
        with open(OUTPUTS_PATH, "w") as fout:
            with open(INPUTS_PATH, "w") as fin:
                for seq_ix, seq in enumerate(test_data):
                    sentence = " ".join(seq)
                    predicted_labels = []
                    output_obj = nlp(sentence)
                    for tok in output_obj:
                        if '##' not in tok['word']:
                            predicted_labels.append(tok['entity'])
                    logging.info(f"\n----- {seq_ix+1} -----\n{seq}\nPRED:{predicted_labels}")
                    fin.write(sentence+"\n")
                    fout.write(" ".join(predicted_labels)+"\n")
