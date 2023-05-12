"""
    This is the training code for the BERT Binary Token Classifier [ABBR, NO_ABBR]
"""

from typing import List, Dict, Tuple
import random, time, os
import torch
from torch.nn import CrossEntropyLoss
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
import logging, sys, argparse
from transformers import AutoTokenizer, AutoModelForTokenClassification

import utils.bert_utils as utils


if __name__ == "__main__":
    """
    
    RUN EXAMPLE:
        
        python3 bert_token_classifier.py -t data/sbl-51abbr.tok.train.json -d data/sbl-51abbr.tok.dev.json --bert_model 'EMBEDDIA/sloberta' \
         --save_model_dir saved_models/BERT_ABBR_SL_876972 --epochs 1 --batch_size 32 --info_every 10 --seed_val 876972
        
        python3 bert_token_classifier.py -t data/outputs/german/apis-de-abbr.tok.train.json --bert_model 'bert-base-german-cased' \
         --save_model_dir saved_models/BERT_ABBR_DE_876972 --epochs 1 --batch_size 16 --info_every 10 --seed_val 876972

    """

    # =====================================================================================
    #                    GET PARAMETERS
    # =====================================================================================
    # Read arguments from command line
    parser = argparse.ArgumentParser()

    # GENERAL SYSTEM PARAMS
    parser.add_argument('-t', '--train_path', help='Filepath containing the Training JSON', required=True)
    parser.add_argument('-d', '--dev_path', help='Filepath containing the Validation JSON', default=None)
    parser.add_argument('-s', '--save_model_dir', required=True)
    parser.add_argument('-b', '--bert_model', default="bert-base-cased")
    parser.add_argument('-r', '--recover_epoch', default=None)
    parser.add_argument('-g', '--gpu', type=int, default=0)

    # NEURAL NETWORK PARAMS
    parser.add_argument('-sv', '--seed_val', type=int, default=1373)
    parser.add_argument('-ep', '--epochs', type=int, default=1)
    parser.add_argument('-bs', '--batch_size', type=int, default=8)
    parser.add_argument('-inf', '--info_every', type=int, default=100)
    parser.add_argument('-mx', '--max_len', type=int, default=20)
    parser.add_argument('-lr', '--learning_rate', type=float, default=2e-5)
    parser.add_argument('-gr', '--gradient_clip', type=float, default=1.0)

    args = parser.parse_args()

    # =====================================================================================
    #                    INITIALIZE PARAMETERS
    # =====================================================================================
    # To resume training of a model...
    if args.recover_epoch:
        START_EPOCH = int(args.recover_epoch)
        RECOVER_CHECKPOINT = True
    else:
        START_EPOCH = 0
        RECOVER_CHECKPOINT = False

    EPOCHS = args.epochs
    BERT_MODEL_NAME = args.bert_model
    DO_LOWERCASE = False
    GPU_RUN_IX=args.gpu
    WORDPIECE_SYMBOL="▁" # '##'

    SEED_VAL = args.seed_val
    SEQ_MAX_LEN = args.max_len
    PRINT_INFO_EVERY = args.info_every
    GRADIENT_CLIP = args.gradient_clip
    LEARNING_RATE = args.learning_rate
    BATCH_SIZE = args.batch_size

    TRAIN_DATA_PATH = args.train_path
    DEV_DATA_PATH = args.dev_path
    MODEL_DIR = args.save_model_dir
    LABELS_FILENAME = f"{MODEL_DIR}/label2index.json"
    LOSS_TRN_FILENAME = f"{MODEL_DIR}/Losses_Train_{EPOCHS}.json"
    LOSS_DEV_FILENAME = f"{MODEL_DIR}/Losses_Dev_{EPOCHS}.json"

    PAD_TOKEN_LABEL_ID = CrossEntropyLoss().ignore_index # -100

    if not os.path.exists(args.save_model_dir):
        os.makedirs(args.save_model_dir)

    # =====================================================================================
    #                    LOGGING INFO ...
    # =====================================================================================
    console_hdlr = logging.StreamHandler(sys.stdout)
    file_hdlr = logging.FileHandler(filename=f"{MODEL_DIR}/BERT_TokenClassifier_train_{START_EPOCH}_{EPOCHS}.log")
    logging.basicConfig(level=logging.INFO, handlers=[console_hdlr, file_hdlr])
    logging.info("Start Logging")
    logging.info(args)

    # Initialize Random seeds and validate if there's a GPU available...
    device, USE_CUDA = utils.get_torch_device(GPU_RUN_IX)
    random.seed(SEED_VAL)
    np.random.seed(SEED_VAL)
    torch.manual_seed(SEED_VAL)
    torch.cuda.manual_seed_all(SEED_VAL)

    # ==========================================================================================
    #               LOAD TRAIN & DEV DATASETS
    # ==========================================================================================
    # Initialize Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME, do_lower_case=DO_LOWERCASE, do_basic_tokenize=True)
    # Load Train Dataset
    train_data, train_labels, train_label2index = utils.read_abbr_tokens(TRAIN_DATA_PATH, has_labels=True, sample_limit=10000000)

    train_inputs, train_masks, train_labels, seq_lengths = utils.abbr_data_to_tensors(train_data, 
                                                                                 tokenizer, 
                                                                                 max_len=SEQ_MAX_LEN, 
                                                                                 labels=train_labels, 
                                                                                 label2index=train_label2index,
                                                                                 pad_token_label_id=PAD_TOKEN_LABEL_ID)
    utils.save_label_dict(train_label2index, filename=LABELS_FILENAME)
    index2label = {v: k for k, v in train_label2index.items()}

    # Create the DataLoader for our training set.
    train_data = TensorDataset(train_inputs, train_masks, train_labels)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=BATCH_SIZE)

    # Load Dev Dataset
    if DEV_DATA_PATH:
        dev_data, dev_labels, _ = utils.read_abbr_tokens(DEV_DATA_PATH, has_labels=True)
        dev_inputs, dev_masks, dev_labels, dev_lens = utils.abbr_data_to_tensors(dev_data, 
                                                                                 tokenizer, 
                                                                                max_len=SEQ_MAX_LEN, 
                                                                                labels=dev_labels, 
                                                                                label2index=train_label2index,
                                                                                pad_token_label_id=PAD_TOKEN_LABEL_ID)

        # Create the DataLoader for our Development set.
        dev_data = TensorDataset(dev_inputs, dev_masks, dev_labels, dev_lens)
        dev_sampler = RandomSampler(dev_data)
        dev_dataloader = DataLoader(dev_data, sampler=dev_sampler, batch_size=BATCH_SIZE)

    # ==========================================================================================
    #              LOAD MODEL & OPTIMIZER
    # ==========================================================================================
    if RECOVER_CHECKPOINT:
        model, tokenizer = utils.load_model(AutoModelForTokenClassification, AutoTokenizer, f"{MODEL_DIR}/EPOCH_{START_EPOCH}")
    else:
        model = AutoModelForTokenClassification.from_pretrained(BERT_MODEL_NAME, num_labels=len(train_label2index))
        model.config.finetuning_task = 'token-classification'
        model.config.id2label = index2label
        model.config.label2id = train_label2index
    if USE_CUDA: model.cuda()

    # Total number of training steps is number of batches * number of epochs.
    total_steps = len(train_dataloader) * EPOCHS

    # Create optimizer and the learning rate scheduler.
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,
                                                num_training_steps=total_steps)

    # ==========================================================================================
    #                          TRAINING ...
    # ==========================================================================================
    # Store the average loss after each epoch so we can plot them.
    loss_trn_values, loss_dev_values = [], []

    # For each epoch...
    for epoch_i in range(START_EPOCH+1, EPOCHS+1):
        # Perform one full pass over the training set.
        logging.info("")
        logging.info('======== Epoch {:} / {:} ========'.format(epoch_i, EPOCHS))
        logging.info('Training...')

        t0 = time.time()
        total_loss = 0
        model.train()

        # For each batch of training data...
        for step, batch in enumerate(train_dataloader):
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            model.zero_grad()

            # Perform a forward pass (evaluate the model on this training batch).
            outputs = model(b_input_ids, attention_mask=b_input_mask, labels=b_labels)
            loss = outputs[0]
            total_loss += loss.item()

            # Perform a backward pass to calculate the gradients.
            loss.backward()

            # Clip the norm of the gradients to 1.0.
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)

            # Update parameters
            optimizer.step()
            scheduler.step()

            # Progress update
            if step % PRINT_INFO_EVERY == 0 and step != 0:
                # Calculate elapsed time in minutes.
                elapsed = utils.format_time(time.time() - t0)
                # Report progress.
                logging.info('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.    Loss: {}.'.format(step, len(train_dataloader),
                                                                                                elapsed, loss.item()))

        # Calculate the average loss over the training data.
        avg_train_loss = total_loss / len(train_dataloader)

        # Store the loss value for plotting the learning curve.
        loss_trn_values.append(avg_train_loss)

        logging.info("")
        logging.info("  Average training loss: {0:.4f}".format(avg_train_loss))
        logging.info("  Training Epoch took: {:}".format(utils.format_time(time.time() - t0)))

        # ================================================
        #               Save Checkpoint for this Epoch
        # ================================================
        utils.save_model(f"{MODEL_DIR}/EPOCH_{epoch_i}", {"args":[]}, model, tokenizer)


        # ========================================
        #               Validation
        # ========================================
        if DEV_DATA_PATH:
            # After the completion of each training epoch, measure our performance on
            # our validation set.
            t0 = time.time()
            results, preds_list = utils.evaluate_bert_model(dev_dataloader, BATCH_SIZE, model, tokenizer, index2label, PAD_TOKEN_LABEL_ID, prefix="Validation Set", wordpiece_symbol=WORDPIECE_SYMBOL)
            loss_dev_values.append(results['loss'])
            logging.info("  Validation Loss: {0:.2f}".format(results['loss']))
            logging.info("  Precision: {0:.2f} || Recall: {1:.2f} || F1: {2:.2f}".format(results['precision']*100, results['recall']*100, results['f1']*100))
            logging.info("  Validation took: {:}".format(utils.format_time(time.time() - t0)))

            with open(f"{MODEL_DIR}/EPOCH_{epoch_i}/inputs.txt", "w") as fout:
                with open(f"{MODEL_DIR}/EPOCH_{epoch_i}/outputs.txt", "w") as fin:
                    for sent, pred in preds_list:
                        fin.write(" ".join(sent)+"\n")
                        fout.write(" ".join(pred)+"\n")

    

    utils.save_losses(loss_trn_values, filename=LOSS_TRN_FILENAME)
    utils.save_losses(loss_dev_values, filename=LOSS_DEV_FILENAME)

    logging.info("")
    logging.info("Training complete!")