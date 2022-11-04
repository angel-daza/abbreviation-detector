# abbreviation-detector
Code to train classifiers for abbreviation detection and expansion in context. This repository also contains the evaluation code that complements the paper [Dealing with Abbreviations in the Slovenian Biographical Lexicon](https://2022.emnlp.org/) to be presented at The 2022 Conference on Empirical Methods in Natural Language Processing [EMNLP 2022](https://2022.emnlp.org/)


## Installation

Download repo

```bash
git clone git@github.com:angel-daza/abbreviation-detector.git
```

Create a new environment:
```bash
conda create -n abbr-detector python=3.9
conda activate abbr-detector
```

Install Requirements:

```bash
pip install -r requirements
```

## Paper Results

### Abbreviation Detection

Create the Dataset Train/Dev/Test Partitions:

```bash
python3 slovene_abbr_preprocess.py
```

To Reproduce the Baseline Results:

```bash
python3 naive_baselines.py
```

To Reproduce the BERT Abbreviation Classifier Results:

```bash
# 1) Train the Binary BERT Classifier [ABBR, NO_ABBR]
python3 bert_token_classifier.py -t data/sbl-51abbr.tok.train.json -d data/sbl-51abbr.tok.dev.json\
     --bert_model 'EMBEDDIA/sloberta' --save_model_dir saved_models/BERT_ABBR_876972\
     --epochs 5 --batch_size 32 --info_every 10 --seed_val 876972

# 2) Make predictions using the BERT Classifier
python3 bert_token_classifier_predict.py -m saved_models/BERT_ABBR_876972 --bert_model 'EMBEDDIA/sloberta'\
     --epoch 1 --test_path data/sbl-51abbr.tok.test.json --gold_labels True
```

### Abbreviation Expansion

To Reproduce BERT Abbreviation Expansion Results:

```bash
python3 bert_abbrev_expansion.py
```