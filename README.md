# OpinioNet

OpinioNet is a Chinese sentiment analysis and opinion mining project using BERT-based models. The project includes training, evaluation, and prediction pipelines for aspect-based sentiment analysis (ABSA).

## Features

- Pre-trained Chinese BERT models (RoBERTa-wwm-ext)
- Aspect-based sentiment analysis
- Data augmentation techniques
- Cross-validation support
- Ensemble evaluation methods

## Requirements

- Python 3.6+
- PyTorch 1.1.0
- Transformers (pytorch-pretrained-bert 0.6.2)
- jieba for Chinese text segmentation
- tqdm for progress bars

See `requirements.txt` for details.

## Installation

1. Clone the repository:
   ```bash
   git clone git@github.com:guohuiyuan/OpinioNet.git
   cd OpinioNet
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download pre-trained models:
   ```bash
   python download_models.py
   ```

## Usage

### Training
```bash
bash train_script.sh
```
or
```bash
python src/train.py
```

### Evaluation
```bash
bash eval_script.sh
```
or
```bash
python src/eval.py
```

### Prediction with LLM
```bash
python llm_predict.py --mode train  # for training data
python llm_predict.py --mode test   # for test data
```

### Data Conversion
Convert ABSA data to JSONL format:
```bash
python convert_absa_to_jsonl.py
```

## Project Structure

```
OpinioNet/
├── README.md                 # This file
├── README.txt               # Original README with screen commands
├── requirements.txt         # Python dependencies
├── .gitignore              # Git ignore rules
├── download_models.py      # Script to download pre-trained models
├── convert_absa_to_jsonl.py # Convert ABSA data to JSONL format
├── llm_predict.py          # LLM-based prediction script
├── llm_eval.py             # LLM evaluation script
├── label_corpus.sh         # Shell script for labeling corpus
├── train_script.sh         # Training script
├── eval_script.sh          # Evaluation script
├── data/                   # Data directory
│   ├── TRAIN/              # Training data
│   │   ├── train.jsonl
│   │   ├── Train_reviews.csv
│   │   ├── Train_labels.csv
│   │   ├── Result.csv
│   │   └── intermediate_results.jsonl
│   └── TEST/               # Test data
│       ├── Test_reviews.csv
│       ├── Result.csv
│       ├── Result(example).csv
│       └── intermediate_results.jsonl
├── models/                 # Model files
│   ├── thresh_dict.json   # Threshold dictionary
│   ├── chinese_roberta_wwm_ext_pytorch/  # RoBERTa-wwm-ext model
│   │   ├── config.json
│   │   ├── pytorch_model.bin
│   │   ├── tokenizer.json
│   │   ├── vocab.txt
│   │   └── ...
│   └── chinese_wwm_ext_pytorch/          # BERT-wwm-ext model
│       ├── config.json
│       ├── pytorch_model.bin
│       ├── tokenizer.json
│       ├── vocab.txt
│       └── ...
└── src/                    # Source code
    ├── config.py           # Configuration settings
    ├── dataset.py          # Dataset loading and preprocessing
    ├── model.py            # Model architecture
    ├── train.py            # Training pipeline
    ├── train_round2.py     # Second round training
    ├── train_cv.py         # Cross-validation training
    ├── eval.py             # Evaluation pipeline
    ├── eval_round2.py      # Second round evaluation
    ├── eval_ensemble.py    # Ensemble evaluation
    ├── eval_ensemble_round2.py
    ├── eval_ensemble_final.py
    ├── pretrain.py         # Pre-training
    ├── pretrain2.py        # Second pre-training
    ├── pretrain2_cv.py     # Cross-validation pre-training
    ├── finetune_cv.py      # Cross-validation fine-tuning
    ├── test_cv.py          # Cross-validation testing
    ├── test_ensemble_cv.py # Ensemble cross-validation testing
    ├── data_aug.py         # Data augmentation
    ├── data_augmentation.py
    └── lr_scheduler.py     # Learning rate scheduler
```

## File Descriptions

### Key Scripts

- **download_models.py**: Downloads pre-trained BERT models from Hugging Face
- **convert_absa_to_jsonl.py**: Converts CSV data to JSONL format for training
- **llm_predict.py**: Runs LLM-based predictions on training or test data
- **llm_eval.py**: Evaluates LLM predictions

### Source Code

- **src/train.py**: Main training script
- **src/eval.py**: Main evaluation script
- **src/model.py**: BERT-based model architecture
- **src/dataset.py**: Data loading and preprocessing
- **src/config.py**: Configuration parameters

### Data Files

- **data/TRAIN/train.jsonl**: Training data in JSONL format
- **data/TRAIN/Train_reviews.csv**: Original training reviews
- **data/TRAIN/Train_labels.csv**: Training labels
- **data/TEST/Test_reviews.csv**: Test reviews
- **data/TEST/Result.csv**: Prediction results

### Model Files

- **models/chinese_roberta_wwm_ext_pytorch/**: Pre-trained RoBERTa-wwm-ext model
- **models/chinese_wwm_ext_pytorch/**: Pre-trained BERT-wwm-ext model
- **models/thresh_dict.json**: Threshold dictionary for classification

## Training Commands

The original README.txt includes screen commands for running predictions:

```bash
screen -L -Logfile test.log -dmS test bash -c "/new_disk/med_group/ghy/miniconda3/bin/python /new_disk/med_group/ghy/code/OpinioNet/llm_predict.py --mode test"
screen -L -Logfile train.log -dmS train bash -c "/new_disk/med_group/ghy/miniconda3/bin/python /new_disk/med_group/ghy/code/OpinioNet/llm_predict.py --mode train"
```

## License

This project is for research purposes.

## Contact

For questions or issues, please contact the repository maintainer.
