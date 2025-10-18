import os
import sys
import torch
import random
import numpy as np
import pickle
import matplotlib.pyplot as plt

# 1) Add code directory to path
HERE = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(HERE, 'code'))

# Project modules
from utils.utils import load_dataset, compute_label_aggregations, select_data, preprocess_signals
from models.fastai_model import fastai_model

# ------------------- Configuration -------------------
DATA_PATH            = 'data/ptbxl/'                               # Data root directory
SAMPLING_RATE        = 100                                         # Sampling rate
CTYPE                = 'superdiagnostic'                           # Task type
MIN_SAMPLES          = 0                                           # Minimum samples per class
# Fine-tuning specific output directory, no longer using output/exp0/data
FT_EXPERIMENT        = 'exp0_ft'
OUTPUT_DATA_FT       = f'output/{FT_EXPERIMENT}/data/'              # Fine-tuning output directory
OUTPUT_MODELS_FT     = f'output/{FT_EXPERIMENT}/models/'            # Fine-tuning model storage directory

PRETRAINED_NAME      = 'fastai_xresnet1d101'                       # Pretrained model identifier
PRETRAINED_FOLDER    = 'output/exp0/models/fastai_xresnet1d101/'   # Pretrained weights directory
EPOCHS_FT            = 10                                          # Fine-tuning epochs
SEED                 = 42
DEVICE               = 'cuda' if torch.cuda.is_available() else 'cpu'

# Assume all signals are 1000 length, 12 leads
INPUT_SHAPE = (1000, 12)

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main():
    set_seed()

    # Ensure output directories exist
    os.makedirs(OUTPUT_DATA_FT, exist_ok=True)
    os.makedirs(OUTPUT_MODELS_FT, exist_ok=True)

    # 1. Load and preprocess labels
    data, raw_labels = load_dataset(DATA_PATH, SAMPLING_RATE)
    labels_df       = compute_label_aggregations(raw_labels, DATA_PATH, CTYPE)
    data, labels_df, Y, _ = select_data(
        data, labels_df, CTYPE,
        min_samples=MIN_SAMPLES,
        outputfolder=OUTPUT_DATA_FT    # Changed to fine-tuning directory
    )

    print(f"Total samples: {len(data)}, label matrix shape: {Y.shape}")

    # 2. Use official strat_fold for train/validation split
    X_train = data[labels_df.strat_fold < 10]
    y_train = Y[labels_df.strat_fold < 10]
    X_val   = data[labels_df.strat_fold == 10]
    y_val   = Y[labels_df.strat_fold == 10]

    print(f"Train samples: {len(X_train)}, Val samples: {len(X_val)}")

    # 3. Standardization - exactly the same as pretraining
    #    preprocess_signals will save to OUTPUT_DATA_FT/standard_scaler.pkl
    X_train, X_val, _ = preprocess_signals(X_train, X_val, [], OUTPUT_DATA_FT)

    # Convert to list of arrays for fastai_model.fit call
    X_train_fin = [x for x in X_train]
    X_val_fin   = [x for x in X_val]
    y_train_fin = [y for y in y_train]
    y_val_fin   = [y for y in y_val]

    # 4. Instantiate and fine-tune fastai_model
    model = fastai_model(
        name                = PRETRAINED_NAME,
        n_classes           = y_train.shape[1],   # e.g. 5
        freq                = SAMPLING_RATE,
        outputfolder        = OUTPUT_MODELS_FT,   # Fine-tuning weights storage directory
        input_shape         = INPUT_SHAPE,
        pretrained          = True,
        pretrainedfolder    = PRETRAINED_FOLDER,
        n_classes_pretrained= 71,                 # Number of pretrained classes
        epochs_finetuning   = EPOCHS_FT,
        chunkify_train      = False,
    )

    # 5. Run fine-tuning (includes lr_find, earlystop, etc. internally)
    model.fit(X_train_fin, y_train_fin, X_val_fin, y_val_fin)

    # After fit, fastai_model has already called learn.save(self.name) internally
    # and exposes the learner through the following line
    learn = model.learn

    # 6. Additional manual save of state_dict (optional)
    torch.save(
        {'model': learn.model.state_dict()},
        os.path.join(OUTPUT_MODELS_FT, f'{PRETRAINED_NAME}_finetuned_manual.pth')
    )

    # 7. Plot and save LR Finder curve
    learn.recorder.plot()
    plt.savefig(os.path.join(OUTPUT_MODELS_FT, 'lr_find.png'))

if __name__ == '__main__':
    main()
