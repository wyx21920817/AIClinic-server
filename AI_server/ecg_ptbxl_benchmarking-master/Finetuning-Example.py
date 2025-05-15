from utils import utils

sampling_frequency=100
datafolder='../data/ptbxl/'
task='superdiagnostic'
outputfolder='../output/'

# Load PTB-XL data
data, raw_labels = utils.load_dataset(datafolder, sampling_frequency)
# Preprocess label data
labels = utils.compute_label_aggregations(raw_labels, datafolder, task)
# Select relevant data and convert to one-hot
data, labels, Y, _ = utils.select_data(data, labels, task, min_samples=0, outputfolder=outputfolder)

# 1-9 for training 
X_train = data[labels.strat_fold < 10]
y_train = Y[labels.strat_fold < 10]
# 10 for validation
X_val = data[labels.strat_fold == 10]
y_val = Y[labels.strat_fold == 10]

num_classes = 5         # <=== number of classes in the finetuning dataset
input_shape = [1000,12] # <=== shape of samples, [None, 12] in case of different lengths

X_train.shape, y_train.shape, X_val.shape, y_val.shape

from models.fastai_model import fastai_model

experiment = 'exp0'
modelname = 'fastai_xresnet1d101'
pretrainedfolder = '../output/'+experiment+'/models/'+modelname+'/'
mpath='../output/' # <=== path where the finetuned model will be stored
n_classes_pretrained = 71 # <=== because we load the model from exp0, this should be fixed because this depends the experiment

model = fastai_model(
    modelname, 
    num_classes, 
    sampling_frequency, 
    mpath, 
    input_shape=input_shape, 
    pretrainedfolder=pretrainedfolder,
    n_classes_pretrained=n_classes_pretrained, 
    pretrained=True,
    epochs_finetuning=2,
)

import pickle

standard_scaler = pickle.load(open('../output/'+experiment+'/data/standard_scaler.pkl', "rb"))

X_train = utils.apply_standardizer(X_train, standard_scaler)
X_val = utils.apply_standardizer(X_val, standard_scaler)

model.fit(X_train, y_train, X_val, y_val)

y_val_pred = model.predict(X_val)
utils.evaluate_experiment(y_val, y_val_pred)

