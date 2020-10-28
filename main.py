import argparse
import os
import torch
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch import nn
import torch.nn.functional as F
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from torch.utils.data import random_split
from transformers import BertModel, BertConfig

from data import AskUbuntuTrainDataset, AskUbuntuDevDataset
from models import LitBertModel
from utils import CheckpointEveryNSteps



parser = argparse.ArgumentParser(description='Train a model to do information retrieval on askubuntu dataset')

parser.add_argument('--save_iters', type=int, default=500,
                    help='The amount of steps to save a model')
parser.add_argument('--data_dir', default="./data/askubuntu/",
                    help='The root of the data directory for the askubuntu dataset')
parser.add_argument('--model_dir', default="./models/",
                    help='The directory to save trained/checkpointed models')
parser.add_argument('--model_save_name', default="test_name",
                    help='The name to save the model with')
parser.add_argument('--model_type', default="bert",
                    help='The type of model that we use for evaluation')
parser.add_argument('--train_batch_size', default=16,
                    help='The batch size for training')
parser.add_argument('--val_batch_size', default=16,
                    help='The batch size for validating')
parser.add_argument('--cache_dir', default="./cache/",
                    help='The directory to store all caches that we generate.')

args = parser.parse_args()

data_dir = args.data_dir
model_name = args.model_save_name
model_type = args.model_type
model_dir = args.model_dir
save_iters = args.save_iters
train_batch_size = args.train_batch_size
val_batch_size = args.val_batch_size
train_loader = DataLoader(AskUbuntuTrainDataset(toy_n=20, toy_pad=20, root_dir=data_dir), batch_size=train_batch_size, shuffle=True)
val_loader = DataLoader(AskUbuntuDevDataset(root_dir=data_dir), batch_size=16)
# init model
autoencoder = LitBertModel()

# most basic trainer, uses good defaults (auto-tensorboard, checkpoints, logs, and more)
# trainer = pl.Trainer(gpus=8) (if you have GPUs)
# train on 8 CPUs
# trainer = pl.Trainer(num_processes=8)
# train on 1 GPU
# trainer = pl.Trainer(gpus=1)

# train on TPUs using 16 bit precision
# using only half the training data and checking validation every quarter of a training epoch
# trainer = pl.Trainer(
#     tpu_cores=8,
#     precision=16,
#     limit_train_batches=0.5,
#     val_check_interval=0.25
# )

# TODO Logging
# TODO Early stopping or epochs?
# TODO auto lr finder
# TODO Hyperparameters
# TODO FINISH DATA CACHE.
# TODO model dirs
# tODO scripts
# todo tensorboard
# TODO METRICS https://pytorch-lightning.readthedocs.io/en/latest/metrics.html


# checkpoints
checkpoints = CheckpointEveryNSteps(save_iters)
#Early stopping #TODO model must implement a logging for 'val_loss'
early_stopping = EarlyStopping(monitor='val_loss')
trainer = pl.Trainer(callbacks=[checkpoints,early_stopping],check_val_every_n_epoch=1, val_check_interval=0.25)
trainer.fit(autoencoder, train_loader, val_loader)
trainer.run_evaluation()
