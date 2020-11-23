from argparse import ArgumentParser
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
from pytorch_lightning.loggers import TensorBoardLogger

from data import AskUbuntuDataModule
from models import LitKnowBERTModel,LitBertModel,LitOutputBertModel, LitInputBertModel
from utils import CheckpointEveryNSteps
torch.autograd.set_detect_anomaly(True)
parser = ArgumentParser(description='Train a model to do information retrieval on askubuntu dataset')

parser.add_argument('--save_iters', type=int, default=500,
                    help='The amount of steps to save a model')
parser.add_argument('--epochs', type=int, default=5,
                    help='The amount of epochs to run during training')
parser.add_argument('--data_dir', default="./data/askubuntu-master",
                    help='The root of the data directory for the askubuntu dataset')
parser.add_argument('--model_dir', default="./models/",
                    help='The directory to save trained/checkpointed models')
parser.add_argument('--model_save_name', default="test_name",
                    help='The name to save the model with')
parser.add_argument('--model_type', default="bert",
                    help='The type of model that we use for evaluation')
parser.add_argument('--train_batch_size', default=16, type=int,
                    help='The batch size for training')
parser.add_argument('--val_batch_size', default=16, type=int,
                    help='The batch size for validating')
parser.add_argument('--cache_dir', default="./cache/",
                    help='The directory to store all caches that we generate.')
parser.add_argument('--use_gpu', default=False, action="store_true",
                    help='The directory to store all caches that we generate.')
parser.add_argument('--fp16', default=False, action="store_true",
                    help='Use mixed precision training.')

args = parser.parse_args()

data_dir = args.data_dir
model_name = args.model_save_name
model_type = args.model_type
model_dir = args.model_dir
save_iters = args.save_iters
train_batch_size = args.train_batch_size
val_batch_size = args.val_batch_size
cache_dir = args.cache_dir
use_gpu = args.use_gpu
epochs = args.epochs
fp16 = 16 if args.fp16 else 32
num_workers = 0
if cache_dir is not None:
    os.makedirs(cache_dir, exist_ok=True)
datamodule = AskUbuntuDataModule(data_dir=data_dir,batch_size=train_batch_size,cache_dir=cache_dir,num_workers=num_workers)
# init model
# autoencoder = LitBertModel()

# autoencoder = LitKnowBERTModel() ##Requires --train_batch_size 2
autoencoder = LitInputBertModel()
tb_logger =TensorBoardLogger(save_dir="tb_logs",name=model_name)

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
# TODO model dirs
# tODO scripts


# checkpoints
checkpoints = CheckpointEveryNSteps(save_iters)
# Early stopping #TODO model must implement a logging for 'val_loss'
early_stopping = EarlyStopping(monitor='val_loss')
trainer = pl.Trainer(callbacks=[checkpoints, early_stopping], gpus=1 if args.use_gpu else None,
                     auto_select_gpus=True,max_epochs=epochs,check_val_every_n_epoch=5,
                     logger=tb_logger,precision=fp16,num_sanity_val_steps=0)
trainer.fit(autoencoder,datamodule=datamodule)
trainer.test(datamodule=datamodule)
