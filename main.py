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
from models import *
from utils import CheckpointEveryNSteps
# torch.autograd.set_detect_anomaly(True)
parser = ArgumentParser(description='Train a model to do information retrieval on askubuntu dataset')

parser.add_argument('--save_iters', type=int, default=1000,
                    help='The amount of steps to save a model')
parser.add_argument('--epochs', type=int, default=5,
                    help='The amount of epochs to run during training')
parser.add_argument('--data_dir', default="./data/askubuntu/",
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
parser.add_argument('--pad_len',default=128,type=int,
                    help="The amount of padding that will be added or the length that we will cut the sequences to.")
parser.add_argument('--gradient_acc_batches',default=1,type=int,
                    help="The amount batches that we accumulate the gradient to")
parser.add_argument('--toy_n',default=None,type=int,
                    help="N amount of examples will be used in training.")

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
pad_len = args.pad_len
fp16 = 16 if args.fp16 else 32
num_workers = 0
grad_acc_batchs = args.gradient_acc_batches
toy_n = args.toy_n

if cache_dir is not None:
    os.makedirs(cache_dir, exist_ok=True)


datasets_and_models = [
    # ({"pad_len":pad_len,"data_dir":data_dir,"batch_size":train_batch_size,"cache_dir":cache_dir,"num_workers":num_workers,"toy_n":toy_n},
    #        LitBertModel,{"accumulate_grad_batches":grad_acc_batchs},"basebert"),
    ({"pad_len":pad_len,"data_dir":data_dir,"batch_size":train_batch_size,"cache_dir":cache_dir,"num_workers":num_workers,"toy_n":toy_n},
           LitInputBertModel,{"accumulate_grad_batches":grad_acc_batchs},"basebert"),
    # ({"pad_len":pad_len,"data_dir":data_dir,"batch_size":train_batch_size,"cache_dir":cache_dir,"num_workers":num_workers,"toy_n":toy_n},
    #        LitOutputBertModel, {"accumulate_grad_batches":grad_acc_batchs},"outputbert"),
    # ({"pad_len":pad_len,"data_dir":data_dir,"batch_size":train_batch_size,"cache_dir":cache_dir,"num_workers":num_workers,"toy_n":toy_n},
    #        LitOutputBaseModel, {"accumulate_grad_batches":grad_acc_batchs},"outputbase"),
    # ({"pad_len":pad_len,"data_dir":data_dir,"batch_size":2,"cache_dir":cache_dir,"num_workers":num_workers,"toy_n":toy_n},
    #        LitKnowBERTModel,{"accumulate_grad_batches":8},"knowbert")
]
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

# TODO Early stopping or epochs?
# TODO auto lr finder
# TODO Hyperparameters
# TODO model dirs
# tODO scripts



for idx,tup in enumerate(datasets_and_models):
    try:
        ds = tup[0]
        model = tup[1]
        train_p = tup[2]
        model_name = tup[3]
        # checkpoints
        checkpoints = CheckpointEveryNSteps(save_iters)
        # Early stopping
        early_stopping = EarlyStopping(monitor='val_loss_step')
        tb_logger = TensorBoardLogger(save_dir="tb_logs", name=
                                      "model_"+str(idx))

        print("Starting...")
        dataset = AskUbuntuDataModule(**ds)
        model = model()
        #Additional trainer params
        accumulate_grad_batches = train_p["accumulate_grad_batches"] if "accumulate_grad_batches" in train_p.keys() else None
        print(dataset,model)
        print("Initializing the trainer")
        trainer = pl.Trainer(callbacks=[checkpoints, early_stopping], gpus=1 if args.use_gpu else None,
                             auto_select_gpus=True,max_epochs=epochs,val_check_interval=0.1,check_val_every_n_epoch=1,
                             logger=tb_logger,precision=fp16,num_sanity_val_steps=0,
                             accumulate_grad_batches=accumulate_grad_batches)
        print("Fitting...")
        trainer.fit(model,datamodule=dataset)
        print("Testing...")
        trainer.test(datamodule=dataset)
        print("Done!")
        os.makedirs("models/"+model_name+"/")
        trainer.save_checkpoint("models/"+model_name+"/"+model_name+"_trained")
    except Exception as e:
        print(e)
