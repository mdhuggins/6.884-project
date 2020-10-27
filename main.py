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

from data import AskUbuntuTrainDataset
from utils import CheckpointEveryNSteps


class LitAutoEncoder(pl.LightningModule):

    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28*28, 64),
            nn.ReLU(),
            nn.Linear(64, 3)
        )
        self.decoder = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 28*28)
        )

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        embedding = self.encoder(x)
        return embedding

    def training_step(self, batch, batch_idx):
        print(batch)

        labels = batch['label']

        pred = torch.FloatTensor([1]*len(labels))

        loss = F.mse_loss(pred, labels)

        # Logging to TensorBoard by default
        self.log('train_loss', loss)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

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

args = parser.parse_args()

data_dir = args.data_dir
model_name = args.model_save_name
model_type = args.model_type
model_dir = args.model_dir
save_iters = args.save_iters

train_loader = DataLoader(AskUbuntuTrainDataset(root_dir=data_dir))

# init model
autoencoder = LitAutoEncoder()

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
# TODO auto lr finder
# TODO BERT
# TODO Hyperparameters
# TODO model dirs
# tODO scripts
# todo tensorboard



# checkpoints
checkpoints = CheckpointEveryNSteps(save_iters)
#Early stopping #TODO model must implement a logging for 'val_loss'
early_stopping = EarlyStopping(monitor='val_loss')
trainer = pl.Trainer(callbacks=[checkpoints,early_stopping])
trainer.fit(autoencoder, train_loader)
