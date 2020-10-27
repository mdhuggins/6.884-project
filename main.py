import os
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from torch.utils.data import random_split
from transformers import BertModel, BertConfig

from data import AskUbuntuTrainDataset


class LitAutoEncoder(pl.LightningModule):

    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained(
            "bert-base-uncased", # Use the 12-layer BERT model, with an uncased vocab.
            output_attentions = False, # Whether the model returns attentions weights.
            output_hidden_states = False, # Whether the model returns all hidden-states.
        )

        self.cosine_sim = nn.CosineSimilarity(dim=1, eps=1e-6)

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        embedding = self.encoder(x)
        return embedding

    def training_step(self, batch, batch_idx):

        labels = batch['label'].float()

        query_dict = batch['query_enc_dict']
        response_dict = batch['response_enc_dict']

        query_last_hidden_state, query_pooler_output = self.bert(query_dict['input_ids'], token_type_ids=query_dict['token_type_ids'], attention_mask=query_dict['attention_mask'])
        response_last_hidden_state, response_pooler_output = self.bert(response_dict['input_ids'], token_type_ids=response_dict['token_type_ids'], attention_mask=response_dict['attention_mask'])

        preds = self.cosine_sim(query_pooler_output, response_pooler_output)

        loss = F.mse_loss(preds, labels)

        # Logging to TensorBoard by default
        self.log('train_loss', loss)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


# dataset = MNIST(os.getcwd(), download=True, transform=transforms.ToTensor())
train_loader = DataLoader(AskUbuntuTrainDataset(toy_n=20, toy_pad=20), batch_size=16, shuffle=True)

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
# TODO batch size param
# TODO Save encodings
# TODO Early stopping or epochs?
# TODO auto lr finder
# TODO BERT
# TODO checkpoints
# TODO Hyperparameters
# TODO model dirs
# tODO scripts
# todo tensorboard

trainer = pl.Trainer()

trainer.fit(autoencoder, train_loader)
