import pytorch_lightning as pl
import torch
from allennlp.common import Params
from torch import nn as nn
from torch.nn.functional import mse_loss

from models import transfer_batch_to_device


class LitBertArchitectureModel(pl.LightningModule):

    def __init__(self):
        super().__init__()
        archive_file = 'https://allennlp.s3-us-west-2.amazonaws.com/knowbert/models/knowbert_wiki_wordnet_model.tar.gz'

        # load model and batcher
        params = Params({"archive_file": archive_file})
        model = ModelArchiveFromParams.from_params(params=params)
        self.batcher = KnowBertBatchifier(archive_file)
        self.bert = model
        self.query_rescale_layer = nn.Linear(768, 768)
        self.response_rescale_layer = nn.Linear(768, 768)

        self.cosine_sim = nn.CosineSimilarity(dim=1, eps=1e-6)
        # self.save_hyperparameters()
        self.accuracy = pl.metrics.Accuracy()

    def on_train_epoch_start(self):
        print("Starting training epoch...")
    def on_train_epoch_end(self, outputs):
        print("Finished training epoch...")

    def training_step(self, batch, batch_idx):

        labels = batch['label'].float()

        query_text = batch['query_text']
        response_text = batch['response_text']
        b = next(self.batcher.iter_batches(response_text,verbose=False))
        b = transfer_batch_to_device(b,self.device)

        query_pooler_output = self.bert(**b)['pooled_output']
        b = next(self.batcher.iter_batches(response_text,verbose=False))
        b = transfer_batch_to_device(b,self.device)

        response_pooler_output = self.bert(**b)['pooled_output']
        # response_last_hidden_state, response_pooler_output = self.bert(response_dict['input_ids'], token_type_ids=response_dict['token_type_ids'], attention_mask=response_dict['attention_mask'])
        #
        preds = torch.sigmoid(self.cosine_sim(self.query_rescale_layer(query_pooler_output), self.response_rescale_layer(response_pooler_output)))
        #
        loss =mse_loss(preds, labels)

        # Logging to TensorBoard by default
        self.log('train_loss', loss,on_step=True)

        return loss



    def validation_step(self, batch, batch_idx):

        batch = transfer_batch_to_device(batch,self.device)
        labels = batch['label'].float()

        query_dict = batch['query_enc_dict']
        response_dict = batch['response_enc_dict']
        #
        # query_last_hidden_state, query_pooler_output = self.bert(query_dict['input_ids'],
        #                                                          token_type_ids=query_dict['token_type_ids'],
        #                                                          attention_mask=query_dict['attention_mask'])
        # response_last_hidden_state, response_pooler_output = self.bert(response_dict['input_ids'],
        #                                                                token_type_ids=response_dict['token_type_ids'],
        #                                                                attention_mask=response_dict['attention_mask'])
        query_text = batch['query_text']
        response_text = batch['response_text']
        b = next(self.batcher.iter_batches(response_text,verbose=False))
        b = transfer_batch_to_device(b,self.device)
        query_pooler_output = self.bert(**b)['pooled_output']

        b = next(self.batcher.iter_batches(response_text,verbose=False))
        b = transfer_batch_to_device(b,self.device)

        response_pooler_output = self.bert(**b)['pooled_output']
        preds = self.cosine_sim(self.query_rescale_layer(query_pooler_output), self.response_rescale_layer(response_pooler_output)).to(self.device)

        val_loss = torch.nn.BCEWithLogitsLoss()(preds, labels)
        self.log('val_loss_step', val_loss)
        self.log('test_acc_step', self.accuracy(preds, labels))
        return val_loss

    def on_validation_epoch_end(self):
        print("Calculating validation accuracy...")
        vacc = self.accuracy.compute()
        print("Validation accuracy:", vacc)
        self.log('val_acc_epoch', vacc)

    def test_step(self, batch, batch_idx):
        batch = transfer_batch_to_device(batch,self.device)
        labels = batch['label'].float()

        query_dict = batch['query_enc_dict']
        response_dict = batch['response_enc_dict']

        # query_last_hidden_state, query_pooler_output = self.bert(query_dict['input_ids'],
        #                                                          token_type_ids=query_dict['token_type_ids'],
        #                                                          attention_mask=query_dict['attention_mask'])
        # response_last_hidden_state, response_pooler_output = self.bert(response_dict['input_ids'],
        #                                                                token_type_ids=response_dict['token_type_ids'],
        #                                                                attention_mask=response_dict['attention_mask'])
        query_text = batch['query_text']
        response_text = batch['response_text']
        b = next(self.batcher.iter_batches(response_text,verbose=False))
        b = transfer_batch_to_device(b,self.device)

        query_pooler_output = self.bert(**b)['pooled_output']
        b = next(self.batcher.iter_batches(response_text,verbose=False))
        b = transfer_batch_to_device(b,self.device)

        response_pooler_output = self.bert(**b)['pooled_output']
        preds = self.cosine_sim(self.query_rescale_layer(query_pooler_output), self.response_rescale_layer(response_pooler_output)).to(self.device)

        test_loss =  torch.nn.BCEWithLogitsLoss()(preds,labels)
        self.log('test_loss', test_loss)
        self.log('test_acc_step', self.accuracy(preds, labels))

        return test_loss


    def on_test_epoch_end(self):
        print("Calculating test accuracy...")
        self.log('test_acc_epoch', self.accuracy.compute())

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer