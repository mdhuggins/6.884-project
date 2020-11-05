from typing import Dict, List

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from transformers import BertModel
import sklearn
from kb.include_all import ModelArchiveFromParams
from kb.knowbert_utils import KnowBertBatchifier
from allennlp.common import Params

def get_accuracy(y_true, y_prob):
    accuracy = sklearn.metrics.accuracy_score(y_true, [1 if x > 0.5 else 0 for x in y_prob])
    return accuracy


class LitBertModel(pl.LightningModule):

    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained(
            "bert-base-uncased", # Use the 12-layer BERT model, with an uncased vocab.
            output_attentions = False, # Whether the model returns attentions weights.
            output_hidden_states = False, # Whether the model returns all hidden-states.
        )
        self.query_rescale_layer = nn.Linear(768, 768)
        self.response_rescale_layer = nn.Linear(768, 768)

        self.cosine_sim = nn.CosineSimilarity(dim=1, eps=1e-6)
        # self.save_hyperparameters()


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

        preds = torch.nn.functional.sigmoid(self.cosine_sim(self.query_rescale_layer(query_pooler_output), self.response_rescale_layer(response_pooler_output)))

        loss = F.mse_loss(preds, labels)

        # Logging to TensorBoard by default
        self.log('train_loss', loss)

        return loss

    def transfer_batch_to_device(self, batch, device):
        # print("### DEVICE CHECK", device)
        for k in batch.keys():
            if isinstance(batch[k],Dict):
                for i in batch[k].keys():
                    batch[k][i] = batch[k][i].to(device)
            else:
                batch[k] = batch[k].to(device)
        return batch

    def validation_step(self, batch, batch_idx):
        batch = self.transfer_batch_to_device(batch,self.device)
        labels = batch['label'].float()

        query_dict = batch['query_enc_dict']
        response_dict = batch['response_enc_dict']

        query_last_hidden_state, query_pooler_output = self.bert(query_dict['input_ids'],
                                                                 token_type_ids=query_dict['token_type_ids'],
                                                                 attention_mask=query_dict['attention_mask'])
        response_last_hidden_state, response_pooler_output = self.bert(response_dict['input_ids'],
                                                                       token_type_ids=response_dict['token_type_ids'],
                                                                       attention_mask=response_dict['attention_mask'])

        preds = torch.nn.functional.sigmoid(self.cosine_sim(self.query_rescale_layer(query_pooler_output), self.response_rescale_layer(response_pooler_output))).to(self.device)

        val_loss = F.binary_cross_entropy(preds,labels)
        self.log('val_loss', val_loss)
        return val_loss

    def test_step(self, batch, batch_idx):
        batch = self.transfer_batch_to_device(batch,self.device)
        labels = batch['label'].float()

        query_dict = batch['query_enc_dict']
        response_dict = batch['response_enc_dict']

        query_last_hidden_state, query_pooler_output = self.bert(query_dict['input_ids'],
                                                                 token_type_ids=query_dict['token_type_ids'],
                                                                 attention_mask=query_dict['attention_mask'])
        response_last_hidden_state, response_pooler_output = self.bert(response_dict['input_ids'],
                                                                       token_type_ids=response_dict['token_type_ids'],
                                                                       attention_mask=response_dict['attention_mask'])

        preds = torch.nn.functional.sigmoid(self.cosine_sim(self.query_rescale_layer(query_pooler_output), self.response_rescale_layer(response_pooler_output))).to(self.device)

        val_loss = F.binary_cross_entropy(preds,labels)
        self.log('test_loss', val_loss)
        return val_loss

    def on_validation_epoch_end(self):
        results = []
        gold = []
        print("Calculating validation accuracy...")
        for idx, batch in enumerate(tqdm(self.val_dataloader(),leave=True)):
            batch = self.transfer_batch_to_device(batch, self.device)
            labels = batch['label'].float()
            gold.extend([int(x) for x in labels.cpu().numpy()])
            query_dict = batch['query_enc_dict']
            response_dict = batch['response_enc_dict']

            query_last_hidden_state, query_pooler_output = self.bert(query_dict['input_ids'],
                                                                     token_type_ids=query_dict['token_type_ids'],
                                                                     attention_mask=query_dict['attention_mask'])
            response_last_hidden_state, response_pooler_output = self.bert(response_dict['input_ids'],
                                                                           token_type_ids=response_dict[
                                                                               'token_type_ids'],
                                                                           attention_mask=response_dict[
                                                                               'attention_mask'])

            preds = torch.nn.functional.sigmoid(self.cosine_sim(self.query_rescale_layer(query_pooler_output), self.response_rescale_layer(response_pooler_output)))
            results.extend([float(x) for x in preds.cpu().numpy()])
        val_acc = get_accuracy(gold, results)
        print("Val acc",val_acc)
        self.log('val_accuracy',val_acc)

    def on_test_epoch_end(self):
        results = []
        gold = []
        print("Calculating test accuracy...")
        for idx, batch in enumerate(tqdm(self.test_dataloader(),leave=True)):
            batch = self.transfer_batch_to_device(batch, self.device)
            labels = batch['label'].float()
            gold.extend([int(x) for x in labels.cpu().numpy()])
            query_dict = batch['query_enc_dict']
            response_dict = batch['response_enc_dict']

            query_last_hidden_state, query_pooler_output = self.bert(query_dict['input_ids'],
                                                                     token_type_ids=query_dict['token_type_ids'],
                                                                     attention_mask=query_dict['attention_mask'])
            response_last_hidden_state, response_pooler_output = self.bert(response_dict['input_ids'],
                                                                           token_type_ids=response_dict[
                                                                               'token_type_ids'],
                                                                           attention_mask=response_dict[
                                                                               'attention_mask'])

            preds = torch.nn.functional.sigmoid(self.cosine_sim(self.query_rescale_layer(query_pooler_output),
                                                                self.response_rescale_layer(response_pooler_output)))
            results.extend([float(x) for x in preds.cpu().numpy()])
        val_acc = get_accuracy(gold, results)
        print("Test acc", val_acc)
        self.log('test_accuracy', val_acc)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


class LitKnowBERTModel(pl.LightningModule):

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

    def training_step(self, batch, batch_idx):

        labels = batch['label'].float()

        query_text = batch['query_text']
        response_text = batch['response_text']
        b = next(self.batcher.iter_batches(response_text,verbose=False))
        b = self.transfer_batch_to_device(b,self.device)

        query_pooler_output = self.bert(**b)['pooled_output']
        b = next(self.batcher.iter_batches(response_text,verbose=False))
        b = self.transfer_batch_to_device(b,self.device)

        response_pooler_output = self.bert(**b)['pooled_output']
        # response_last_hidden_state, response_pooler_output = self.bert(response_dict['input_ids'], token_type_ids=response_dict['token_type_ids'], attention_mask=response_dict['attention_mask'])
        #
        preds = torch.nn.functional.sigmoid(self.cosine_sim(self.query_rescale_layer(query_pooler_output), self.response_rescale_layer(response_pooler_output)))
        #
        loss = F.mse_loss(preds, labels)

        # Logging to TensorBoard by default
        self.log('train_loss', loss)

        return loss

    def transfer_batch_to_device(self, batch, device):
        # print("### DEVICE CHECK", device)
        for k in batch.keys():
            if isinstance(batch[k],Dict):
                batch[k] = self.transfer_batch_to_device(batch[k],device)
            elif isinstance(batch[k],List):
                print(batch[k])
                continue
            else:
                batch[k] = batch[k].to(device)
        return batch

    def validation_step(self, batch, batch_idx):

        batch = self.transfer_batch_to_device(batch,self.device)
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
        b = self.transfer_batch_to_device(b,self.device)
        query_pooler_output = self.bert(**b)['pooled_output']

        b = next(self.batcher.iter_batches(response_text,verbose=False))
        b = self.transfer_batch_to_device(b,self.device)

        response_pooler_output = self.bert(**b)['pooled_output']
        preds = self.cosine_sim(self.query_rescale_layer(query_pooler_output), self.response_rescale_layer(response_pooler_output)).to(self.device)

        val_loss = torch.nn.BCEWithLogitsLoss()(preds,labels)
        self.log('val_loss', val_loss)
        return val_loss

    def test_step(self, batch, batch_idx):
        batch = self.transfer_batch_to_device(batch,self.device)
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
        b = self.transfer_batch_to_device(b,self.device)

        query_pooler_output = self.bert(**b)['pooled_output']
        b = next(self.batcher.iter_batches(response_text,verbose=False))
        b = self.transfer_batch_to_device(b,self.device)

        response_pooler_output = self.bert(**b)['pooled_output']
        preds = self.cosine_sim(self.query_rescale_layer(query_pooler_output), self.response_rescale_layer(response_pooler_output)).to(self.device)

        val_loss = torch.nn.BCEWithLogitsLoss()(preds,labels)
        self.log('test_loss', val_loss)
        return val_loss

    def on_validation_epoch_end(self):
        results = []
        gold = []
        return
        print("Calculating validation accuracy...")
        for idx, batch in enumerate(tqdm(self.val_dataloader(),leave=True)):
            batch = self.transfer_batch_to_device(batch, self.device)
            labels = batch['label'].float()
            gold.extend([int(x) for x in labels.cpu().numpy()])
            query_dict = batch['query_enc_dict']
            response_dict = batch['response_enc_dict']
            query_text = batch['query_text']
            response_text = batch['response_text']
            b = next(self.batcher.iter_batches(response_text,verbose=False))
            b = self.transfer_batch_to_device(b, self.device)

            query_pooler_output = self.bert(**b)['pooled_output']
            b = next(self.batcher.iter_batches(response_text,verbose=False))
            b = self.transfer_batch_to_device(b, self.device)

            response_pooler_output = self.bert(**b)['pooled_output']
            # query_last_hidden_state, query_pooler_output = self.bert(query_dict['input_ids'],
            #                                                          token_type_ids=query_dict['token_type_ids'],
            #                                                          attention_mask=query_dict['attention_mask'])
            # response_last_hidden_state, response_pooler_output = self.bert(response_dict['input_ids'],
            #                                                                token_type_ids=response_dict[
            #                                                                    'token_type_ids'],
            #                                                                attention_mask=response_dict[
            #                                                                    'attention_mask'])

            preds = torch.nn.functional.sigmoid(self.cosine_sim(self.query_rescale_layer(query_pooler_output), self.response_rescale_layer(response_pooler_output)))
            results.extend([float(x) for x in preds.cpu().numpy()])
        val_acc = get_accuracy(gold, results)
        print("Val acc",val_acc)
        self.log('val_accuracy',val_acc)

    def on_test_epoch_end(self):
        results = []
        gold = []
        print("Calculating test accuracy...")
        for idx, batch in enumerate(tqdm(self.test_dataloader(),leave=True)):
            batch = self.transfer_batch_to_device(batch, self.device)
            labels = batch['label'].float()
            gold.extend([int(x) for x in labels.cpu().numpy()])
            query_dict = batch['query_enc_dict']
            response_dict = batch['response_enc_dict']

            # query_last_hidden_state, query_pooler_output = self.bert(query_dict['input_ids'],
            #                                                          token_type_ids=query_dict['token_type_ids'],
            #                                                          attention_mask=query_dict['attention_mask'])
            # response_last_hidden_state, response_pooler_output = self.bert(response_dict['input_ids'],
            #                                                                token_type_ids=response_dict[
            #                                                                    'token_type_ids'],
            #                                                                attention_mask=response_dict[
            #                                                                    'attention_mask'])
            query_text = batch['query_text']
            response_text = batch['response_text']
            b = next(self.batcher.iter_batches(response_text,verbose=False))
            b = self.transfer_batch_to_device(b, self.device)

            query_pooler_output = self.bert(**b)['pooled_output']
            b = next(self.batcher.iter_batches(response_text,verbose=False))
            b = self.transfer_batch_to_device(b, self.device)

            response_pooler_output = self.bert(**b)['pooled_output']
            preds = torch.nn.functional.sigmoid(self.cosine_sim(self.query_rescale_layer(query_pooler_output),
                                                                self.response_rescale_layer(response_pooler_output)))
            results.extend([float(x) for x in preds.cpu().numpy()])
        val_acc = get_accuracy(gold, results)
        print("Test acc", val_acc)
        self.log('test_accuracy', val_acc)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
