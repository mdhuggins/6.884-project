import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from transformers import BertModel
import sklearn

class LitBertModel(pl.LightningModule):

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

    def validation_step(self, batch, batch_idx):
        labels = batch['label'].float()

        query_dict = batch['query_enc_dict']
        response_dict = batch['response_enc_dict']

        query_last_hidden_state, query_pooler_output = self.bert(query_dict['input_ids'],
                                                                 token_type_ids=query_dict['token_type_ids'],
                                                                 attention_mask=query_dict['attention_mask'])
        response_last_hidden_state, response_pooler_output = self.bert(response_dict['input_ids'],
                                                                       token_type_ids=response_dict['token_type_ids'],
                                                                       attention_mask=response_dict['attention_mask'])

        preds = torch.sigmoid(self.cosine_sim(query_pooler_output, response_pooler_output))

        val_loss = F.binary_cross_entropy(preds,labels)
        self.log('val_loss', val_loss)
        return val_loss

    def on_validation_epoch_end(self):
        def get_accuracy(y_true, y_prob):
            accuracy = sklearn.metrics.accuracy_score(y_true, [1 if x>0.5 else 0 for x in y_prob])
            return accuracy
        results = []
        gold = []
        print("Calculating validation accuracy...")
        for idx, batch in enumerate(tqdm(self.val_dataloader())):
            labels = batch['label'].float()
            gold.extend([int(x) for x in labels.numpy()])
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

            preds = torch.sigmoid(self.cosine_sim(query_pooler_output, response_pooler_output))
            results.extend([float(x) for x in preds.numpy()])
        val_acc = get_accuracy(gold, results)
        print("Val acc",val_acc)
        self.log('val_accuracy',val_acc)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
