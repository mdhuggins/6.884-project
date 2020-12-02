import pytorch_lightning as pl
import torch
from torch import nn as nn
from torch.nn import BCEWithLogitsLoss
from torch.nn.functional import mse_loss
from torch.optim.lr_scheduler import ReduceLROnPlateau, LambdaLR, StepLR
from transformers import BertModel

from evaluation import Evaluation
from utils import transfer_batch_to_device
import numpy as np

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
        self.accuracy = pl.metrics.Accuracy()
        self.testaccuracy = pl.metrics.Accuracy()
        self.val_res = []
        self.eval_res = []

    def training_step(self, batch, batch_idx):

        labels = batch['label'].float()

        query_dict = batch['query_enc_dict']
        response_dict = batch['response_enc_dict']

        query_last_hidden_state, query_pooler_output = self.bert(query_dict['input_ids'],
                                                                 token_type_ids=query_dict['token_type_ids'],
                                                                 attention_mask=query_dict['attention_mask'])
        # query_pooler_output = torch.mean(query_last_hidden_state, 1)

        response_last_hidden_state, response_pooler_output = self.bert(response_dict['input_ids'],
                                                                       token_type_ids=response_dict['token_type_ids'],
                                                                       attention_mask=response_dict['attention_mask'])
        # response_pooler_output = torch.mean(response_last_hidden_state, 1)
        # c = self.cosine_sim(self.query_rescale_layer(query_pooler_output), self.response_rescale_layer(response_pooler_output))
        c = self.cosine_sim(query_pooler_output, response_pooler_output)
        # c = self.cosine_sim(self.query_rescale_layer(query_pooler_output),
        #                         self.query_rescale_layer(response_pooler_output))
        # preds = torch.sigmoid(c)
        preds=c
        loss = mse_loss(preds, labels)
        # loss = BCEWithLogitsLoss()(c,labels)
        # Logging to TensorBoard by default
        self.log('train_loss', loss, on_step=True)

        return loss

    def validation_step(self, batch, batch_idx):
        batch = transfer_batch_to_device(batch,self.device)
        all_preds = []
        val_loss = 0
        query_dict = batch['query_enc_dict']
        query_last_hidden_state, query_pooler_output = self.bert(query_dict['input_ids'],
                                                                 token_type_ids=query_dict['token_type_ids'],
                                                                 attention_mask=query_dict['attention_mask'])

        for i in range(len(batch['label'])):
            labels = batch['label'][i].float()

            response_dict = batch['response_enc_dict'][i]

            # query_pooler_output = torch.mean(query_last_hidden_state,1)

            response_last_hidden_state, response_pooler_output = self.bert(response_dict['input_ids'],
                                                                           token_type_ids=response_dict['token_type_ids'],
                                                                           attention_mask=response_dict['attention_mask'])
            # response_pooler_output = torch.mean(response_last_hidden_state,1)

            # preds = self.cosine_sim(self.query_rescale_layer(query_pooler_output), self.query_rescale_layer(response_pooler_output))

            # preds = self.cosine_sim(self.query_rescale_layer(query_pooler_output), self.response_rescale_layer(response_pooler_output)).to(self.device)
            preds = self.cosine_sim(query_pooler_output, response_pooler_output)#.to(self.device)
            # preds = torch.sigmoid(preds)
            preds = preds.to('cpu')
            labels = labels.to('cpu')
            val_loss += mse_loss(preds, labels)
            all_preds.append(torch.unsqueeze(preds,1))


            # val_loss =  torch.nn.BCEWithLogitsLoss()(preds,labels)
            self.log('val_acc_step', self.accuracy(preds, labels))

        self.log('val_loss_step', val_loss/len(batch['label']))
        all_preds = torch.cat(all_preds,1)
        all_preds_np = all_preds.cpu().numpy()
        np_labels = [x.cpu().numpy() for x in batch['label']]
        np_labels = np.array(np_labels).transpose()
        res=[]
        for i in range(len(batch['label'][0])):
            b_i = all_preds_np[i,:]
            idxs = (-b_i).argsort()
            labels_idx = np_labels[i,idxs]
            res.append(labels_idx)
        self.val_res.extend(res)
        return val_loss

    def on_validation_epoch_end(self):
        print("Calculating validation accuracy...")
        vacc = self.accuracy.compute()
        e = Evaluation(self.val_res)
        MAP = e.MAP() * 100
        MRR = e.MRR() * 100
        P1 = e.Precision(1) * 100
        P5 = e.Precision(5) * 100
        # print(e)
        print("Validation accuracy:",vacc),
        print(MAP,MRR,P1,P5)
        self.log('val_acc_epoch',vacc)
        self.log('v_MAP',MAP,sync_dist=True)
        self.log('v_Mrr', MRR,sync_dist=True)
        self.log('v_p1', P1,sync_dist=True)
        self.log('v_p5',P5,sync_dist=True)


    def test_step(self, batch, batch_idx):
        batch = transfer_batch_to_device(batch, self.device)
        all_preds = []
        val_loss = 0
        query_dict = batch['query_enc_dict']
        query_last_hidden_state, query_pooler_output = self.bert(query_dict['input_ids'],
                                                                 token_type_ids=query_dict['token_type_ids'],
                                                                 attention_mask=query_dict['attention_mask'])
        # query_pooler_output = torch.mean(query_last_hidden_state, 1)

        for i in range(len(batch['label'])):
            labels = batch['label'][i].float()

            response_dict = batch['response_enc_dict'][i]


            response_last_hidden_state, response_pooler_output = self.bert(response_dict['input_ids'],
                                                                           token_type_ids=response_dict[
                                                                               'token_type_ids'],
                                                                           attention_mask=response_dict[
                                                                               'attention_mask'])
            # response_pooler_output = torch.mean(response_last_hidden_state,1)
            # preds = self.cosine_sim(self.query_rescale_layer(query_pooler_output), self.query_rescale_layer(response_pooler_output))

            # preds = self.cosine_sim(self.query_rescale_layer(query_pooler_output), self.response_rescale_layer(response_pooler_output))

            # preds = self.cosine_sim(self.query_rescale_layer(query_pooler_output), self.response_rescale_layer(response_pooler_output)).to(self.device)
            preds = self.cosine_sim(query_pooler_output, response_pooler_output)  # .to(self.device)

            # preds = torch.sigmoid(preds)
            preds = preds.to('cpu')
            labels = labels.to('cpu')
            val_loss += mse_loss(preds, labels)
            all_preds.append(torch.unsqueeze(preds, 1))

            # val_loss =  torch.nn.BCEWithLogitsLoss()(preds,labels)
            self.log('test_acc_step', self.testaccuracy(preds, labels))

        self.log('test_loss_step', val_loss / len(batch['label']))
        all_preds = torch.cat(all_preds, 1)
        all_preds_np = all_preds.cpu().numpy()
        np_labels = [x.cpu().numpy() for x in batch['label']]
        np_labels = np.array(np_labels).transpose()
        res = []
        for i in range(len(batch['label'][0])):
            b_i = all_preds_np[i, :]
            idxs = (-b_i).argsort()
            labels_idx = np_labels[i, idxs]
            res.append(labels_idx)
        self.eval_res.extend(res)
        return val_loss

    def on_test_epoch_end(self):
        print("Calculating test accuracy...")
        vacc = self.testaccuracy.compute()
        e = Evaluation(self.eval_res)
        MAP = e.MAP() * 100
        MRR = e.MRR() * 100
        P1 = e.Precision(1) * 100
        P5 = e.Precision(5) * 100
        # print(e)
        print("Test accuracy:", vacc),
        print(MAP, MRR, P1, P5)
        self.log('test_acc_epoch', vacc)
        self.log('t_MAP', MAP,sync_dist=True)
        self.log('t_Mrr', MRR,sync_dist=True)
        self.log('t_p1', P1,sync_dist=True)
        self.log('t_p5', P5,sync_dist=True)

    # def configure_optimizers(self):
    #     optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
    #     return optimizer

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=5e-5)
        scheduler = StepLR(optimizer, step_size=25, gamma=0.7)
        # scheduler  = torch.optim.lr_scheduler.OneCycleLR(optimizer, 5e-3, total_steps=len(self.train_dataloader()), epochs=1, steps_per_epoch=None,
        #                                     pct_start=0.3, anneal_strategy='linear', cycle_momentum=True,
        #                                     base_momentum=0.85, max_momentum=0.95, div_factor=25.0,
        #                                     final_div_factor=10000.0, last_epoch=-1, verbose=False)
        return [optimizer], [scheduler]
