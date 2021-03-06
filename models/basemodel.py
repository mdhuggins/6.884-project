import pytorch_lightning as pl
import torch
from torch import nn as nn
from torch.nn import BCEWithLogitsLoss
from torch.nn.functional import mse_loss
from torch.optim.lr_scheduler import ReduceLROnPlateau, LambdaLR, StepLR
from torch.utils.data.dataloader import default_collate
from transformers import BertModel, AdamW, get_linear_schedule_with_warmup

from evaluation import Evaluation
from utils import transfer_batch_to_device
import numpy as np

class LitBertModel(pl.LightningModule):
    @staticmethod
    def col_fn(x):
        return default_collate(x)

    def __init__(self, name,lr, total_steps, concat=None):
        super().__init__()
        self.bert = BertModel.from_pretrained(
            "bert-base-uncased", # Use the 12-layer BERT model, with an uncased vocab.
            output_attentions = False, # Whether the model returns attentions weights.
            output_hidden_states = False, # Whether the model returns all hidden-states.
        )
        self.query_rescale_layer = nn.Linear(768, 768)
        self.cosine_sim = nn.CosineSimilarity(dim=1, eps=1e-6)
        # self.save_hyperparameters()
        self.accuracy = pl.metrics.Accuracy()
        self.testaccuracy = pl.metrics.Accuracy()
        self.val_res = []
        self.eval_res = []
        self.lr = lr
        self.total_steps = total_steps

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.lr)
        scheduler = get_linear_schedule_with_warmup(optimizer, int(0.1 * self.total_steps), self.total_steps)
        # scheduler = StepLR(optimizer, step_size=25, gamma=0.8)
        scheduler = {
            'scheduler': scheduler,
            'interval': 'step',  # or 'epoch'
            'frequency': 1
        }
        return [optimizer], [scheduler]
    def training_step(self, batch, batch_idx):

        labels = batch['label'].float()

        query_dict = batch['query_enc_dict']
        response_dict = batch['response_enc_dict']

        out = self.bert(query_dict['input_ids'],
                                                                 token_type_ids=query_dict['token_type_ids'],
                                                                 attention_mask=query_dict['attention_mask'])
        query_last_hidden_state = out.last_hidden_state
        query_pooler_output = out.pooler_output

        query_pooler_output = torch.mean(query_last_hidden_state, 1)

        out = self.bert(response_dict['input_ids'],
                                                                       token_type_ids=response_dict['token_type_ids'],
                                                                       attention_mask=response_dict['attention_mask'])
        response_last_hidden_state = out.last_hidden_state
        response_pooler_output = out.pooler_output

        response_pooler_output = torch.mean(response_last_hidden_state, 1)
        # c = self.cosine_sim(self.query_rescale_layer(query_pooler_output), self.response_rescale_layer(response_pooler_output))
        # c = self.cosine_sim(query_pooler_output, response_pooler_output)
        c = self.cosine_sim(self.query_rescale_layer(query_pooler_output),
                                self.query_rescale_layer(response_pooler_output))
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
        out = self.bert(query_dict['input_ids'],
                                                                 token_type_ids=query_dict['token_type_ids'],
                                                                 attention_mask=query_dict['attention_mask'])
        query_last_hidden_state = out.last_hidden_state
        query_pooler_output = out.pooler_output
        query_pooler_output = torch.mean(query_last_hidden_state, 1)

        for i in range(len(batch['label'])):
            labels = batch['label'][i].float()

            response_dict = batch['response_enc_dict'][i]


            out = self.bert(response_dict['input_ids'],
                                                                           token_type_ids=response_dict['token_type_ids'],
                                                                           attention_mask=response_dict['attention_mask'])

            response_last_hidden_state = out.last_hidden_state
            response_pooler_output = out.pooler_output
            response_pooler_output = torch.mean(response_last_hidden_state,1)

            preds = self.cosine_sim(self.query_rescale_layer(query_pooler_output), self.query_rescale_layer(response_pooler_output))

            # preds = self.cosine_sim(self.query_rescale_layer(query_pooler_output), self.response_rescale_layer(response_pooler_output)).to(self.device)
            # preds = self.cosine_sim(query_pooler_output, response_pooler_output)#.to(self.device)
            # preds = torch.sigmoid(preds)
            preds = preds.to('cpu')
            labels = labels.to('cpu')
            val_loss += mse_loss(preds, labels)
            all_preds.append(torch.unsqueeze(preds,1))


            # val_loss =  torch.nn.BCEWithLogitsLoss()(preds,labels)
            self.log('val_acc_step', self.accuracy(torch.clamp(preds, 0, 1), labels.int()))

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
        self.log('v_MAP',MAP)
        self.log('v_Mrr', MRR)
        self.log('v_p1', P1)
        self.log('v_p5',P5)
        return vacc,MAP,MRR,P1,P5


    def test_step(self, batch, batch_idx):
        batch = transfer_batch_to_device(batch, self.device)
        all_preds = []
        val_loss = 0
        query_dict = batch['query_enc_dict']
        out = self.bert(query_dict['input_ids'],
                                                                 token_type_ids=query_dict['token_type_ids'],
                                                                 attention_mask=query_dict['attention_mask'])
        query_last_hidden_state = out.last_hidden_state
        query_pooler_output = out.pooler_output

        query_pooler_output = torch.mean(query_last_hidden_state, 1)

        for i in range(len(batch['label'])):
            labels = batch['label'][i].float()

            response_dict = batch['response_enc_dict'][i]


            out = self.bert(response_dict['input_ids'],
                                                                           token_type_ids=response_dict[
                                                                               'token_type_ids'],
                                                                           attention_mask=response_dict[
                                                                               'attention_mask'])
            response_last_hidden_state = out.last_hidden_state
            response_pooler_output = out.pooler_output
            response_pooler_output = torch.mean(response_last_hidden_state,1)
            preds = self.cosine_sim(self.query_rescale_layer(query_pooler_output), self.query_rescale_layer(response_pooler_output))

            # preds = self.cosine_sim(self.query_rescale_layer(query_pooler_output), self.response_rescale_layer(response_pooler_output))

            # preds = self.cosine_sim(self.query_rescale_layer(query_pooler_output), self.response_rescale_layer(response_pooler_output)).to(self.device)
            # preds = self.cosine_sim(query_pooler_output, response_pooler_output)  # .to(self.device)

            # preds = torch.sigmoid(preds)
            preds = preds.to('cpu')
            labels = labels.to('cpu')
            val_loss += mse_loss(preds, labels)
            all_preds.append(torch.unsqueeze(preds, 1))

            # val_loss =  torch.nn.BCEWithLogitsLoss()(preds,labels)
            self.log('test_acc_step', self.testaccuracy(torch.clamp(preds, 0, 1), labels.int()))

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
        self.log('t_MAP', MAP)
        self.log('t_Mrr', MRR)
        self.log('t_p1', P1)
        self.log('t_p5', P5)
        return vacc,MAP,MRR,P1,P5

