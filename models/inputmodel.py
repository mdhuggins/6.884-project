import os
import re

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import spacy
import torch
from spacy.matcher import PhraseMatcher
from torch import nn as nn
from torch.nn.functional import mse_loss
from torch.optim.lr_scheduler import StepLR
from torch.utils.data.dataloader import default_collate
from tqdm import tqdm
from transformers import BertTokenizerFast, BertModel, AdamW, get_linear_schedule_with_warmup
from transformers.modeling_outputs import BaseModelOutputWithPooling

from evaluation import Evaluation
from utils import transfer_batch_to_device, load_vectors_pandas, get_retro_embeds
import dask.dataframe as dd


class LitInputBertModel(pl.LightningModule):
    @staticmethod
    def col_fn(x):
        res = default_collate(x)
        query_dict = res['query_enc_dict']
        q_retro = get_retro_embeds(query_dict["input_ids"])
        response_dict = res['response_enc_dict']
        if isinstance(response_dict,list):
            r_retro = []
            for i in  range(len(response_dict)):
                r_retro.append(get_retro_embeds(response_dict[i]["input_ids"]))
        else:
            r_retro = get_retro_embeds(response_dict["input_ids"])
        res['q_retro'] = q_retro
        res['r_retro'] = r_retro
        return res

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

    def __init__(self,name,lr, total_steps, concat=None):
        super().__init__()
        self.bert = ExtraInputBertModel.from_pretrained(
            "bert-base-uncased", # Use the 12-layer BERT model, with an uncased vocab.
            output_attentions = False, # Whether the model returns attentions weights.
            output_hidden_states = False, # Whether the model returns all hidden-states.
            input_process_fn=self.knowledge_infusion
        )
        self.query_rescale_layer = nn.Linear(768, 768)
        self.response_rescale_layer = nn.Linear(768, 768)

        self.concat_rescale_layer = nn.Linear(1068, 768)

        self.accuracy = pl.metrics.Accuracy()
        self.testaccuracy = pl.metrics.Accuracy()
        self.val_res = []
        self.eval_res = []
        self.concat = concat
        self.sum = True
        self.decompressor = nn.Linear(300,768)
        self.cosine_sim = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.lr = lr
        self.total_steps = total_steps
        # self.save_hyperparameters()

        # self.tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
        # path = "models/graphembeddings/entities_glove_format"
        # self.numberbatch = load_vectors_pandas(path, "wiki.h5", clean_names=True)
        # self.numberbatch.index = self.numberbatch.index.map(str)
        # ddf = dd.from_pandas(self.numberbatch, npartitions=20)
        # self.numberbatch = ddf
        # self.nlp = spacy.load('en_core_web_sm')
        # self.phraseMatcher = PhraseMatcher(self.nlp.vocab, attr='LOWER')
        # terms = [str(x) for x in self.numberbatch.index]
        # patterns = [self.nlp.make_doc(text) for text in terms]
        # self.phraseMatcher.add("Match_By_Phrase", None, *patterns)
        # print(self.numberbatch.loc["cat"])


    def knowledge_infusion(self, entity_embeds, embedding_output):
        if self.concat:
            concat = torch.cat((entity_embeds, embedding_output), 2)
            representation = self.concat_rescale_layer(concat)
        elif self.sum:
            # print(entity_embeds.shape)
            entity_embeds = self.decompressor(entity_embeds)
            representation = entity_embeds+embedding_output
        else:
            representation = embedding_output

        return representation

    def training_step(self, batch, batch_idx):

        labels = batch['label'].float()

        query_dict = batch['query_enc_dict']
        response_dict = batch['response_enc_dict']

        out = self.bert(query_dict['input_ids'],
                                                                 token_type_ids=query_dict['token_type_ids'],
                                                                 attention_mask=query_dict['attention_mask'],
                                                                 entity_embeds=batch["q_retro"]
                                                                 )
        query_last_hidden_state = out.last_hidden_state
        query_pooler_output = out.pooler_output

        out = self.bert(response_dict['input_ids'],
                                                                       token_type_ids=response_dict['token_type_ids'],
                                                                       attention_mask=response_dict['attention_mask'],
                                                                       entity_embeds=batch["r_retro"])
        response_last_hidden_state = out.last_hidden_state
        response_pooler_output = out.pooler_output

        # preds = torch.sigmoid(self.cosine_sim(self.query_rescale_layer(query_pooler_output), self.response_rescale_layer(response_pooler_output)))
        a = torch.mean(query_last_hidden_state,1)
        b = torch.mean(response_last_hidden_state, 1)

        c = self.cosine_sim(self.query_rescale_layer(a), self.query_rescale_layer(b))

        loss =mse_loss(c, labels)
        # loss =  torch.nn.BCEWithLogitsLoss()(c,labels)

        # Logging to TensorBoard by default
        self.log('train_loss', loss,on_step=True)

        return loss




    def validation_step(self, batch, batch_idx):
        batch = transfer_batch_to_device(batch,self.device)
        all_preds = []
        val_loss = 0
        query_dict = batch['query_enc_dict']
        out = self.bert(query_dict['input_ids'],
                                                                 token_type_ids=query_dict['token_type_ids'],
                                                                 attention_mask=query_dict['attention_mask'],
                                                                 entity_embeds=batch["q_retro"]
                                                                 )
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
                                                                               'attention_mask'],
                                                                           entity_embeds=batch["r_retro"][i])
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
        self.log('v_MAP', MAP)
        self.log('v_Mrr', MRR)
        self.log('v_p1', P1)
        self.log('v_p5', P5)
        return vacc,MAP,MRR,P1,P5


    def test_step(self, batch, batch_idx):
        batch = transfer_batch_to_device(batch, self.device)
        all_preds = []
        val_loss = 0
        query_dict = batch['query_enc_dict']
        out = self.bert(query_dict['input_ids'],
                                                                 token_type_ids=query_dict['token_type_ids'],
                                                                 attention_mask=query_dict['attention_mask'],
                                                                 entity_embeds=batch["q_retro"]
                                                                 )
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
                                                                               'attention_mask'],
                                                                           entity_embeds=batch["r_retro"][i])
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




class ExtraInputBertModel(BertModel):
    def __init__(self, config, add_pooling_layer=True, input_process_fn=lambda x: x):
        super().__init__(config, add_pooling_layer=add_pooling_layer)
        self.input_process_fn = input_process_fn

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        entity_embeds=None
    ):
        r"""
        encoder_hidden_states  (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
            if the model is configured as a decoder.
        encoder_attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask
            is used in the cross-attention if the model is configured as a decoder.
            Mask values selected in ``[0, 1]``:
            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds
        )

        processed_embedding_output = self.input_process_fn(entity_embeds, embedding_output)

        encoder_outputs = self.encoder(
            processed_embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )