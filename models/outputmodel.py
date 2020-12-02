import os
import re

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning.core.step_result import Result
from torch import nn as nn
from torch.nn.functional import mse_loss
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
from transformers import BertTokenizerFast, BertModel, BertConfig

from utils import transfer_batch_to_device
#
#
# class LitOutputBaseModel(pl.LightningModule):
#
#     def __init__(self):
#         super().__init__()
#         bertmodel = "bert-base-uncased"
#         # self.bert = BertModel.from_pretrained(
#         #     bertmodel,# Use the 12-layer BERT model, with an uncased vocab.
#         #     output_attentions=False,# Whether the model returns attentions weights.
#         #     output_hidden_states=False,# Whether the model returns all hidden-states.
#         # )
#         # print("Freeze the bert model!!!")
#         # for parameter in self.bert.parameters():
#         #     parameter.requires_grad = False
#
#         self.query_rescale_layer = nn.Linear(768, 768)
#         self.knowledge_infusion_layer = nn.Linear(300,768)
#         self.compressor = nn.Linear(300,64)
#         self.decompressor = nn.Linear(64,300)
#         self.mover = nn.Linear(1068,768)
#         self.retro_expander = nn.Linear(300,768)
#         self.retro_next_dim = nn.Linear(1024,1024)
#         self.relu = nn.ReLU()
#         self.retro_compressor = nn.Linear(1024,768)
#         self.mover2 = nn.Linear(1536,768)
#         self.convovler1 = nn.Conv1d(128,64,3)
#         self.convovler2 = nn.Conv1d(64, 32, 4)
#         self.convovler3 = nn.Conv1d(32, 16, 5)
#         self.maxpool = nn.MaxPool1d(3)
#         # bert_config = BertConfig.from_pretrained(bertmodel,hidden_size=300,num_hidden_layers=1)
#
#         # self.k_att = BertLayer(bert_config)
#         # untouched_config = BertConfig.from_pretrained(bertmodel)
#         # self.recontextualizer = BertLayer(untouched_config)
#         self.nonlinear = nn.Tanh()
#         # self.pooler = BertPooler(untouched_config)
#
#         self.cosine_sim = nn.CosineSimilarity(dim=1, eps=1e-6)
#         # self.save_hyperparameters()
#
#         self.accuracy = pl.metrics.Accuracy()
#
#         # self.nlp = spacy.load('en_core_web_sm')
#         #
#         # pattern = [{'POS': 'VERB', 'OP': '?'},
#         #            {'POS': 'ADV', 'OP': '*'},
#         #            {'POS': 'AUX', 'OP': '*'},
#         #            {'POS': 'VERB', 'OP': '+'}]
#
#         # instantiate a Matcher instance
#         # self.matcher = Matcher(self.nlp.vocab)
#         # self.matcher.add("Verb phrase", None, pattern)
#         self.tokenizer = BertTokenizerFast.from_pretrained(bertmodel)
#         # self.numberbatch = pd.read_hdf("models/mini-2.h5","mat")
#         if os.path.exists("nb.h5"):
#             print("Using cached!")
#             self.numberbatch = pd.read_hdf("nb.h5","mat")
#         else:
#             names = []
#             vecs = []
#             with open("graphembeddings/numberbatch-en-19.08.txt") as nb:
#                 for line in tqdm(nb):
#                     line = line.strip()
#                     if len(line.split())==2:
#                         continue
#                     name = line.split()[0]
#                     d = pd.Series([float(x) for x in line.split()[1:]])
#                     vecs.append(d)
#                     names.append(name)
#             self.numberbatch = pd.DataFrame(data=vecs,index=names)
#             self.numberbatch.to_hdf("nb.h5","mat")
#         # self.numberbatch = self.numberbatch.transpose()
#         print(self.numberbatch.loc["cat"])
#         # self.numberbatch = self.numberbatch.loc[[x for x in self.numberbatch.index if '/c/en/' in x]]
#         self.lm_loss_ent = nn.Linear(768,300)
#         # self.numberbatch_dict = {}
#         # for word in self.numberbatch.index:
#         #     if "/c/en/" in word:
#         #         self.numberbatch_dict[word]=self.numberbatch.loc[word]
#         # print("Configured numberbatch!")
#         # print("Sanitychecking cat")
#         # print("/c/en/cat",self.numberbatch_dict["/c/en/cat"])
#
#     def on_train_epoch_start(self):
#         print("Starting training epoch...")
#         # if self.current_epoch == 2:
#         #     print("Unfreezing in second epoch")
#         #     for p in self.bert.parameters():
#         #         p.requires_grad = True
#
#         print("Done")
#     def on_train_epoch_end(self, outputs):
#         print("Finished training epoch...")
#
#     def knowledge_infusion(self, input_ids):
#         stacked_sentences = []
#         prefix = ""
#         stacked_retroembeds = []
#         for inpt_id_idx, sentence in enumerate(input_ids):
#             dec_sentence = self.tokenizer.decode(sentence.cpu().numpy())
#             sentence_vecs = []
#             counts = {}
#             def find_sub_list(sl, l):
#                 if str(sl) not in counts.keys():
#                     counts[str(sl)] = 0
#                 counts[str(sl)] += 1
#                 results = []
#                 sll = len(sl)
#                 for ind in (i for i, e in enumerate(l) if e == sl[0]):
#                     if l[ind:ind + sll] == sl:
#                         results.append((ind, ind + sll - 1))
#                 try:
#                     r = results[counts[str(sl)] - 1]
#
#                     return r
#                 except Exception as t:
#                     return None
#
#             words = re.findall(r'\w+', dec_sentence)
#             retroembeds = []
#             for word in words:
#                 try:
#                     vec = self.numberbatch.loc[prefix + word]
#                     to_append = np.array(vec).reshape(300, )
#                 except:
#                     to_append = np.zeros((300,))
#                 retroembeds.append(to_append)
#             # retroembeds = retroembeddings.get_embeddings_from_input_ids(words).contiguous()
#             # retroembeds  = retro_vecs[sample]
#             stacked_retroembeds.append(retroembeds)
#             replacement_list = []
#             for word in words:
#                 toks = self.tokenizer.encode(word, add_special_tokens=False)
#                 locs = find_sub_list(toks, [int(x) for x in sentence.cpu().numpy()])
#                 if locs is None:
#                     continue
#                 replacement_list.append(locs)
#             final_list = []
#             for idx, id in enumerate(sentence):
#                 # Id iN SPECIAL TOKENS
#                 added = False
#                 for rep_idx, rep_tup in enumerate(replacement_list):
#                     if idx >= rep_tup[0] and idx <= rep_tup[1]:
#                         final_list.append(retroembeds[rep_idx])
#                         added = True
#                         break
#                 if not added:
#                     t = np.zeros((300,))
#                     final_list.append(t)
#             stacked_sentences.append(final_list)
#         retro_embeds = torch.tensor(stacked_sentences).float().to(self.device)
#         # retro_embeds = retro_embeds.reshape(retro_embeds.shape[0],retro_embeds.shape[2],retro_embeds.shape[1])
#
#
#         retro_embeds = self.convovler1(retro_embeds)
#         retro_embeds = self.relu(retro_embeds)
#         retro_embeds = self.maxpool(retro_embeds)
#         retro_embeds = self.convovler2(retro_embeds)
#         retro_embeds = self.relu(retro_embeds)
#         retro_embeds = self.maxpool(retro_embeds)
#         retro_embeds = torch.flatten(retro_embeds,1)
#         retro_embeds = self.retro_compressor(retro_embeds)
#         # print("Here")
#         # entity_embeds = self.compressor(retro_embeds)
#         # entity_embeds = self.nonlinear(entity_embeds)
#         # entity_embeds = self.decompressor(entity_embeds)
#         # representation = torch.cat([last_hidden_state,entity_embeds],dim=2)
#         # before_avg = self.mover(representation)
#         # token_avg = torch.mean(before_avg[:,1:], 1)
#         # pool_tok = before_avg[:,0]
#         # representation = torch.cat([pool_tok,token_avg],1)
#         # representation = self.mover2(representation)
#         # representation = self.nonlinear(representation)
#         # pool_toks = last_hidden_state[:,0]
#         # entity_embeds = self.compressor(entity_embeds)
#         # entity_embeds = self.nonlinear(entity_embeds)
#         # entity_embeds = self.decompressor(entity_embeds)
#         # entity_embeds = self.nonlinear(entity_embeds)
#         # extended_att_maskk = get_extended_attention_mask(attention_mask, entity_embeds.shape, self.device)
#         # input_dict = {'hidden_states':entity_embeds,"attention_mask":extended_att_maskk}
#         # att_entity_embeds = self.k_att(**input_dict)[0]
#         # expanded_entity_embeds = self.knowledge_infusion_layer(att_entity_embeds)
#         #
#         # # representation = last_hidden_state #+ expanded_entity_embeds
#         # representation = torch.cat([last_hidden_state,expanded_entity_embeds],dim=2)
#         # representation = self.mover(representation)
#         # representation = self.nonlinear(representation)
#         # input_dict["hidden_states"] = representation
#         # representation = self.recontextualizer(**input_dict)[0]
#         # # representation = torch.mean(representation, 1)
#         # representation = self.pooler(representation)
#
#         # avg_retro_embeds = [[torch.tensor(e).float() for e in x if not torch.tensor(e).float().sum()==0] for x in stacked_retroembeds]
#         # avg_retro_embeds = [torch.stack(x) for x in avg_retro_embeds]
#         # avg_retro_embeds = [torch.mean(x,0) for x in avg_retro_embeds]
#         # avg_retro_embeds = torch.stack(avg_retro_embeds).to(self.device)
#         # avg_retro_embeds = self.retro_expander(avg_retro_embeds)
#         # avg_retro_embeds = self.retro_next_dim(avg_retro_embeds)
#         # avg_retro_embeds = self.relu(avg_retro_embeds)
#         # avg_retro_embeds = self.retro_compressor(avg_retro_embeds)
#
#         # representation = torch.cat([pool_toks, retro_embeds], dim=1)
#         # representation = self.mover2(representation)
#         # representation = se
#         # lf.nonlinear(representation)
#         representation = retro_embeds
#         return representation#, retro_embeds, before_avg,
#
#     def mm_loss(self, transformer_outputs, retro_vecs):
#         margin_calc_amount = 5
#         comp = self.lm_loss_ent(transformer_outputs)
#         lambda_param = 0.3
#         ent_loss = 0
#         for i in range(margin_calc_amount):
#             rp = torch.randperm(retro_vecs.size()[1])
#             a = torch.cosine_similarity(comp, retro_vecs, 2)
#             b = torch.cosine_similarity(comp, retro_vecs[:, rp, :], 2)
#             loss_t = torch.clamp(a - b + lambda_param, min=0.0)
#             loss_t = torch.sum(loss_t, dim=0) / (a.shape[0] * 1.0)
#             loss_t = torch.sum(loss_t, dim=0) / (a.shape[1] * 1.0)
#             ent_loss += loss_t
#         ent_loss /= (margin_calc_amount * 1.0)
#         return ent_loss
#
#     def training_step(self, batch, batch_idx):
#
#
#         labels = batch['label'].float()
#
#         query_dict = batch['query_enc_dict']
#         response_dict = batch['response_enc_dict']
#
#         # query_last_hidden_state, query_pooler_output = self.bert(query_dict['input_ids'], token_type_ids=query_dict['token_type_ids'], attention_mask=query_dict['attention_mask'])
#         query_pooler_output = self.knowledge_infusion(query_dict["input_ids"])
#         # l1 = self.mm_loss(query_last_hidden_infused,query_retroembeds)
#         # response_last_hidden_state, response_pooler_output = self.bert(response_dict['input_ids'], token_type_ids=response_dict['token_type_ids'], attention_mask=response_dict['attention_mask'])
#         response_pooler_output = self.knowledge_infusion(response_dict["input_ids"])#,response_last_hidden_state, attention_mask=response_dict['attention_mask'])
#         # l2 = self.mm_loss(response_last_hidden_infused,response_retroembeds)
#         preds = torch.sigmoid(self.cosine_sim(self.query_rescale_layer(query_pooler_output), self.query_rescale_layer(response_pooler_output)))
#
#         loss = mse_loss(preds, labels)
#         alpha = 4
#         # mm_loss = alpha*(l1+l2)/2.0
#         # loss += mm_loss
#         # Logging to TensorBoard by default
#         self.log('train_loss', loss,on_step=True)
#         # self.log('mm_train_loss', mm_loss,on_step=True)
#
#         return loss
#
#     def validation_step(self, batch, batch_idx):
#         batch = transfer_batch_to_device(batch,self.device)
#         labels = batch['label'].float()
#
#         query_dict = batch['query_enc_dict']
#         response_dict = batch['response_enc_dict']
#
#         # query_last_hidden_state, query_pooler_output = self.bert(query_dict['input_ids'], token_type_ids=query_dict['token_type_ids'], attention_mask=query_dict['attention_mask'])
#         query_pooler_output = self.knowledge_infusion(query_dict["input_ids"])
#         # l1 = self.mm_loss(query_last_hidden_infused,query_retroembeds)
#         # response_last_hidden_state, response_pooler_output = self.bert(response_dict['input_ids'], token_type_ids=response_dict['token_type_ids'], attention_mask=response_dict['attention_mask'])
#         response_pooler_output = self.knowledge_infusion(
#             response_dict["input_ids"])  # ,response_last_hidden_state, attention_mask=response_dict['attention_mask'])
#         # l2 = self.mm_loss(response_last_hidden_infused,response_retroembeds)
#
#
#         preds =self.cosine_sim(self.query_rescale_layer(query_pooler_output), self.query_rescale_layer(response_pooler_output)).to(self.device)
#
#         val_loss =  torch.nn.BCEWithLogitsLoss()(preds,labels)
#         self.log('val_loss_step', val_loss)
#         self.log('test_acc_step', self.accuracy(preds, labels))
#         return val_loss
#
#     def on_validation_epoch_end(self):
#         print("Calculating validation accuracy...")
#         vacc = self.accuracy.compute()
#         print("Validation accuracy:", vacc)
#         self.log('val_acc_epoch', vacc)
#
#     def test_step(self, batch, batch_idx):
#         batch = transfer_batch_to_device(batch,self.device)
#         labels = batch['label'].float()
#
#         query_dict = batch['query_enc_dict']
#         response_dict = batch['response_enc_dict']
#
#         # query_last_hidden_state, query_pooler_output = self.bert(query_dict['input_ids'], token_type_ids=query_dict['token_type_ids'], attention_mask=query_dict['attention_mask'])
#         query_pooler_output = self.knowledge_infusion(query_dict["input_ids"])
#         # l1 = self.mm_loss(query_last_hidden_infused,query_retroembeds)
#         # response_last_hidden_state, response_pooler_output = self.bert(response_dict['input_ids'], token_type_ids=response_dict['token_type_ids'], attention_mask=response_dict['attention_mask'])
#         response_pooler_output = self.knowledge_infusion(
#             response_dict["input_ids"])  # ,response_last_hidden_state, attention_mask=response_dict['attention_mask'])
#         # l2 = self.mm_loss(response_last_hidden_infused,response_retroembeds)
#
#         preds = self.cosine_sim(self.query_rescale_layer(query_pooler_output), self.query_rescale_layer(response_pooler_output)).to(self.device)
#
#         test_loss = torch.nn.BCEWithLogitsLoss()(preds,labels)
#         self.log('test_loss', test_loss)
#         self.log('test_acc_step', self.accuracy(preds, labels))
#         return test_loss
#
#     def on_test_epoch_end(self):
#         print("Calculating test accuracy...")
#         self.log('test_acc_epoch', self.accuracy.compute())
#
#     def configure_optimizers(self):
#         optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
#         return optimizer
#

class LitOutputBertModel(pl.LightningModule):

    def __init__(self, learning_rate = 5e-4):
        super().__init__()
        bertmodel = "bert-base-uncased"
        self.bert = BertModel.from_pretrained(
            bertmodel,# Use the 12-layer BERT model, with an uncased vocab.
            output_attentions=False,# Whether the model returns attentions weights.
            output_hidden_states=False,# Whether the model returns all hidden-states.
        )
        print("Freeze the bert model!!!")
        # for parameter in self.bert.parameters():
        #     parameter.requires_grad = False
        # self.learning_rate = learning_rate
        self.query_rescale_layer = nn.Linear(768, 768)
        self.response_rescale_layer = nn.Linear(768, 768)
        self.compressor = nn.Linear(300,64)
        self.decompressor = nn.Linear(300,768)

        # self.decompressor = nn.Linear(64,300)
        self.downscale = nn.Linear(768,64)
        s = torch.nn.init.xavier_uniform_(self.downscale.weight, gain=0.0001)
        self.downscale.weight = s

        self.mm_scaler = nn.Linear(768,300)
        self.upscale = nn.Linear(64,768)
        s = torch.nn.init.xavier_uniform_(self.upscale.weight, gain=0.0001)
        self.upscale.weight = s
        self.relu = nn.ReLU()
        self.convovler1 = nn.Conv1d(128,64,3)
        self.maxpool = nn.MaxPool1d(3)
        self.convolver4 = nn.Conv1d(768,256,3)

        # self.nonlinear = nn.Tanh()
        self.dropout = nn.Dropout(0.3)

        self.cosine_sim = nn.CosineSimilarity(dim=1, eps=1e-6)
        # self.save_hyperparameters()

        self.accuracy = pl.metrics.Accuracy()
        self.testaccuracy = pl.metrics.Accuracy()

        self.tokenizer = BertTokenizerFast.from_pretrained(bertmodel)
        # self.numberbatch = pd.read_hdf("models/mini-2.h5","mat")
        if os.path.exists("nb.h5"):
            print("Using cached!")
            self.numberbatch = pd.read_hdf("nb.h5","mat")
        else:
            names = []
            vecs = []
            with open("models/graphembeddings/numberbatch-en-19.08.txt") as nb:
                for line in tqdm(nb):
                    line = line.strip()
                    if len(line.split())==2:
                        continue
                    name = line.split()[0]
                    d = pd.Series([float(x) for x in line.split()[1:]])
                    vecs.append(d)
                    names.append(name)
            self.numberbatch = pd.DataFrame(data=vecs,index=names)
            self.numberbatch.to_hdf("nb.h5","mat")
        print(self.numberbatch.loc["cat"])


    def on_train_epoch_start(self):
        print("Starting training epoch...")
        # if self.current_epoch == 2:
        #     print("Unfreezing in second epoch")
        print("Freezing bert...")
        # for p in self.bert.parameters():
        #     p.requires_grad = True
        print("Done")
    def on_train_epoch_end(self, outputs):
        print("Finished training epoch...")

    def knowledge_infusion(self, input_ids,last_hidden_state,attention_mask):
        stacked_sentences = []
        prefix = ""
        stacked_retroembeds = []
        for inpt_id_idx, sentence in enumerate(input_ids):
            dec_sentence = self.tokenizer.decode(sentence.cpu().numpy())
            sentence_vecs = []
            counts = {}
            def find_sub_list(sl, l):
                if str(sl) not in counts.keys():
                    counts[str(sl)] = 0
                counts[str(sl)] += 1
                results = []
                sll = len(sl)
                for ind in (i for i, e in enumerate(l) if e == sl[0]):
                    if l[ind:ind + sll] == sl:
                        results.append((ind, ind + sll - 1))
                try:
                    r = results[counts[str(sl)] - 1]

                    return r
                except Exception as t:
                    return None

            words = re.findall(r'\w+', dec_sentence)
            retroembeds = []
            for word in words:
                try:
                    vec = self.numberbatch.loc[prefix + word]
                    to_append = np.array(vec).reshape(300, )
                except:
                    to_append = np.zeros((300,))
                retroembeds.append(to_append)
            # retroembeds = retroembeddings.get_embeddings_from_input_ids(words).contiguous()
            # retroembeds  = retro_vecs[sample]
            stacked_retroembeds.append(retroembeds)
            replacement_list = []
            for word in words:
                toks = self.tokenizer.encode(word, add_special_tokens=False)
                locs = find_sub_list(toks, [int(x) for x in sentence.cpu().numpy()])
                if locs is None:
                    continue
                replacement_list.append(locs)
            final_list = []
            for idx, id in enumerate(sentence):
                # Id iN SPECIAL TOKENS
                added = False
                for rep_idx, rep_tup in enumerate(replacement_list):
                    if idx >= rep_tup[0] and idx <= rep_tup[1]:
                        final_list.append(retroembeds[rep_idx])
                        added = True
                        break
                if not added:
                    t = np.zeros((300,))
                    final_list.append(t)
            stacked_sentences.append(final_list)
        retro_embeds = torch.tensor(stacked_sentences).float().to(self.device)
        original_retro_embeds = retro_embeds
        retro_embeds = self.decompressor(retro_embeds)
        adapter_down = retro_embeds+last_hidden_state
        # adapter_down = self.downscale(last_hidden_state) + retro_embeds
        # adapter_down = self.relu(adapter_down)
        # adapter_down = self.upscale(adapter_down) + last_hidden_state

        # retro_embeds = self.convovler1(adapter_down)
        # retro_embeds = self.relu(retro_embeds)
        # adapter_down = self.maxpool(retro_embeds)
        # retro_embeds = self.convovler2(retro_embeds)
        # retro_embeds = self.relu(retro_embeds)
        # retro_embeds = self.maxpool(retro_embeds)
        # retro_embeds = retro_embeds.reshape(retro_embeds.shape[0],retro_embeds.shape[2],retro_embeds.shape[1])
        # retro_embeds = self.convolver4(retro_embeds)
        # retro_embeds = torch.flatten(retro_embeds,1)
        # retro_embeds = self.comp(retro_embeds)
        pre_pool = adapter_down
        retro_embeds = torch.mean(adapter_down,1)
        return retro_embeds,original_retro_embeds, pre_pool

    def mm_loss(self, transformer_outputs, retro_vecs):
        margin_calc_amount = 5
        comp = transformer_outputs
        lambda_param = 0.3
        ent_loss = 0
        for i in range(margin_calc_amount):
            rp = torch.randperm(retro_vecs.size()[1])
            a = torch.cosine_similarity(comp, retro_vecs, 2)
            b = torch.cosine_similarity(comp, retro_vecs[:, rp, :], 2)
            loss_t = torch.clamp(a - b + lambda_param, min=0.0)
            loss_t = torch.sum(loss_t, dim=0) / (a.shape[0] * 1.0)
            loss_t = torch.sum(loss_t, dim=0) / (a.shape[1] * 1.0)
            ent_loss += loss_t
        ent_loss /= (margin_calc_amount * 1.0)
        return ent_loss

    def training_step(self, batch, batch_idx):


        labels = batch['label'].float()

        query_dict = batch['query_enc_dict']
        response_dict = batch['response_enc_dict']

        query_last_hidden_state, query_pooler_output = self.bert(query_dict['input_ids'], token_type_ids=query_dict['token_type_ids'], attention_mask=query_dict['attention_mask'])
        query_pooler_output, query_retroembeds, query_pre_pooler_output = self.knowledge_infusion(query_dict["input_ids"],query_last_hidden_state, attention_mask=query_dict['attention_mask'])
        # l1 = self.mm_loss(self.mm_scaler(query_pre_pooler_output),query_retroembeds)
        response_last_hidden_state, response_pooler_output = self.bert(response_dict['input_ids'], token_type_ids=response_dict['token_type_ids'], attention_mask=response_dict['attention_mask'])
        response_pooler_output, response_retroembeds, response_pre_pooler_output = self.knowledge_infusion(response_dict["input_ids"],response_last_hidden_state, attention_mask=response_dict['attention_mask'])
        # l2 = self.mm_loss(self.mm_scaler(response_pre_pooler_output),response_retroembeds)
        # a = self.query_rescale_layer(query_pooler_output)
        # a = self.dropout(a)
        # b = self.query_rescale_layer(response_pooler_output)
        # b  = self.dropout(b)
        a = query_pooler_output
        b = response_pooler_output

        c = self.cosine_sim(a, b)
        # preds = torch.sigmoid(c)

        # base_train_loss = mse_loss(preds, labels)
        base_train_loss = torch.nn.BCEWithLogitsLoss()(c,labels)
        self.log('base_train_loss', base_train_loss,on_step=True)
        self.log('train_loss',base_train_loss)
        alpha = 1
        # mm_loss = alpha*((l1+l2)/2.0)
        # loss = base_train_loss + mm_loss
        # self.log('train_loss', loss,on_step=True)

        # Logging to TensorBoard by default
        # self.log('mm_loss', mm_loss)
        r = {'train_loss':base_train_loss,'minimize':base_train_loss}
        res = Result(minimize=r['minimize'])
        res.update(r)
        # minimize = loss
        return res#loss,mm_loss, base_train_loss,minimize

    def validation_step(self, batch, batch_idx):
        batch = transfer_batch_to_device(batch,self.device)
        labels = batch['label'].float()

        query_dict = batch['query_enc_dict']
        response_dict = batch['response_enc_dict']

        query_last_hidden_state, query_pooler_output = self.bert(query_dict['input_ids'],
                                                                 token_type_ids=query_dict['token_type_ids'],
                                                                 attention_mask=query_dict['attention_mask'])
        query_pooler_output,_,_ = self.knowledge_infusion(query_dict["input_ids"],query_last_hidden_state, attention_mask=query_dict['attention_mask'])

        response_last_hidden_state, response_pooler_output = self.bert(response_dict['input_ids'],
                                                                       token_type_ids=response_dict['token_type_ids'],
                                                                       attention_mask=response_dict['attention_mask'])
        response_pooler_output,_,_= self.knowledge_infusion(response_dict["input_ids"],response_last_hidden_state, attention_mask=response_dict['attention_mask'])

        # a = self.query_rescale_layer(query_pooler_output)
        # b = self.response_rescale_layer(response_pooler_output)
        a = query_pooler_output
        b = response_pooler_output
        preds =self.cosine_sim(a, b).to(self.device)

        val_loss = torch.nn.BCEWithLogitsLoss()(preds,labels)
        self.log('val_loss_step', val_loss)
        self.log('val_acc_step', self.accuracy(preds, labels))
        # self.print(self.accuracy.compute())
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

        query_last_hidden_state, query_pooler_output = self.bert(query_dict['input_ids'],
                                                                 token_type_ids=query_dict['token_type_ids'],
                                                                 attention_mask=query_dict['attention_mask'])
        query_pooler_output , _  ,_= self.knowledge_infusion(query_dict["input_ids"],query_last_hidden_state, attention_mask=query_dict['attention_mask'])

        response_last_hidden_state, response_pooler_output = self.bert(response_dict['input_ids'],
                                                                       token_type_ids=response_dict['token_type_ids'],
                                                                       attention_mask=response_dict['attention_mask'])
        response_pooler_output, _ ,_= self.knowledge_infusion(response_dict["input_ids"],response_last_hidden_state, attention_mask=response_dict['attention_mask'])

        # preds = self.cosine_sim(self.query_rescale_layer(query_pooler_output), self.query_rescale_layer(response_pooler_output)).to(self.device)
        preds = self.cosine_sim(query_pooler_output, response_pooler_output).to(self.device)

        test_loss = torch.nn.BCEWithLogitsLoss()(preds,labels)
        self.log('test_loss', test_loss)
        self.log('test_acc_step', self.testaccuracy(preds, labels))
        return test_loss

    def on_test_epoch_end(self):
        print("Calculating test accuracy...")
        self.log('test_acc_epoch', self.testaccuracy.compute())

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=5e-5)
        scheduler = StepLR(optimizer, step_size=25, gamma=0.8)
        return [optimizer], [scheduler]
        # return optimizer#, [scheduler]
        # return optimizer
