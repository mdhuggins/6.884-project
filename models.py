import re
from typing import Dict, List

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch as F
from torch.nn.functional import mse_loss
from tqdm import tqdm
from transformers import BertModel, BertTokenizerFast
import sklearn
from kb.include_all import ModelArchiveFromParams
from kb.knowbert_utils import KnowBertBatchifier
from allennlp.common import Params
import spacy
from spacy.matcher import Matcher
from spacy.util import filter_spans
import pandas as pd
import numpy as np

def get_accuracy(y_true, y_prob):
    accuracy = sklearn.metrics.accuracy_score(y_true, [1 if x > 0.5 else 0 for x in y_prob])
    return accuracy
def transfer_batch_to_device(batch, device):
    # print("### DEVICE CHECK", device)
    for k in batch.keys():
        if isinstance(batch[k],Dict):
            batch[k] = transfer_batch_to_device(batch[k],device)
        elif isinstance(batch[k],List):
            # print(batch[k])
            continue
        else:
            batch[k] = batch[k].to(device)
    return batch

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


    def training_step(self, batch, batch_idx):

        labels = batch['label'].float()

        query_dict = batch['query_enc_dict']
        response_dict = batch['response_enc_dict']

        query_last_hidden_state, query_pooler_output = self.bert(query_dict['input_ids'], token_type_ids=query_dict['token_type_ids'], attention_mask=query_dict['attention_mask'])
        response_last_hidden_state, response_pooler_output = self.bert(response_dict['input_ids'], token_type_ids=response_dict['token_type_ids'], attention_mask=response_dict['attention_mask'])

        preds = torch.sigmoid(self.cosine_sim(self.query_rescale_layer(query_pooler_output), self.response_rescale_layer(response_pooler_output)))

        loss =mse_loss(preds, labels)

        # Logging to TensorBoard by default
        self.log('train_loss', loss,on_step=True)

        return loss



    def validation_step(self, batch, batch_idx):
        batch = transfer_batch_to_device(batch,self.device)
        labels = batch['label'].float()

        query_dict = batch['query_enc_dict']
        response_dict = batch['response_enc_dict']

        query_last_hidden_state, query_pooler_output = self.bert(query_dict['input_ids'],
                                                                 token_type_ids=query_dict['token_type_ids'],
                                                                 attention_mask=query_dict['attention_mask'])
        response_last_hidden_state, response_pooler_output = self.bert(response_dict['input_ids'],
                                                                       token_type_ids=response_dict['token_type_ids'],
                                                                       attention_mask=response_dict['attention_mask'])

        preds = self.cosine_sim(self.query_rescale_layer(query_pooler_output), self.response_rescale_layer(response_pooler_output)).to(self.device)

        val_loss =  torch.nn.BCEWithLogitsLoss()(preds,labels)
        self.log('val_loss', val_loss,on_step=True)
        return val_loss

    def test_step(self, batch, batch_idx):
        batch = transfer_batch_to_device(batch,self.device)
        labels = batch['label'].float()

        query_dict = batch['query_enc_dict']
        response_dict = batch['response_enc_dict']

        query_last_hidden_state, query_pooler_output = self.bert(query_dict['input_ids'],
                                                                 token_type_ids=query_dict['token_type_ids'],
                                                                 attention_mask=query_dict['attention_mask'])
        response_last_hidden_state, response_pooler_output = self.bert(response_dict['input_ids'],
                                                                       token_type_ids=response_dict['token_type_ids'],
                                                                       attention_mask=response_dict['attention_mask'])

        preds = self.cosine_sim(self.query_rescale_layer(query_pooler_output), self.response_rescale_layer(response_pooler_output)).to(self.device)

        val_loss =  torch.nn.BCEWithLogitsLoss()(preds,labels)
        self.log('test_loss', val_loss,on_step=True)
        return val_loss

    def on_validation_epoch_end(self):
        results = []
        gold = []
        print("Calculating validation accuracy...")
        for idx, batch in enumerate(tqdm(self.val_dataloader(),leave=True)):
            batch = transfer_batch_to_device(batch, self.device)
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

            preds = torch.sigmoid(self.cosine_sim(self.query_rescale_layer(query_pooler_output), self.response_rescale_layer(response_pooler_output)))
            results.extend([float(x) for x in preds.cpu().numpy()])
        val_acc = get_accuracy(gold, results)
        print("Val acc",val_acc)
        self.log('val_accuracy',val_acc,on_epoch=True)

    def on_test_epoch_end(self):
        results = []
        gold = []
        print("Calculating test accuracy...")
        for idx, batch in enumerate(tqdm(self.test_dataloader(),leave=True)):
            batch = transfer_batch_to_device(batch, self.device)
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

            preds = torch.sigmoid(self.cosine_sim(self.query_rescale_layer(query_pooler_output),
                                                                self.response_rescale_layer(response_pooler_output)))
            results.extend([float(x) for x in preds.cpu().numpy()])
        val_acc = get_accuracy(gold, results)
        print("Test acc", val_acc)
        self.log('test_accuracy', val_acc,on_epoch=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

class LitOutputBertModel(pl.LightningModule):

    def __init__(self):
        super().__init__()
        bertmodel = "bert-base-uncased"
        self.bert = BertModel.from_pretrained(
            bertmodel, # Use the 12-layer BERT model, with an uncased vocab.
            output_attentions = False, # Whether the model returns attentions weights.
            output_hidden_states = False, # Whether the model returns all hidden-states.
        )
        self.query_rescale_layer = nn.Linear(768, 768)
        self.response_rescale_layer = nn.Linear(768, 768)
        self.knowledge_infusion_layer = nn.Linear(300,768)
        self.cosine_sim = nn.CosineSimilarity(dim=1, eps=1e-6)
        # self.save_hyperparameters()


        self.nlp = spacy.load('en_core_web_sm')

        pattern = [{'POS': 'VERB', 'OP': '?'},
                   {'POS': 'ADV', 'OP': '*'},
                   {'POS': 'AUX', 'OP': '*'},
                   {'POS': 'VERB', 'OP': '+'}]

        # instantiate a Matcher instance
        self.matcher = Matcher(self.nlp.vocab)
        self.matcher.add("Verb phrase", None, pattern)
        self.tokenizer = BertTokenizerFast.from_pretrained(bertmodel)
        self.numberbatch = pd.read_hdf("models/mini-2.h5","mat")

    def knowledge_infusion(self, input_ids,last_hidden_state):
        stacked_sentences = []
        prefix = "/c/en/"
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
            # for word in dec_sentence.split():
            #     enc_word = self.tokenizer.encode(word, add_special_tokens=False)
            #     try:
            #         vec = self.numberbatch.loc[prefix + word]
            #         to_append = np.array(vec).reshape(300, )
            #     except:
            #         to_append = np.zeros((300,))
            #     for i in range(len(enc_word)):
            #         sentence_vecs.append(to_append)
            stacked_sentences.append(final_list)
        entity_embeds = torch.tensor(stacked_sentences).float().to(self.device)
        expanded_entity_embeds = self.knowledge_infusion_layer(entity_embeds)
        representation = last_hidden_state + expanded_entity_embeds
        representation = torch.mean(representation, 1)
        return representation

    def training_step(self, batch, batch_idx):

        labels = batch['label'].float()

        query_dict = batch['query_enc_dict']
        response_dict = batch['response_enc_dict']

        query_last_hidden_state, query_pooler_output = self.bert(query_dict['input_ids'], token_type_ids=query_dict['token_type_ids'], attention_mask=query_dict['attention_mask'])
        query_pooler_output = self.knowledge_infusion(query_dict["input_ids"],query_last_hidden_state)

        response_last_hidden_state, response_pooler_output = self.bert(response_dict['input_ids'], token_type_ids=response_dict['token_type_ids'], attention_mask=response_dict['attention_mask'])
        response_pooler_output = self.knowledge_infusion(response_dict["input_ids"],response_last_hidden_state)

        preds = torch.sigmoid(self.cosine_sim(self.query_rescale_layer(query_pooler_output), self.response_rescale_layer(response_pooler_output)))

        loss = mse_loss(preds, labels)

        # Logging to TensorBoard by default
        self.log('train_loss', loss,on_step=True)

        return loss


    def validation_step(self, batch, batch_idx):
        batch = transfer_batch_to_device(batch,self.device)
        labels = batch['label'].float()

        query_dict = batch['query_enc_dict']
        response_dict = batch['response_enc_dict']

        query_last_hidden_state, query_pooler_output = self.bert(query_dict['input_ids'],
                                                                 token_type_ids=query_dict['token_type_ids'],
                                                                 attention_mask=query_dict['attention_mask'])
        query_pooler_output = self.knowledge_infusion(query_dict["input_ids"], query_last_hidden_state)

        response_last_hidden_state, response_pooler_output = self.bert(response_dict['input_ids'],
                                                                       token_type_ids=response_dict['token_type_ids'],
                                                                       attention_mask=response_dict['attention_mask'])
        response_pooler_output = self.knowledge_infusion(response_dict["input_ids"], response_last_hidden_state)

        preds =self.cosine_sim(self.query_rescale_layer(query_pooler_output), self.response_rescale_layer(response_pooler_output)).to(self.device)

        val_loss =  torch.nn.BCEWithLogitsLoss()(preds,labels)
        self.log('val_loss', val_loss,on_step=True)
        return val_loss

    def test_step(self, batch, batch_idx):
        batch = transfer_batch_to_device(batch,self.device)
        labels = batch['label'].float()

        query_dict = batch['query_enc_dict']
        response_dict = batch['response_enc_dict']

        query_last_hidden_state, query_pooler_output = self.bert(query_dict['input_ids'],
                                                                 token_type_ids=query_dict['token_type_ids'],
                                                                 attention_mask=query_dict['attention_mask'])
        query_pooler_output = self.knowledge_infusion(query_dict["input_ids"], query_last_hidden_state)

        response_last_hidden_state, response_pooler_output = self.bert(response_dict['input_ids'],
                                                                       token_type_ids=response_dict['token_type_ids'],
                                                                       attention_mask=response_dict['attention_mask'])
        response_pooler_output = self.knowledge_infusion(response_dict["input_ids"], response_last_hidden_state)

        preds = self.cosine_sim(self.query_rescale_layer(query_pooler_output), self.response_rescale_layer(response_pooler_output)).to(self.device)

        val_loss = torch.nn.BCEWithLogitsLoss()(preds,labels)
        self.log('test_loss', val_loss,on_step=True)
        return val_loss

    def on_validation_epoch_end(self):
        results = []
        gold = []
        print("Calculating validation accuracy...")
        for idx, batch in enumerate(tqdm(self.val_dataloader(),leave=True)):
            batch = transfer_batch_to_device(batch, self.device)
            labels = batch['label'].float()
            gold.extend([int(x) for x in labels.cpu().numpy()])
            query_dict = batch['query_enc_dict']
            response_dict = batch['response_enc_dict']

            query_last_hidden_state, query_pooler_output = self.bert(query_dict['input_ids'],
                                                                     token_type_ids=query_dict['token_type_ids'],
                                                                     attention_mask=query_dict['attention_mask'])
            query_pooler_output = self.knowledge_infusion(query_dict["input_ids"], query_last_hidden_state)

            response_last_hidden_state, response_pooler_output = self.bert(response_dict['input_ids'],
                                                                           token_type_ids=response_dict[
                                                                               'token_type_ids'],
                                                                           attention_mask=response_dict[
                                                                               'attention_mask'])
            response_pooler_output = self.knowledge_infusion(response_dict["input_ids"], response_last_hidden_state)

            preds = torch.sigmoid(self.cosine_sim(self.query_rescale_layer(query_pooler_output), self.response_rescale_layer(response_pooler_output)))
            results.extend([float(x) for x in preds.cpu().numpy()])
        val_acc = get_accuracy(gold, results)
        print("Val acc",val_acc)
        self.log('val_accuracy',val_acc,on_epoch=True)

    def on_test_epoch_end(self):
        results = []
        gold = []
        print("Calculating test accuracy...")
        for idx, batch in enumerate(tqdm(self.test_dataloader(),leave=True)):
            batch = transfer_batch_to_device(batch, self.device)
            labels = batch['label'].float()
            gold.extend([int(x) for x in labels.cpu().numpy()])
            query_dict = batch['query_enc_dict']
            response_dict = batch['response_enc_dict']

            query_last_hidden_state, query_pooler_output = self.bert(query_dict['input_ids'],
                                                                     token_type_ids=query_dict['token_type_ids'],
                                                                     attention_mask=query_dict['attention_mask'])
            query_pooler_output = self.knowledge_infusion(query_dict["input_ids"], query_last_hidden_state)

            response_last_hidden_state, response_pooler_output = self.bert(response_dict['input_ids'],
                                                                           token_type_ids=response_dict[
                                                                               'token_type_ids'],
                                                                           attention_mask=response_dict[
                                                                               'attention_mask'])
            response_pooler_output = self.knowledge_infusion(response_dict["input_ids"], response_last_hidden_state)

            preds = torch.sigmoid(self.cosine_sim(self.query_rescale_layer(query_pooler_output),
                                                                self.response_rescale_layer(response_pooler_output)))
            results.extend([float(x) for x in preds.cpu().numpy()])
        val_acc = get_accuracy(gold, results)
        print("Test acc", val_acc)
        self.log('test_accuracy', val_acc,on_epoch=True)

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

        val_loss = torch.nn.BCEWithLogitsLoss()(preds,labels)
        self.log('val_loss', val_loss,on_step=True)
        return val_loss

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

        val_loss = torch.nn.BCEWithLogitsLoss()(preds,labels)
        self.log('test_loss', val_loss,on_step=True)
        return val_loss

    def on_validation_epoch_end(self):
        results = []
        gold = []
        return
        print("Calculating validation accuracy...")
        for idx, batch in enumerate(tqdm(self.val_dataloader(),leave=True)):
            batch = transfer_batch_to_device(batch, self.device)
            labels = batch['label'].float()
            gold.extend([int(x) for x in labels.cpu().numpy()])
            query_dict = batch['query_enc_dict']
            response_dict = batch['response_enc_dict']
            query_text = batch['query_text']
            response_text = batch['response_text']
            b = next(self.batcher.iter_batches(response_text,verbose=False))
            b = transfer_batch_to_device(b, self.device)

            query_pooler_output = self.bert(**b)['pooled_output']
            b = next(self.batcher.iter_batches(response_text,verbose=False))
            b = transfer_batch_to_device(b, self.device)

            response_pooler_output = self.bert(**b)['pooled_output']
            # query_last_hidden_state, query_pooler_output = self.bert(query_dict['input_ids'],
            #                                                          token_type_ids=query_dict['token_type_ids'],
            #                                                          attention_mask=query_dict['attention_mask'])
            # response_last_hidden_state, response_pooler_output = self.bert(response_dict['input_ids'],
            #                                                                token_type_ids=response_dict[
            #                                                                    'token_type_ids'],
            #                                                                attention_mask=response_dict[
            #                                                                    'attention_mask'])

            preds = torch.sigmoid(self.cosine_sim(self.query_rescale_layer(query_pooler_output), self.response_rescale_layer(response_pooler_output)))
            results.extend([float(x) for x in preds.cpu().numpy()])
        val_acc = get_accuracy(gold, results)
        print("Val acc",val_acc)
        self.log('val_accuracy',val_acc,on_epoch=True)

    def on_test_epoch_end(self):
        results = []
        gold = []
        print("Calculating test accuracy...")
        for idx, batch in enumerate(tqdm(self.test_dataloader(),leave=True)):
            batch = transfer_batch_to_device(batch, self.device)
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
            b = transfer_batch_to_device(b, self.device)

            query_pooler_output = self.bert(**b)['pooled_output']
            b = next(self.batcher.iter_batches(response_text,verbose=False))
            b = transfer_batch_to_device(b, self.device)

            response_pooler_output = self.bert(**b)['pooled_output']
            preds = torch.sigmoid(self.cosine_sim(self.query_rescale_layer(query_pooler_output),
                                                                self.response_rescale_layer(response_pooler_output)))
            results.extend([float(x) for x in preds.cpu().numpy()])
        val_acc = get_accuracy(gold, results)
        print("Test acc", val_acc)
        self.log('test_accuracy', val_acc,on_epoch=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
