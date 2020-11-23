import re
from typing import Dict, List

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch as F
from torch.nn.functional import mse_loss
from tqdm import tqdm
from transformers import BertModel, BertTokenizerFast
from transformers.modeling_outputs import BaseModelOutputWithPooling
import sklearn
# from kb.include_all import ModelArchiveFromParams
# from kb.knowbert_utils import KnowBertBatchifier
# from allennlp.common import Params
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

        loss = mse_loss(preds, labels)

        # Logging to TensorBoard by default
        self.log('train_loss', loss, on_step=True)

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

        processed_embedding_output = self.input_process_fn(input_ids, embedding_output)

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


class LitInputBertModel(pl.LightningModule):

    def __init__(self):
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


        self.cosine_sim = nn.CosineSimilarity(dim=1, eps=1e-6)
        # self.save_hyperparameters()

        self.tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
        self.numberbatch = pd.read_hdf("models/mini-2.h5","mat")

    def knowledge_infusion(self, input_ids, embedding_output):
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

        concat = torch.cat((entity_embeds, embedding_output), 2)
        representation = self.concat_rescale_layer(concat)

        return representation

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
