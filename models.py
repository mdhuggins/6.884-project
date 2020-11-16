import re
from typing import Dict, List

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch as F
from torch.nn.functional import mse_loss
from tqdm import tqdm
from transformers import BertModel, BertTokenizerFast, BertLayer, BertConfig
import sklearn
from kb.include_all import ModelArchiveFromParams
from kb.knowbert_utils import KnowBertBatchifier
from allennlp.common import Params
import spacy
from spacy.matcher import Matcher
from spacy.util import filter_spans
import pandas as pd
import numpy as np

# def get_accuracy(y_true, y_prob):
#     accuracy = sklearn.metrics.accuracy_score(y_true, [1 if x > 0.5 else 0 for x in y_prob])
#     return accuracy
from transformers.modeling_bert import BertEncoder, BertPooler

def get_extended_attention_mask(attention_mask, input_shape , device):
    """
    Makes broadcastable attention and causal masks so that future and masked tokens are ignored.

    Arguments:
        attention_mask (:obj:`torch.Tensor`):
            Mask with ones indicating tokens to attend to, zeros for tokens to ignore.
        input_shape (:obj:`Tuple[int]`):
            The shape of the input to the model.
        device: (:obj:`torch.device`):
            The device of the input to the model.

    Returns:
        :obj:`torch.Tensor` The extended attention mask, with a the same dtype as :obj:`attention_mask.dtype`.
    """
    # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
    # ourselves in which case we just need to make it broadcastable to all heads.
    if attention_mask.dim() == 3:
        extended_attention_mask = attention_mask[:, None, :, :]
    elif attention_mask.dim() == 2:
        # Provided a padding mask of dimensions [batch_size, seq_length]
        # - if the model is a decoder, apply a causal mask in addition to the padding mask
        # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
        # if self.config.is_decoder:
        #     batch_size, seq_length = input_shape
        #     seq_ids = torch.arange(seq_length, device=device)
        #     causal_mask = seq_ids[None, None, :].repeat(batch_size, seq_length, 1) <= seq_ids[None, :, None]
        #     # causal and attention masks must have same type with pytorch version < 1.3
        #     causal_mask = causal_mask.to(attention_mask.dtype)
        #     extended_attention_mask = causal_mask[:, None, :, :] * attention_mask[:, None, None, :]
        # else:
        extended_attention_mask = attention_mask[:, None, None, :]
    else:
        raise ValueError(
            "Wrong shape for input_ids (shape {}) or attention_mask (shape {})".format(
                input_shape, attention_mask.shape
            )
        )

    # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
    # masked positions, this operation will create a tensor which is 0.0 for
    # positions we want to attend and -10000.0 for masked positions.
    # Since we are adding it to the raw scores before the softmax, this is
    # effectively the same as removing these entirely.
    # extended_attention_mask = extended_attention_mask.to(dtype=self.dtype)  # fp16 compatibility
    extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
    return extended_attention_mask

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
        self.accuracy = pl.metrics.Accuracy()
    def on_train_epoch_start(self):
        print("Starting training epoch...")
    def on_train_epoch_end(self, outputs):
        print("Finished training epoch...")

    def training_step(self, batch, batch_idx):

        labels = batch['label'].float()

        query_dict = batch['query_enc_dict']
        response_dict = batch['response_enc_dict']

        query_last_hidden_state, query_pooler_output = self.bert(query_dict['input_ids'], token_type_ids=query_dict['token_type_ids'], attention_mask=query_dict['attention_mask'])
        response_last_hidden_state, response_pooler_output = self.bert(response_dict['input_ids'], token_type_ids=response_dict['token_type_ids'], attention_mask=response_dict['attention_mask'])

        preds = torch.sigmoid(self.cosine_sim(self.query_rescale_layer(query_pooler_output), self.response_rescale_layer(response_pooler_output)))

        loss = mse_loss(preds, labels)

        # Logging to TensorBoard by default
        self.log('train_loss', loss)

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
        self.log('val_loss_step', val_loss)
        self.log('test_acc_step', self.accuracy(preds, labels))
        return val_loss

    def on_validation_epoch_end(self):
        print("Calculating validation accuracy...")
        vacc = self.accuracy.compute()
        print("Validation accuracy:",vacc)
        self.log('val_acc_epoch',vacc)

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

class LitOutputBertModel(pl.LightningModule):

    def __init__(self):
        super().__init__()
        bertmodel = "bert-base-uncased"
        self.bert = BertModel.from_pretrained(
            bertmodel,# Use the 12-layer BERT model, with an uncased vocab.
            output_attentions=False,# Whether the model returns attentions weights.
            output_hidden_states=False,# Whether the model returns all hidden-states.
        )
        self.query_rescale_layer = nn.Linear(768, 768)
        self.knowledge_infusion_layer = nn.Linear(300,768)
        self.compressor = nn.Linear(300,64)
        self.decompressor = nn.Linear(64,300)
        self.mover = nn.Linear(1068,768)
        self.mover2 = nn.Linear(1536,768)
        bert_config = BertConfig.from_pretrained(bertmodel,hidden_size=300,num_hidden_layers=1)

        self.k_att = BertLayer(bert_config)
        untouched_config = BertConfig.from_pretrained(bertmodel)
        self.recontextualizer = BertLayer(untouched_config)
        self.nonlinear = nn.Tanh()
        self.pooler = BertPooler(untouched_config)

        self.cosine_sim = nn.CosineSimilarity(dim=1, eps=1e-6)
        # self.save_hyperparameters()

        self.accuracy = pl.metrics.Accuracy()

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
        self.numberbatch = self.numberbatch.loc[[x for x in self.numberbatch.index if '/c/en/' in x]]
        # self.numberbatch_dict = {}
        # for word in self.numberbatch.index:
        #     if "/c/en/" in word:
        #         self.numberbatch_dict[word]=self.numberbatch.loc[word]
        # print("Configured numberbatch!")
        # print("Sanitychecking cat")
        # print("/c/en/cat",self.numberbatch_dict["/c/en/cat"])

    def on_train_epoch_start(self):
        print("Starting training epoch...")
    def on_train_epoch_end(self, outputs):
        print("Finished training epoch...")

    def knowledge_infusion(self, input_ids,last_hidden_state, attention_mask=None):
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
        representation = torch.cat([last_hidden_state,entity_embeds],dim=2)
        representation = self.mover(representation)
        token_avg = torch.mean(representation[:,1:], 1)
        pool_tok = representation[:,0]
        representation = torch.cat([pool_tok,token_avg],1)
        representation = self.mover2(representation)
        representation = self.nonlinear(representation)

        # entity_embeds = self.compressor(entity_embeds)
        # entity_embeds = self.nonlinear(entity_embeds)
        # entity_embeds = self.decompressor(entity_embeds)
        # entity_embeds = self.nonlinear(entity_embeds)
        # extended_att_maskk = get_extended_attention_mask(attention_mask, entity_embeds.shape, self.device)
        # input_dict = {'hidden_states':entity_embeds,"attention_mask":extended_att_maskk}
        # att_entity_embeds = self.k_att(**input_dict)[0]
        # expanded_entity_embeds = self.knowledge_infusion_layer(att_entity_embeds)
        #
        # # representation = last_hidden_state #+ expanded_entity_embeds
        # representation = torch.cat([last_hidden_state,expanded_entity_embeds],dim=2)
        # representation = self.mover(representation)
        # representation = self.nonlinear(representation)
        # input_dict["hidden_states"] = representation
        # representation = self.recontextualizer(**input_dict)[0]
        # # representation = torch.mean(representation, 1)
        # representation = self.pooler(representation)
        return representation

    def training_step(self, batch, batch_idx):

        labels = batch['label'].float()

        query_dict = batch['query_enc_dict']
        response_dict = batch['response_enc_dict']

        query_last_hidden_state, query_pooler_output = self.bert(query_dict['input_ids'], token_type_ids=query_dict['token_type_ids'], attention_mask=query_dict['attention_mask'])
        query_pooler_output = self.knowledge_infusion(query_dict["input_ids"],query_last_hidden_state, attention_mask=query_dict['attention_mask'])

        response_last_hidden_state, response_pooler_output = self.bert(response_dict['input_ids'], token_type_ids=response_dict['token_type_ids'], attention_mask=response_dict['attention_mask'])
        response_pooler_output = self.knowledge_infusion(response_dict["input_ids"],response_last_hidden_state, attention_mask=response_dict['attention_mask'])

        preds = torch.sigmoid(self.cosine_sim(self.query_rescale_layer(query_pooler_output), self.query_rescale_layer(response_pooler_output)))

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
        query_pooler_output = self.knowledge_infusion(query_dict["input_ids"],query_last_hidden_state, attention_mask=query_dict['attention_mask'])

        response_last_hidden_state, response_pooler_output = self.bert(response_dict['input_ids'],
                                                                       token_type_ids=response_dict['token_type_ids'],
                                                                       attention_mask=response_dict['attention_mask'])
        response_pooler_output = self.knowledge_infusion(response_dict["input_ids"],response_last_hidden_state, attention_mask=response_dict['attention_mask'])

        preds =self.cosine_sim(self.query_rescale_layer(query_pooler_output), self.query_rescale_layer(response_pooler_output)).to(self.device)

        val_loss =  torch.nn.BCEWithLogitsLoss()(preds,labels)
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

        query_last_hidden_state, query_pooler_output = self.bert(query_dict['input_ids'],
                                                                 token_type_ids=query_dict['token_type_ids'],
                                                                 attention_mask=query_dict['attention_mask'])
        query_pooler_output = self.knowledge_infusion(query_dict["input_ids"],query_last_hidden_state, attention_mask=query_dict['attention_mask'])

        response_last_hidden_state, response_pooler_output = self.bert(response_dict['input_ids'],
                                                                       token_type_ids=response_dict['token_type_ids'],
                                                                       attention_mask=response_dict['attention_mask'])
        response_pooler_output = self.knowledge_infusion(response_dict["input_ids"],response_last_hidden_state, attention_mask=response_dict['attention_mask'])

        preds = self.cosine_sim(self.query_rescale_layer(query_pooler_output), self.query_rescale_layer(response_pooler_output)).to(self.device)

        test_loss = torch.nn.BCEWithLogitsLoss()(preds,labels)
        self.log('test_loss', test_loss)
        self.log('test_acc_step', self.accuracy(preds, labels))
        return test_loss

    def on_test_epoch_end(self):
        print("Calculating test accuracy...")
        self.log('test_acc_epoch', self.accuracy.compute())

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
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
