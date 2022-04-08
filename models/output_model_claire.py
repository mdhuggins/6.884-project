import pickle
import sys

import pytorch_lightning as pl
import spacy
import torch
from pandas import Series
from torch import nn as nn
from torch.nn import BCEWithLogitsLoss
from torch.nn.functional import mse_loss
from torch.optim.lr_scheduler import ReduceLROnPlateau, LambdaLR, StepLR
from torch.utils.data.dataloader import default_collate
from torchmetrics import Accuracy
from tqdm import tqdm
from transformers import BertModel, AdamW, get_linear_schedule_with_warmup, BertTokenizer
sys.path.insert(0, 'components/')

from evaluation import Evaluation
from utils import transfer_batch_to_device, get_embeddings, get_phrase_matcher
import numpy as np
def find_sub_list(sl,l):
    results=[]
    sll=len(sl)
    for ind in (i for i,e in enumerate(l) if e==sl[0]):
        if l[ind:ind+sll]==sl:
            results.append((ind,ind+sll-1))

    return results


class LitBertClaireOutputModel(pl.LightningModule):
    @staticmethod
    def col_fn(x):
        return default_collate(x)

    def __init__(self, name, lr, total_steps, concat=None, embedding_model="", embedding_file="", graph_file=""):
        super().__init__()
        self.bert = BertModel.from_pretrained(
            "bert-base-uncased",  # Use the 12-layer BERT model, with an uncased vocab.
            output_attentions=False,  # Whether the model returns attentions weights.
            output_hidden_states=False,  # Whether the model returns all hidden-states.
        )
        self.query_rescale_layer = nn.Linear(768, 768)
        self.knowledge_infuser = nn.Linear(768+300,768)
        self.cosine_sim = nn.CosineSimilarity(dim=1, eps=1e-6)
        # self.save_hyperparameters()
        self.accuracy = Accuracy()
        self.testaccuracy = Accuracy()
        self.val_res = []
        self.eval_res = []
        self.lr = lr
        self.total_steps = total_steps
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.rgcn = torch.load(embedding_model)
        self.graph = pickle.load(open(graph_file,'rb'))
        embedding_map = pickle.load(open(embedding_file,'rb'))
        mp = {}
        self.emb_size = 0
        for vocab in tqdm(self.rgcn.vocabulary):
            mp[vocab] = Series(embedding_map[self.rgcn.vocabulary.index(vocab),:].detach())
            if self.emb_size==0:
                self.emb_size = embedding_map[self.rgcn.vocabulary.index(vocab),:].shape[0]
                print("Emb size set to...",self.emb_size)
        self.embedding_resize = nn.Linear(self.emb_size,300)

        self.embedding_map = mp
        self.nlp = spacy.load('en_core_web_sm')
        self.matcher = get_phrase_matcher(numberbatch=self.embedding_map,nlp=self.nlp)
        self.dropout = nn.Dropout(0.1)

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


    def get_knowledge(self,input_ids):
        sentences = [self.tokenizer.decode(x,skip_special_tokens=True,clean_up_tokenization_spaces=False)for x in input_ids.tolist()]
        k_embeds_list = []
        for sentence_id in range(len(sentences)):
            sentence = sentences[sentence_id]
            bert_ids_sentence = input_ids[sentence_id,:].tolist()
            k_embs = [np.zeros((1,self.emb_size)) for i in range(len(bert_ids_sentence))]
            doc = self.nlp(sentence)
            matches_dict = {}
            for match_id, start, end in self.matcher(doc):
                string_id = self.nlp.vocab.strings[match_id]  # Get string representation
                span = doc[start:end].text # The matched span
                if string_id not in matches_dict.keys():
                    matches_dict[string_id]=-1
                matches_dict[string_id]+=1
                tok_span = self.tokenizer(span,add_special_tokens=False)["input_ids"]
                # tst = self.tokenizer.decode([101,2129,2079])
                results = find_sub_list(tok_span,bert_ids_sentence)
                if len(results) == 0 or len(results)<=matches_dict[string_id]:
                    # print(len(results), matches_dict[string_id])
                    continue
                results = results[matches_dict[string_id]]
                for rep_idx in range(results[0],results[1]+1):
                    k_embs[rep_idx] = np.expand_dims(self.embedding_map[string_id].to_numpy(),axis=0)
            k_embeds_list.append(torch.tensor(k_embs))
        k_embeds = torch.cat(k_embeds_list,dim=1).to(self.bert.embeddings.word_embeddings.weight.device).transpose(0,1)
        k_embeds = self.embedding_resize(k_embeds.float())
        # k_embeds = self.dropout(k_embeds)
        # print("here")
        return k_embeds
    def training_step(self, batch, batch_idx):

        labels = batch['label'].float()

        query_dict = batch['query_enc_dict']
        response_dict = batch['response_enc_dict']
        knowledge_embeds = self.get_knowledge(query_dict['input_ids'])
        out = self.bert(query_dict['input_ids'],
                        token_type_ids=query_dict['token_type_ids'],
                        attention_mask=query_dict['attention_mask'])
        query_last_hidden_state = out.last_hidden_state
        query_last_hidden_state = self.knowledge_infuser(torch.cat([knowledge_embeds,query_last_hidden_state],dim=2).float())
        query_pooler_output = torch.mean(query_last_hidden_state, 1)

        knowledge_embeds = self.get_knowledge(response_dict['input_ids'])
        out = self.bert(response_dict['input_ids'],
                        token_type_ids=response_dict['token_type_ids'],
                        attention_mask=response_dict['attention_mask'])
        response_last_hidden_state = out.last_hidden_state
        response_last_hidden_state = self.knowledge_infuser(torch.cat([knowledge_embeds,response_last_hidden_state],dim=2).float())

        response_pooler_output = torch.mean(response_last_hidden_state, 1)
        # c = self.cosine_sim(self.query_rescale_layer(query_pooler_output), self.response_rescale_layer(response_pooler_output))
        # c = self.cosine_sim(query_pooler_output, response_pooler_output)
        c = self.cosine_sim(self.query_rescale_layer(query_pooler_output),
                            self.query_rescale_layer(response_pooler_output))
        # preds = torch.sigmoid(c)
        preds = c
        loss = mse_loss(preds, labels)
        # loss = BCEWithLogitsLoss()(c,labels)
        # Logging to TensorBoard by default
        self.log('train_loss', loss, on_step=True)

        return loss

    def validation_step(self, batch, batch_idx):
        batch = transfer_batch_to_device(batch, self.device)
        all_preds = []
        val_loss = 0
        query_dict = batch['query_enc_dict']
        knowledge_embeds = self.get_knowledge(query_dict['input_ids'])
        out = self.bert(query_dict['input_ids'],
                        token_type_ids=query_dict['token_type_ids'],
                        attention_mask=query_dict['attention_mask'])
        query_last_hidden_state = out.last_hidden_state
        query_last_hidden_state = self.knowledge_infuser(
            torch.cat([knowledge_embeds, query_last_hidden_state], dim=2).float())
        query_pooler_output = torch.mean(query_last_hidden_state, 1)

        for i in range(len(batch['label'])):
            labels = batch['label'][i].float()

            response_dict = batch['response_enc_dict'][i]

            knowledge_embeds = self.get_knowledge(response_dict['input_ids'])
            out = self.bert(response_dict['input_ids'],
                            token_type_ids=response_dict['token_type_ids'],
                            attention_mask=response_dict['attention_mask'])
            response_last_hidden_state = out.last_hidden_state
            response_last_hidden_state = self.knowledge_infuser(
                torch.cat([knowledge_embeds, response_last_hidden_state], dim=2).float())

            response_pooler_output = torch.mean(response_last_hidden_state, 1)


            preds = self.cosine_sim(self.query_rescale_layer(query_pooler_output),
                                    self.query_rescale_layer(response_pooler_output))

            # preds = self.cosine_sim(self.query_rescale_layer(query_pooler_output), self.response_rescale_layer(response_pooler_output)).to(self.device)
            # preds = self.cosine_sim(query_pooler_output, response_pooler_output)#.to(self.device)
            # preds = torch.sigmoid(preds)
            preds = preds.to('cpu')
            labels = labels.to('cpu')
            val_loss += mse_loss(preds, labels)
            all_preds.append(torch.unsqueeze(preds, 1))

            # val_loss =  torch.nn.BCEWithLogitsLoss()(preds,labels)
            self.log('val_acc_step', self.accuracy(torch.clamp(preds, 0, 1), labels.int()))

        self.log('val_loss_step', val_loss / len(batch['label']))
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
        print("Validation accuracy:", vacc),
        print(MAP, MRR, P1, P5)
        self.log('val_acc_epoch', vacc)
        self.log('v_MAP', MAP)
        self.log('v_Mrr', MRR)
        self.log('v_p1', P1)
        self.log('v_p5', P5)
        return vacc, MAP, MRR, P1, P5

    def test_step(self, batch, batch_idx):
        batch = transfer_batch_to_device(batch, self.device)
        all_preds = []
        val_loss = 0
        query_dict = batch['query_enc_dict']
        knowledge_embeds = self.get_knowledge(query_dict['input_ids'])
        out = self.bert(query_dict['input_ids'],
                        token_type_ids=query_dict['token_type_ids'],
                        attention_mask=query_dict['attention_mask'])
        query_last_hidden_state = out.last_hidden_state
        query_last_hidden_state = self.knowledge_infuser(
            torch.cat([knowledge_embeds, query_last_hidden_state], dim=2).float())
        query_pooler_output = torch.mean(query_last_hidden_state, 1)

        for i in range(len(batch['label'])):
            labels = batch['label'][i].float()

            response_dict = batch['response_enc_dict'][i]

            knowledge_embeds = self.get_knowledge(response_dict['input_ids'])
            out = self.bert(response_dict['input_ids'],
                            token_type_ids=response_dict['token_type_ids'],
                            attention_mask=response_dict['attention_mask'])
            response_last_hidden_state = out.last_hidden_state
            response_last_hidden_state = self.knowledge_infuser(
                torch.cat([knowledge_embeds, response_last_hidden_state], dim=2).float())

            response_pooler_output = torch.mean(response_last_hidden_state, 1)

            preds = self.cosine_sim(self.query_rescale_layer(query_pooler_output),
                                    self.query_rescale_layer(response_pooler_output))

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
        return vacc, MAP, MRR, P1, P5
