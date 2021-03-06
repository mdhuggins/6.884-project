import collections

import numpy as np
import pytorch_lightning as pl
import torch
from torch import nn as nn
from torch.nn import CrossEntropyLoss
from torch.nn.functional import mse_loss
from torch.utils.data.dataloader import default_collate
from transformers import BertTokenizerFast, BertModel, BertConfig, AdamW, \
    get_linear_schedule_with_warmup
from transformers.modeling_outputs import MaskedLMOutput
from transformers.models.bert.modeling_bert import BertOnlyMLMHead

from evaluation import Evaluation
from utils import transfer_batch_to_device, get_retro_embeds


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
    @staticmethod
    def col_fn(x):
        res = default_collate(x)
        query_dict = res['query_enc_dict']
        q_retro = get_retro_embeds(query_dict["input_ids"])
        response_dict = res['response_enc_dict']
        if isinstance(response_dict, list):
            r_retro = []
            for i in range(len(response_dict)):
                r_retro.append(get_retro_embeds(response_dict[i]["input_ids"]))
        else:
            r_retro = get_retro_embeds(response_dict["input_ids"])
        res['q_retro'] = q_retro
        res['r_retro'] = r_retro
        return res

    def __init__(self, name, config=None, pretrained_name="bert-base-uncased", tokenizer=None, lr=2e-5,
                 total_steps=100,concat=False):
        super().__init__()
        self.name = name
        self.lr = lr
        self.total_steps = total_steps
        bertmodel = pretrained_name
        if tokenizer is None:
            tokenizer = BertTokenizerFast.from_pretrained(bertmodel)
        if config is None:
            config = BertConfig.from_pretrained(bertmodel)
        self.config = config
        self.bert = BertModel.from_pretrained(bertmodel, add_pooling_layer=False)

        self.query_rescale_layer = nn.Linear(768, 768)
        self.concat_dim = 64
        self.compressor = nn.Linear(300, self.concat_dim)
        self.decompressor = nn.Linear(300, 768)
        gain = 0.0001
        s = torch.nn.init.uniform_(self.decompressor.weight) * gain
        self.decompressor.weight = nn.parameter.Parameter(s)
        self.decompressor.bias.data.fill_(gain)
        self.concat_rescale_layer = nn.Linear(768+self.concat_dim, 768)
        self.dropout = nn.Dropout(0.3)
        self.cosine_sim = nn.CosineSimilarity(dim=1)
        self.nonlinear = nn.ReLU()
        # self.save_hyperparameters()

        self.accuracy = pl.metrics.Accuracy()
        self.testaccuracy = pl.metrics.Accuracy()
        self.val_res = []
        self.eval_res = []
        self.tokenizer = tokenizer
        self.concat = concat
        self.sum = not self.concat
        self.freeze_encoder_ = True
        self.unfreeze_epoch = 200 # zero indexed
        # if self.freeze_encoder_:
        #     self.freeze_encoder()
        self.additional_down = nn.Linear(300,100)
        self.mixer = nn.Linear(100,100)
        self.additional_up = nn.Linear(100,300)

    def knowledge_infusion(self, input_ids, last_hidden_state, attention_mask, retro_embeds):
        original_retro_embeds = retro_embeds
        retro_embeds = self.additional_down(retro_embeds)
        retro_embeds = self.nonlinear(retro_embeds)
        retro_embeds = self.mixer(retro_embeds)
        retro_embeds = self.nonlinear(retro_embeds)
        retro_embeds = self.additional_up(retro_embeds)
        retro_embeds = self.nonlinear(retro_embeds)
        if self.concat:
            retro_embeds = self.compressor(retro_embeds)
            concat = torch.cat((retro_embeds, last_hidden_state), 2)
            adapter_down = self.concat_rescale_layer(concat)
        elif self.sum:
            retro_embeds = self.decompressor(retro_embeds)
            # print(retro_embeds)
            # print(last_hidden_state)
            adapter_down = retro_embeds + last_hidden_state
        else:
            adapter_down = last_hidden_state
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
        # pre_pool = adapter_down
        mean_retro_embeds = torch.mean(adapter_down, 1)
        return adapter_down, original_retro_embeds, mean_retro_embeds

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

    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                encoder_hidden_states=None,
                encoder_attention_mask=None,
                labels=None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=None,
                retro_vecs=None
                ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        r = self.bert(input_ids,
                      attention_mask=attention_mask,
                      token_type_ids=token_type_ids,
                      position_ids=position_ids,
                      head_mask=head_mask,
                      inputs_embeds=inputs_embeds,
                      encoder_hidden_states=encoder_hidden_states,
                      encoder_attention_mask=encoder_attention_mask,
                      output_attentions=output_attentions,
                      output_hidden_states=output_hidden_states,
                      return_dict=return_dict,
                      retro_vecs=retro_vecs)
        for name,param in self.bert.named_parameters(recurse=True):
            if "retro" in name or "norm" in name:
                param.requires_grad = True
                print(name,"requires grad:",True)
            else:
                param.requires_grad = False
                print(name,"requires grad:",False)

        query_last_hidden_state = r.last_hidden_state
        query_pooler_output, query_retroembeds, query_pre_pooler_output = self.knowledge_infusion(input_ids,
                                                                                                  query_last_hidden_state,
                                                                                                  attention_mask=attention_mask,
                                                                                                  retro_embeds=retro_vecs)
        r.last_hidden_state = query_pooler_output
        outputs = r
        sequence_output = outputs[0]
        prediction_scores = self.training_bert(sequence_output)

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()  # -100 index = padding token
            y_pred = prediction_scores.view(-1, self.config.vocab_size)
            y = labels.view(-1)
            masked_lm_loss = loss_fct(y_pred, y)

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    def freeze_encoder(self):
        print("Freezing the encoder.")
        for parameter in self.bert.parameters(recurse=True):
            parameter.requires_grad = False
        self.bert.train(False)

    def unfreeze_encoder(self):
        print("Unfreezing the encoder.")
        for parameter in self.bert.parameters(recurse=True):
            parameter.requires_grad = True
        self.bert.train(True)

    def on_train_batch_start(self, batch, batch_idx,dataloader_idx):
        # if batch_idx == self.unfreeze_epoch:
        #     self.unfreeze_encoder()
        #     opt = self.optimizers()
        #     opt.state = collections.defaultdict(dict)
        pass
    def training_step(self, batch, batch_idx):
        labels = batch['label'].float()
        query_dict = batch['query_enc_dict']
        response_dict = batch['response_enc_dict']


        r = self.bert(query_dict['input_ids'], token_type_ids=query_dict['token_type_ids'],
                      attention_mask=query_dict['attention_mask'])
        query_last_hidden_state = r.last_hidden_state
        query_pooler_output = r.pooler_output
        query_pre_pooler_output, query_retroembeds, query_pooler_output = self.knowledge_infusion(
            query_dict["input_ids"],
            query_last_hidden_state,
            attention_mask=query_dict['attention_mask'],
            retro_embeds=batch["q_retro"])
        r = self.bert(response_dict['input_ids'], token_type_ids=response_dict['token_type_ids'],
                      attention_mask=response_dict['attention_mask'])
        response_last_hidden_state = r.last_hidden_state
        response_pooler_output = r.pooler_output
        response_pre_pooler_output, response_retroembeds, response_pooler_output = self.knowledge_infusion(
            response_dict["input_ids"],
            response_last_hidden_state,
            attention_mask=response_dict['attention_mask'],
            retro_embeds=batch["r_retro"])
        a = self.query_rescale_layer(query_pooler_output)
        b = self.query_rescale_layer(response_pooler_output)
        c = self.cosine_sim(a, b)
        loss = mse_loss(c, labels)
        self.log('train_loss', loss)
        return loss  # loss,mm_loss, base_train_loss,minimize

    def validation_step(self, batch, batch_idx):
        batch = transfer_batch_to_device(batch, self.device)
        all_preds = []
        val_loss = 0
        query_dict = batch['query_enc_dict']
        r = self.bert(query_dict['input_ids'], token_type_ids=query_dict['token_type_ids'],
                      attention_mask=query_dict['attention_mask'])
        query_last_hidden_state = r.last_hidden_state
        query_pooler_output = r.pooler_output
        query_pre_pooler_output, query_retroembeds, query_pooler_output = self.knowledge_infusion(
            query_dict["input_ids"], query_last_hidden_state,
            attention_mask=query_dict['attention_mask'], retro_embeds=batch["q_retro"])

        for i in range(len(batch['label'])):
            labels = batch['label'][i].float()

            response_dict = batch['response_enc_dict'][i]

            r = self.bert(response_dict['input_ids'],
                          token_type_ids=response_dict[
                              'token_type_ids'],
                          attention_mask=response_dict[
                              'attention_mask'])
            response_last_hidden_state = r.last_hidden_state
            response_pooler_output = r.pooler_output
            response_pre_pooler_output, response_retroembeds, response_pooler_output = self.knowledge_infusion(
                response_dict["input_ids"], response_last_hidden_state, attention_mask=response_dict['attention_mask'],
                retro_embeds=batch["r_retro"][i])

            preds = self.cosine_sim(self.query_rescale_layer(query_pooler_output),
                                    self.query_rescale_layer(response_pooler_output))
            # print(preds)
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
        r = self.bert(query_dict['input_ids'],
                      token_type_ids=query_dict['token_type_ids'],
                      attention_mask=query_dict['attention_mask'])
        query_last_hidden_state = r.last_hidden_state
        query_pre_pooler_output, query_retroembeds, query_pooler_output = self.knowledge_infusion(
            query_dict["input_ids"], query_last_hidden_state,
            attention_mask=query_dict['attention_mask'], retro_embeds=batch["q_retro"])

        for i in range(len(batch['label'])):
            labels = batch['label'][i].float()

            response_dict = batch['response_enc_dict'][i]

            r = self.bert(response_dict['input_ids'],
                          token_type_ids=response_dict[
                              'token_type_ids'],
                          attention_mask=response_dict[
                              'attention_mask'])
            response_last_hidden_state = r.last_hidden_state
            response_pre_pooler_output, response_retroembeds, response_pooler_output = self.knowledge_infusion(
                response_dict["input_ids"], response_last_hidden_state, attention_mask=response_dict['attention_mask'],
                retro_embeds=batch["r_retro"][i])
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
