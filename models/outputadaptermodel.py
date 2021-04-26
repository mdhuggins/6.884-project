import collections

import numpy as np
import pytorch_lightning as pl
import torch
from torch import nn as nn
from torch.nn import CrossEntropyLoss
from torch.nn.functional import mse_loss
from torch.utils.data.dataloader import default_collate
from transformers import BertTokenizerFast, AdamW, \
    get_linear_schedule_with_warmup
from transformers.modeling_outputs import MaskedLMOutput

from evaluation import Evaluation
from models.configuration_bert_adapter import BertConfig
from models.modeling_bert_kadapter import BertOnlyMLMHead, BertModel
from utils import transfer_batch_to_device, get_retro_embeds





class LitOutputAdapterBertModel(pl.LightningModule):
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
                 total_steps=100,concat=False,sum=False,injection_indices=[],ignore_injection=False,use_summation=True):
        super().__init__()
        self.name = name
        self.lr = lr
        self.total_steps = total_steps
        bertmodel = pretrained_name
        if tokenizer is None:
            tokenizer = BertTokenizerFast.from_pretrained(bertmodel)
        if config is None:
            config = BertConfig.from_pretrained(bertmodel)
        config.ignore_injection = ignore_injection
        config.adapter_inject_layers = [x in injection_indices for x in range(12)]
        config.adapter_infusion_amount = ["nb"]
        config.adapter_use_summation =use_summation
        self.config = config

        self.training_bert = BertOnlyMLMHead(config)
        self.bert = BertModel.from_pretrained(bertmodel, add_pooling_layer=False,config=config)

        self.query_rescale_layer = nn.Linear(768, 768)
        self.concat_dim = 64
        self.compressor = nn.Linear(300, self.concat_dim)
        self.decompressor = nn.Linear(300, 768)
        # gain = 0.0001
        # s = torch.nn.init.uniform_(self.decompressor.weight) * gain
        # self.decompressor.weight = nn.parameter.Parameter(s)
        # self.decompressor.bias.data.fill_(gain)
        self.concat_rescale_layer = nn.Linear(768+self.concat_dim, 768)
        self.dropout = nn.Dropout(0.3)
        self.cosine_sim = nn.CosineSimilarity(dim=1)
        # self.save_hyperparameters()

        self.accuracy = pl.metrics.Accuracy()
        self.testaccuracy = pl.metrics.Accuracy()
        self.val_res = []
        self.eval_res = []
        self.tokenizer = tokenizer
        self.concat = concat
        self.sum = not self.concat and sum
        self.freeze_encoder_ = True
        self.unfreeze_epoch = 200 # zero indexed
        if self.freeze_encoder_:
            self.freeze_encoder()

    def knowledge_infusion(self, input_ids, last_hidden_state, attention_mask, retro_embeds):
        original_retro_embeds = retro_embeds
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
        for name,parameter in self.bert.named_parameters(recurse=True):
            if "retro" in name or "Norm" in name or "cls" in name:
                parameter.requires_grad = True
            else:
                parameter.requires_grad =False
            print(name,parameter.requires_grad)
        # self.bert.train()

    # def unfreeze_encoder(self):
    #     print("Unfreezing the encoder.")
    #     for parameter in self.bert.parameters(recurse=True):
    #         parameter.requires_grad = True
    #     self.bert.train(True)

    # def on_train_batch_start(self, batch, batch_idx,dataloader_idx):
    #     if batch_idx == self.unfreeze_epoch:
    #         self.unfreeze_encoder()
    #         opt = self.optimizers()
    #         opt.state = collections.defaultdict(dict)

    def training_step(self, batch, batch_idx):
        labels = batch['label'].float()
        query_dict = batch['query_enc_dict']
        response_dict = batch['response_enc_dict']


        r = self.bert(query_dict['input_ids'], token_type_ids=query_dict['token_type_ids'],
                      attention_mask=query_dict['attention_mask'],retro_vecs=batch["q_retro"])
        query_last_hidden_state = r.last_hidden_state
        query_pooler_output = r.pooler_output
        query_pre_pooler_output, query_retroembeds, query_pooler_output = self.knowledge_infusion(
            query_dict["input_ids"],
            query_last_hidden_state,
            attention_mask=query_dict['attention_mask'],
            retro_embeds=batch["q_retro"])
        r = self.bert(response_dict['input_ids'], token_type_ids=response_dict['token_type_ids'],
                      attention_mask=response_dict['attention_mask'],retro_vecs=batch["r_retro"])
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
                      attention_mask=query_dict['attention_mask'],retro_vecs=batch["q_retro"])
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
                              'attention_mask'],retro_vecs=batch["r_retro"][i])
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
                      attention_mask=query_dict['attention_mask'],retro_vecs=batch["q_retro"])
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
                              'attention_mask'],
                          retro_vecs=batch["r_retro"][i])
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
        my_list = ['1.retro_adapter_downproject.weight','1.retro_adapter_upproject.weight', '1.retro_adapter_downproject.bias','1.retro_adapter_upproject.bias',
                   'retro_adapter_remixer','retro_adapter_remixer2']

        params = []
        base_params = []
        for name,param in self.bert.named_parameters(recurse=True):
            added =False
            for target in my_list:
                if target in name:
                    added = True
                    params.append(param)
                    break
            if not added:
                base_params.append(param)



        optimizer = AdamW([{'params': base_params}, {'params': params, 'lr': self.lr*10}], lr=self.lr)
        scheduler = get_linear_schedule_with_warmup(optimizer, int(0.1 * self.total_steps), self.total_steps)
        # scheduler = StepLR(optimizer, step_size=25, gamma=0.8)
        scheduler = {
            'scheduler': scheduler,
            'interval': 'step',  # or 'epoch'
            'frequency': 1
        }
        return [optimizer], [scheduler]
