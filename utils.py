import os
import re
from typing import Dict, List

import pytorch_lightning as pl
import spacy
import torch
from spacy.matcher import PhraseMatcher
from tqdm import tqdm
from transformers import BertTokenizerFast
import numpy as np

class CheckpointEveryNSteps(pl.Callback):
    """
    Save a checkpoint every N steps, instead of Lightning's default that checkpoints
    based on validation loss.
    """

    def __init__(
            self,
            save_step_frequency,
            prefix="N-Step-Checkpoint",
            use_modelcheckpoint_filename=False,
    ):
        """
        Args:
            save_step_frequency: how often to save in steps
            prefix: add a prefix to the name, only used if
                use_modelcheckpoint_filename=False
            use_modelcheckpoint_filename: just use the ModelCheckpoint callback's
                default filename, don't use ours.
        """
        self.save_step_frequency = save_step_frequency
        self.prefix = prefix
        self.use_modelcheckpoint_filename = use_modelcheckpoint_filename

    def on_batch_end(self, trainer: pl.Trainer, _):
        """ Check if we should save a checkpoint after every train batch """
        epoch = trainer.current_epoch
        global_step = trainer.global_step
        model_name = trainer.model.name
        if global_step % self.save_step_frequency == 0:
            if self.use_modelcheckpoint_filename:
                filename = trainer.checkpoint_callback.filename
            else:
                filename = f"{self.prefix}_{epoch}_{global_step}_{model_name}.ckpt"
            ckpt_path = os.path.join(trainer.checkpoint_callback.dirpath, filename)
            trainer.save_checkpoint(ckpt_path)


def get_extended_attention_mask(attention_mask, input_shape, device):
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
        if isinstance(batch[k], Dict):
            batch[k] = transfer_batch_to_device(batch[k], device)
        elif isinstance(batch[k], List):
            # print(batch[k])
            continue
        else:
            batch[k] = batch[k].to(device)
    return batch


import pandas as pd


def load_vectors_pandas(path, cache="nb.h5",clean_names = False):
    numberbatch = None
    if os.path.exists(cache):
        print("Using cached!")
        numberbatch = pd.read_hdf(cache, "mat")
    else:
        names = []
        vecs = []
        with open(path) as nb:
            for line in tqdm(nb):
                line = line.strip()
                if len(line.split()) == 2:
                    continue
                name = line.split()[0]
                if clean_names:
                    name = name.lower().replace("_"," ")
                d = pd.Series([float(x) for x in line.split()[1:]])
                vecs.append(d)
                names.append(str(name))
        numberbatch = pd.DataFrame(data=vecs, index=names)
        numberbatch.to_hdf(cache, "mat")
    return numberbatch
import dask.dataframe as dd
def get_embeddings(path):
    numberbatch = load_vectors_pandas(path, "wiki.h5", clean_names=True)
    numberbatch.index = numberbatch.index.map(str)
    ddf = dd.from_pandas(numberbatch, npartitions=20)
    numberbatch = ddf
    return numberbatch
def get_phrase_matcher(numberbatch,nlp):
    phraseMatcher = PhraseMatcher(nlp.vocab, attr='LOWER')
    print("Creating matcher")
    terms = map(str, numberbatch.index)
    # terms = [str(x) for x in self.numberbatch.index]
    patterns = [nlp.make_doc(text) for text in terms]

    phraseMatcher.add("Match_By_Phrase", None, *patterns)
    return phraseMatcher

def get_retro_embeds(input_ids):
    global initialized
    if not initialized :
        print("Initializing in thread...")
        initialize_retro()
        print("ready.")

    stacked_sentences = []
    prefix = ""
    stacked_retroembeds = []
    for inpt_id_idx, sentence in enumerate(input_ids):
        dec_sentence = tokenizer.decode(sentence.cpu().numpy())
        doc = nlp(dec_sentence)
        matches = phraseMatcher(doc)

        # cg_res = self.cg.get_mentions_raw_text(dec_sentence)
        # d = {
        #     "contextual_embeddings":last_hidden_state,'tokens_mask':attention_mask,
        #     "tokenized_text":cg_res["tokenized_text"], 'candidate_spans':cg_res['candidate_spans'],
        #     'candidate_entities':cg_res["candidate_entities"],"candidate_entity_priors":cg_res['candidate_entity_priors'],
        #     'candidate_segment_ids':None
        # }
        # tst = self.el(**d)
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
        words_2 = []
        for match_id, start, end in matches:
            span = doc[start:end]
            words_2.append(span.text)

        retroembeds = []
        for word in words:
            try:
                vec = numberbatch.loc[prefix + word]
                to_append = np.array(vec).reshape(300, )
            except:
                to_append = np.zeros((300,))
            retroembeds.append(to_append)

        retroembeds2 = []
        for word in words_2:
            if len(word.split(' ')) > 1:
                try:
                    vec = numberbatch.loc[prefix + word]
                    to_append = np.array(vec).reshape(300, )
                except:
                    to_append = np.zeros((300,))
            else:
                to_append = np.zeros((300,))
            retroembeds2.append(to_append)
        # retroembeds = retroembeddings.get_embeddings_from_input_ids(words).contiguous()
        # retroembeds  = retro_vecs[sample]
        stacked_retroembeds.append(retroembeds)
        replacement_list = []
        for word in words:
            toks = tokenizer.encode(word, add_special_tokens=False)
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
        counts = {}
        replacement_list = []
        for word in words_2:
            toks = tokenizer.encode(word, add_special_tokens=False)
            locs = find_sub_list(toks, [int(x) for x in sentence.cpu().numpy()])
            if locs is None:
                continue
            replacement_list.append(locs)
        for idx, id in enumerate(sentence):
            # Id iN SPECIAL TOKENS
            for rep_idx, rep_tup in enumerate(replacement_list):
                if idx >= rep_tup[0] and idx <= rep_tup[1]:
                    final_list[idx] = (final_list[idx] + retroembeds2[rep_idx]) / 2.0
                    break

        stacked_sentences.append(final_list)
    retro_embeds = torch.tensor(stacked_sentences).float()
    return retro_embeds
tokenizer = numberbatch = nlp = phraseMatcher = None
initialized = False
def initialize_retro():
    global tokenizer
    global numberbatch
    global nlp
    global phraseMatcher
    global initialized
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased')
    numberbatch = get_embeddings("models/graphembeddings/entities_glove_format")
    nlp = spacy.load('en_core_web_sm')
    phraseMatcher = get_phrase_matcher(numberbatch, nlp)
    initialized = True