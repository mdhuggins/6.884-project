import os
import pickle
import re
from typing import Dict, List

import h5py
import pytorch_lightning as pl
import spacy
import torch
from spacy.matcher import PhraseMatcher
from tqdm import tqdm
from transformers import BertTokenizerFast
import numpy as np
from filelock import Timeout, FileLock

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
def load_vectors_wordnet(path,key="tucker_gensen"):
    ents = []
    with open("models/graphembeddings/wordnet_synsets_mask_null_vocab.txt") as f:
        for line in f:
            ents.append(line.strip())
    with h5py.File(path, 'r') as fin:
        emb = fin[key][...]
    return emb

# load_vectors_wordnet('models/graphembeddings/wordnet_synsets_mask_null_vocab_embeddings_tucker_gensen.hdf5')

def load_vectors_pandas(path, cache="nb.h5",clean_names = False):
    numberbatch = None
    if os.path.exists(cache):
        print("Using cached!")
        # numberbatch = pd.read_hdf(cache, "mat")
        numberbatch = pickle.load(open(cache,"rb"))
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
                    name = name.replace("_"," ")
                d = pd.Series([float(x) for x in line.split()[1:]])
                vecs.append(d)
                names.append(str(name))
        # numberbatch = pd.DataFrame(data=vecs, index=names)
        # numberbatch.to_hdf(cache, "mat")
        numberbatch = dict([x for x in zip(names,vecs)])
        lock = FileLock(cache+".lock")
        with lock:
            # open("high_ground.txt", "a").write("You were the chosen one.")
            pickle.dump(numberbatch,open(cache,"wb"))
    return numberbatch
def get_embeddings(path,cache="wiki.h5"):
    numberbatch = load_vectors_pandas(path, cache, clean_names=True)
    vecs = numberbatch
    # numberbatch.index = numberbatch.index.map(str)
    # vecs = {}
    # for i in numberbatch.index:
    #     vecs[i] = numberbatch.loc[i]
    # ddf = dd.from_pandas(numberbatch, npartitions=20)
    # numberbatch = ddf
    return vecs
def get_phrase_matcher(numberbatch,nlp,cache=False):
    print("Creating matcher")
    if os.path.exists("matcher") and cache:
        print("Loading cache...")
        lock = FileLock("matcher.lock")
        with lock:
            # open("high_ground.txt", "a").write("You were the chosen one.")
            phraseMatcher = pickle.load(open("matcher.bin","rb"))

    else:
        print("Not loading cache...")
        phraseMatcher = PhraseMatcher(nlp.vocab, attr='LOWER')
        terms = numberbatch.keys() if isinstance(numberbatch,dict) else  map(str, numberbatch.index)
        # terms = [str(x) for x in self.numberbatch.index]
        for text in terms:
            patterns = [nlp.make_doc(text)]
            phraseMatcher.add(text, None, *patterns)
        lock = FileLock("matcher.lock")
        with lock:
            # open("high_ground.txt", "a").write("You were the chosen one.")
            pickle.dump(phraseMatcher,open("matcher.bin","wb"))

    return phraseMatcher

def get_retro_embeds(input_ids,path_to_embs="models/graphembeddings/numberbatch-en-19.08.txt"):
    global initialized
    if not initialized :
        print("Initializing in thread...")
        initialize_retro(path_to_embs)
        print("ready.")

    stacked_sentences = []
    prefix = ""
    stacked_retroembeds = []
    for inpt_id_idx, sentence in enumerate(input_ids):
        dec_sentence = tokenizer.decode(sentence.cpu().numpy())
        doc = nlp.tokenizer(dec_sentence)
        matches = phraseMatcher(doc)


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

        # words = re.findall(r'\w+', dec_sentence)
        words_2 = []
        for match_id, start, end in matches:
            span = doc[start:end]
            words_2.append(span.text)
        # print("NB MATCHES:",len(matches))
        retroembeds_2 = []
        for word in words_2:
            try:
                vec = numberbatch[word]#.compute()
                to_append = np.array(vec).reshape(300, )
            except:
                to_append = np.zeros((300,))
            retroembeds_2.append(to_append)

        # res = numberbatch.loc[numberbatch.index.isin(words_2)].compute()
        # res = numberbatch.loc[numberbatch.index.isin(words_2)].compute()
        counts = {}
        replacement_list = []
        for word in words_2:
            toks = tokenizer.encode(word, add_special_tokens=False)
            locs = find_sub_list(toks, [int(x) for x in sentence.cpu().numpy()])
            if locs is None:
                continue
            replacement_list.append(locs)
        final_list = [np.zeros((300,)) for x in sentence]
        for idx, id in enumerate(sentence):
            # Id iN SPECIAL TOKENS
            for rep_idx, rep_tup in enumerate(replacement_list):
                if idx >= rep_tup[0] and idx <= rep_tup[1]:
                    if np.sum(final_list[idx]) == 0:
                        final_list[idx] = retroembeds_2[rep_idx]  # ) / 2.0
                    else:
                        final_list[idx] *= retroembeds_2[rep_idx]  #

        stacked_sentences.append(final_list)
    retro_embeds = torch.tensor(stacked_sentences).float()
    return retro_embeds

tokenizer = numberbatch = nlp = phraseMatcher = None
initialized = False
def initialize_retro(path_to_embeddings,
                     cache="wiki.h5"):
    global tokenizer
    global numberbatch
    global nlp
    global phraseMatcher
    global initialized
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased')
    # numberbatch = get_embeddings("/Users/pedro/Documents/Documents - Pedro’s MacBook Pro/"
    #                              "git/6.884-project/models/numberbatch-en-19.08.txt")
    numberbatch = get_embeddings(path_to_embeddings,cache)
    nlp = spacy.load('en_core_web_sm')
    phraseMatcher = get_phrase_matcher(numberbatch, nlp)
    initialized = True