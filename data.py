import os
import pickle
import random

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast
from tqdm import tqdm

tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased', do_lower_case=True)


class AskUbuntuTrainDataset(Dataset):

    def __init__(self, root_dir="./data/askubuntu-master", neg_pos_ratio=20, use_bert_tokenizer=True, toy_n=None, toy_pad=None, cache_dir=None):
        self.use_bert_tokenizer = use_bert_tokenizer
        self.root_dir = root_dir
        self.neg_pos_ratio = neg_pos_ratio
        self.pad_len = 0
        if cache_dir is not None:
            train_cache_file = os.path.join(cache_dir, "training_data.pkl")

        print("Loading training data...")

        # Load token sequences
        with open(f"{self.root_dir}/text_tokenized.txt", "r") as f:
            lines = f.readlines()

        # id (int) -> (title ([str, ...]), body ([str, ...]))
        self.id_to_tokens = dict([
            (int(line.split("\t")[0]),
             (line.split("\t")[1].split(), line.split("\t")[2].split()))
            for line in lines])

        # Load train
        with open(f"{self.root_dir}/train_random.txt", "r") as f:
            lines = f.readlines()

        # id (int) -> (pos ids ([int, ...]), neg ids ([int, ...]))
        train_dict = dict([
            (int(line.split("\t")[0]),
             ([int(p) for p in line.split("\t")[1].split()], [int(n) for n in line.split("\t")[2].split()]))
            for line in lines])

        # Convert into triples
        triples = []  # Each triple is (query id, candidate id, label (1 for correct, else 0))

        if toy_n:
                print(f"TOY DATASET: Only training on {toy_n} queries")

        for query_id, cands in list(train_dict.items())[:toy_n] if toy_n else train_dict.items():
            pos, all_neg = cands

            neg = random.sample(all_neg, min(100, self.neg_pos_ratio*len(pos)))

            for pos_id in pos:
                triples.append((query_id, pos_id, 1))

            for neg_id in neg:
                triples.append((query_id, neg_id, 0))

        # Set padding length to 95 percentile length
        lens = []
        for idx in range(len(triples)):
            query_id, response_id, label = triples[idx]

            query = self.combine_title_body(query_id)
            response = self.combine_title_body(response_id)

            lens.append(len(query.split()))
            lens.append(len(response.split()))

        self.pad_len = int(np.percentile(lens, 95)) if not toy_pad else toy_pad

        if toy_pad:
            print(f"TOY DATASET: Only padding to {toy_pad} tokens")

        # Embed/pad/etc.
        self.examples = []
        if cache_dir is not None:
            print("Checking if cache...")
            if os.path.exists(train_cache_file):
                print("Cache found",train_cache_file,"loading it...")
                self.examples = pickle.load(open(train_cache_file,'rb'))
                return

        print("Tokenizing training set with BERT tokenizer...")
        for idx in tqdm(range(len(triples))):
            query_id, response_id, label = triples[idx]

            query = self.combine_title_body(query_id)
            response = self.combine_title_body(response_id)

            if self.use_bert_tokenizer:
                query_enc_dict = tokenizer.encode_plus(
                    query,  # Sentence to encode.
                    add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
                    max_length=self.pad_len,  # Pad & truncate all sentences.
                    pad_to_max_length=True,
                    return_attention_mask=True,  # Construct attn. masks.
                    return_tensors='pt',  # Return pytorch tensors.
                )

                response_enc_dict = tokenizer.encode_plus(
                    response,  # Sentence to encode.
                    add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
                    max_length=self.pad_len,  # Pad & truncate all sentences.
                    pad_to_max_length=True,
                    return_attention_mask=True,  # Construct attn. masks.
                    return_tensors='pt',  # Return pytorch tensors.
                )
            else:
                raise NotImplementedError()

            sample = {
                "query_enc_dict": dict([(k, torch.squeeze(v)) for k, v in query_enc_dict.items()]),
                "response_enc_dict": dict([(k, torch.squeeze(v)) for k, v in response_enc_dict.items()]),
                "label": label
            }

            self.examples.append(sample)
        if train_cache_file is not None:
            pickle.dump(self.examples,open(train_cache_file,"wb"))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return self.examples[idx]

    def combine_title_body(self, query_id):
        query_title_tokens, query_body_tokens = self.id_to_tokens[query_id]

        # TODO Better combination method?
        return " ".join(query_title_tokens) + "\t" + " ".join(query_body_tokens)


class AskUbuntuDevDataset(Dataset):

    def __init__(self, root_dir="./data/askubuntu-master", neg_pos_ratio=20,cache_dir=None):
        self.root_dir = root_dir
        self.neg_pos_ratio = neg_pos_ratio
        self.pad_len = 128
        if cache_dir is not None:
            val_cache_file = os.path.join(cache_dir, "validation_data.pkl")

        # Load token sequences
        with open(f"{self.root_dir}/text_tokenized.txt", "r") as f:
            lines = f.readlines()

        # id (int) -> (title ([str, ...]), body ([str, ...]))
        self.id_to_tokens = dict([
            (int(line.split("\t")[0]),
             (line.split("\t")[1].split(), line.split("\t")[2].split()))
            for line in lines])

        # Load dev
        with open(f"{self.root_dir}/dev.txt", "r") as f:
            lines = f.readlines()

        # Each element is query id, positive ids, candidate ids, similarity scores
        self.tuples = [
            (int(line.split("\t")[0]),
             [int(p) for p in line.split("\t")[1].split()],
             [int(n) for n in line.split("\t")[2].split()],
             [float(n) for n in line.split("\t")[3].split()])
            for line in lines]

        self.examples = []
        if cache_dir is not None:
            print("Checking if cache...")
            if os.path.exists(val_cache_file):
                print("Cache found",val_cache_file,"loading it...")
                self.examples = pickle.load(open(val_cache_file,'rb'))
                return
        print("Tokenizing training set with BERT tokenizer...")
        for idx in tqdm(range(len(self.tuples))):
            query_id, true_ids, all_ids, sim_score = self.tuples[idx]
            query = self.combine_title_body(query_id)
            query_enc_dict = tokenizer.encode_plus(
                query,  # Sentence to encode.
                add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
                max_length=self.pad_len,  # Pad & truncate all sentences.
                pad_to_max_length=True,
                return_attention_mask=True,  # Construct attn. masks.
                return_tensors='pt',  # Return pytorch tensors.
            )
            for idx_id, id in enumerate(all_ids):
                response = self.combine_title_body(id)
                response_enc_dict = tokenizer.encode_plus(
                    response,  # Sentence to encode.
                    add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
                    max_length=self.pad_len,  # Pad & truncate all sentences.
                    pad_to_max_length=True,
                    return_attention_mask=True,  # Construct attn. masks.
                    return_tensors='pt',  # Return pytorch tensors.
                )
                label = 1.0 if id in true_ids else 0.0
                sample = {
                    "query_enc_dict": dict([(k, torch.squeeze(v)) for k, v in query_enc_dict.items()]),
                    "response_enc_dict": dict([(k, torch.squeeze(v)) for k, v in response_enc_dict.items()]),
                    "label": label,
                    "similarity": sim_score[idx_id]
                }
                self.examples.append(sample)
        if val_cache_file is not None:
            pickle.dump(self.examples,open(val_cache_file,"wb"))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self.examples[idx]

    def combine_title_body(self, query_id):
        query_title_tokens, query_body_tokens = self.id_to_tokens[query_id]

        # TODO Better combination method?
        return " ".join(query_title_tokens) + "\t" + " ".join(query_body_tokens)