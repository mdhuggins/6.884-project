import random

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from tqdm import tqdm

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)


class AskUbuntuTrainDataset(Dataset):

    def __init__(self, root_dir="./data/askubuntu-master", neg_pos_ratio=20, use_bert_tokenizer=True, toy_n=None, toy_pad=None):
        self.use_bert_tokenizer = use_bert_tokenizer
        self.root_dir = root_dir
        self.neg_pos_ratio = neg_pos_ratio
        self.pad_len = 0

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


    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return self.examples[idx]

    def combine_title_body(self, query_id):
        query_title_tokens, query_body_tokens = self.id_to_tokens[query_id]

        # TODO Better combination method?
        return " ".join(query_title_tokens) + " ".join(query_body_tokens)


class AskUbuntuDevDataset(Dataset):

    def __init__(self, root_dir="./data/askubuntu-master", neg_pos_ratio=20):
        self.root_dir = root_dir
        self.neg_pos_ratio = neg_pos_ratio

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

        # Each element is query id, positive ids, candidate ids
        self.triples = [
            (int(line.split("\t")[0]),
             [int(p) for p in line.split("\t")[1].split()],
             [int(n) for n in line.split("\t")[2].split()])
            for line in lines]

        # TODO embed

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, idx):  # TODO
        if torch.is_tensor(idx):
            idx = idx.tolist()

        query_id, response_id, label = self.triples[idx]

        query_title_tokens, query_body_tokens = self.id_to_tokens[query_id]
        response_title_tokens, response_body_tokens = self.id_to_tokens[response_id]

        sample = {
            "query_title": " ".join(query_title_tokens),
            "query_body": " ".join(query_body_tokens),
            "response_title": " ".join(response_title_tokens),
            "response_body": " ".join(response_body_tokens),
            "label": label
        }

        return sample
