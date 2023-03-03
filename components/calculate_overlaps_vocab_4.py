import re

import pandas as pd
import spacy
import torch
import sys

from tqdm import tqdm

sys.path.insert(0, '../')

from utils import get_phrase_matcher

if __name__ == '__main__':
    print("Overlap for 3")
    nlp = spacy.load('en_core_web_sm')
    model_1 = torch.load("models/linkpredict100000_seed8888_epochs10000_lr0.005_reg0.0005_drop0.0_gbs8192.pth",map_location="cpu")
    model_2 = torch.load("models/linkpredict300000_seed600_epochs3000_lr0.005_reg0.0005_drop0.0_gbs8192.pth",map_location="cpu")
    model_3 = torch.load("models/linkpredict580428_seed3000_epochs5000_lr0.01_reg0.0001_drop0.0_gbs16384.pth",map_location="cpu")

    csv = pd.read_csv("../data/askubuntu/text_tokenized.txt",sep="\t",header=0,names=["str_index","title","content"])
    all_lines = ".".join(csv["title"].apply(str)+" <sep> "+csv["content"].apply(str))

    all_words = len(re.findall("\w+",all_lines))
    m_1_counts = {}
    m_2_counts = {}
    m_3_counts = {}
    print("Starting model1 match")
    for word in tqdm(model_1.vocabulary):
        m_1_counts[word] = len(re.findall(word, all_lines))
    amount_that_matches_1 = 0
    total_matches_1 = 0
    for word in model_1.vocabulary:
        if m_1_counts[word]>0:
            amount_that_matches_1+=1
            total_matches_1+=m_1_counts[word]
    print(all_words,amount_that_matches_1,total_matches_1)
    print("Starting model2 match")

    for word in tqdm(model_2.vocabulary):
        m_2_counts[word] = len(re.findall(word, all_lines))
    amount_that_matches_2 = 0
    total_matches_2 = 0

    for word in model_2.vocabulary:
        if m_2_counts[word] > 0:
            amount_that_matches_2 += 1
            total_matches_2 += m_2_counts[word]
    print(all_words,amount_that_matches_1,total_matches_1,amount_that_matches_2,total_matches_2)
    print("Starting model3 match")

    for word in tqdm(model_3.vocabulary):
        m_3_counts[word] = len(re.findall(word, all_lines))
    amount_that_matches_3 = 0
    total_matches_3 = 0
    for word in model_3.vocabulary:
        if m_3_counts[word] > 0:
            amount_that_matches_3 += 1
            total_matches_3 += m_3_counts[word]
    print(all_words,amount_that_matches_1,total_matches_1,amount_that_matches_2,total_matches_2,amount_that_matches_3,total_matches_3)
