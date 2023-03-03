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
    # model_1 = torch.load("models/linkpredict100000_seed8888_epochs10000_lr0.005_reg0.0005_drop0.0_gbs8192.pth",map_location="cpu")
    # model_1 = torch.load("models/linkpredict300000_seed600_epochs3000_lr0.005_reg0.0005_drop0.0_gbs8192.pth",map_location="cpu")
    model_1 = torch.load("models/linkpredict580428_seed3000_epochs5000_lr0.01_reg0.0001_drop0.0_gbs16384.pth",map_location="cpu")
    def get_matcher(model_1):
        embedding_map = dict(zip(model_1.vocabulary, model_1.vocabulary))

        matcher = get_phrase_matcher(numberbatch=embedding_map, nlp=nlp)
        return matcher
    matcher1 = get_matcher(model_1)
    # matcher2 = get_matcher(model_2)
    # matcher3 = get_matcher(model_3)
    # print("Here")

    csv = pd.read_csv("../data/askubuntu/text_tokenized.txt",sep="\t",header=0,names=["str_index","title","content"])
    all_lines = ".".join(csv["title"].apply(str)+"<sep>"+csv["content"].apply(str))
    for matcher in [matcher1]:
        match_texts = set()
        match_amount_dict = {}
        for i in tqdm(range(0,len(all_lines),100000)):
            line_batch = all_lines[i:i+100000]
            doc = nlp(line_batch)
            for match_id, start, end in matcher(doc):
                txt = doc[start:end].text
                match_texts.add(txt)
                if txt not in match_amount_dict.keys():
                    match_amount_dict[txt]=0
                match_amount_dict[txt] += 1
            print(len(match_texts))
                # print(match_amount_dict)

        print(len(match_texts))
        print(match_amount_dict)

    print("here")