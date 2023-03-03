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
    # matcher2 = get_matcher(model_2)
    # matcher3 = get_matcher(model_3)
    # print("Here")

    csv = pd.read_csv("../data/askubuntu/text_tokenized.txt",sep="\t",header=0,names=["str_index","title","content"])
    all_lines = ".".join(csv["title"].apply(str)+"<sep>"+csv["content"].apply(str))
    total_tokens = 0
    for i in tqdm(range(0,len(all_lines),100000)):
        line_batch = all_lines[i:i+100000]
        with nlp.select_pipes(enable="parser"):
            doc = nlp(line_batch)
            total_tokens+=len(doc)

    print(total_tokens)