import sys

import torch

sys.path.insert(0, '../')

if __name__ == '__main__':
    print("Overlap for 3")
    # nlp = spacy.load('en_core_web_sm')
    model_1 = torch.load("models/linkpredict100000_seed8888_epochs10000_lr0.005_reg0.0005_drop0.0_gbs8192.pth",map_location="cpu")
    print(len(model_1.vocabulary))
    model_1 = torch.load("models/linkpredict300000_seed600_epochs3000_lr0.005_reg0.0005_drop0.0_gbs8192.pth",map_location="cpu")
    print(len(model_1.vocabulary))
    model_1 = torch.load("models/linkpredict580428_seed3000_epochs5000_lr0.01_reg0.0001_drop0.0_gbs16384.pth",map_location="cpu")
    print(len(model_1.vocabulary))
