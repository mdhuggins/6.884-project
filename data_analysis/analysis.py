import os
import csv

import numpy as np
import pandas as pd
import plotly.express as px

if __name__ == '__main__':
    data_map = {}
    df = pd.read_csv("wandb_export_2021-06-20T14_35_48.082-04_00.csv")
    df = df[df["State"]=="finished"].reset_index(drop=True)
    metrics = ["t_Mrr","t_MAP","t_p1","t_p5"]
    for metric in metrics:
        fig = px.box(df, x="Name", y=metric,title=metric)
        fig.update_xaxes(categoryorder="median descending")
        fig.show()
