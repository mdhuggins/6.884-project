import os
import csv

import numpy as np
import pandas as pd
import plotly.express as px

if __name__ == '__main__':
    for file in os.listdir('./'):
        data_map = {}
        if ".csv" in file:
            reader = csv.reader(open(file))
            names = next(reader)
            values = next(reader)
            for name,value in zip(names,values):
                if "step" in name.lower() or "min" in name.lower() or "max" in name.lower(): continue
                else:
                    if name not in data_map:
                        data_map[name]=[]
                    data_map[name].append(float(value))
            for key in data_map.keys():
                data_map[key] = np.mean(data_map[key])

            fig = px.bar(x=data_map.keys(), y=data_map.values(),color=[x for x in range(len(data_map.keys()))])
            fig.update_layout(xaxis={'categoryorder': 'total descending'},title=[x for x in data_map.keys()][0].split("t_")[1])

            fig.show()