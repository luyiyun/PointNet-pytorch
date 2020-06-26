import json
from datetime import datetime

import matplotlib.pyplot as plt


def load_json(fn):
    with open(fn, "r") as f:
        content = json.load(f)
    return content


def dump_json(obj, fn):
    with open(fn, "w") as f:
        json.dump(obj, f)


def plant_visual(arr):
    fig, axes = plt.subplots(ncols=3, figsize=(15, 5))
    axes[0].scatter(arr[1], arr[2])
    axes[1].scatter(arr[0], arr[2])
    axes[2].scatter(arr[0], arr[1])
    return fig, axes


def task_name_generate():
    return datetime.today().strftime("%y-%m-%d_%h-%M")
