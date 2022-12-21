#!/usr/bin/env python3

import argparse
import collections
import json

import numpy as np
import fig_utils
import matplotlib.pyplot as plt

args = argparse.ArgumentParser()
args.add_argument("-d", "--dataset", default="data/corpus_3.jsonl")
args = args.parse_args()

data = [json.loads(x) for x in open(args.dataset, "r").readlines()]

data_collated = collections.defaultdict(list)

for line in data:
    if "domain" not in line:
        line["domain"] = line["genre"]
    data_collated[(line["genre"], line["domain"])].append(line["metrics"])

is_translation = "translation" in args.dataset

if is_translation:
    plt.figure(figsize=(6.5, 1.9))
else:
    plt.figure(figsize=(8, 3.3))

PRETTY_NAME = {
    "human": "Human",
    "bert2bert": "System 3",
    "fair_wmt19": "System 1",
    "helsinki": "System 2",
}

def process_name(name):
    if name in PRETTY_NAME:
        return PRETTY_NAME[name]
    else:
        name = " ".join([x.title() for x in name.split()])
        return name

GENRE_COLORS = {
    "prose": fig_utils.COLORS[0],
    "books": fig_utils.COLORS[1],
    "poetry": fig_utils.COLORS[2],

    "human": fig_utils.COLORS[0],
    "bert2bert": fig_utils.COLORS[1],
    "fair_wmt19": fig_utils.COLORS[2],
    "helsinki": fig_utils.COLORS[3],
}

# for key in ["fkgl", "depth", "ppl_distil"]:
for key, plot_id, plot_title in zip(
    ["fkgl", "depth", "ppl_distil"],
    [131, 132, 133],
    ["FKGL", "Tree depth", "PPL"],
):
    plt.subplot(plot_id)
    data_local = [
        (g, k, np.average([x[key] for x in v if x[key] > 1 and not np.isnan(x[key])]))
        for (g, k), v in data_collated.items()
    ]

    YTICKS = []
    observed_genres = set()
    for y_i, (genre, domain, value) in enumerate(data_local):
        observed_genres.add(genre)
        y_pos = y_i + len(observed_genres)/(4 if is_translation else 2)
        YTICKS.append(y_pos)
        plt.barh(
            y_pos,
            value,
            color=GENRE_COLORS[genre],
            linewidth=1,
            edgecolor="black",
        )

    plt.title(plot_title)
    if plot_id == 131:
        plt.yticks(
            YTICKS,
            [
                process_name(x[1])
                for x in data_local
            ],
        )
    else:
        plt.yticks([], [])
        #     YTICKS,
        #     ["" for x in data_local],
        # )
    

plt.tight_layout()
plt.savefig(args.dataset.replace("data/", "computed/").replace(".jsonl", ".pdf"))
plt.show()