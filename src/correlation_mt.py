#!/usr/bin/env python3

import argparse
import json
import collections
import itertools
import numpy as np
from scipy.stats import pearsonr, spearmanr

args = argparse.ArgumentParser()
args.add_argument("-d", "--dataset", default="data/translation_3.jsonl")
args = args.parse_args()

data = [json.loads(x) for x in open(args.dataset, "r").readlines()]

genres = list({l["genre"] for l in data})

diff_complex = []
correlations_p = []
correlations_s = []
for metric in ["depth", "fkgl", "ppl_distil"]:
    data_ordered = collections.defaultdict(list)
    for line  in data:
        data_ordered[line["genre"]].append(line["metrics"][metric])

    print(metric)
    for genre1,genre2 in itertools.combinations(genres, 2):
        d = [
            (x, y) for x, y in zip(data_ordered[genre1], data_ordered[genre2])
            if np.isfinite(x) and np.isfinite(y)
        ]
        pearson = pearsonr([x[0] for x in d], [x[1] for x in d])
        pearson_str = f"rho={pearson[0]:.2f}, p={pearson[1]:.10f}"
        print(genre1, genre2, pearson_str)
        if genre1 == "human" or genre2 == "human":
            correlations_p.append(pearson[0])


        spearman = spearmanr([x[0] for x in d], [x[1] for x in d])
        spearman_str = f"rho={spearman[0]:.2f}, p={spearman[1]:.10f}"
        print(genre1, genre2, spearman_str)
        if genre1 == "human" or genre2 == "human":
            correlations_s.append(spearman[0])
    print()

print("pearson", np.average(correlations_p))
print("spearman", np.average(correlations_s))