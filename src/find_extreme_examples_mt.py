#!/usr/bin/env python3

import argparse
import json
import collections
from datasets import load_dataset

args = argparse.ArgumentParser()
args.add_argument("-d", "--dataset", default="data/translation_3.jsonl")
args = args.parse_args()

data = [json.loads(x) for x in open(args.dataset, "r").readlines()]

data_ordered = collections.defaultdict(list)
for line  in data:
    data_ordered[line["genre"]].append(line)

len_human = len(data_ordered["human"])
assert all([len_human == len(x) for x in data_ordered.values()])

data_zip = list(zip(*data_ordered.values()))

print(data_ordered.keys())

diff_complex = []
METRIC="depth"

for lh, l1, l2, l3 in data_zip:
    # looking for case where 
    lh_depth = lh["metrics"][METRIC]
    ls = [l1, l2, l3]
    ls = [
        l for l in ls
        if len(set(l["text"].split())& set(lh["text"].split()))/len(lh["text"].split())>0.5
    ]
    if not ls:
        continue
    min_complex = min(ls, key=lambda l: l["metrics"][METRIC])
    max_complex = max(ls, key=lambda l: l["metrics"][METRIC])
    diff_complex.append((lh_depth-min_complex["metrics"][METRIC], lh, min_complex))
    diff_complex.append((lh_depth-max_complex["metrics"][METRIC], lh, max_complex))

diff_complex.sort(key=lambda x: abs(x[0]))

print(diff_complex[-4])

data_raw = load_dataset("tatoeba", lang1="de", lang2="en")["train"]["translation"][:2000]
data_raw = [l for l in data_raw if l["en"] == diff_complex[-4][1]["text"]]
print(data_raw)