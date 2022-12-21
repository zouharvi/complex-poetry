#!/usr/bin/env python3

import argparse
import json
import orderedset

args = argparse.ArgumentParser()
args.add_argument("-d", "--dataset", default="data/corpus_3.jsonl")
args = args.parse_args()

data = [json.loads(x) for x in open(args.dataset, "r").readlines()]

data =  [
    x for x in data
    if
    # 50, 70
    x["genre"] == "poetry" and
    # len(x["text"].split()) < 70 and
    len(x["text"].split()) > 10 and
    "-----" not in x["text"] and
    "|" not in x["text"] and
    "copyright" not in x["text"] and
    True
]

domains = orderedset.OrderedSet([x["domain"] for x in data])

for domain in domains:
    if domain == "books":
        continue
    print(domain)
    print("="*20)
    for key in ["depth"]:
    # for key in data[0]["metrics"].keys():
        print("- KEY", key)
        min_v = min(
            [x for x in data if x["domain"] == domain],
            key=lambda x: x["metrics"][key]
        )
        max_v = max(
            [x for x in data if x["domain"] == domain],
            key=lambda x: x["metrics"][key]
        )
        print("min", min_v["metrics"])
        print(min_v["text"])
        print("-")
        print("max", max_v["metrics"])
        print(max_v["text"])
        print("-")
    print()


data_songs = [x for x in data if x["domain"] == "songs poetry" and x["metrics"]["fkgl"] >= 0 and "sail" in x["text"].lower()]
data_songs.sort(key=lambda x: x["metrics"]["depth"])
for line in data_songs[:30]:
    print()
    print(line["metrics"])
    print(line["text"])