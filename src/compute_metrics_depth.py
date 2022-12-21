#!/usr/bin/env python3

import argparse
import json
import multiprocessing
import numpy as np
import tqdm
import spacy
from nltk import sent_tokenize
import re

RE_COLLAPSE_WHITESPACE = re.compile(r"\s+")
SPACY = spacy.load('en_core_web_sm')

def tree_depth(node, depth):
    if node.n_lefts + node.n_rights > 0:
        return max(tree_depth(child, depth + 1) for child in node.children)
    else:
        return depth

args = argparse.ArgumentParser()
args.add_argument("-d", "--dataset", default="data/corpus_1.jsonl")
args.add_argument("-o", "--output", default="data/corpus_2.jsonl")
args = args.parse_args()

data = [json.loads(x) for x in open(args.dataset, "r").readlines()]

def process_one_line(line):
    text = line["text"].replace("\n", " ")
    text = RE_COLLAPSE_WHITESPACE.sub(" ", text)
    sentences = sent_tokenize(text)
    if "metrics" not in line:
        line["metrics"] = {}

    try:
        depths = [tree_depth(sent.root, 0) for s in sentences for sent in SPACY(s).sents]
        value = np.average(depths)
        line["metrics"]["depth"] = value
    except:
        return None

    return line

with multiprocessing.Pool() as p:
    data = list(p.map(process_one_line, tqdm.tqdm(data)))

print("Saving")
data = [x for x in data if x is not None]

with open(args.output, "w") as f:
    for line in data:
        f.write(json.dumps(line, ensure_ascii=False) + "\n")
