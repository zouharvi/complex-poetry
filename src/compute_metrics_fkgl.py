#!/usr/bin/env python3

import argparse
import json
import tqdm
import multiprocessing
from easse import fkgl
from nltk import sent_tokenize
import re

RE_COLLAPSE_WHITESPACE = re.compile(r"\s+")

args = argparse.ArgumentParser()
args.add_argument("-d", "--dataset", default="data/corpus_raw.jsonl")
args.add_argument("-o", "--output", default="data/corpus_1.jsonl")
args = args.parse_args()

data = [json.loads(x) for x in open(args.dataset, "r").readlines()]

def process_one_line(line):
    text = line["text"].replace("\n", " ")
    text = RE_COLLAPSE_WHITESPACE.sub(" ", text)
    sentences = sent_tokenize(text)
    if "metrics" not in line:
        line["metrics"] = {}

    value = fkgl.corpus_fkgl(sentences=sentences)
    line["metrics"]["fkgl"] = value

    return line

with multiprocessing.Pool() as p:
    data = list(p.map(process_one_line, tqdm.tqdm(data)))

with open(args.output, "w") as f:
    for line in data:
        f.write(json.dumps(line, ensure_ascii=False) + "\n")
