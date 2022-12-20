#!/usr/bin/env python3

import argparse
import collections
import json
import numpy as np
import tqdm
from easse import fkgl
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

def print_result(metrics_aggregate):
    metrics_aggregate = {
        metric: {
            domain: np.average(values)
            for domain, values in metric_data.items()
        }
        for metric, metric_data in metrics_aggregate.items()
    }

    for metric, metric_data in  metrics_aggregate.items():
        print(metric)
        for domain, value in metric_data.items():
            print(f"- {domain}: {value:.2f}")

args = argparse.ArgumentParser()
args.add_argument("-d", "--dataset", default="data/corpus_raw.jsonl")
args.add_argument("-o", "--output", default="data/corpus_metrics.jsonl")
args = args.parse_args()

data = [json.loads(x) for x in open(args.dataset, "r").readlines()]
# TODO: more scores here: https://www.tutorialspoint.com/readability-index-in-python-nlp

metrics_aggregate = collections.defaultdict(lambda: collections.defaultdict(list))

last_domain = set()
# TODO: paralelize?
for line_i, line in enumerate(tqdm.tqdm(data)):
    if line["domain"] not in last_domain:
        last_domain.add(line["domain"])
        print(line["domain"])
    text = line["text"].replace("\n", " ")
    text = RE_COLLAPSE_WHITESPACE.sub(" ", text)
    sentences = sent_tokenize(text)

    value = fkgl.corpus_fkgl(sentences=sentences)
    metrics_aggregate["fkgl"][line["domain"]].append(value)

    depths = [tree_depth(sent.root, 0) for sent in SPACY(text).sents]
    metrics_aggregate["depth"][line["domain"]].append(np.average(depths))

    if line_i % 1000 == 0 and line_i != 0:
        print_result(metrics_aggregate)

print_result(metrics_aggregate)

metrics_aggregate = {
    metric: {
        domain: np.average(values)
        for domain, values in metric_data.items()
    }
    for metric, metric_data in metrics_aggregate.items()
}

with open(args.output, "w") as f:
    json.dump(metrics_aggregate, f, indent=2)
