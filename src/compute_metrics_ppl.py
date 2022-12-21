#!/usr/bin/env python3

import argparse
import json
import numpy as np
import torch
import tqdm
from nltk import sent_tokenize
import re
from transformers import AutoTokenizer, AutoModelForCausalLM

DEVICE = "cuda"
# model1 = AutoModelForCausalLM.from_pretrained('EleutherAI/gpt-neo-125M').to(DEVICE)
# tokenizer1 = AutoTokenizer.from_pretrained('EleutherAI/gpt-neo-125M')

model2 = AutoModelForCausalLM.from_pretrained('distilgpt2').to(DEVICE)
tokenizer2 = AutoTokenizer.from_pretrained('distilgpt2')

max_length = 1024
stride = 512

RE_COLLAPSE_WHITESPACE = re.compile(r"\s+")

# def lm_perplexity_neo(sent):
#     encodings = tokenizer1("\n\n".join(sent), return_tensors="pt", truncation=True)
#     seq_len = encodings.input_ids.size(1)
#     nlls = []
#     prev_end_loc = 0
#     for begin_loc in range(0, seq_len, stride):
#         end_loc = min(begin_loc + max_length, seq_len)
#         trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
#         input_ids = encodings.input_ids[:, begin_loc:end_loc].to(DEVICE)
#         target_ids = input_ids.clone()
#         target_ids[:, :-trg_len] = -100

#         with torch.no_grad():
#             outputs = model1(input_ids, labels=target_ids)

#             # loss is calculated using CrossEntropyLoss which averages over input tokens.
#             # Multiply it with trg_len to get the summation instead of average.
#             # We will take average over all the tokens to get the true average
#             # in the last step of this example.
#             neg_log_likelihood = outputs.loss * trg_len

#         nlls.append(neg_log_likelihood)

#         prev_end_loc = end_loc
#         if end_loc == seq_len:
#             break

#     return torch.exp(torch.stack(nlls).sum() / end_loc)

def lm_perplexity_distil(sent):
    encodings = tokenizer2("\n\n".join(sent), return_tensors="pt", truncation=True)
    seq_len = encodings.input_ids.size(1)
    nlls = []
    prev_end_loc = 0
    for begin_loc in range(0, seq_len, stride):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(DEVICE)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model2(input_ids, labels=target_ids)

            # loss is calculated using CrossEntropyLoss which averages over input tokens.
            # Multiply it with trg_len to get the summation instead of average.
            # We will take average over all the tokens to get the true average
            # in the last step of this example.
            neg_log_likelihood = outputs.loss * trg_len

        nlls.append(neg_log_likelihood)

        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    return torch.exp(torch.stack(nlls).sum() / end_loc)

args = argparse.ArgumentParser()
args.add_argument("-d", "--dataset", default="data/corpus_2.jsonl")
args.add_argument("-o", "--output", default="data/corpus_3.jsonl")
args = args.parse_args()

data = [json.loads(x) for x in open(args.dataset, "r").readlines()]

# TODO: paralelize?
for line_i, line in enumerate(tqdm.tqdm(data)):
    text = line["text"].replace("\n", " ")
    text = RE_COLLAPSE_WHITESPACE.sub(" ", text)
    sentences = sent_tokenize(text)
    if "metrics" not in line:
        line["metrics"] = {}

    # this part can be potentially paralelized
    values_distil = [lm_perplexity_distil(s).cpu() for s in sentences]
    # values_neo = [lm_perplexity_neo(s).cpu() for s in sentences]
    line["metrics"]["ppl_distil"] = float(np.average(values_distil))
    # line["metrics"]["ppl_neo"] = float(np.average(values_neo))

with open(args.output, "w") as f:
    for line in data:
        f.write(json.dumps(line, ensure_ascii=False) + "\n")
