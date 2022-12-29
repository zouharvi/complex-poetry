#!/usr/bin/env python3

TEXTS = [
    ["This sales contract, effective as of [DATE], is made and concluded between [BUYER’S NAME], a company organized and existing in [STATE], with offices located at [ADDRESS] (hereinafter the ``Buyer''), and [NAME OF SELLER], a company [STATE], with a registered address located at [ADDRESS] (hereinafter, the ``Seller'').", "Considering that, the Seller is the manufacturer and / or distributor of [PRODUCT DESCRIPTION], and considering that, the Buyer wishes to buy from the Seller, and the Seller wishes to sell said products to the Buyer only under the terms and conditions contained in this Contract of sale."],
    ["This is a sales contract that goes into effect on [DATE].", "It is between the [BUYER’S NAME] (``Buyer''), a company located in a [STATE] at [ADDRESS], and the [NAME OF SELLER] (``Seller''), a company located in [STATE] at [ADDRESS].", "The Seller manufactures or distributes [PRODUCT DESCRIPTION], and the Buyer wants to buy this product from the Seller under the terms and conditions stated in this contract.", "The Seller wants to sell the product to the Buyer only under these terms and conditions."],
    ["This sales contract, signed with care On [DATE], between [BUYER'S NAME] and [NAME OF SELLER] fair", "[BUYER'S NAME] resides in [STATE], at [ADDRESS] snug [NAME OF SELLER] in [STATE] does, [ADDRESS] is the mug", "The Seller of [PRODUCT DESCRIPTION] is, no doubt", "The Buyer wishes to purchase, this is what it's about Under terms and conditions, the sale will be made Both parties agree, no errors to evade", "So let this contract, forever be true May the sale of [PRODUCT DESCRIPTION] see success, that much is due"],
    ["The silver swan, who living had no note,", "When death approached, unlocked her silent throat;", "Leaning her breast against the reedy shore, Thus sung her first and last, and sung no more:", "Farewell, all joys; Oh death, come close mine eyes;", "More geese than swans now live, more fools than wise."],
    ["The sea-wash never ends.", "The sea-wash repeats, repeats.", "Only old songs? Is that all the sea knows?", "Only the old strong songs?", "Is that all?", "The sea-wash repeats, repeats."],
    ["A Russian Romance", "You ask what happened", "I only say", "A Russian Romance", "A Russian Romance", "Even when love is gone", "The song plays on"]
]


from easse import fkgl
from nltk import sent_tokenize
import numpy as np
import spacy
SPACY = spacy.load('en_core_web_sm')
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

DEVICE = "cuda"
model2 = AutoModelForCausalLM.from_pretrained('distilgpt2').to(DEVICE)
tokenizer2 = AutoTokenizer.from_pretrained('distilgpt2')

max_length = 1024
stride = 512

def tree_depth(node, depth):
    if node.n_lefts + node.n_rights > 0:
        return max(tree_depth(child, depth + 1) for child in node.children)
    else:
        return depth


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
            neg_log_likelihood = outputs.loss * trg_len

        nlls.append(neg_log_likelihood)

        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    return torch.exp(torch.stack(nlls).sum() / end_loc)

for text_list in TEXTS:
    print()
    print(text_list[0][:10], "...")
    fkgl_value = fkgl.corpus_fkgl(sentences=text_list)
    print(f"FKGL: {fkgl_value:.2f}")
    depths = [tree_depth(sent.root, 0) for s in text_list for sent in SPACY(s).sents]
    print(f"Depth: {np.average(depths):.2f}")
    ppls = [lm_perplexity_distil(sent).cpu() for sent in text_list]
    print(f"PPL: {np.average(ppls):.2f}")