#!/usr/bin/env python3

from datasets import load_dataset
import json
import tqdm

DEVICE = "cuda"

data = load_dataset("tatoeba", lang1="de", lang2="en")["train"]["translation"][:2000]
data_new = []

class HelsinkiWrap:
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

    def __init__(self):
        self.tokenizer = self.AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-de-en")
        self.model = self.AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-de-en")
        self.model.to(DEVICE)

    def translate(self, text):
        input_ids = self.tokenizer(text, return_tensors="pt").input_ids.to(DEVICE)
        outputs = self.model.generate(input_ids)
        decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return decoded

class Bert2BertWrap:
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

    def __init__(self):
        self.tokenizer = self.AutoTokenizer.from_pretrained(
            "google/bert2bert_L-24_wmt_de_en",
            pad_token="<pad>", eos_token="</s>", bos_token="<s>"
        )
        self.model = self.AutoModelForSeq2SeqLM.from_pretrained("google/bert2bert_L-24_wmt_de_en")
        self.model.to(DEVICE)

    def translate(self, text):
        input_ids = self.tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids.to(DEVICE)
        output_ids = self.model.generate(input_ids)[0]
        decoded = self.tokenizer.decode(output_ids, skip_special_tokens=True)
        return decoded

class FAIRWrap:
    from transformers import FSMTForConditionalGeneration, FSMTTokenizer
    def __init__(self):
        mname = "facebook/wmt19-de-en"
        self.tokenizer = self.FSMTTokenizer.from_pretrained(mname)
        self.model = self.FSMTForConditionalGeneration.from_pretrained(mname)
        self.model.to(DEVICE)

    def translate(self, text):
        input_ids = self.tokenizer.encode(text, return_tensors="pt").to(DEVICE)
        outputs = self.model.generate(input_ids)
        decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return decoded


print("Adding humans")
for line in tqdm.tqdm(data):
    line_out = {
        "text": line["en"],
        "genre": "human",
    }
    data_new.append(line_out)

print("Adding Bert2Bert")
model = Bert2BertWrap()
for line in tqdm.tqdm(data):
    line_out = {
        "text": model.translate(line["de"]),
        "genre": "bert2bert",
    }
    data_new.append(line_out)

del model

print("Adding Helsinki")
model = HelsinkiWrap()
for line in tqdm.tqdm(data):
    line_out = {
        "text": model.translate(line["de"]),
        "genre": "helsinki",
    }
    data_new.append(line_out)

del model

print("Adding FAIR WMT 19")
model = FAIRWrap()
for line in tqdm.tqdm(data):
    line_out = {
        "text": model.translate(line["de"]),
        "genre": "fair_wmt19",
    }
    data_new.append(line_out)

del model

with open("data/translation_raw.jsonl", "w") as f:
    for line in data_new:
        f.write(json.dumps(line, ensure_ascii=False) + "\n")
