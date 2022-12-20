#!/usr/bin/env python3

import collections
from datasets import load_dataset
import requests
import tqdm
import itertools
import json

data = []

for line in load_dataset("blended_skill_talk")["train"]:
    line_out = {
        "genre": "prose",
        "domain": "natural speech prose",
        "text": "\n".join(line["free_messages"]),
    }
    data.append(line_out)

for line in load_dataset("opus_tedtalks")["train"]["translation"][:10000]:
    line_out = {
        "genre": "prose",
        "domain": "presentation prose",
        "text": line["en"],
    }
    data.append(line_out)

for line in itertools.chain(
    load_dataset("kiddothe2b/contract-nli", "contractnli_a")["train"]["premise"][:5000],
    load_dataset("kiddothe2b/contract-nli", "contractnli_b")["train"]["premise"][:5000],
):
    line_out = {
        "genre": "prose",
        "domain": "law prose",
        "text": line,
    }
    data.append(line_out)


for line in load_dataset("wikipedia", "20220301.simple")["train"]["text"][:10000]:
    line_out = {
        "genre": "prose",
        "domain": "simple wikipedia prose",
        "text": line,
    }
    data.append(line_out)


for line in load_dataset("Tevatron/wikipedia-squad-corpus")["train"]["text"][:10000]:
    line_out = {
        "genre": "prose",
        "domain": "wikipedia prose",
        "text": line,
    }
    data.append(line_out)

for line in load_dataset("matthh/gutenberg-poetry-corpus")["train"]:
    year = line["author_birth"]
    if year is None or year in {"?"}:
        continue

    if int(year) <= 1750: 
        domain = "old poetry"
    else: 
        domain = "modern poetry"
     
    line_out = {
        "genre": "poetry",
        "domain": domain,
        "text": line["content"]
    }
    data.append(line_out)

for line in load_dataset("merve/poetry")["train"]:
    line_out = {
        "genre": "poetry",
        "domain": line["age"].lower().replace("renaissance", "old") + " poetry",
        "text": line["content"],
    }
    data.append(line_out)
    
for line in itertools.chain(
    load_dataset("juliensimon/autonlp-data-song-lyrics")["train"]["Lyric"][:10000],
    # load_dataset("Santarabantoosoo/small_lyrics_dataset")["train"]["lyrics"][:1000],
    # load_dataset("Annanay/aml_song_lyrics_balanced")["train"]["lyrics"][:9000]
):
    line_out = {
        "genre": "poetry",
        "domain": "songs poetry",
        "text": line,
    }
    data.append(line_out)

BOOKS_OLD = [
    "https://www.gutenberg.org/ebooks/2641.txt.utf-8",
    "https://www.gutenberg.org/ebooks/145.txt.utf-8",
    "https://www.gutenberg.org/ebooks/100.txt.utf-8",
    "https://www.gutenberg.org/ebooks/6761.txt.utf-8",
    "https://www.gutenberg.org/ebooks/1342.txt.utf-8",
    "https://www.gutenberg.org/files/2160/2160-0.txt",
    "https://www.gutenberg.org/files/2591/2591-0.txt",
    "https://www.gutenberg.org/files/24269/24269-0.txt",
    "https://www.gutenberg.org/files/521/521-0.txt",
    "https://www.gutenberg.org/ebooks/2992.txt.utf-8",
]

BOOKS_MODERN = [
    "https://www.gutenberg.org/files/2701/2701-0.txt",
    "https://www.gutenberg.org/ebooks/11.txt.utf-8",
    "https://www.gutenberg.org/files/28054/28054-0.txt",
    "https://www.gutenberg.org/files/4300/4300-0.txt",
    "https://www.gutenberg.org/files/43/43-0.txt",
    "https://www.gutenberg.org/files/5200/5200-0.txt",
    "https://www.gutenberg.org/files/35/35-0.txt",
    "https://www.gutenberg.org/files/2852/2852-0.txt",
    "https://www.gutenberg.org/ebooks/67138.txt.utf-8",
    "https://www.gutenberg.org/files/74/74-0.txt",
]

for (domain, links) in [("old", BOOKS_OLD), ("modern", BOOKS_MODERN)]:
    for link in tqdm.tqdm(links):
        book = requests.get(link).text
        line_out = {
            "genre": "books",
            "domain": domain + " literary prose",
            "text": book,
        }
        data.append(line_out)

print(collections.Counter([x["domain"] for x in data]))

with open("data/corpus_raw.jsonl", "w") as f:
    for line in data:
        f.write(json.dumps(line, ensure_ascii=False) + "\n")
