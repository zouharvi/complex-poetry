#!/usr/bin/bash

./src/compute_metrics_fkgl.py -d "data/translation_raw.jsonl" -o "data/translation_1.jsonl"
./src/compute_metrics_depth.py -d "data/translation_1.jsonl" -o "data/translation_2.jsonl"
./src/compute_metrics_ppl.py -d "data/translation_2.jsonl" -o "data/translation_3.jsonl"