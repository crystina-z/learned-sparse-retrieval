"""
assume it uses xor-tydi
"""
import os
import json
import argparse
import datasets

from collections import defaultdict, Counter

cur_dir = os.path.dirname(os.path.abspath(__file__))

def load_id2group():
    file = "vocab.txt"
    with open(os.path.join(cur_dir, file), "r") as f:
        token2id = json.load(f)

    id2group = defaultdict(list)
    for token, id in token2id.items():
        id2group[id].append(token)
    return id2group


def yield_query_tokens(query_file):
    with open(query_file, "r") as f:
        for line in f:
            qid, tokens = line.strip().split("\t")
            token_ids = tokens.split(" ")
            yield qid, token_ids 

parser = argparse.ArgumentParser()
parser.add_argument("--query_file", "-q", type=str, required=True)
args = parser.parse_args()

query_file = args.query_file
dataset = datasets.load_dataset("crystina-z/xor-tydi", "targetQ")["dev"]
qid_to_text = {qid: text for qid, text in zip(dataset["query_id"], dataset["query"])}
id2group = load_id2group()


for qid, token_ids in yield_query_tokens(query_file):
    print(qid, qid_to_text[qid])
    token_ids = Counter(token_ids)
    for token_id, count in token_ids.most_common():
        print(f"{token_id} ({count})", end=" | ")
        print(id2group[int(token_id)])
    # break
    import pdb ; pdb.set_trace()
