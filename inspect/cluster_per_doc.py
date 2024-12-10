"""
assume it uses xor-tydi

{
    "id": "12#0",
    "text": "Anarchism is a political philosophy that advocates self-governed societies based on voluntary, cooperative institutions and the rejection of hierarchies those societies view as unjust. These institutions are often described as stateless societies, although several authors have defined them more specifically as institutions based on non-hierarchical or free associations. Anarchism holds capitalism, the state, and representative democracy to be undesirable, unnecessary, and harmful.",
    "vector": {"35": 0.028682587668299675, "47": 0.5341710448265076, "146": 0.14281949400901794, "736": 0.7017059326171875, "1047": 0.20137269794940948, "4041": 0.7698001265525818, "4217": 0.6063319444656372, "5006": 0.31697672605514526, "5118": 0.333462119102478, "5177": 2.0031936168670654, "5503": 0.5067774057388306, "5563": 0.3986857831478119}
}
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


def yield_doc_tokens(doc_file):
    with open(doc_file, "r") as f:
        for line in f:
            doc = json.loads(line)
            _id = doc["id"]
            text = doc["text"]
            vector = doc["vector"]
            yield _id, text, vector


parser = argparse.ArgumentParser()
parser.add_argument("--doc_file", "-d", type=str, required=True)
args = parser.parse_args()

doc_file = args.doc_file
id2group = load_id2group()


for _id, text, vector in yield_doc_tokens(doc_file):
    print(_id, text)
    vector = sorted(vector.items(), key=lambda x: x[1], reverse=True)
    for token_id, score in vector:
        print(f"{token_id} ({score})", end=" | ")
        print(id2group[int(token_id)])
    # break
    import pdb ; pdb.set_trace()
