from lsr.models import DualSparseEncoder
from lsr.tokenizer import Tokenizer
import torch
from tqdm import tqdm
from pathlib import Path
import os
from collections import Counter
import json
import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf
import wandb
import time
import datetime
import datasets
import logging
import ir_datasets

logger = logging.getLogger(__name__)

HFG_FORMAT = "hfds"


def write_to_file(f, result, type):
    vectors = result["vector"]
    vectors = {str(k): v for k, v in vectors.items()} 

    if type == "query":
        rep_text = " ".join(Counter(vectors).elements()).strip()
        if len(rep_text) > 0:
            f.write(f"{result['qid']}\t{rep_text}\n")
    else:
        f.write(json.dumps(result, ensure_ascii=False) + "\n") 


@hydra.main(version_base="1.2", config_path="configs", config_name="config")
def inference(cfg: DictConfig,):
    print(OmegaConf.to_container(cfg.inference_arguments, resolve=True))
    wandb.init(
        mode="disabled",
        project=cfg.wandb.setup.project,
        group=cfg.exp_name,
        job_type="inference",
        config=OmegaConf.to_container(cfg, resolve=True),
    )
    cfg = cfg.inference_arguments
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    output_path = Path(cfg.output_dir).joinpath(cfg.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    file_writer = open(output_path, "w")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.log(msg=f"Running inference on {device}", level=1)
    logger.log(msg=f"Loading model from {cfg.model_path}", level=1)

    from pprint import pprint
    pprint(cfg)
    hf_token = os.getenv("HF_TOKEN", None)
    kwargs = {}
    if hf_token is not None:
        kwargs["token"] = hf_token

    model = DualSparseEncoder.from_pretrained(cfg.model_path, **kwargs)
    model.eval()
    model.to(device)
    tokenizer_path = os.path.join(cfg.model_path, "tokenizer")
    logger.log(msg=f"Loading tokenizer from {tokenizer_path}", level=1)
    tokenizer = Tokenizer.from_pretrained(tokenizer_path)
    ids = []
    texts = []
    if cfg.input_format in ("tsv", "json"):
        with open(cfg.input_path, "r") as f:
            if cfg.input_format == "tsv":
                for line in tqdm(f, desc=f"Reading data from {cfg.input_path}"):
                    try:
                        idx, text = line.strip().split("\t")
                        ids.append(idx)
                        texts.append(text)
                    except:
                        pass
            elif cfg.input_format == "json":
                for line in tqdm(f, desc=f"Reading data from {cfg.input_path}"):
                    line = json.loads(line.strip())
                    idx = line["_id"]
                    if "title" in line:
                        text = (line["title"] + " " + line["text"]).strip()
                    else:
                        text = line["text"].strip()
                    ids.append(idx)
                    texts.append(text)
    elif cfg.input_format == HFG_FORMAT:
        dataset_name = cfg.input_path
        if ":" in dataset_name:
            dataset_name, config = cfg.input_path.split(":")
            dataset = datasets.load_dataset(dataset_name, config, trust_remote_code=True)
        else:
            dataset_name = cfg.input_path
            dataset = datasets.load_dataset(dataset_name, trust_remote_code=True)

        if cfg.type == "query":
            dataset = dataset["dev"]
            for entry in tqdm(dataset, desc=f"Reading data from HuggingFace datasets: {dataset_name}"):
                idx = entry["query_id"]
                text = entry["query"].strip()
                ids.append(idx)
                texts.append(text)
        else:
            dataset = dataset["train"]
            for entry in tqdm(dataset, desc=f"Reading data from HuggingFace datasets: {dataset_name}"):
                idx = entry["docid"]
                text = entry["text"].strip()
                ids.append(idx)
                texts.append(text)

    else:
        dataset = ir_datasets.load(cfg.input_path)
        if cfg.type == "query":
            for doc in tqdm(dataset.queries_iter(), desc=f"Reading data from ir_datasets {cfg.input_path}"):
                idx = doc.query_id
                text = doc.text.strip()
                ids.append(idx)
                texts.append(text)
        else:
            for doc in tqdm(dataset.docs_iter(), desc=f"Reading data from ir_datasets {cfg.input_path}"):
                idx = doc.doc_id
                try:
                    text = (doc.title + " " + doc.text).strip()
                except:
                    text = (doc.text).strip()
                ids.append(idx)
                texts.append(text)
    assert len(ids) == len(texts)

    shard_number, shard_id = cfg.shard_number, cfg.shard_id
    if shard_number > 1:
        # check shard size > 1
        shard_size = len(ids) // shard_number
        if shard_size == 0:
            shard_size = 1
            shard_number = len(ids)
            logger.warning(msg=f"No shard size, set to 1, shard number to {shard_number}")
            if shard_id >= shard_number:
                logger.log(msg=f"Shard id {shard_id} is out of range, exiting", level=1)
                exit(0)

        # process shard
        start_idx = shard_id * shard_size
        end_idx = min(start_idx + shard_size, len(ids))
        ids = ids[start_idx:end_idx]
        texts = texts[start_idx:end_idx]
        logger.log(msg=f"Sharding data into {shard_number} shards, using shard {shard_id}; each shard has size {shard_size}", level=1)

        assert len(ids) == len(texts)
        if not ids or not texts:
            logger.log(msg=f"No data to process, exiting", level=1)
            exit(0)

    all_token_ids = list(range(tokenizer.get_vocab_size()))
    # all_tokens = np.array(tokenizer.convert_ids_to_tokens(all_token_ids))
    for idx in tqdm(range(0, len(ids), cfg.batch_size)):
        logger.log(msg={"batch": idx}, level=1)
        batch_texts = texts[idx: idx + cfg.batch_size]
        batch_ids = ids[idx: idx + cfg.batch_size]
        batch_tkn = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=cfg.input_max_length,
            return_special_tokens_mask=True,
            return_tensors="pt",
        ).to(device)
        if cfg.fp16:
            with torch.no_grad(), torch.cuda.amp.autocast():
                if cfg.type == "query":
                    batch_output = model.encode_queries(**batch_tkn).to("cpu")
                else:
                    batch_output = model.encode_docs(**batch_tkn).to("cpu")
        else:
            with torch.no_grad():
                if cfg.type == "query":
                    batch_output = model.encode_queries(**batch_tkn).to("cpu")
                else:
                    batch_output = model.encode_docs(**batch_tkn).to("cpu")
        batch_output = batch_output.float()

        id_key = "qid" if cfg.type == "query" else "id"
        if cfg.top_k > 0:
            # do top_k selection in batch
            top_k_res = batch_output.topk(dim=1, k=cfg.top_k, sorted=False)
            batch_values = top_k_res.values
            if cfg.type == "query":
                batch_values = (top_k_res.values * cfg.scale_factor).to(torch.int)

            indices = top_k_res.indices
            # batch_tokens = all_tokens[indices]
            batch_token_ids = all_token_ids[indices]
            for text_id, text, tokens, weights in zip(
                # batch_ids, batch_texts, batch_tokens, batch_values
                batch_ids, batch_texts, batch_token_ids, batch_values
            ):
                mask = weights > 0
                tokens = tokens[mask]
                weights = weights[mask]
                write_to_file(
                    file_writer,
                    {
                        id_key: text_id,
                        "text": text,
                        "vector": dict(zip(tokens.tolist(), weights.tolist())),
                    },
                    cfg.type,
                )
        else:
            # do non-zero selection
            if cfg.type == "query":
                batch_output = (batch_output * cfg.scale_factor).to(torch.int)

            # batch_output = (batch_output * cfg.scale_factor).to(torch.int)
            batch_tokens = [[] for _ in range(len(batch_ids))]
            batch_weights = [[] for _ in range(len(batch_ids))]
            for row_col in batch_output.nonzero():
                row, col = row_col
                batch_tokens[row].append(all_token_ids[col])
                # batch_tokens[row].append(all_tokens[col].item())
                batch_weights[row].append(batch_output[row, col].item())
            for text_id, text, tokens, weights in zip(
                batch_ids, batch_texts, batch_tokens, batch_weights
            ):
                write_to_file(
                    file_writer,
                    {id_key: text_id,
                     "text": text,
                     "vector": dict(zip(tokens, weights))},
                    cfg.type,
                )


if __name__ == "__main__":
    start_time = time.time()
    inference()
    run_time = time.time() - start_time
    logger.log(
        msg=f"Finished! Runing time {str(datetime.timedelta(seconds=666))}", level=1)
