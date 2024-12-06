import random
from lsr.utils.dataset_utils import (
    read_collection,
    read_qrels,
    read_queries,
    read_triplets,
)
from typing import List
from torch.utils.data import Dataset
from collections import defaultdict


class MultipleDatasets(Dataset):
    """
    A dataset that concatenates multiple datasets.
    Specifically designed for mMARCO cross-lingual:
    Queries are mapping to passages from all languages (doc id are prepanded with language id)
    TODO: We do not have the version of the same queries in different languages yet, but maybe
    it is beneficial for NeuCLIR tasks.
    """
    def __init__(
        self,
        datasets: List[Dataset],
    ):
        self.datasets = datasets
        self.dataset_keys = [i for i in range(len(datasets))]

        self.q_dict = self.process_q_dict()
        self.qids = list(self.q_dict.keys())

        # self.qids = list(self.q_dict.keys())
        train_group_sizes = {ds.train_group_size for ds in datasets}
        assert len(train_group_sizes) == 1, "All datasets must have the same train_group_size"
        self.train_group_size = train_group_sizes.pop()

    def process_q_dict(self):
        """ merge q_dicts from different datasets """
        q_dict = {}
        for ds in self.datasets:
            q_dict.update(ds.q_dict)
        return q_dict

    def process_query2pos_and_query2neg(self):
        """
        merge query2pos and query2neg from different datasets,
        where the doc ids are refommated 
        """
        query2pos = defaultdict(list)
        query2neg = defaultdict(list)

        for ds_id, ds in zip(self.dataset_keys, self.datasets):
            for qid in ds.query2pos:
                pos_ids = ds.query2pos[qid]
                neg_ids = ds.query2neg[qid]
                query2pos[qid].extend([self._format_doc_id(ds_id, pos_id) for pos_id in pos_ids])
                query2neg[qid].extend([self._format_doc_id(ds_id, neg_id) for neg_id in neg_ids])
        return query2pos, query2neg

    def _format_doc_id(self, ds_id, doc_id):
        return f"{ds_id}:{doc_id}"

    def parse_doc_id(self, doc_id):
        """ return ds_id, doc_id """
        return doc_id.split(":")

    def __len__(self):
        return len(self.q_dict)

    def __getitem__(self, item):
        qid = self.qids[item]
        query = self.q_dict[qid]
        pos_id = random.choice(self.query2pos[qid])

        # different from the single dataset version, here we need to parse the doc_id
        ds_id, pos_id = self.parse_doc_id(pos_id)
        pos_psg = self.datasets[ds_id].docs_dict[pos_id]
        # end of the difference

        group_batch = []
        group_batch.append(pos_psg)
        if len(self.query2neg[qid]) < self.train_group_size - 1:
            negs = random.choices(self.query2neg[qid], k=self.train_group_size - 1)
        else:
            negs = random.sample(self.query2neg[qid], k=self.train_group_size - 1)
        group_batch.extend([self.docs_dict[neg] for neg in negs])
        return query, group_batch
