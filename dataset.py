import numpy as np
import torch

from torch.utils.data import Dataset
from util import flatten_query, list2tuple, parse_time, set_global_seed, eval_tuple, flatten

class TestDataset(Dataset):
    def __init__(self, queries, nentity, nrelation):
        # queries is a list of (query, query_structure) pairs
        self.len = len(queries)
        self.queries = queries
        self.nentity = nentity
        self.nrelation = nrelation

    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        query = self.queries[idx][0]
        query_structure = self.queries[idx][1]
        return flatten(query), query, query_structure
    
    @staticmethod
    def collate_fn(data):
        query = [_[0] for _ in data]
        query_unflatten = [_[1] for _ in data]
        query_structure = [_[2] for _ in data]
        return query, query_unflatten, query_structure
    
# rewriting
class TestRewriteDataset(Dataset):

    def __init__(self, queries, nentity, nrelation, specializations):
        # queries is a list of (query, query_structure) pairs
        self.len = len(queries)
        self.queries = queries
        self.nentity = nentity
        self.nrelation = nrelation
        self.specializations = specializations
        #print('SPE: ')
        #print(specializations)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        query = self.queries[idx][0]
        query_structure = self.queries[idx][1]

        spe_queries = self.specializations[query]
        spe_list = []
        for (spe_query, spe_conf, spe_structure) in spe_queries:
            spe_list.append((flatten(spe_query), spe_conf, spe_structure))

        return flatten(query), query, query_structure, spe_list

    @staticmethod
    def collate_fn(data):
        query = [_[0] for _ in data]
        query_unflatten = [_[1] for _ in data]
        query_structure = [_[2] for _ in data]
        specializations = [_[3] for _ in data]

        return query, query_unflatten, query_structure, specializations