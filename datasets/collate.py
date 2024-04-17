"""
Inspired from https://github.com/pytorch/pytorch/blob/master/torch/utils/data/_utils/collate.py
"""
import re

import numpy as np
from torch.utils.data._utils.collate import default_collate


np_str_obj_array_pattern = re.compile(r"[SaUO]")

 

def meshreg_collate(batch, extend_queries=None):
    """
    Collate function, duplicating the items in extend_queries along the
    first dimension so that they all have the same length.
    Typically applies to faces and vertices, which have different sizes
    depending on the object.
    """

    pop_queries = []
    for poppable_query in extend_queries:
        if poppable_query in batch[0]:
            pop_queries.append(poppable_query)

    # Remove fields that don't have matching sizes
    for pop_query in pop_queries:
        max_size = max([sample[pop_query].shape[0] for sample in batch])
        for sample in batch:
            pop_value = sample[pop_query]
            # Repeat vertices so all have the same number
            pop_value = np.concatenate([pop_value] * int(max_size / pop_value.shape[0] + 1))[:max_size]
            sample[pop_query] = pop_value
    batch = default_collate(batch)
    return batch

 

def seq_extend_flatten_collate(seq, extend_queries=None):
    batch=[]    
    seq_len = len(seq[0])#len(seq) is batch size, seq_len is num frames per sample

    for sample in seq:
        for seq_idx in range(seq_len):
            batch.append(sample[seq_idx])
    return meshreg_collate(batch,extend_queries)


 