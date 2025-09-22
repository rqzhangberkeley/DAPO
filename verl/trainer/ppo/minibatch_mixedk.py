# verl/trainer/ppo/minibatch_mixedk.py
from __future__ import annotations
from typing import Iterator, List, Sequence, Optional
import numpy as np
from random import Random


def make_minibatch_plan(
    uids: Sequence,
    rows_per_minibatch: int,
    rows_per_microbatch: Optional[int] = None,
    shuffle: bool = True,
    seed: int = 0,
    keep_uid_whole: bool = False,
) -> Iterator[List[List[int]]]:
    """
    Yield mini-batches as lists of micro-batches of row indices.
    We split by ACTUAL row count; never reshape to [B, N].

    keep_uid_whole=True keeps all rows of a UID in the same micro-batch
    (useful if the actor computes RLOO inside the minibatch).
    """
    n = len(uids)
    order = list(range(n))
    if shuffle:
        rng = Random(seed); rng.shuffle(order)

    minis: List[List[int]] = []
    if keep_uid_whole:
        uid2idxs = {}
        for i in order:
            uid2idxs.setdefault(uids[i], []).append(i)
        cur: List[int] = []
        for arr in uid2idxs.values():
            if rows_per_minibatch and len(cur) + len(arr) > rows_per_minibatch:
                if cur:
                    minis.append(cur); cur = []
            cur += arr
            if rows_per_minibatch and len(cur) >= rows_per_minibatch:
                minis.append(cur); cur = []
        if cur:
            minis.append(cur)
    else:
        # free split by row count
        for i in range(0, n, rows_per_minibatch):
            minis.append(order[i : i + rows_per_minibatch])

    for mb in minis:
        if not rows_per_microbatch or rows_per_microbatch <= 0:
            yield [mb]
        else:
            yield [mb[i : i + rows_per_microbatch] for i in range(0, len(mb), rows_per_microbatch)]
