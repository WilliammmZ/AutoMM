import itertools
import logging
import numpy as np
import torch
import torch.utils.data as data
from torch.utils.data.sampler import Sampler

__all__ = ["ToIterableDataset"]

logger = logging.getLogger(__name__)

def _shard_iterator_dataloader_worker(iterable):
    # Shard the iterable if we're currently inside pytorch dataloader worker.
    worker_info = data.get_worker_info()
    if worker_info is None or worker_info.num_workers == 1:
        # do nothing
        yield from iterable
    else:
        yield from itertools.islice(iterable, worker_info.id, None, worker_info.num_workers)


class ToIterableDataset(data.IterableDataset):
    """
    Convert an old indices-based (also called map-style) dataset
    to an iterable-style dataset.
    """

    def __init__(self, dataset: data.Dataset, sampler: Sampler, shard_sampler: bool = True):
        """
        Args:
            dataset: an old-style dataset with ``__getitem__``
            sampler: a cheap iterable that produces indices to be applied on ``dataset``.
            shard_sampler: whether to shard the sampler based on the current pytorch data loader
                worker id. When an IterableDataset is forked by pytorch's DataLoader into multiple
                workers, it is responsible for sharding its data based on worker id so that workers
                don't produce identical data.

                Most samplers (like our TrainingSampler) do not shard based on dataloader worker id
                and this argument should be set to True. But certain samplers may be already
                sharded, in that case this argument should be set to False.
        """
        assert not isinstance(dataset, data.IterableDataset), dataset
        assert isinstance(sampler, Sampler), sampler
        self.dataset = dataset
        self.sampler = sampler
        self.shard_sampler = shard_sampler

    def __iter__(self):
        if not self.shard_sampler:
            sampler = self.sampler
        else:
            # With map-style dataset, `DataLoader(dataset, sampler)` runs the
            # sampler in main process only. But `DataLoader(ToIterableDataset(dataset, sampler))`
            # will run sampler in every of the N worker. So we should only keep 1/N of the ids on
            # each worker. The assumption is that sampler is cheap to iterate so it's fine to
            # discard ids in workers.
            sampler = _shard_iterator_dataloader_worker(self.sampler)
        for idx in sampler:
            yield self.dataset[idx]

    def __len__(self):
        return len(self.sampler)

