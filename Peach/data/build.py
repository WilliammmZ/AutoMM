import torch
import torch.utils.data as torchdata
from typing import Any, Callable, Dict, List, Optional, Union

from Peach.config import configurable
from Peach.utils.logger import _log_api_usage
from Peach.utils.comm import get_world_size
from Peach.utils.env import seed_all_rng

from .datasets import build_dataset
from .samplers import TrainingSampler, InferenceSampler
from .common import ToIterableDataset


def get_dataset_list(
    names,
    cfg,
):
    """
    Load and prepare dataset dicts for video task and other task.

    Args:
        names (str or list[str]): a dataset name or a list of dataset names
        filter_empty (bool): whether to filter out images without instance annotations
    
    Returns:
        list[dict]: a list of dicts following the standard dataset dict format.
    """
    if isinstance(names, str):
        names = [names]
    assert len(names), names
    
    dataset_list = []
    for dataset_name in names:
        train_dataset = build_dataset(dataset_name, cfg)
        dataset_list.append(train_dataset)

    for dataset_name, dataset in zip(names, dataset_list):
        assert len(dataset), "Dataset '{}' is empty!".format(dataset_name)

    # dataset_dicts = list(itertools.chain.from_iterable(dataset_dicts))

    assert len(dataset_list), "No valid data found in {}.".format(",".join(names))
    return dataset_list


def _train_loader_from_config(cfg, *, dataset=None, sampler=None):
    if dataset == None:
        dataset_list = get_dataset_list(names=cfg.DATASETS.TRAIN, cfg=cfg)
        _log_api_usage("dataset." + cfg.DATASETS.TRAIN[0])
        if len(dataset_list) == 1:
            dataset = dataset_list[0]
        else:
            dataset = torchdata.ConcatDataset(dataset_list)

    if sampler is None:
        sampler = TrainingSampler(len(dataset))

    return {
        "dataset": dataset,
        "sampler": sampler,
        "total_batch_size": cfg.SOLVER.SAMPLE_PER_BATCH,
        "num_workers": cfg.DATALOADER.NUM_WORKERS,
    }


@configurable(from_config=_train_loader_from_config)
def build_train_loader(
    dataset,
    *,
    sampler=None,
    total_batch_size,
    num_workers=0,
    collate_fn=None,
):
    """
    Build a dataloader with some default features.

    Args:
        dataset (list or torch.utils.data.Dataset): a list of dataset,
            or a pytorch dataset (either map-style or iterable).

        sampler (torch.utils.data.sampler.Sampler or None): a sampler that produces
            indices to be applied on ``dataset``.

        total_batch_size (int): total batch size across all workers.
        [not implement]aspect_ratio_grouping (bool): whether to group images with similar
            aspect ratio for efficiency. When enabled, it requires each
            element in dataset be a dict with keys "width" and "height".
        num_workers (int): number of parallel data loading workers
        collate_fn: a function that determines how to do batching, same as the argument of
            `torch.utils.data.DataLoader`. Defaults to do no collation and return a list of
            data. No collation is OK for small batch size and simple data structures.
            If your batch size is large and each sample contains too many small tensors,
            it's more efficient to collate them in data loader.

    Returns:
        torch.utils.data.DataLoader:
            a dataloader. 
    """
    world_size = get_world_size()
    assert (
        total_batch_size > 0 and total_batch_size % world_size == 0
    ), "Total train batch size ({}) must be divisible by the number of gpus ({}).".format(
        total_batch_size, world_size
    )

    if isinstance(dataset, torchdata.IterableDataset):
        assert sampler is None, "sampler must be None if dataset is IterableDataset"
    else:
        dataset = ToIterableDataset(dataset, sampler)

    batch_size = total_batch_size // world_size

    return torchdata.DataLoader(
            dataset,
            batch_size=batch_size,
            drop_last=True,
            num_workers=num_workers,
            collate_fn=collate_fn,
            worker_init_fn=worker_init_reset_seed,
        )


def _test_loader_from_config(cfg, dataset_name):
    """
    Uses the given `dataset_name` argument (instead of the names in cfg), because the
    standard practice is to evaluate each test set individually (not combining them).
    """

    dataset_list = get_dataset_list(dataset_name, cfg=cfg)
    _log_api_usage("dataset." + cfg.DATASETS.TEST[0])
    dataset = dataset_list[0]
    
    world_size = get_world_size()
    total_batch_size = cfg.DATASETS.TEST_BATCH
    assert (
        total_batch_size > 0 and total_batch_size % world_size == 0
    ), "Total test batch size ({}) must be divisible by the number of gpus ({}).".format(
        total_batch_size, world_size
    )
    batch_size = total_batch_size // world_size
    
    return {
        "dataset": dataset,
        "num_workers": cfg.DATALOADER.NUM_WORKERS,
        "sampler": InferenceSampler(len(dataset)),
        "batch_size":batch_size,
    }


@configurable(from_config=_test_loader_from_config)
def build_test_loader(
    dataset: Union[List[Any], torchdata.Dataset],
    *,
    sampler: Optional[torchdata.Sampler] = None,
    batch_size: int = 1,
    num_workers: int = 0,
    collate_fn: Optional[Callable[[List[Any]], Any]] = None,
) -> torchdata.DataLoader:
    
    return torchdata.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        drop_last=False,
        num_workers=num_workers,
        # collate_fn=trivial_batch_collator if collate_fn is None else collate_fn,
    )


def trivial_batch_collator(batch):
    """
    A batch collator that does nothing.
    """
    return batch


def worker_init_reset_seed(worker_id):
    initial_seed = torch.initial_seed() % 2**31
    seed_all_rng(initial_seed + worker_id)
