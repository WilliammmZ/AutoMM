from Peach.utils.registry import Registry

DATASETS_REGISTRY = Registry("DATASETS")  # noqa F401 isort:skip
DATASETS_REGISTRY.__doc__ = """
Registry for Datasets.
"""
def build_dataset(datasetname, cfg):
    dataset = DATASETS_REGISTRY.get(datasetname)(cfg)
    return dataset
