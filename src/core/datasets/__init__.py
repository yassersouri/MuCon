from yacs.config import CfgNode

from .breakfast import (
    create_breakfast_dataset,
    create_fully_supervised_breakfast_dataset,
    create_mixed_supervision_breakfast_dataset,
)
from .general_dataset import (
    GeneralDataset,
    GeneralFullySupervisedDataset,
    Batch,
    GeneralMixedSupervisionDataset,
)


def handel_dataset(cfg: CfgNode, train: bool) -> GeneralDataset:
    dataset_name = cfg.dataset.name
    if dataset_name == "breakfast":
        return create_breakfast_dataset(cfg=cfg, train=train)
    else:
        raise Exception(f"Invalid dataset name. ({dataset_name})")


def handel_fully_supervised_dataset(
    cfg: CfgNode, train: bool
) -> GeneralFullySupervisedDataset:
    if cfg.dataset.name == "breakfast":
        return create_fully_supervised_breakfast_dataset(cfg=cfg, train=train)
    else:
        raise Exception("Invalid dataset name.")


def handel_mixed_supervision_dataset(
    cfg: CfgNode, train: bool
) -> GeneralMixedSupervisionDataset:
    if cfg.dataset.name == "breakfast":
        return create_mixed_supervision_breakfast_dataset(cfg=cfg, train=train)
    else:
        raise Exception("Invalid dataset name.")
