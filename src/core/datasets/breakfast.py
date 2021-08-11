from pathlib import Path

from yacs.config import CfgNode

from .general_dataset import (
    GeneralDataset,
    GeneralFullySupervisedDataset,
    GeneralMixedSupervisionDataset,
)

POSSIBLE_SPLITS = [1, 2, 3, 4]
MAX_TRANSCRIPT_LENGTH = 30
KINETICS_FEAT_NAME = "i3d"
IDT_FEAT_NAME = "idt"
COIN_FEAT_NAME = "coin"
I3DPCA_FEAT_NAME = "i3dpca"
CONCAT_FEAT_NAME = "concat"  # concatenation of idt and i3dpca
FEAT_DIM_MAPPING = {
    KINETICS_FEAT_NAME: 2048,
    IDT_FEAT_NAME: 64,
    COIN_FEAT_NAME: 512,
    I3DPCA_FEAT_NAME: 64,
    CONCAT_FEAT_NAME: 128
}


def create_breakfast_dataset(cfg: CfgNode, train: bool = True) -> GeneralDataset:
    split = cfg.dataset.split
    feat_name = cfg.dataset.feat_name
    root = Path(cfg.dataset.root)
    set_name = "train" if train else "test"
    assert split in POSSIBLE_SPLITS
    db_path = root / "{db_name}_{feat_name}".format(
        db_name="breakfast", feat_name=feat_name
    )

    file_list = db_path / "split{split}.{set_name}".format(
        split=split, set_name=set_name
    )
    train_file_list = db_path / "split{split}.{set_name}".format(
        split=split, set_name="train"
    )

    mapping = db_path / cfg.dataset.mapping_file_name

    db = GeneralDataset(
        cfg=cfg,
        root=db_path,
        relative_path_to_list=file_list,
        relative_path_to_mapping=mapping,
        feat_dim=FEAT_DIM_MAPPING[feat_name],
        relative_path_to_train_list=train_file_list,
    )
    db.end_class_id = 0
    db.mof_eval_ignore_classes = []
    db.background_class_ids = [0]
    db.convenient_name = "breakfast_split{split}_{set_name}".format(
        split=split, set_name=set_name
    )
    db.split = split
    db.max_transcript_length = MAX_TRANSCRIPT_LENGTH

    return db


def create_fully_supervised_breakfast_dataset(
    cfg: CfgNode, train: bool = True
) -> GeneralFullySupervisedDataset:
    split = cfg.dataset.split
    feat_name = cfg.dataset.feat_name
    root = Path(cfg.dataset.root)
    set_name = "train" if train else "test"
    assert split in POSSIBLE_SPLITS
    db_path = root / "{db_name}_{feat_name}".format(
        db_name="breakfast", feat_name=feat_name
    )

    file_list = db_path / "split{split}.{set_name}".format(
        split=split, set_name=set_name
    )

    mapping = db_path / cfg.dataset.mapping_file_name

    db = GeneralFullySupervisedDataset(
        cfg=cfg,
        root=db_path,
        relative_path_to_list=file_list,
        relative_path_to_mapping=mapping,
        feat_dim=FEAT_DIM_MAPPING[feat_name],
    )
    db.end_class_id = 0
    db.mof_eval_ignore_classes = []
    db.background_class_ids = [0]
    db.convenient_name = "fully_supervised_breakfast_split{split}_{set_name}".format(
        split=split, set_name=set_name
    )
    db.split = split
    db.max_transcript_length = MAX_TRANSCRIPT_LENGTH

    return db


def create_mixed_supervision_breakfast_dataset(
    cfg: CfgNode, train: bool = True
) -> GeneralMixedSupervisionDataset:
    split = cfg.dataset.split
    feat_name = cfg.dataset.feat_name
    root = Path(cfg.dataset.root)
    full_supervision_percentage = cfg.dataset.mixed.full_supervision_percentage

    set_name = "train" if train else "test"
    assert split in POSSIBLE_SPLITS
    db_path = root / "{db_name}_{feat_name}".format(
        db_name="breakfast", feat_name=feat_name
    )

    file_list = db_path / "split{split}.{set_name}".format(
        split=split, set_name=set_name
    )

    mapping = db_path / cfg.dataset.mapping_file_name

    db = GeneralMixedSupervisionDataset(
        cfg=cfg,
        root=db_path,
        relative_path_to_list=file_list,
        relative_path_to_mapping=mapping,
        feat_dim=FEAT_DIM_MAPPING[feat_name],
        full_supervision_percentage=full_supervision_percentage,
    )
    db.end_class_id = 0
    db.mof_eval_ignore_classes = []
    db.background_class_ids = [0]
    db.convenient_name = "mixed_supervision_percentage_{percentage}_breakfast_split{split}_{set_name}".format(
        split=split, set_name=set_name, percentage=full_supervision_percentage
    )
    db.split = split
    db.max_transcript_length = MAX_TRANSCRIPT_LENGTH

    return db
