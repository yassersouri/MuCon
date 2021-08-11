from yacs.config import CfgNode as CN
import os

dataset_cfg = CN()
dataset_cfg.root = os.path.expanduser("~/work/MuCon/datasets")
dataset_cfg.name = "breakfast"  # "breakfast", "hollywood"
dataset_cfg.feat_name = "i3d"  # "i3d", "idt", "i3dpca", "concat"
dataset_cfg.mapping_file_name = "mapping.txt"
dataset_cfg.split = 1
dataset_cfg.mixed = CN()
dataset_cfg.mixed.full_supervision_percentage = 50.0


system_cfg = CN()
system_cfg.device = "cuda"
system_cfg.num_workers = 2
system_cfg.seed = 1
