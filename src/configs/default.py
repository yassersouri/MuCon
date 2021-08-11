from yacs.config import CfgNode as CN
from core.config import dataset_cfg, system_cfg

_C = CN()

_C.system = system_cfg
_C.dataset = dataset_cfg


def get_cfg_defaults():
    return _C.clone()
