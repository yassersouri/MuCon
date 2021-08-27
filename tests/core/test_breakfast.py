from unittest import TestCase

from yacs.config import CfgNode

from core.config import dataset_cfg
from core.datasets import create_breakfast_dataset


class TestBreakfast(TestCase):
    def test_smoke_test_creation(self):
        cfg = CfgNode()
        cfg.dataset = dataset_cfg
        db = create_breakfast_dataset(cfg=cfg, train=True)
        print(len(db))

        for f in db.feat_file_paths:
            self.assertTrue(f.exists())

        for f in db.gt_file_paths:
            self.assertTrue(f.exists())

        for f in db.tr_file_paths:
            self.assertTrue(f.exists())
