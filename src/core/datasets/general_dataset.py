import random
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import torch
from fandak import Dataset
from fandak.core.datasets import GeneralBatch
from torch import Tensor
from yacs.config import CfgNode

from core.datasets.utils import create_tf_input, create_tf_target
from core.viterbi.grammar import ModifiedPathGrammar


@dataclass(repr=False)
class Batch(GeneralBatch):
    """
    T: the video length
    D: the feat dim
    M: number of actions in the video.
    N: length of the transcript
    """

    feats: Tensor  # [1 x T x D] float
    gt_label: Tensor  # [T] long
    transcript: Tensor  # [N] long, 0 <= each item < M
    transcript_tf_input: Tensor  # [N + 1] long: equal to BOS + transcript
    # 0 <= each item < M + 2, +2 because of EOS and BOS in decoder dictionary
    transcript_tf_target: Tensor  # [N + 1] long: equal to transcript + EOS
    # 0 <= each item < M + 2, +2 because of EOS and BOS in decoder dictionary
    video_name: str  # the name of the video


@dataclass(repr=False)
class FullySupervisedBatch(Batch):
    absolute_lengths: Tensor  # [N]  float


@dataclass(repr=False)
class MixedSupervisionBatch(FullySupervisedBatch):
    fully_supervised: bool  # whether full supervision should be used or not.


class GeneralDataset(Dataset):
    def __init__(
        self,
        cfg: CfgNode,
        root: Path,
        relative_path_to_list: Path = "split1.train",
        relative_path_to_mapping: Path = "mapping.txt",
        feat_dim: int = -1,
        relative_path_to_train_list: Path = None,
    ):
        """
        relative_path_to_train_list is added because sometimes we need to do full decoding,
        which means we need access to the full set of training transcripts.
        """
        super().__init__(cfg)
        self.root = root
        self.file_list = root / relative_path_to_list
        if relative_path_to_train_list is not None:
            train_file_list = root / relative_path_to_train_list
        else:
            train_file_list = None
        self.mapping_file = root / relative_path_to_mapping
        self.end_class_id = 0
        self.mof_eval_ignore_classes = []
        self.background_class_ids = [0]

        # following are defaults, should be set
        self.feat_dim = feat_dim
        self.convenient_name = None
        self.split = -1
        self.max_transcript_length = 100

        with open(self.file_list) as f:
            self.file_names = [x.strip() for x in f if len(x.strip()) > 0]

        self.action_id_to_name = {}
        self.action_name_to_id = {}
        if self.mapping_file is not None:
            with open(self.mapping_file) as f:
                the_mapping = [tuple(x.strip().split()) for x in f]

                for (i, l) in the_mapping:
                    self.action_id_to_name[int(i)] = l
                    self.action_name_to_id[l] = int(i)

        self.num_actions = len(self.action_id_to_name)

        self.feat_file_paths = [
            self.root / "features" / f"{x}.npy" for x in self.file_names
        ]
        self.gt_file_paths = [
            self.root / "labels" / f"{x}.npy" for x in self.file_names
        ]
        self.tr_file_paths = [
            self.root / "transcripts" / f"{x}.npy" for x in self.file_names
        ]

        self.eos_token = "_EOS_"  # end of sentence
        self.sos_token = "_SOS_"  # start of sentence
        self.eos_token_id = self.num_actions  # = M, 48 for breakfast
        self.sos_token_id = self.num_actions + 1  # = M + 1, 49 for breakfast
        self.action_id_to_name[self.eos_token_id] = self.eos_token
        self.action_name_to_id[self.eos_token] = self.eos_token_id
        self.action_id_to_name[self.sos_token_id] = self.sos_token
        self.action_name_to_id[self.sos_token] = self.sos_token_id

        # loading the training transcripts
        if train_file_list is not None:
            with open(train_file_list) as f:
                train_file_names = [x.strip() for x in f if len(x.strip()) > 0]
            tr_train_file_paths = [
                self.root / "transcripts" / f"{x}.npy" for x in train_file_names
            ]
            training_transcripts = set()
            for tr_file_path in tr_train_file_paths:
                transcript = tuple(np.load(str(tr_file_path)))
                training_transcripts.add(transcript)

            self.training_transcripts_list = []
            for t in training_transcripts:
                self.training_transcripts_list.append(list(t))

            self.training_path_grammar = ModifiedPathGrammar(
                transcripts=self.training_transcripts_list, num_classes=self.num_actions
            )

    def get_num_classes(self) -> int:
        return self.num_actions

    def __len__(self) -> int:
        return len(self.feat_file_paths)

    def __getitem__(self, item: int) -> Batch:
        feat_file_path = str(self.feat_file_paths[item])
        gt_file_path = str(self.gt_file_paths[item])
        tr_file_path = str(self.tr_file_paths[item])

        numpy_feats = np.load(feat_file_path)  # T x D
        vid_feats = torch.tensor(numpy_feats).float()  # T x D
        vid_feats.unsqueeze_(0)  # 1 x T x D

        gt_labels = np.load(gt_file_path)  # T
        transcript = np.load(tr_file_path)  # N

        gt_labels = torch.tensor(gt_labels).long()  # T
        transcript = torch.tensor(transcript).long()  # N

        transcript_tf_input = torch.tensor(
            create_tf_input(transcript, sos_i=self.sos_token_id)
        ).long()  # [N + 1]
        transcript_tf_target = torch.tensor(
            create_tf_target(transcript, eos_i=self.eos_token_id)
        ).long()  # [N + 1]

        return Batch(
            feats=vid_feats,
            gt_label=gt_labels,
            transcript=transcript,
            transcript_tf_input=transcript_tf_input,
            transcript_tf_target=transcript_tf_target,
            video_name=self.file_names[item],
        )

    def collate_fn(self, items: List[Batch]) -> Batch:
        # We assume batch_size = 1
        assert len(items) == 1

        return items[0]


class GeneralFullySupervisedDataset(GeneralDataset):
    def __init__(
        self,
        cfg: CfgNode,
        root: Path,
        relative_path_to_list: Path = "split1.train",
        relative_path_to_mapping: Path = "mapping.txt",
        feat_dim: int = -1,
    ):
        super().__init__(
            cfg, root, relative_path_to_list, relative_path_to_mapping, feat_dim
        )
        self.len_file_paths = [
            self.root / "lengths" / f"{x}.npy" for x in self.file_names
        ]

    def __getitem__(self, item: int) -> FullySupervisedBatch:
        the_item = super().__getitem__(item)

        len_file_path = str(self.len_file_paths[item])
        absolute_lengths = np.load(len_file_path)  # [N]
        absolute_lengths = torch.tensor(absolute_lengths, dtype=torch.float32)

        return FullySupervisedBatch(
            feats=the_item.feats,
            gt_label=the_item.gt_label,
            transcript=the_item.transcript,
            transcript_tf_target=the_item.transcript_tf_target,
            transcript_tf_input=the_item.transcript_tf_input,
            video_name=the_item.video_name,
            absolute_lengths=absolute_lengths,
        )


class GeneralMixedSupervisionDataset(GeneralFullySupervisedDataset):
    def __init__(
        self,
        cfg: CfgNode,
        root: Path,
        full_supervision_percentage: float,
        relative_path_to_list: Path = "split1.train",
        relative_path_to_mapping: Path = "mapping.txt",
        feat_dim: int = -1,
    ):
        super().__init__(
            cfg, root, relative_path_to_list, relative_path_to_mapping, feat_dim
        )
        assert 0.0 < full_supervision_percentage < 100.0
        self.full_supervision_percentage = full_supervision_percentage

        self.number_of_full_supervision_examples = min(
            len(self.feat_file_paths),
            max(
                1,
                int(
                    round(
                        (
                            len(self.feat_file_paths)
                            * self.full_supervision_percentage
                            / 100.0
                        )
                    )
                ),
            ),
        )
        self.is_it_supervised = [False] * len(self.feat_file_paths)
        self.is_it_supervised[: self.number_of_full_supervision_examples] = [
            True for _ in range(self.number_of_full_supervision_examples)
        ]
        random.seed(
            f"{self.cfg.system.seed}-{self.number_of_full_supervision_examples}"
        )
        random.shuffle(self.is_it_supervised)

    def __getitem__(self, item: int) -> MixedSupervisionBatch:
        fully_supervised_batch = super().__getitem__(item)
        is_this_supervised = self.is_it_supervised[item]

        return MixedSupervisionBatch(
            feats=fully_supervised_batch.feats,
            gt_label=fully_supervised_batch.gt_label,
            transcript=fully_supervised_batch.transcript,
            transcript_tf_input=fully_supervised_batch.transcript_tf_input,
            transcript_tf_target=fully_supervised_batch.transcript_tf_target,
            video_name=fully_supervised_batch.video_name,
            absolute_lengths=fully_supervised_batch.absolute_lengths,
            fully_supervised=is_this_supervised,
        )
