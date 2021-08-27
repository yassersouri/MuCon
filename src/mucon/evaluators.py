import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
from fandak import Evaluator
from fandak.core.evaluators import GeneralEvaluatorResult
from fandak.utils.torch import tensor_to_numpy
from torch.utils.data import DataLoader

from core.datasets.general_dataset import Batch
from core.metrics import Metric
from core.metrics.fully_supervised import Edit, F1Score
from core.metrics.segmentation import (
    MoFAccuracyMetric,
    IoDMetric,
    IoUMetric,
)
from core.metrics.transcript import MatchingScoreMetric, AbsLenDiffMetric
from core.utils import make_same_size_interpolate
from core.viterbi.grammar import SingleTranscriptGrammar
from core.viterbi.length_model import PoissonModel, MultiPoissonModel
from core.viterbi.viterbi import Viterbi
from mucon.models import MuConPredictOut, MuConForwardOut


def create_segmentation_from_segments(
    actions: np.ndarray, lengths: np.ndarray, n_frames: int
) -> np.ndarray:
    lengths = lengths * n_frames
    lengths = np.around(lengths).astype(int)
    lengths[lengths < 0] = 0
    gt = np.repeat(actions, lengths)
    return gt


@dataclass
class MuConEvaluatorResult(GeneralEvaluatorResult):
    y_mof: float
    y_mof_nbg: float
    y_iod: float
    y_iou: float

    s_mof: float
    s_mof_nbg: float
    s_iod: float
    s_iou: float
    s_iod_nbg: float
    s_iou_nbg: float

    s_mat_score: float
    s_len_diff: float

    vit_mof: float
    vit_mof_nbg: float
    vit_iod: float
    vit_iou: float
    vit_iod_nbg: float
    vit_iou_nbg: float

    vit_edit_score: float
    vit_f1_score: Tuple[float, float, float]
    y_edit_score: float
    y_f1_score: Tuple[float, float, float]
    s_edit_score: float
    s_f1_score: Tuple[float, float, float]


# fixme: change this, this is slightly memory inefficient
def one_hot(a: np.ndarray, num_classes: int) -> np.ndarray:
    return np.eye(num_classes)[a.reshape(-1)]


class MuConEvaluator(Evaluator):
    def __init__(self, cfg, test_db, model, device):
        super().__init__(cfg, test_db, model, device)
        self.enable_viterbi = False
        self.viterbi_multi_length = self.cfg.evaluator.viterbi.multi_length
        self.vi_decoder = Viterbi(None, None, frame_sampling=30)
        number_of_action_classes = self.test_db.get_num_classes()
        background_class_ids = self.test_db.background_class_ids

        self.y_mof_metric = MoFAccuracyMetric()
        self.y_mof_nbg_metric = MoFAccuracyMetric(ignore_ids=background_class_ids)
        self.y_iod_metric = IoDMetric()
        self.y_iou_metric = IoUMetric()

        self.s_mof_metric = MoFAccuracyMetric()
        self.s_mof_nbg_metric = MoFAccuracyMetric(ignore_ids=background_class_ids)
        self.s_iod_metric = IoDMetric()
        self.s_iou_metric = IoUMetric()
        self.s_iod_nbg_metric = IoDMetric(ignore_ids=background_class_ids)
        self.s_iou_nbg_metric = IoUMetric(ignore_ids=background_class_ids)

        self.vit_mof_metric = MoFAccuracyMetric()
        self.vit_mof_nbg_metric = MoFAccuracyMetric(ignore_ids=background_class_ids)
        self.vit_iod_metric = IoDMetric()
        self.vit_iou_metric = IoUMetric()
        self.vit_iod_nbg_metric = IoDMetric(ignore_ids=background_class_ids)
        self.vit_iou_nbg_metric = IoUMetric(ignore_ids=background_class_ids)

        self.s_mat_score_metric = MatchingScoreMetric()
        self.s_abs_len_diff_metric = AbsLenDiffMetric()

        self.vit_edit_score_metric = Edit()
        self.y_edit_score_metric = Edit()
        self.s_edit_score_metric = Edit()
        self.vit_f1_score_metric = F1Score()
        self.y_f1_score_metric = F1Score()
        self.s_f1_score_metric = F1Score()

    def viterbi_mode(self, mode=True):
        self.enable_viterbi = mode

    @staticmethod
    def _call_on_metrics(metrics: List[Metric], **kwargs):
        for m in metrics:
            m(**kwargs)

    def batch_eval_calculation(self, batch: Batch, forward_out: MuConForwardOut):
        prediction_out: MuConPredictOut = self.model.predict(batch, forward_out)

        # FIXME: here we should call .predit on the model.
        feature_length = batch.feats.shape[1]
        number_of_action_classes = self.test_db.get_num_classes()

        target_transcript_list = list(tensor_to_numpy(batch.transcript))
        target_labels = tensor_to_numpy(batch.gt_label)
        # -1, because the last one should be EOS.
        predicted_transcript_s_head_list = prediction_out.transcript[:-1]
        predicted_relative_lengths = prediction_out.lengths

        y_head_logits_log_softmaxed = prediction_out.segmentation_logits
        y_head_prediction = prediction_out.segmentation_logits.argmax(dim=1)

        self.s_mat_score_metric.add(
            target_transcript=target_transcript_list,
            predicted_transcript=predicted_transcript_s_head_list,
        )
        self.s_abs_len_diff_metric.add(
            target_transcript=target_transcript_list,
            predicted_transcript=predicted_transcript_s_head_list,
        )

        # Viterbi
        if self.enable_viterbi:
            self.vi_decoder.grammar = SingleTranscriptGrammar(
                predicted_transcript_s_head_list, number_of_action_classes
            )

            if not self.viterbi_multi_length:
                # the code below calculates the average length of each class
                # based on the lengths predicted for each action by the s head.
                actions = one_hot(
                    np.array(predicted_transcript_s_head_list),
                    number_of_action_classes,
                )
                lengths = np.dot(tensor_to_numpy(predicted_relative_lengths), actions)
                lengths *= feature_length
                k = actions.sum(0)
                k[k == 0] = 1
                lengths /= k  # [N], contains **absolute** estimated length, float
                # just setting the value of lengths so they are not zero
                lengths[lengths == 0] = 1

                self.vi_decoder.length_model = PoissonModel(lengths)
                self.vi_decoder.set_multi_length(False)
            else:
                lengths = (
                    tensor_to_numpy(predicted_relative_lengths) * feature_length
                )  # [N]
                self.vi_decoder.length_model = MultiPoissonModel(
                    lengths.tolist(), number_of_action_classes
                )
                self.vi_decoder.set_multi_length(True)

            viterbi_score, viterbi_labels, viterbi_segments = self.vi_decoder.decode(
                tensor_to_numpy(y_head_logits_log_softmaxed)
            )

        s_head_prediction = create_segmentation_from_segments(
            actions=np.array(predicted_transcript_s_head_list),
            lengths=tensor_to_numpy(predicted_relative_lengths),
            n_frames=feature_length,
        )

        # make same size predictions
        s_head_prediction_same_size = make_same_size_interpolate(
            prediction=s_head_prediction,
            target=target_labels,
        )
        y_head_prediction_same_size = make_same_size_interpolate(
            prediction=tensor_to_numpy(y_head_prediction),
            target=target_labels,
        )
        self._call_on_metrics(
            [
                self.s_mof_metric,
                self.s_mof_nbg_metric,
                self.s_iod_metric,
                self.s_iod_nbg_metric,
                self.s_iou_metric,
                self.s_iou_nbg_metric,
                self.s_edit_score_metric,
                self.s_f1_score_metric,
            ],
            targets=target_labels,
            predictions=s_head_prediction_same_size,
        )
        self._call_on_metrics(
            [
                self.y_mof_metric,
                self.y_mof_nbg_metric,
                self.y_iod_metric,
                self.y_iou_metric,
                self.y_edit_score_metric,
                self.y_f1_score_metric,
            ],
            targets=target_labels,
            predictions=y_head_prediction_same_size,
        )

        if self.enable_viterbi:
            vit_prediction_same_size = make_same_size_interpolate(
                prediction=np.array(viterbi_labels),
                target=target_labels,
            )

            self._call_on_metrics(
                [
                    self.vit_mof_metric,
                    self.vit_mof_nbg_metric,
                    self.vit_iod_metric,
                    self.vit_iod_nbg_metric,
                    self.vit_iou_metric,
                    self.vit_iou_nbg_metric,
                    self.vit_edit_score_metric,
                    self.vit_f1_score_metric,
                ],
                targets=target_labels,
                predictions=vit_prediction_same_size,
            )

        if self.enable_viterbi:
            self.vit_segs.append(vit_prediction_same_size)
        else:
            self.vit_segs.append(s_head_prediction_same_size)

        self.y_segs.append(y_head_prediction_same_size)
        self.s_segs.append(s_head_prediction_same_size)
        self.s_lens.append(tensor_to_numpy(predicted_relative_lengths))
        self.s_transcript.append(predicted_transcript_s_head_list)
        self.target_segs.append(target_labels)
        self.target_transcripts.append(target_transcript_list)

    def create_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_db,
            batch_size=1,
            shuffle=False,
            num_workers=self.cfg.system.num_workers,
            collate_fn=self.test_db.collate_fn,
            pin_memory=True,
        )

    def on_finish_eval(self) -> MuConEvaluatorResult:
        # noinspection PyAttributeOutsideInit
        self.to_save = {
            "y_segs": self.y_segs,
            "s_segs": self.s_segs,
            "vit_segs": self.vit_segs,
            "s_lens": self.s_lens,
            "s_transcript": self.s_transcript,
            "target_segs": self.target_segs,
            "target_transcripts": self.target_transcripts,
        }

        return MuConEvaluatorResult(
            s_mat_score=self.s_mat_score_metric.summary(),
            s_len_diff=self.s_abs_len_diff_metric.summary(),
            s_mof=self.s_mof_metric.summary(),
            s_mof_nbg=self.s_mof_nbg_metric.summary(),
            s_iod=self.s_iod_metric.summary(),
            s_iod_nbg=self.s_iod_nbg_metric.summary(),
            s_iou=self.s_iou_metric.summary(),
            s_iou_nbg=self.s_iou_nbg_metric.summary(),
            y_mof=self.y_mof_metric.summary(),
            y_mof_nbg=self.y_mof_nbg_metric.summary(),
            y_iod=self.y_iod_metric.summary(),
            y_iou=self.y_iou_metric.summary(),
            vit_mof=self.vit_mof_metric.summary(),
            vit_mof_nbg=self.vit_mof_nbg_metric.summary(),
            vit_iod=self.vit_iod_metric.summary(),
            vit_iod_nbg=self.vit_iod_nbg_metric.summary(),
            vit_iou=self.vit_iou_metric.summary(),
            vit_iou_nbg=self.vit_iou_nbg_metric.summary(),
            y_edit_score=self.y_edit_score_metric.summary(),
            y_f1_score=self.y_f1_score_metric.summary(),
            s_edit_score=self.s_edit_score_metric.summary(),
            s_f1_score=self.s_f1_score_metric.summary(),
            vit_edit_score=self.vit_edit_score_metric.summary(),
            vit_f1_score=self.vit_f1_score_metric.summary(),
        )

    def get_name(self) -> str:
        return self.name

    # noinspection PyAttributeOutsideInit
    def set_name(self, name: str):
        self.name = name

    # noinspection PyAttributeOutsideInit
    def set_checkpointing_folder(self, folder: Path):
        self.checkpointing_folder = folder

    def save_stuff(self):
        with open(str(self.checkpointing_folder / f"data_{self.name}.pkl"), "wb") as f:
            pickle.dump(self.to_save, f)

    # noinspection PyAttributeOutsideInit
    def on_start_eval(self):
        # setting teacher forcing to false
        # should not happen in we want alignment performance.
        self.model.set_teacher_forcing(False)

        # will be used to store results and save them to disk.
        self.y_segs = []
        self.s_segs = []
        self.vit_segs = []
        self.s_lens = []
        self.s_transcript = []
        self.target_segs = []
        self.target_transcripts = []

        # let's reset all the metrics specified in the init function.
        for attrname in dir(self):
            attr = self.__getattribute__(attrname)
            if isinstance(attr, Metric):
                attr.reset()


class MuConAlignmentEvaluator(MuConEvaluator):
    def on_start_eval(self):
        super().on_start_eval()
        # Because we are doing alignment here.
        self.model.set_teacher_forcing(True)
