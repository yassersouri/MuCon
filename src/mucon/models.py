import math
from dataclasses import dataclass
from typing import List

import torch
import torch.nn as nn
from fandak import GeneralLoss, GeneralForwardOut, Model
from fandak.utils.torch import tensor_to_numpy
from torch import Tensor
from torch.nn import functional as F
from yacs.config import CfgNode

from core.datasets.general_dataset import (
    FullySupervisedBatch,
    MixedSupervisionBatch,
    Batch,
)
from core.modules.temporal import WaveNetBlock, MSTCNPPFirstStage, NoFt
from mucon.masks import project_lengths_softmax, create_masks


def rand_t(*sz):
    return torch.randn(sz) / math.sqrt(sz[0])


def rand_p(*sz):
    return nn.Parameter(rand_t(*sz), requires_grad=True)


def create_model(
    cfg: CfgNode,
    num_classes: int,
    max_decoding_steps: int,
    input_feature_size: int,
) -> "MuCon":
    model_name = cfg.model.name

    if model_name == "mucon":
        return MuCon(
            cfg=cfg,
            input_feature_size=input_feature_size,
            num_classes=num_classes,
            max_decoding_steps=max_decoding_steps,
        )
    else:
        raise Exception("Invalid model name")


def create_fully_supervised_model(
    cfg: CfgNode,
    num_classes: int,
    max_decoding_steps: int,
    input_feature_size: int,
) -> "MuConFullySupervised":
    model_name = cfg.model.name
    if model_name == "mucon":
        return MuConFullySupervised(
            cfg=cfg,
            input_feature_size=input_feature_size,
            num_classes=num_classes,
            max_decoding_steps=max_decoding_steps,
        )
    else:
        raise Exception("Invalid model name")


def create_mixed_supervision_model(
    cfg: CfgNode,
    num_classes: int,
    max_decoding_steps: int,
    input_feature_size: int,
) -> "MuConMixedSupervision":
    model_name = cfg.model.name
    if model_name == "mucon":
        return MuConMixedSupervision(
            cfg=cfg,
            input_feature_size=input_feature_size,
            num_classes=num_classes,
            max_decoding_steps=max_decoding_steps,
        )
    else:
        raise Exception("Invalid model name")


@dataclass(repr=False)
class MuConLoss(GeneralLoss):
    transcript_loss: Tensor
    mucon_loss: Tensor
    length_loss: Tensor
    smoothing_loss: Tensor


@dataclass(repr=False)
class MuConFullySupervisedLoss(MuConLoss):
    classification_loss: Tensor
    supervised_length_loss: Tensor


@dataclass(repr=False)
class MuConForwardOut(GeneralForwardOut):
    # list of the logits of the actions from the s head
    # it includes the expected EOS as well.
    transcript: Tensor  # [(N + 1) x (M + 1)], float32

    # the un-normalized log length estimates
    lengths: Tensor  # [N], float32

    # logits from the y head (without any log_softmax, etc)
    segmentation: Tensor  # [Tf x M] float


@dataclass(repr=False)
class MuConPredictOut(GeneralForwardOut):
    # predicted list of action from s head
    # fixme: don't return list of int!
    transcript: List[int]  # length = N + 1, it includes the EOS

    # predicted lengths of actions, after softmax (all positive), they sum to 1.
    lengths: Tensor  # [N], size is N, float32

    # logits from the y head, **after log_softmax**
    # can be used for viterbi_decoding
    segmentation_logits: Tensor  # [T x M], float 32

    # fixme: add later
    # # per-frame segmentation from y head
    # segmentation: Tensor  # [T] long
    #
    # # pre-frame segmentation from s head
    # segmentation_s_head: Tensor  # [T] long


class MuCon(Model):
    def __init__(
        self,
        cfg: CfgNode,
        input_feature_size: int,
        num_classes: int,
        max_decoding_steps: int,
    ):
        super().__init__(cfg)
        self.input_feature_size = input_feature_size
        self.num_classes = num_classes
        self.max_decoding_steps = max_decoding_steps

        # the teacher forcing parameter
        self.teacher_forcing = True

        # the eos token_id, for knowing when to stop the decoding process
        # when doing the decoding in inference. E.G. for breakfast it is 48.
        self.EOS_token_id = num_classes

        # loss multipliers
        self.loss_mul_mucon = self.cfg.model.loss.mul_mucon
        self.loss_mul_transcript = self.cfg.model.loss.mul_transcript
        self.loss_mul_smoothing = self.cfg.model.loss.mul_smoothing
        self.loss_mul_length = self.cfg.model.loss.mul_length

        # the temporal modeling network
        ft_type = self.cfg.model.ft.type
        if ft_type == "wavenet":
            self.ft = WaveNetBlock(
                in_channels=self.input_feature_size,
                stages=self.cfg.model.ft.stages,
                out_dims=self.cfg.model.ft.hidden_size,
                pooling=self.cfg.model.ft.pooling,
                pooling_type=self.cfg.model.ft.pooling_type,
                pooling_layers=self.cfg.model.ft.pooling_layers,
                leaky=self.cfg.model.ft.leaky_relu,
                dropout_rate=self.cfg.model.ft.dropout_rate,
            )
        elif ft_type == "mstcnpp":
            self.ft = MSTCNPPFirstStage(
                input_dim=self.input_feature_size,
                num_layers=len(self.cfg.model.ft.stages),
                output_dim=self.cfg.model.ft.hidden_size,
                num_f_maps=self.cfg.model.ft.hidden_size,
                pooling_layers=self.cfg.model.ft.pooling_layers,
            )
        elif ft_type == "noft":
            self.ft = NoFt(
                in_chnnels=self.input_feature_size,
                out_dims=self.cfg.model.ft.hidden_size,
            )
        else:
            raise Exception(f"Invalid ft type ({ft_type})")

        self.ft_last_gn = nn.GroupNorm(
            num_groups=self.cfg.model.ft.last_gn_num_groups,
            num_channels=self.cfg.model.ft.hidden_size,
        )
        self.ft_last_dropout = nn.Dropout(p=self.cfg.model.ft.last_dropout_rate)

        # sequence generation module
        self.fs_encoder_lstm = nn.LSTM(
            input_size=self.cfg.model.ft.hidden_size,
            hidden_size=self.cfg.model.fs.encoder.hidden_size,
            batch_first=True,
            dropout=self.cfg.model.fs.encoder.dropout,
            bidirectional=self.cfg.model.fs.encoder.bidirectional,
        )
        fs_encoder_hidden_out_input_size = (
            (2 * self.cfg.model.fs.encoder.hidden_size)
            if self.cfg.model.fs.encoder.bidirectional
            else self.cfg.model.fs.encoder.hidden_size
        )
        self.fs_encoder_hidden_out = nn.Linear(
            in_features=fs_encoder_hidden_out_input_size,
            out_features=self.cfg.model.fs.encoder.hidden_size,
        )
        self.fs_encoder_cn_out = nn.Linear(
            in_features=fs_encoder_hidden_out_input_size,
            out_features=self.cfg.model.fs.encoder.hidden_size,
        )

        # sequence generation attention parameters
        self.fs_decoder_attention_W1 = rand_p(
            self.cfg.model.fs.encoder.hidden_size * 2,
            self.cfg.model.fs.decoder.hidden_size,
        )
        self.fs_decoder_attention_l2 = nn.Linear(
            self.cfg.model.fs.decoder.hidden_size, self.cfg.model.fs.decoder.hidden_size
        )
        self.fs_decoder_attention_l3 = nn.Linear(
            self.cfg.model.fs.decoder.hidden_size
            + self.cfg.model.fs.decoder.hidden_size,
            self.cfg.model.fs.decoder.hidden_size,
        )
        self.fs_decoder_attention_V = rand_p(self.cfg.model.fs.decoder.hidden_size)

        # sequence generation decoder
        # attn_combine, lstm, trn_fc, trn_out, size_fc, size_out
        self.fs_decoder_embedding = nn.Embedding(
            num_embeddings=self.num_classes + 2,  # not sure if +2 is a better than +1.
            embedding_dim=self.cfg.model.fs.decoder.hidden_size,
        )
        self.fs_decoder_embedding_drop = nn.Dropout(
            p=self.cfg.model.fs.decoder.embedding_dropout
        )
        self.fs_decoder_attn_combine = nn.Linear(
            in_features=fs_encoder_hidden_out_input_size
            + self.cfg.model.fs.decoder.hidden_size,
            out_features=self.cfg.model.fs.decoder.hidden_size,
        )
        self.fs_decoder_lstm = nn.LSTM(
            input_size=self.cfg.model.fs.decoder.hidden_size,
            hidden_size=self.cfg.model.fs.decoder.hidden_size,
            dropout=self.cfg.model.fs.decoder.dropout,
        )
        self.fs_decoder_transcript = nn.Sequential(
            nn.Linear(
                in_features=self.cfg.model.fs.decoder.hidden_size,
                out_features=self.cfg.model.fs.decoder.hidden_size,
            ),
            nn.ReLU(),
            nn.Linear(
                in_features=self.cfg.model.fs.decoder.hidden_size,
                out_features=self.num_classes + 1,
            ),
        )
        # fixme: a/b test
        self.fs_decoder_length = nn.Sequential(
            nn.Linear(
                in_features=self.cfg.model.fs.decoder.hidden_size
                + self.num_classes
                + 1,
                out_features=int(self.cfg.model.fs.decoder.hidden_size / 2),
            ),
            nn.ReLU(),
            nn.Linear(
                in_features=int(self.cfg.model.fs.decoder.hidden_size / 2),
                out_features=1,
            ),
        )

        self.conv_classifier = nn.Conv1d(
            self.cfg.model.ft.hidden_size, self.num_classes, kernel_size=1
        )

        # the reason for the following is that, one set of experiments showed
        # that applying the clip_grad_norm is good when we do it separately
        # for different set of parameters.
        # the first set of parameters
        self.params_set_encode = [
            self.ft,
            self.ft_last_gn,
            self.fs_encoder_lstm,
            self.fs_encoder_hidden_out,
            self.fs_encoder_cn_out,
        ]
        # the second set of parameters
        self.params_set_decode = [
            self.fs_decoder_attention_W1,
            self.fs_decoder_attention_l2,
            self.fs_decoder_attention_l3,
            self.fs_decoder_attention_V,
            self.fs_decoder_embedding,
            self.fs_decoder_attn_combine,
            self.fs_decoder_lstm,
            self.fs_decoder_transcript,
            self.fs_decoder_length,
            self.conv_classifier,
        ]

        self.encode_params = []
        for pset in self.params_set_encode:
            try:
                self.encode_params.extend(list(pset.parameters()))
            except:
                self.encode_params.extend([pset])

        self.decode_params = []
        for pset in self.params_set_decode:
            try:
                self.decode_params.extend(list(pset.parameters()))
            except:
                self.decode_params.extend([pset])

    def forward(self, batch: Batch) -> MuConForwardOut:
        features = batch.feats  # [1 x T x D] float
        tf_transcript_target = batch.transcript_tf_target  # [N + 1] long

        features_length = features.shape[1]
        tf_transcript_target_length = tf_transcript_target.shape[0]

        # Temporal modeling
        temporal_encoded = self.temporal_modeling_forward(
            input=features
        )  # [1 x Tz x Dt]

        # Decoding
        predicted_transcripts, predicted_lengths = self.sequence_generation_forward(
            temporal_encoded=temporal_encoded,
            tf_transcript_target_length=tf_transcript_target_length,
            transcript_tf_input=batch.transcript_tf_input,
            transcript_tf_target=batch.transcript_tf_target,
        )

        # Classification of Frames
        temporal_encoded_permuted = temporal_encoded.permute((0, 2, 1))  # [1 x Ds x Tz]
        predicted_segmentation = self.frame_classifier_forward(
            temporal_encoded=temporal_encoded_permuted, target_length=features_length
        )  # [1 x num_classes x Tf]

        # transposing
        predicted_segmentation = predicted_segmentation.squeeze(0).permute(
            1, 0
        )  # [Tf x num_classes]

        # Preparation of output
        final_predicted_lengths = torch.stack(predicted_lengths[:-1])
        final_predicted_transcripts = torch.cat(predicted_transcripts, dim=0)

        return MuConForwardOut(
            transcript=final_predicted_transcripts,
            lengths=final_predicted_lengths,
            segmentation=predicted_segmentation,
        )

    def predict(self, batch: Batch, forward_out: MuConForwardOut) -> MuConPredictOut:
        if self.teacher_forcing:  # this is set to True for alignment
            transcript = tensor_to_numpy(batch.transcript_tf_target).tolist()
        else:
            transcript = [
                word_probs.argmax().item() for word_probs in forward_out.transcript
            ]
        lengths = F.softmax(forward_out.lengths, dim=0)
        segmentation_logits = F.log_softmax(forward_out.segmentation, dim=1)

        return MuConPredictOut(
            transcript=transcript,
            lengths=lengths,
            segmentation_logits=segmentation_logits,
        )

    def loss(self, batch: Batch, forward_out: MuConForwardOut) -> MuConLoss:
        # fixme: for fast version.
        transcript_loss = self.transcript_loss(batch, forward_out)
        length_loss = self.length_loss(batch, forward_out)
        mucon_loss = self.mucon_loss(batch, forward_out)
        smoothing_loss = self.smoothing_loss(batch, forward_out)

        main_loss = (
            self.loss_mul_transcript * transcript_loss
            + self.loss_mul_length * length_loss
            + self.loss_mul_mucon * mucon_loss
            + self.loss_mul_smoothing * smoothing_loss
        )

        return MuConLoss(
            main=main_loss,
            transcript_loss=transcript_loss,
            length_loss=length_loss,
            mucon_loss=mucon_loss,
            smoothing_loss=smoothing_loss,
        )

    def smoothing_loss(self, batch: Batch, forward_out: MuConForwardOut) -> Tensor:
        loss = self.calculate_smoothing_loss_for_logits(forward_out.segmentation)

        return loss

    def calculate_smoothing_loss_for_logits(self, logits):
        if self.cfg.model.loss.smoothing.log_softmax_before:
            logits = F.log_softmax(logits, dim=1)
        values = F.mse_loss(logits[1:, :], logits[:-1, :].detach())
        if self.cfg.model.loss.smoothing.clamp:
            clamp_min = self.cfg.model.loss.smoothing.clamp_min
            clamp_max = self.cfg.model.loss.smoothing.clamp_max
            values = torch.clamp(values, min=clamp_min, max=clamp_max)
        loss = torch.mean(values)
        return loss

    def mucon_loss(self, batch: Batch, forward_out: MuConForwardOut) -> Tensor:
        lengths = forward_out.lengths
        if self.teacher_forcing:
            target_transcript = batch.transcript
        else:
            target_transcript = torch.tensor(
                [
                    word_probs.argmax().item()
                    for word_probs in forward_out.transcript[:-1]
                ],
                dtype=torch.long,
                device=batch.transcript.device,
            )  # -1 because EOS
            target_transcript[target_transcript >= self.num_classes] = 0  # make sure
        segmentation_size = forward_out.segmentation.shape[0]

        absolute_lengths = project_lengths_softmax(
            T=segmentation_size, L=lengths
        )  # [N], float, sums to Tz

        mask_template = self.cfg.model.loss.mucon.template
        mask_overlap = self.cfg.model.loss.mucon.overlap
        masks = create_masks(
            T=segmentation_size,
            L=absolute_lengths,
            template=mask_template,
            overlap=mask_overlap,
        )  # [N x T]

        mucon_loss = self.calculate_mucon_loss_using_masks(
            absolute_lengths,
            masks,
            forward_out.segmentation,
            target_transcript,
        )

        return mucon_loss

    def calculate_mucon_loss_using_masks(
        self, absolute_lengths, masks, segmentation, target_transcript
    ):
        mucon_type = self.cfg.model.loss.mucon.type
        if mucon_type == "flint":
            num_decoded_words = absolute_lengths.shape[0]
            predictions = []
            for i in range(num_decoded_words):
                masked_segmentation = masks[i].unsqueeze(1) * segmentation
                window_prediction = (
                    masked_segmentation.sum(0) / absolute_lengths[i]
                )  # [M], num_classes
                window_prediction = F.log_softmax(window_prediction, dim=0)
                predictions.append(window_prediction)
            predictions_stacked = torch.stack(
                predictions
            )  # [N, M] log softmaxed logits

            if not self.cfg.model.loss.mucon_weight_background:
                return F.nll_loss(
                    predictions_stacked, target_transcript, reduction="mean"
                )
            else:
                weight_index = self.cfg.model.loss.mucon_weight_background_index
                weight_value = self.cfg.model.loss.mucon_weight_background_value
                weight = torch.ones(
                    self.num_classes,
                    dtype=torch.float32,
                    device=absolute_lengths.device,
                )
                weight[weight_index] = weight_value
                return F.nll_loss(
                    predictions_stacked,
                    target_transcript,
                    weight=weight,
                    reduction="mean",
                )
        elif mucon_type == "arithmetic":
            num_decoded_words = absolute_lengths.shape[0]
            segmentation_size = segmentation.size(0)
            losses = 0
            for i in range(num_decoded_words):
                target = (
                    target_transcript[i]
                    .clone()
                    .detach()
                    .repeat(segmentation_size)
                    .long()
                    .to(device=segmentation.device)
                )
                if not self.cfg.model.loss.mucon_weight_background:
                    loss = (
                        F.cross_entropy(segmentation, target, reduction="none")
                        * masks[i]
                    )
                else:
                    weight_index = self.cfg.model.loss.mucon_weight_background_index
                    weight_value = self.cfg.model.loss.mucon_weight_background_value
                    weight = torch.ones(
                        self.num_classes,
                        dtype=torch.float32,
                        device=absolute_lengths.device,
                    )
                    weight[weight_index] = weight_value
                    loss = (
                        F.cross_entropy(
                            segmentation, target, reduction="none", weight=weight
                        )
                        * masks[i]
                    )
                losses += loss.sum()
            return losses / segmentation_size
        else:
            raise Exception(f"Invalid mucon type ({mucon_type})")

    def length_loss(self, batch: Batch, forward_out: MuConForwardOut) -> Tensor:
        """
        F.relu(s - w) + F.relu(- w - s)
        """
        width = self.cfg.model.loss.length_width

        predicted_lengths = forward_out.lengths  # [N]
        y1 = F.relu(predicted_lengths - width)
        y2 = F.relu(-width - predicted_lengths)

        return y1.sum() + y2.sum()

    def transcript_loss(self, batch: Batch, forward_out: MuConForwardOut) -> Tensor:
        if self.cfg.model.loss.transcript_average:
            reduction = "mean"
        else:
            reduction = "sum"

        if not self.cfg.model.loss.transcript_weight_background:
            return F.nll_loss(
                input=forward_out.transcript,
                target=batch.transcript_tf_target,
                reduction=reduction,
            )
        else:
            weight_index = self.cfg.model.loss.transcript_weight_background_index
            weight_value = self.cfg.model.loss.transcript_weight_background_value
            weight = torch.ones(
                self.num_classes + 1,
                dtype=torch.float32,
                device=forward_out.transcript.device,
            )
            weight[weight_index] = weight_value
            return F.nll_loss(
                input=forward_out.transcript,
                target=batch.transcript_tf_target,
                reduction=reduction,
                weight=weight,
            )

    def frame_classifier_forward(self, temporal_encoded: Tensor, target_length: int):
        """
        :param temporal_encoded: [1 x Ds x Tz], float32
        :param target_length: int, Tf
        :return: [1 x num_classes x Tf]
        """

        interpolated_input = F.interpolate(
            temporal_encoded, target_length
        )  # [1 x Ds x Tf]

        segmentation = self.conv_classifier.forward(
            interpolated_input
        )  # [1 x num_classes, Tf]

        return segmentation  # [1 x num_classes x Tf]

    # fixme: jit?
    def sequence_generation_forward(
        self,
        temporal_encoded: Tensor,
        tf_transcript_target_length: int,
        transcript_tf_input: Tensor,
        transcript_tf_target: Tensor,
    ):
        """
        :param temporal_encoded: [1 x T' x D']
        :param tf_transcript_target_length: int
        :param transcript_tf_input: [N + 1] long
        :param transcript_tf_target: [N + 1] long
        :return:
        """

        # bidirectional lstm encoder
        # fs_encoder_lstm_out: [1 x T x 2*D']  because it is bidirectional
        # fs_encoder_lstm_hidden, fs_encoder_lstm_c: [2 x 1 x D'], 2 because it is bidirectional.
        (
            fs_encoder_lstm_out,
            (fs_encoder_lstm_hidden, fs_encoder_lstm_c),
        ) = self.fs_encoder_lstm.forward(temporal_encoded)

        fs_encoder_lstm_hidden_flat = fs_encoder_lstm_hidden.view(1, -1)
        fs_encoder_lstm_c_flat = fs_encoder_lstm_c.view(1, -1)

        fs_encoder_lstm_hidden = self.fs_encoder_hidden_out.forward(
            fs_encoder_lstm_hidden_flat
        ).unsqueeze(
            0
        )  # [1 x 1 x D'']
        fs_encoder_lstm_c = self.fs_encoder_cn_out.forward(
            fs_encoder_lstm_c_flat
        ).unsqueeze(
            0
        )  # [1 x 1 x D'']

        # fixme: fix naming
        fs_decoder_hidden, fs_decoder_c = fs_encoder_lstm_hidden, fs_encoder_lstm_c

        # Prepare for attention
        # here we are again assuming batch size of 1.
        fs_encoder_lstm_out_without_batch = fs_encoder_lstm_out[0]
        encoder_result_ready_for_attention = (
            fs_encoder_lstm_out_without_batch @ self.fs_decoder_attention_W1
        )  # [T x D']

        # Prepare for decoding loop
        predicted_lengths = []
        predicted_transcripts = []
        if self.teacher_forcing or self.training:
            decoding_iter_length = tf_transcript_target_length
        else:
            decoding_iter_length = self.max_decoding_steps

        # decoding for loop
        for decoding_step in range(decoding_iter_length):
            if self.teacher_forcing:
                decoder_input = transcript_tf_input[decoding_step].unsqueeze(0)
            else:
                if decoding_step == 0:
                    # taking the zeroth element from the input, since it is always SOS
                    decoder_input = transcript_tf_input[0].unsqueeze(0)
                else:
                    # input will be set automatically at the end of the previous loop
                    pass

            # Calculated input embedding.
            # fixme: this can be done outside and all at once if teacher forcing is on.
            # noinspection PyUnboundLocalVariable
            decoder_input_embedded = self.fs_decoder_embedding.forward(
                decoder_input
            )  # [1 x Ds]
            decoder_input_embedded = self.fs_decoder_embedding_drop.forward(
                F.relu(decoder_input_embedded)
            )

            # Apply Attention
            attention_weights = self._calculate_attention(
                current_hidden_state=fs_decoder_hidden,
                encoder_result_ready_for_attention=encoder_result_ready_for_attention,
            )  # [Tz]

            # we are using broadcasting here for multiplication, so first the hidden size dimension
            # is added to attention_weights with unsqueeze(1).
            # at the end, the summation is applied in the temporal dimension.
            # attention_weights: [Tz]
            # fs_encoder_lstm_out_without_batch: [Tz x 2Ds]
            attention_applied = (
                attention_weights.unsqueeze(1) * fs_encoder_lstm_out_without_batch
            ).sum(
                dim=0, keepdim=True
            )  # [1 x 2Ds]

            attn_concat_input = torch.cat(
                (decoder_input_embedded, attention_applied), 1
            )  # [1 x 2Ds + Ds]

            output_attn = self.fs_decoder_attn_combine.forward(
                attn_concat_input
            ).unsqueeze(0)
            output_attn = F.relu(output_attn)  # [1 x 1 x Ds]

            # Decode
            # fs_decoder_out [1 x 1 x Ds]
            # fs_decoder_hidden, fs_decoder_s : [1 x 1 x Ds]
            (
                fs_decoder_out,
                (fs_decoder_hidden, fs_decoder_c),
            ) = self.fs_decoder_lstm.forward(
                output_attn, (fs_decoder_hidden, fs_decoder_c)
            )

            # Generate Transcript and Length
            # pred_transcript [1 x 1 x num_decoder_words]
            pred_transcript = self.fs_decoder_transcript.forward(fs_decoder_out)

            s_input = torch.cat((output_attn, pred_transcript), 2)
            s_input = F.relu(s_input)

            pred_length = self.fs_decoder_length.forward(s_input).squeeze()  # []

            # This means that we have to care about loss
            # and use nll_loss instead of cross entropy.
            pred_transcript = F.log_softmax(
                pred_transcript.squeeze(0), dim=1
            )  # [1 x num_decoder_words]

            # Append to the list of predictions
            predicted_transcripts.append(pred_transcript)
            predicted_lengths.append(pred_length)

            # Handling inference and non-teacher forcing case.
            predicted_word = pred_transcript.argmax(dim=1)
            if not self.teacher_forcing and not self.training:
                # check if we should break
                if predicted_word.item() == self.EOS_token_id:
                    break

            if not self.teacher_forcing:
                # set the decoder input for next step
                decoder_input = predicted_word

        return predicted_transcripts, predicted_lengths

    def _calculate_attention(
        self, current_hidden_state: Tensor, encoder_result_ready_for_attention: Tensor
    ) -> Tensor:
        # current_hidden_state: [1 x 1 x Ds]
        # encoder_result_ready_for_attention: [Tz x Ds]
        # returns: attention: [Tz], float 32, sums to 1.

        current_hidden_flat = current_hidden_state.view(1, -1)  # [1 x Ds]
        current_hidden_encoded = self.fs_decoder_attention_l2(
            current_hidden_flat
        )  # [1 x Ds]
        # adding the hidden_encoded representation to every temporal position.
        u = torch.tanh(encoder_result_ready_for_attention + current_hidden_encoded)
        attention = F.softmax(u @ self.fs_decoder_attention_V, dim=0)
        return attention

    def temporal_modeling_forward(self, input: Tensor) -> Tensor:
        """
        :param input: The input features to the input modelling.
        [B x T x D] float 32.
        :return: The output [B x T' x D']
        """
        # first reshape
        features = input.permute(0, 2, 1)  # [B x D x T]

        # forward pass
        features = self.ft.forward(features)  # [B x D' x T']

        # group norm
        if self.cfg.model.ft.last_gn:
            features = self.ft_last_gn.forward(features)  # [B x D' x T']

        # relu
        if self.cfg.model.ft.last_relu:
            features = F.relu(features)  # [B x D' x T']

        # dropout
        if self.cfg.model.ft.last_dropout:
            features = self.ft_last_dropout.forward(features)  # [B x D' x T']

        # reshape again
        output = features.permute(0, 2, 1)  # [B x T' x D']

        return output

    def set_teacher_forcing(self, teacher_forcing: bool = True):
        # this can be optionally set during training.
        # or can be set during inference for action alignment.
        self.teacher_forcing = teacher_forcing


class MuConFullySupervised(MuCon):
    def __init__(
        self,
        cfg: CfgNode,
        input_feature_size: int,
        num_classes: int,
        max_decoding_steps: int,
    ):
        super().__init__(
            cfg,
            input_feature_size=input_feature_size,
            num_classes=num_classes,
            max_decoding_steps=max_decoding_steps,
        )

        # loss multipliers
        self.loss_mul_mucon = self.cfg.model.loss.mul_mucon
        self.loss_mul_transcript = self.cfg.model.loss.mul_transcript
        self.loss_mul_smoothing = self.cfg.model.loss.mul_smoothing
        self.loss_mul_length = self.cfg.model.loss.mul_length
        self.loss_mul_classification = (
            self.cfg.model.loss.fully_supervised.mul_classification
        )
        self.loss_mul_supervised_length = (
            self.cfg.model.loss.fully_supervised.mul_supervised_length
        )

    def classification_loss(
        self, batch: FullySupervisedBatch, forward_out: MuConForwardOut
    ) -> Tensor:
        target_length = batch.gt_label.shape[0]  # Tt
        target_labels = batch.gt_label

        classification_loss = self.calculate_classification_loss_for_logit(
            forward_out.segmentation, target_labels, target_length
        )

        return classification_loss

    def calculate_classification_loss_for_logit(
        self, segmentation, target_labels, target_length
    ):
        if segmentation.shape[0] != target_length:
            # first transpose, then interpolate, then transpose back.
            segmentation_shuffled = F.interpolate(
                input=segmentation.transpose(0, 1).unsqueeze(0), size=target_length
            ).squeeze(0)
            segmentation = segmentation_shuffled.transpose(0, 1)

        # segmentation: [Tt x M]
        return F.cross_entropy(segmentation, target_labels, reduction="mean")

    def supervised_length_loss(
        self, batch: FullySupervisedBatch, forward_out: MuConForwardOut
    ) -> Tensor:
        relative_lengths = batch.absolute_lengths / batch.absolute_lengths.sum()
        relative_predicted_lengths = F.softmax(forward_out.lengths, dim=0)

        return F.mse_loss(
            relative_lengths, relative_predicted_lengths, reduction="mean"
        )

    def loss(
        self, batch: FullySupervisedBatch, forward_out: MuConForwardOut
    ) -> MuConFullySupervisedLoss:
        transcript_loss = self.transcript_loss(batch, forward_out)
        length_loss = self.length_loss(batch, forward_out)
        mucon_loss = self.mucon_loss(batch, forward_out)
        smoothing_loss = self.smoothing_loss(batch, forward_out)
        classification_loss = self.classification_loss(batch, forward_out)
        supervised_length_loss = self.supervised_length_loss(batch, forward_out)

        main_loss = (
            self.loss_mul_transcript * transcript_loss
            + self.loss_mul_length * length_loss
            + self.loss_mul_mucon * mucon_loss
            + self.loss_mul_smoothing * smoothing_loss
            + self.loss_mul_classification * classification_loss
            + self.loss_mul_supervised_length * supervised_length_loss
        )

        return MuConFullySupervisedLoss(
            main=main_loss,
            transcript_loss=transcript_loss,
            length_loss=length_loss,
            mucon_loss=mucon_loss,
            smoothing_loss=smoothing_loss,
            classification_loss=classification_loss,
            supervised_length_loss=supervised_length_loss,
        )


class MuConMixedSupervision(MuConFullySupervised):
    def loss(
        self, batch: MixedSupervisionBatch, forward_out: MuConForwardOut
    ) -> MuConFullySupervisedLoss:
        fully_supervised = batch.fully_supervised

        transcript_loss = self.transcript_loss(batch, forward_out)
        length_loss = self.length_loss(batch, forward_out)
        mucon_loss = self.mucon_loss(batch, forward_out)
        smoothing_loss = self.smoothing_loss(batch, forward_out)
        classification_loss = self.classification_loss(batch, forward_out)
        supervised_length_loss = self.supervised_length_loss(batch, forward_out)

        if fully_supervised:
            main_loss = (
                self.loss_mul_transcript * transcript_loss
                + self.loss_mul_length * length_loss
                + self.loss_mul_mucon * mucon_loss
                + self.loss_mul_smoothing * smoothing_loss
                + self.loss_mul_classification * classification_loss
                + self.loss_mul_supervised_length * supervised_length_loss
            )
        else:
            main_loss = (
                self.loss_mul_transcript * transcript_loss
                + self.loss_mul_length * length_loss
                + self.loss_mul_mucon * mucon_loss
                + self.loss_mul_smoothing * smoothing_loss
            )

        return MuConFullySupervisedLoss(
            main=main_loss,
            transcript_loss=transcript_loss,
            length_loss=length_loss,
            mucon_loss=mucon_loss,
            smoothing_loss=smoothing_loss,
            classification_loss=classification_loss,
            supervised_length_loss=supervised_length_loss,
        )
