from pathlib import Path
from typing import Optional, Iterable, Any, Dict, List, Union

import torch
from fandak import Trainer, Evaluator, Model, Dataset
from fandak.core.trainers import Scheduler
from torch import optim
from torch.nn import Parameter
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import MultiStepLR, ReduceLROnPlateau
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from yacs.config import CfgNode

from mucon.evaluators import MuConEvaluatorResult


def create_optimizer(cfg: CfgNode, parameters: Iterable[Parameter]) -> Optimizer:
    learning_rate = cfg.trainer.learning_rate
    momentum = cfg.trainer.momentum
    optimizer_name = cfg.trainer.optimizer
    weight_decay = cfg.trainer.weight_decay

    if optimizer_name == "SGD":
        return optim.SGD(
            params=parameters,
            lr=learning_rate,
            weight_decay=weight_decay,
            momentum=momentum,
        )
    elif optimizer_name == "Adam":
        return optim.Adam(
            params=parameters, lr=learning_rate, weight_decay=weight_decay, amsgrad=True
        )
    else:
        raise Exception("Invalid optimizer name (%s)" % optimizer_name)


def create_scheduler(cfg: CfgNode, optimizer: Optimizer) -> Optional[Scheduler]:
    scheduler_name = cfg.trainer.scheduler.name
    if scheduler_name == "none":
        return None
    elif scheduler_name == "plateau":
        return ReduceLROnPlateau(
            optimizer,
            mode=cfg.trainer.scheduler.plateau.mode,
            factor=cfg.trainer.scheduler.plateau.factor,
            verbose=cfg.trainer.scheduler.plateau.verbose,
            patience=cfg.trainer.scheduler.plateau.patience,
        )
    elif scheduler_name == "step":
        milestones = cfg.trainer.scheduler.step.milestones
        gamma = cfg.trainer.scheduler.step.gamma
        return MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
    else:
        raise Exception("Invalid scheduler name (%s)" % scheduler_name)


class SimpleTrainer(Trainer):
    def update_trainer_using_config(self):
        self.save_every = self.cfg.trainer.save_every
        self.eval_every = self.cfg.trainer.eval_every

    def on_start_epoch(self, epoch_num: int):
        self.model.set_teacher_forcing(self.cfg.model.teacher_forcing)

    # noinspection PyUnresolvedReferences
    def on_finish_epoch(self, epoch_num: int):
        if (epoch_num + 1) % self.eval_every == 0:
            # Saving stuff for streamlit visualization.
            for evaluator in self.evaluators:
                evaluator.set_checkpointing_folder(self._get_checkpointing_folder())
                evaluator.save_stuff()

    def figure_root(self) -> Path:
        return Path(self.cfg.trainer.root)

    def figure_optimizer(self) -> Optimizer:
        original_lr = self.cfg.trainer.learning_rate
        return create_optimizer(self.cfg, self.model.get_params(original_lr))

    def figure_scheduler(self, optimizer: Optimizer) -> Optional[Scheduler]:
        return create_scheduler(self.cfg, optimizer)

    def figure_clip_grad_norm(self) -> Optional[float]:
        if self.cfg.trainer.clip_grad_norm:
            return self.cfg.trainer.clip_grad_norm_value
        else:
            return None

    def figure_accumulate_grad(self) -> int:
        return self.cfg.trainer.accumulate_grad_every

    def figure_num_epochs(self) -> int:
        return self.cfg.trainer.num_epochs

    def create_train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_db,
            batch_size=1,
            shuffle=True,
            num_workers=self.cfg.system.num_workers,
            collate_fn=self.train_db.collate_fn,
            pin_memory=True,
        )

    # noinspection PyUnusedLocal
    def _train_1_batch(self, iter_num: int, batch):
        # callback
        self.on_start_batch(self.iter_num, batch)

        # FIXME: this might be slow depending on the config system
        accumulate_grad_every = self.figure_accumulate_grad()
        if accumulate_grad_every is None:
            accumulate_grad_every = 1

        # TODO: move to the end.
        # TODO: move to infinitely flexible callbacks. like fastai v2.
        # initial setup
        if iter_num % accumulate_grad_every == 0:
            self.optimizer.zero_grad()
        batch.to(self.device)

        # forward pass
        forward_out = self.model.forward(batch)
        loss = self.model.loss(batch, forward_out)

        the_loss = loss.main / accumulate_grad_every

        # backward pass
        the_loss.backward()

        # TODO: refactor fandak with this as a function for easier implementation.
        # optional gradient clipping
        if self.clip_grad_norm is not None:
            if self.cfg.trainer.clip_grad_norm_separate:
                clip_grad_norm_(self.model.encode_params, max_norm=self.clip_grad_norm)
                clip_grad_norm_(self.model.decode_params, max_norm=self.clip_grad_norm)
            else:
                if self.cfg.trainer.clip_grad_norm_every_param:
                    for p in self.model.parameters():
                        clip_grad_norm_(p, self.clip_grad_norm)
                else:
                    clip_grad_norm_(
                        self.model.parameters(), max_norm=self.clip_grad_norm
                    )

        # optimizer step
        if iter_num % accumulate_grad_every == (accumulate_grad_every - 1):
            self.optimizer.step()

        # callback
        self.on_finish_batch(self.iter_num, batch, forward_out, loss)

        return loss, forward_out

    def figure_scheduler_input(
        self, eval_results: List[MuConEvaluatorResult]
    ) -> Dict[str, Any]:
        if self.cfg.trainer.scheduler.name == "plateau":
            return {"metrics": eval_results[0].s_mof_nbg}
        else:
            return {}


class TrainerForTFExperiments(SimpleTrainer):
    def __init__(
        self,
        cfg: CfgNode,
        exp_name: str,
        train_db: Dataset,
        model: Model,
        device: torch.device,
        evaluators: Optional[Union[Iterable[Evaluator], Evaluator]] = None,
        turnoff_tf_after_epoch: int = 1000,
    ):
        super().__init__(
            cfg=cfg,
            exp_name=exp_name,
            train_db=train_db,
            model=model,
            device=device,
            evaluators=evaluators,
        )
        self.turnoff_tf_after_epoch = turnoff_tf_after_epoch

    def on_start_epoch(self, epoch_num: int):
        if epoch_num >= self.turnoff_tf_after_epoch:
            self.model.set_teacher_forcing(teacher_forcing=False)
        else:
            self.model.set_teacher_forcing(teacher_forcing=True)
