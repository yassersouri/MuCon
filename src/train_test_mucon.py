from typing import List

import click
from fandak.utils import common_config
from fandak.utils.config import update_config

from configs.mucon.default import get_cfg_defaults
from core.datasets import handel_dataset
from mucon.evaluators import MuConEvaluator
from mucon.models import create_model
from mucon.trainers import SimpleTrainer


@click.command()
@common_config
@click.option("--exp-name", default="")
def main(file_configs: List[str], set_configs: List[str], exp_name: str):
    cfg = update_config(
        default_config=get_cfg_defaults(),
        file_configs=file_configs,
        set_configs=set_configs,
    )
    if exp_name != "":
        cfg.defrost()
        cfg.experiment_name = exp_name
        cfg.freeze()
    print(cfg)

    train_db = handel_dataset(cfg, train=True)
    test_db = handel_dataset(cfg, train=False)

    model = create_model(
        cfg=cfg,
        num_classes=train_db.get_num_classes(),
        max_decoding_steps=train_db.max_transcript_length + 1,
        # plus one is because of EOS
        input_feature_size=train_db.feat_dim,
    )

    test_evaluator = MuConEvaluator(
        cfg=cfg, test_db=test_db, model=model, device=cfg.system.device
    )

    test_evaluator.set_name("test_eval")

    trainer = SimpleTrainer(
        cfg=cfg,
        exp_name=cfg.experiment_name,
        train_db=train_db,
        model=model,
        device=cfg.system.device,
        evaluators=[test_evaluator],
    )

    trainer.train()
    trainer.save_training()

    # full evaluation with viterbi
    test_evaluator.viterbi_mode(True)
    evaluator_result = test_evaluator.evaluate()
    print(evaluator_result)

    # saving inside epoch folder results.
    test_evaluator.set_checkpointing_folder(trainer._get_checkpointing_folder())
    test_evaluator.save_stuff()

    # resetting the value of the metric and saving.
    trainer.metrics[trainer.eval_metric_name_format.format(1)].set_value(
        evaluator_result, trainer.epoch_num
    )
    trainer.metrics[trainer.eval_metric_name_format.format(1)].save()


if __name__ == "__main__":
    main()
