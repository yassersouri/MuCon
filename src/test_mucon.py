from pathlib import Path

import click

from configs.mucon.default import get_cfg_defaults
from core.datasets import handel_dataset
from mucon.evaluators import MuConEvaluator
from mucon.models import create_model
from mucon.trainers import SimpleTrainer


@click.command()
@click.argument("identifier", type=str)
@click.option("--root", default="")
@click.option("--data-root", default="")
def main(identifier: str, root: str, data_root: str):
    single_main(identifier, root, data_root)


def single_main(
    identifier: str,
    root: str,
    data_root: str,
):
    """
    :param identifier: exp-name/run-number/epoch-number
    :param root: if not the default set it
    :param data_root: if not the default set it
    :param full_test: whether to perform full viterbi decoding (using all training transcripts).
    :param correct_full_test: whether to use the corrected and slower full decoding.
    """
    print(identifier)

    cfg = get_cfg_defaults()
    if root == "":
        root = cfg.trainer.root

    exp_name, run_number, epoch_number = identifier.split("/")
    epoch_number = int(epoch_number)

    first_run_folder = Path(root) / exp_name / f"{run_number}"
    first_config_file_path = first_run_folder / "config.yaml"
    cfg.merge_from_file(str(first_config_file_path))

    if data_root == "":
        data_root = cfg.dataset.root

    cfg.defrost()
    cfg.trainer.root = root
    cfg.dataset.root = data_root
    cfg.freeze()

    test_db = handel_dataset(cfg, train=False)
    model = create_model(
        cfg=cfg,
        num_classes=test_db.get_num_classes(),
        max_decoding_steps=test_db.max_transcript_length + 1,
        # plus one is because of EOS
        input_feature_size=test_db.feat_dim,
    )

    test_evaluator = MuConEvaluator(
        cfg=cfg, test_db=test_db, model=model, device=cfg.system.device
    )
    test_evaluator.set_name("test_eval")

    test_evaluator.viterbi_mode(True)  # setting viterbi to True

    trainer = SimpleTrainer(
        cfg=cfg,
        exp_name=cfg.experiment_name,
        train_db=test_db,
        model=model,
        device=cfg.system.device,
        evaluators=None,
    )

    trainer.load_training(run=run_number, epoch=epoch_number)

    eval_result = test_evaluator.evaluate()
    print(eval_result)

    return eval_result


if __name__ == "__main__":
    main()
