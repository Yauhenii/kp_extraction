# a customizable version of https://github.com/PyTorchLightning/lightning-transformers/blob/0.1.0/lightning_transformers/cli/train.py
import os
import warnings

from hydra import compose, initialize
from pathlib import Path
from typing import Any, Optional

import pprint

from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import LightningDataModule, Trainer
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.tuner.tuning import Tuner
from pytorch_lightning.utilities.distributed import rank_zero_info

from project.data.common.base_data_module import DataModule
from project.models.baselines.ensemble_base_model import EnsembleModel
from project.data.common.kfold_data_module import KFoldDataModule
from project.models.common.base_model import BaseModel
from project.models.common.configs import DataConfig, TaskConfig, TrainerConfig
from project.models.common.hydra_instantiator import Instantiator, HydraInstantiator
from project.models.common.nlp_configs import HFTokenizerConfig
from shutil import copyfile


def set_ignore_warnings():
    warnings.simplefilter("ignore")
    # set os environ variable for multiprocesses
    os.environ["PYTHONWARNINGS"] = "ignore"


def fullname(o):
    klass = o.__class__
    module = klass.__module__
    if module == '__builtin__':
        return klass.__name__  # avoid outputs like '__builtin__.str'
    return module + '.' + klass.__name__


project_dir = Path(__file__).resolve().parents[2]
models_output_path = f"{project_dir}/models"


def kfold_training(
        dm: KFoldDataModule,
        model: BaseModel,
        trainer: Trainer,
        num_folds: int):
    model_paths = []
    dm.setup_folds(num_folds)
    for i in range(num_folds):
        print(f"Starting Fold {i}")
        dm.setup_fold_index(i)
        trainer.fit(model, datamodule=dm)
        out_path = f'{models_output_path}/{model.model_config._name_or_path}_fold_{i}.ckpt'
        copyfile(trainer.checkpoint_callback.best_model_path, out_path)
        model_paths.append(out_path)
    
    ensamble_model = EnsembleModel(model_cls_type=fullname(model), checkpoint_paths=model_paths)
    trainer.test(ensamble_model, datamodule=dm)


def run(
        instantiator: Instantiator,
        batch_size: int,
        kfold_num: int = -1,
        ignore_warnings: bool = True,
        run_test_after_fit: bool = True,
        dataset: DataConfig = DataConfig(),
        task: TaskConfig = TaskConfig(),
        trainer_config: TrainerConfig = TrainerConfig(),
        tokenizer: Optional[HFTokenizerConfig] = None,
        logger: Optional[Any] = None
) -> None:
    if ignore_warnings:
        set_ignore_warnings()
    
    data_module_kwargs = {}
    if tokenizer is not None:
        data_module_kwargs["tokenizer"] = tokenizer
    
    data_module: DataModule = instantiator.data_module(dataset, **data_module_kwargs)
    if data_module is None:
        raise ValueError("No dataset found. Hydra hint: did you set `dataset=...`?")
    if not isinstance(data_module, LightningDataModule):
        raise ValueError(
            "The instantiator did not return a DataModule instance." " Hydra hint: is `dataset._target_` defined?`"
        )
    data_module.setup("fit")
    
    model: BaseModel = instantiator.model(task, model_data_kwargs=getattr(data_module, "model_data_kwargs", None))
    
    # try domain metric such as map_strict if better than val_loss
    early_stopping = EarlyStopping(monitor='val_loss')
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    model_checkpoint = ModelCheckpoint(save_weights_only=True, save_top_k=1, monitor='val_loss')
    callbacks = [early_stopping, lr_monitor, model_checkpoint]
    
    trainer: Trainer = instantiator.trainer(
        trainer_config,
        logger=logger,
        callbacks=callbacks
    )
    
    # auto batch size scaling
    if batch_size is None:
        tuner = Tuner(trainer)
        new_batch_size = tuner.scale_batch_size(model, datamodule=data_module, mode="binsearch")
        trainer.logger.experiment['selected_batch_size'].log(new_batch_size)
        trainer: Trainer = instantiator.trainer(
            trainer_config,
            logger=logger,
            callbacks=callbacks
        )
        data_module: KFoldDataModule = instantiator.data_module(dataset, **data_module_kwargs)
        data_module.batch_size = new_batch_size
        model: BaseModel = instantiator.model(task,
                                              model_data_kwargs=getattr(data_module, "model_data_kwargs", None))
    else:
        data_module.batch_size = batch_size
    
    if kfold_num > 0:
        kfold_training(data_module, model, trainer, kfold_num)
    else:
        trainer.fit(model, datamodule=data_module)
        if run_test_after_fit:
            trainer.test(model, datamodule=data_module, ckpt_path=trainer.checkpoint_callback.best_model_path)
    
    # stop experiment
    trainer.logger.experiment.stop()


project_dir = Path(__file__).resolve().parents[3]
PROCESSED_DATA_PATH = f"{project_dir}/data/processed"


def main() -> None:
    initialize(config_path="./config", job_name="train")
    
    datasets = ['datasets/arg_kp_2021_task1']
    
    experiments = [
        # {
        #     "task_type": "baselines/classification",
        #     "backbone_name": 'bert_base_uncased',
        #     "batch_size": 64,
        #     "kfold_num": -1
        # },
        # {
        #     "task_type": "baselines/triplet_learning",
        #     "backbone_name": 'roberta_base',
        #     "batch_size": 64,
        #     "kfold_num": -1
        # },
        # {
        #     "task_type": "baselines/contrastive_learning",
        #     "backbone_name": 'roberta_base',
        #     "batch_size": 32,
        #     "kfold_num": -1
        # },
        # {
        #     "task_type": "baselines/classification",
        #     "backbone_name": 'bert_tiny',
        #     "batch_size": 1024,
        #     "kfold_num": -1
        # },
        {
            "task_type": "baselines/classification",
            "backbone_name": 'roberta_base',
            "batch_size": 64,
            "kfold_num": -1
        },
        # {
        #     "task_type": "baselines/classification",
        #     "backbone_name": 'st_paraphrase_distilroberta',
        #     "batch_size": 128,
        #     "kfold_num": -1
        # },
        # {
        #     "task_type": "baselines/classification",
        #     "backbone_name": 'all_mpnet_base',
        #     "batch_size": 64,
        #     "kfold_num": -1
        # },
        # {
        #     "task_type": "baselines/classification",
        #     "backbone_name": 'nli_distilroberta_base',
        #     "batch_size": 128,
        #     "kfold_num": 5
        # }
    
    ]
    
    for dataset in datasets:
        for experiment in experiments:
            cfg: DictConfig = compose(config_name="config", overrides=[f"dataset={dataset}",
                                                                       f"task={experiment['task_type']}",
                                                                       f"backbone={experiment['backbone_name']}"])
            # run experiment
            rank_zero_info(OmegaConf.to_yaml(cfg))
            instantiator = HydraInstantiator()
            logger = instantiator.instantiate(cfg.logger)
            logger.experiment['experiment_name'].log(
                f"{experiment['task_type'].replace('/', '_')}_{experiment['backbone_name']}")
            with open('experiment_configs.yaml', "w") as yaml_file:
                yaml_file.write(pprint.pformat(OmegaConf.to_object(cfg)))
            logger.experiment["experiment_configs.yaml"].upload('experiment_configs.yaml')
            run(
                instantiator,
                ignore_warnings=cfg.get("ignore_warnings"),
                run_test_after_fit=cfg.get("training").get("run_test_after_fit"),
                dataset=cfg.get("dataset"),
                tokenizer=cfg.get("tokenizer"),
                task=cfg.get("task"),
                trainer_config=cfg.get("trainer"),
                logger=logger,
                batch_size=experiment["batch_size"] if "batch_size" in experiment else None,
                kfold_num=experiment['kfold_num']
            )


def evaluate_ensemble_models(fold_num, backbone, backbone_config_name, batch_size=16):
    model_paths = []
    for i in range(fold_num):
        model_paths.append(f"{models_output_path}/{backbone}_fold_{i}.ckpt")
    
    initialize(config_path="./config", job_name="train")
    dataset_name = 'datasets/arg_kp_2021_task1'
    cfg: DictConfig = compose(config_name="config",
                              overrides=[f"dataset={dataset_name}", f"backbone={backbone_config_name}"])
    
    instantiator = HydraInstantiator()
    tokenizer = cfg.get("tokenizer")
    dataset = cfg.get("dataset")
    trainer_config = cfg.get("trainer")
    
    data_module_kwargs = {}
    if tokenizer is not None:
        data_module_kwargs["tokenizer"] = tokenizer
    
    dm: DataModule = instantiator.data_module(dataset, **data_module_kwargs)
    dm.batch_size = batch_size
    
    logger = instantiator.instantiate(cfg.logger)
    trainer: Trainer = instantiator.trainer(trainer_config, logger=logger)
    
    ensemble_model = EnsembleModel(model_cls_type='project.models.baselines.classification_model.ClassificationModel',
                                   checkpoint_paths=model_paths)
    trainer.test(ensemble_model, datamodule=dm)
    
    # stop experiment
    trainer.logger.experiment.stop()


if __name__ == "__main__":
    main()
    
    # evaluate_ensemble_models(5, "roberta-base", "roberta_base", batch_size=32)
