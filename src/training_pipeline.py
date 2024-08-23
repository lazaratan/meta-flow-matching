from typing import List, Optional

from pytorch_lightning import Callback, Trainer, seed_everything
from pytorch_lightning.loggers import Logger

import hydra
from omegaconf import DictConfig
from hydra.utils import instantiate

from src import utils

log = utils.get_logger(__name__)

def train(cfg: DictConfig):
    """Contains the training pipeline. Can additionally evaluate model on a testset, using best
    weights achieved during training.

    Args:
        cfg (DictConfig): Configuration composed by Hydra.

    Returns:
        Optional[float]: Metric score for hyperparameter optimization.
    """
    # Initialize the seed
    # Set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        seed_everything(cfg.seed, workers=True)

    # Initialize the data module
    print("cfg.datamodule: ", cfg.datamodule)
    datamodule = instantiate(cfg.datamodule)

    # Initialize the model
    if (
        cfg.datamodule._target_ == "src.datamodule.letters_dataloader.LettersDatamodule"
        or cfg.datamodule._target_ == "src.datamodule.letters_dataloader.LettersBatchDatamodule"
    ):
        if cfg.datamodule.conditional:
            model = instantiate(
                cfg.model,
                datamodule.train_dataset.train_eval_letters,
                num_conditions=datamodule.train_dataset.num_conditions,
            )
        else:
            model = instantiate(
                cfg.model,
                datamodule.train_dataset.train_eval_letters,
            )
    elif cfg.datamodule._target_ == "src.datamodule.trellis_dataloader.TrellisDatamodule":
        if cfg.datamodule.num_components is not None:
            model = instantiate(
                cfg.model,
                datamodule.train_dataset.train_eval_replicas,
                dim=cfg.datamodule.num_components,
                pca=datamodule.train_dataset.pca,
                pca_space=True,
                num_train_replica=datamodule.num_train_replica,
                num_test_replica=datamodule.num_test_replica,
                num_val_replica=datamodule.num_val_replica,
                pca_for_plot=datamodule.pca_for_plot,
                treatments=datamodule.train_dataset.treatment,
            )
        else:
            model = instantiate(
                cfg.model,
                datamodule.train_dataset.train_eval_replicas,
                num_train_replica=datamodule.num_train_replica,
                num_test_replica=datamodule.num_test_replica,
                num_val_replica=datamodule.num_val_replica,
                pca_for_plot=datamodule.pca_for_plot,
                treatments=datamodule.train_dataset.treatment,
            )
    else: 
        model = instantiate(cfg.model)

    # Init lightning callbacks
    callbacks: List[Callback] = []
    if "callbacks" in cfg:
        for _, cb_conf in cfg.callbacks.items():
            if "_target_" in cb_conf:
                log.info(f"Instantiating callback <{cb_conf._target_}>")
                callbacks.append(hydra.utils.instantiate(cb_conf))

    # Init lightning loggers (this can be)
    logger: List[Logger] = []
    if "logger" in cfg:
        for _, lg_conf in cfg.logger.items():
            if "_target_" in lg_conf:
                log.info(f"Instantiating logger <{lg_conf._target_}>")
                logger.append(hydra.utils.instantiate(lg_conf))
    
    # check if checkpoint path exists
    ckpt_path = cfg.trainer.resume_from_checkpoint if cfg.trainer.resume_from_checkpoint else None
    print("Checkpoint path:", ckpt_path)
    
    # Init lightning trainer    
    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer,
        callbacks=callbacks,
        logger=logger,
        _convert_="partial",
        resume_from_checkpoint=ckpt_path,
    )
    
    # Send some parameters from config to all lightning loggers
    log.info("Logging hyperparameters!")
    utils.log_hyperparameters(
        config=cfg,
        model=model,
        datamodule=datamodule,
        trainer=trainer,
        callbacks=callbacks,
        logger=logger,
    )

    # Train the model 
    if cfg.get("train"):
        log.info("Starting training!")
        trainer.fit(model=model, datamodule=datamodule)

    # Test the model
    if cfg.get("test"):
        ckpt_path = "last"
        if not cfg.get("train") or cfg.trainer.get("fast_dev_run"):
            ckpt_path = None
        log.info("Starting testing!")
        trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)

    # Make sure everything closed properly
    log.info("Finalizing!")
    utils.finish(
        config=cfg,
        model=model,
        datamodule=datamodule,
        trainer=trainer,
        callbacks=callbacks,
        logger=logger,
    )

