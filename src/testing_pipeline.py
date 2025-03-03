import os
import torch
import numpy as np
from typing import List

import hydra
from omegaconf import DictConfig
from hydra.utils import instantiate

from pytorch_lightning import (
    LightningDataModule,
    Trainer,
    Callback,
    seed_everything,
)
from pytorch_lightning.loggers import Logger

from src import utils

log = utils.get_logger(__name__)


def test(cfg: DictConfig) -> None:
    """Contains minimal example of the testing pipeline. Evaluates given checkpoint on a testset.

    Args:
        cfg (DictConfig): Configuration composed by Hydra.

    Returns:
        None
    """
    
    if cfg.model._target_ != "src.models.trellis_module.TrellisDummy":   
        cfg.ckpt_path = cfg.ckpt_path + cfg.ckpt
        
        # Convert relative ckpt path to absolute path if necessary
        if not os.path.isabs(cfg.ckpt_path):
            cfg.ckpt_path = os.path.join(
                hydra.utils.get_original_cwd(), cfg.ckpt_path
            )

    # Set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        seed_everything(cfg.seed, workers=True)

    # Init lightning datamodule
    log.info(f"Instantiating datamodule <{cfg.datamodule._target_}>")
    datamodule: LightningDataModule = instantiate(cfg.datamodule)

    # Initialize the model
    if (
        cfg.datamodule._target_ == "src.datamodule.letters_dataloader.LettersDatamodule"
        or cfg.datamodule._target_
        == "src.datamodule.letters_dataloader.LettersBatchDatamodule"
    ):
        if cfg.datamodule.conditional:
            model = instantiate(
                cfg.model,
                datamodule.train_dataset.train_eval_letters,
                num_conditions=datamodule.train_dataset.num_conditions,
            )
        else:
            if not datamodule.save_embeddings:
                model = instantiate(
                    cfg.model,
                    datamodule.train_dataset.train_eval_letters,
                )
            else:
                model = instantiate(
                    cfg.model,
                    datamodule.train_dataset.train_eval_letters,
                    data_for_embed_save=datamodule.data_for_embed_save,
                    save_embeddings=datamodule.save_embeddings,
                )
            
        model_dict = model.state_dict()
        PATH = cfg.ckpt_path
        checkpoint = torch.load(PATH)
        param_dict = {k: v for k, v in checkpoint["state_dict"].items() if k in model_dict}
        model_dict.update(param_dict)
        model.load_state_dict(model_dict)
    
    elif (
        cfg.datamodule._target_ == "src.datamodule.trellis_dataloader.TrellisDatamodule"
    ):  
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
            if cfg.model._target_ == "src.models.trellis_module.TrellisDummy": 
                model = instantiate(
                    cfg.model,
                    datamodule.train_dataset.train_eval_replicas,
                    datamodule.train_dataset.train_populations_means,
                    num_train_replica=datamodule.num_train_replica,
                    num_test_replica=datamodule.num_test_replica,
                    num_val_replica=datamodule.num_val_replica,
                    pca_for_plot=datamodule.pca_for_plot,
                    treatments=datamodule.train_dataset.treatment,
                )
            else: 
                if cfg.model._target_ == "src.models.trellis_module.TrellisMFM":
                    model = instantiate(
                        cfg.model,
                        datamodule.train_dataset.train_eval_replicas,
                        num_train_replica=datamodule.num_train_replica,
                        num_test_replica=datamodule.num_test_replica,
                        num_val_replica=datamodule.num_val_replica,
                        pca_for_plot=datamodule.pca_for_plot,
                        treatments=datamodule.train_dataset.treatment,
                        save_embeddings=datamodule.save_embeddings,
                        data_for_embed_save=datamodule.data_for_embed_save,
                        split=datamodule.split,
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
    
    # Init lightning loggers (this can be)
    logger: List[Logger] = []
    if "logger" in cfg:
        for _, lg_conf in cfg.logger.items():
            if "_target_" in lg_conf:
                log.info(f"Instantiating logger <{lg_conf._target_}>")
                logger.append(instantiate(lg_conf))
                    
    if (
        cfg.datamodule._target_ == "src.datamodule.letters_dataloader.LettersDatamodule"
        or cfg.datamodule._target_
        == "src.datamodule.letters_dataloader.LettersBatchDatamodule"
    ):  
        assert datamodule.batch_size == 1, "Set datamodule.batch_size=1 for testing!"
        assert datamodule.shuffle is False, "Set datamodule.shuffle=False for testing!"
        
        # prefix tag for saving
        fname = cfg.model.name
        print("Generating plots for:", fname)
        
        # Init lightning trainer
        log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
        trainer: Trainer = instantiate(cfg.trainer, logger=logger)
        
        # Log hyperparameters
        trainer.logger.log_hyperparams({"ckpt_path": cfg.ckpt_path})
    
        # Start prediction
        log.info("Starting prediction on train!")
        model.predict_count = 0
        preds_train = trainer.predict(model, datamodule.train_dataloader())

        idcs_train = [2, 11, 23, 31, 45, 52]
        samples_train, trajs_train = [], []
        for i in idcs_train:
            p = preds_train[i]
            traj = p[1]
            idx = p[0]
            source = p[2]
            pred = p[3]
            true = p[4]
            samples_train.append((idx, source[:, :, :], true[:, :, :]))
            trajs_train.append(traj[0])
        
        if cfg.model._target_ == "src.models.letters_module.LettersGNNFM":
            idcs_gif_train = [1, 2, 11, 12, 24, 37, 43, 44, 52]
            samples_gif_train, trajs_gif_train = [], []
            for i in idcs_gif_train:
                p = preds_train[i]
                traj = p[1]
                idx = p[0]
                source = p[2]
                pred = p[3]
                true = p[4]
                samples_gif_train.append((idx, source[:, :, :], true[:, :, :]))
                trajs_gif_train.append(traj[0])
                
            model.gif(
                trajs_gif_train,
                samples_gif_train,
                num_row=3,
                num_step=200,
                tag=f"gif_{fname}_train",
                title=r"Train",
            )
        
        model.plot(
            trajs_train,
            samples_train,
            num_row=6,
            num_step=3,
            tag=f"fig_{fname}_train",
        )
        
        log.info("Starting prediction on val!")
        model.predict_count = 0
        preds_val = trainer.predict(model, datamodule.val_dataloader())

        samples_val, trajs_val = [], []
        for i in range(6):
            p = preds_val[0]
            traj = p[1]
            idx = p[0]
            source = p[2]
            pred = p[3]
            true = p[4]
            samples_val.append((idx[i], source[i].unsqueeze(0), true[i].unsqueeze(0)))
            trajs_val.append(traj[i])

        model.plot(
            trajs_val,
            samples_val,
            num_row=6,
            num_step=3,
            tag=f"fig_{fname}_val",
        )
        
        log.info("Starting prediction on test!")
        model.predict_count = 0 
        preds_test = trainer.predict(model, datamodule.test_dataloader())
  
        samples_test, trajs_test = [], []
        for i in range(6):
            p = preds_test[0]
            traj = p[1]
            idx = p[0]
            source = p[2]
            pred = p[3]
            true = p[4]
            samples_test.append((idx[i], source[i].unsqueeze(0), true[i].unsqueeze(0)))
            trajs_test.append(traj[i])
        
        model.plot(
            trajs_test,
            samples_test,
            num_row=6,
            num_step=3,
            tag=f"fig_{fname}_test",
        )

        if cfg.model._target_ == "src.models.letters_module.LettersGNNFM":
            rng = np.random.default_rng(42)
            idcs_gif_test = [1, 2, 3, 4, 5, 6, 7, 8, 9]
            samples_gif_test, trajs_gif_test = [], []
            for i in idcs_gif_test:
                p = preds_test[0]
                traj = p[1]
                tj = traj[i]
                idcs = rng.choice(
                    np.arange(tj.shape[1]),
                    size=700,
                    replace=False,
                )
                
                idx = p[0]
                source = p[2]
                pred = p[3]
                true = p[4]
                samples_gif_test.append(
                    (idx[i], source[i, idcs].unsqueeze(0), true[i, idcs].unsqueeze(0))
                )
                trajs_gif_test.append(tj[:, idcs, :])

            model.gif(
                trajs_gif_test,
                samples_gif_test,
                num_row=3,
                num_step=200,
                tag=f"gif_{fname}_test",
                title=r"Test",
            )
        
        # plot train and test together
        trajs = trajs_train[3:5] + trajs_test[2:4]
        samples = samples_train[3:5] + samples_test[2:4]
        model.plot(
            trajs,
            samples,
            num_row=4,
            num_step=3,
            tag=f"fig_{fname}_main",
        )
        
        # Start validation
        log.info("Starting validating!")
        trainer.validate(model=model, datamodule=datamodule)

        # Start testing
        log.info("Starting testing!")
        trainer.test(model=model, datamodule=datamodule)
    
    elif cfg.datamodule._target_ == "src.datamodule.trellis_dataloader.TrellisDatamodule":
        
        ### Start prediction        
    
        # Init lightning callbacks
        callbacks: List[Callback] = []
        if "callbacks" in cfg:
            for _, cb_conf in cfg.callbacks.items():
                if "_target_" in cb_conf:
                    log.info(f"Instantiating callback <{cb_conf._target_}>")
                    callbacks.append(hydra.utils.instantiate(cb_conf))
         
        # Init lightning trainer    
        # init tranier from last checkpoint. 
        # This will only run inference if config hparam yaml set the same as used for training
        log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
        trainer: Trainer = hydra.utils.instantiate(
            cfg.trainer,
            callbacks=callbacks,
            logger=logger,
            _convert_="partial",
            resume_from_checkpoint=cfg.ckpt_path,
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
        #if cfg.get("train"):
        log.info("Starting training!")
        trainer.fit(model=model, datamodule=datamodule)
        
        # Test the model
        #if cfg.get("test"):
        log.info("Starting testing!")
        trainer.test(model=model, datamodule=datamodule)
        #trainer.test(model=model, datamodule=datamodule, ckpt_path=cfg.ckpt_path)
        
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

    else:
        raise ValueError("Unknown datamodule!")
