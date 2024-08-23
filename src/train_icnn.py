"""

Check tags before starting!!!

"""

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
import wandb

import hydra
from omegaconf import DictConfig

from hydra.utils import instantiate
import os

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

@hydra.main(config_path="conf", config_name="icnn_train")
def main(cfg: DictConfig):
    wandb._service_wait = 200
    # Initialize the seed
    pl.seed_everything(cfg.seed)

    # Initialize the data module
    print("cfg.datamodule: ", cfg.datamodule)
    datamodule = instantiate(cfg.datamodule)

    # for mlp
    if all(key in cfg.model for key in ["dim", "conds", "batch_size"]):
        cfg.model.dim = datamodule.dims["input_dim"]
        cfg.model.conds = datamodule.dims["conds"]
        cfg.model.batch_size = cfg.datamodule.batch_size

    # cellot
    if "input_dim" in cfg.model:
        cfg.model.input_dim = datamodule.dims["input_dim"]
        cfg.model.target = datamodule.treatment_test

    # Initialize the model
    print("cfg.model: ", cfg.model)
    model = instantiate(cfg.model)

    if "pca" in cfg.datamodule:
        model.pca = datamodule.pca

    wandbcfg = {
        "seed": cfg.seed,
        "sigma": cfg.sigma,
        "lr": model.lr,
    }
    wandbcfg["architecture"] = model.name
    wandbcfg["dataset"] = datamodule.name
    for key in cfg.model:
        if key not in ["name", "lr"]:
            wandbcfg[key] = cfg.model[key]
    for key in cfg.datamodule:
        wandbcfg[key] = cfg.datamodule[key]

    # Set up the logger (wandb)
    wandb_logger = (
        WandbLogger(
            name=f"{datamodule.name}_{model.naming}",
            project="scCFM",
        )
        if cfg.wandb_run
        else None
    )

    # Set up early stopping
    early_stopping = EarlyStopping(**cfg.early_stopping)

    # checkpoint
    checkpoint_callback = ModelCheckpoint(
        dirpath=cfg.checkpoint.dirpath,
        filename=f"{cfg.checkpoint.filename}_{model.name}_{datamodule.name}_{{epoch}}",
        save_top_k=1,
        verbose=True,
        monitor=cfg.checkpoint.monitor,
        mode="min",
    )

    trainer_args = {
        **cfg.trainer,
        "callbacks": [early_stopping, checkpoint_callback],
        "logger": wandb_logger,
        "accelerator": cfg.trainer.accelerator,
        "max_epochs": cfg.trainer.max_epochs,  
        "log_every_n_steps": cfg.trainer.log_every_n_steps,
        "max_time": cfg.trainer.max_time,
    }
    
    # Initialize the PyTorch Lightning trainer
    trainer = pl.Trainer(**trainer_args)

    # Train the model
    if model.name != "Identity":
        trainer.fit(model, datamodule=datamodule)

    # Test
    trainer.test(model, datamodule=datamodule)


if __name__ == "__main__":
    main()
