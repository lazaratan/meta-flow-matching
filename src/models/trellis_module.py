"""
Lightning module for the organoid drug-screen (trellis) dataset.
In this code base, we use the name "trellis" as a short hand for this dataset.
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import wandb

from torchdyn.core import NeuralODE
from util.distribution_distances import compute_distribution_distances
from src.models.components.mlp import (
    Flow, 
    torch_wrapper_flow_cond,
    torch_wrapper_gnn_flow_cond,
)
from src.models.components.gnn import GlobalGNN
from sklearn.decomposition import PCA

from pathlib import Path

script_directory = Path(__file__).absolute().parent
sccfm_directory = script_directory.parent.parent

TREAT_NAMES = {
    "S": "SN-38",
    "F": "5-FU",
    "O": "Oxaliplatin",
    "L": "LGK-974",
    "V": "VX-970",
    "VS": "SN-38 + VX-970",
    "C": "Cetux",
    "CS": "SN-38 + Cetux",
    "CF": "5-FU + Cetux",
    "SF": "SN-38 + 5-FU",
    "CSF": "SN-38 + 5-FU + Cetux",
}

class TrellisFM(pl.LightningModule):
    def __init__(
        self,
        train_eval_batches,
        lr=1e-4,
        dim=43,
        num_hidden=512,
        num_layers=7,
        num_treat_conditions=11,
        base="source",
        integrate_time_steps=500,
        num_train_replica=861,
        num_test_replica=33,
        num_val_replica=33,
        pca_for_plot=None,
        treatments=None,
        pca=None,
        pca_space_eval=True,
        run_validation=True, 
        name="trellis_fm",
        seed=0,
    ) -> None:
        super().__init__()

        # Important: This property controls manual optimization.
        self.automatic_optimization = True

        self.save_hyperparameters(ignore=["train_eval_batches", "pca_for_plot"])

        self.model = Flow(
            D=dim,
            num_conditions=num_treat_conditions,
            num_hidden=num_hidden,
            num_layers=num_layers,
            skip_connections=True,
        ).cuda()
        
        self.lr = lr
        self.dim = dim
        self.num_hidden = num_hidden
        self.integrate_time_steps = integrate_time_steps
        self.num_train_replica = num_train_replica
        self.pca = pca # log if PCA is being used on data
        self.pca_space_eval = pca_space_eval

        self.run_validation = run_validation
        
        assert base in [
            "source",
            "gaussian",
        ], "Invalid base. Must be either 'source' or 'gaussian'"
        self.base = base
        self.name = name
        
        # eval cell batch rng
        self.rng = np.random.default_rng(seed)
        
        # for training data eval
        if train_eval_batches is not None:
            self.train_eval_batches = train_eval_batches
            self.use_pre_train_eval_batches = True
        else:
            self.num_train_evals = 100
            self.train_evals_count = 0
            self.train_eval_batches = []
            self.use_pre_train_eval_batches = False
            
        # for plotting
        self.pca_for_plot = pca_for_plot
        self.treatments = treatments
        n_plot_idcs = min(num_test_replica, num_val_replica)
        interval_plot_idcs = int(n_plot_idcs / 5)
        self.idcs_for_plot = [i for i in range(1, n_plot_idcs, interval_plot_idcs)]
        
        self.train_metrics = {"PDO": [], "PDOF": [], "F": []}
        self.val_metrics = {"PDO": [], "PDOF": [], "F": []}
        self.test_metrics = {"PDO": [], "PDOF": [], "F": []}

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        return optimizer

    def forward(self, t, x):
        return self.model(t.squeeze(-1), x)

    def compute_loss(self, source_samples, target_samples, treat_cond):
        t = torch.rand_like(source_samples[..., 0, None])

        if self.base == "source":
            x = (1.0 - t) * source_samples + t * target_samples
            u = target_samples - source_samples
            b = self.forward(t, torch.cat((x, treat_cond), dim=-1))
            loss = b.norm(dim=-1) ** 2 - 2.0 * (b * u).sum(dim=-1)
        elif self.base == "gaussian":
            z = torch.randn_like(target_samples)
            x = (1.0 - t) * z + t * target_samples
            u = target_samples - z
            b = self.forward(t, torch.cat((x, treat_cond), dim=-1))
            loss = ((b - u) ** 2).sum(dim=-1)
        else:
            raise ValueError(f"unknown base: {self.base}")

        loss = loss.mean()
        return loss

    def training_step(self, batch, batch_idx):
        # save subset of training batches to evaluate on train data later 
        if self.use_pre_train_eval_batches is False:
            if (
                self.current_epoch % (self.trainer.check_val_every_n_epoch - 1)
            ) == 0 and self.train_evals_count < self.num_train_evals:
                self.train_eval_batches.append(batch)
                self.train_evals_count += 1
                        
        _, _, x0, x1, _, _, treat_cond = batch
        loss = self.compute_loss(
            x0.view(-1, x0.shape[-1]).float(),
            x1.view(-1, x1.shape[-1]).float(),
            treat_cond.view(-1, treat_cond.shape[-1]).float(),
        )
        self.log("train/loss", loss, on_step=False, on_epoch=True)
        return loss
        
    def training_epoch_end(self, outputs):
        if self.current_epoch == self.trainer.max_epochs - 1:
            for batch in self.train_eval_batches:
                self.eval_batch(batch, prefix="train")
                
            list_train_metrics = []
            for culture_key in list(self.train_metrics.keys()):
                eval_metrics_mean_train = {
                    k: np.mean([m[k] for m in self.train_metrics[culture_key]])
                    for k in self.train_metrics[culture_key][0]
                }
                for key, value in eval_metrics_mean_train.items():
                    self.log(
                        f"train/{key}-{culture_key}",
                        value,
                        on_step=False,
                        on_epoch=True,
                    )
                list_train_metrics.append(eval_metrics_mean_train)

            avg_train_metrics = dict(pd.DataFrame(list_train_metrics).mean())
            for key, value in avg_train_metrics.items():
                self.log(f"train/{key}-avg", value, on_step=False, on_epoch=True)

            if self.use_pre_train_eval_batches is False:
                self.train_eval_batches = []
                self.train_evals_count = 0
                
            self.train_metrics = {"PDO": [], "PDOF": [], "F": []}

        # Save a checkpoint of the model
        ckpt_name = (
            "last.ckpt"
            if self.current_epoch == self.trainer.max_epochs - 1
            else "ckpt.ckpt"
        )
        ckpt_path = os.path.join(self.trainer.log_dir, "checkpoints", ckpt_name)
        self.trainer.save_checkpoint(ckpt_path)

    def predict_step(self, batch, batch_idx):
        # only used at test time when evaluating models on train data
        if self.train_evals_count < self.num_train_evals:
            self.eval_batch(batch, prefix="train")
            self.train_evals_count += 1
        elif self.train_evals_count == self.num_train_evals:
            # only used at test time when evaluating models on train data
            list_eval_metrics = []
            for culture_key in list(self.train_metrics.keys()):
                if not self.train_metrics[culture_key]:
                    continue
                eval_metrics_mean_ood = {
                    k: np.mean([m[k] for m in self.train_metrics[culture_key]])
                    for k in self.train_metrics[culture_key][0]
                }
                list_eval_metrics.append(eval_metrics_mean_ood)

            avg_eval_metrics = dict(pd.DataFrame(list_eval_metrics).mean())
            self.train_evals_count += 1
            print(avg_eval_metrics) # pyl predict hook doesn't let you log so we print ...
        else:
            pass

    def validation_step(self, batch, batch_idx):
        if self.run_validation:
            self.eval_batch(batch, prefix="val")
        else:
            return None

    def validation_epoch_end(self, outputs):
        if self.run_validation:
            list_val_metrics = []
            for culture_key in list(self.val_metrics.keys()):
                if not self.val_metrics[culture_key]:
                    continue
                eval_metrics_mean_ood = {
                    k: np.mean([m[k] for m in self.val_metrics[culture_key]])
                    for k in self.val_metrics[culture_key][0]
                }
                for key, value in eval_metrics_mean_ood.items():
                    self.log(
                        f"val/{key}-{culture_key}", value, on_step=False, on_epoch=True
                    )
                list_val_metrics.append(eval_metrics_mean_ood)

            avg_val_metrics = dict(pd.DataFrame(list_val_metrics).mean())
            for key, value in avg_val_metrics.items():
                self.log(f"val/{key}-avg", value, on_step=False, on_epoch=True)

            self.val_metrics = {"PDO": [], "PDOF": [], "F": []}
        else:
            return None

    def test_step(self, batch, batch_idx):
        self.eval_batch(batch, prefix="test")

    def test_epoch_end(self, outputs):
        list_test_metrics = []
        for culture_key in list(self.test_metrics.keys()):
            if not self.test_metrics[culture_key]:
                continue
            eval_metrics_mean_ood = {
                k: np.mean([m[k] for m in self.test_metrics[culture_key]])
                for k in self.test_metrics[culture_key][0]
            }
            for key, value in eval_metrics_mean_ood.items():
                self.log(
                    f"test/{key}-{culture_key}", value, on_step=False, on_epoch=True
                )
            list_test_metrics.append(eval_metrics_mean_ood)

        avg_test_metrics = dict(pd.DataFrame(list_test_metrics).mean())
        for key, value in avg_test_metrics.items():
            self.log(f"test/{key}-avg", value, on_step=False, on_epoch=True)

        self.test_metrics = {"PDO": [], "PDOF": [], "F": []}

    def eval_batch(self, batch, prefix):
        idx, culture, x0, x1, x1_full, _, treat_cond = batch
        
        node = NeuralODE(
            torch_wrapper_flow_cond(self.model),
            solver="dopri5",
            sensitivity="adjoint",
            atol=1e-4,
            rtol=1e-4,
        )

        time_span = torch.linspace(0, 1, self.integrate_time_steps)
        ts_for_plot = self.integrate_time_steps / 5
        ts = [i for i in range(int(ts_for_plot - 1), self.integrate_time_steps, int(ts_for_plot))]
        
        with torch.no_grad():
            if len(x0.shape) > 3:
                x0 = x0.squeeze(0)
                x1 = x1.squeeze(0)
                x1_full = x1_full.squeeze(0)
                treat_cond = treat_cond.squeeze(0)

            for i in range(
                x0.shape[0]
            ):  # for loop used to allow for replicate batching for eval
                x0_i = x0[i].float()
                
                if self.base == "gaussian":
                    x0_i = torch.randn_like(x0_i)
                    
                treat_cond_i = treat_cond[i].float()
                pred_batches = []
                traj_batches = []
                idcs_batches = np.arange(x0_i.shape[0])
                for j in range(0, x0_i.shape[0], 1024):
                    idcs = idcs_batches[j : j + 1024]
                    traj = node.trajectory(torch.cat((x0_i[idcs].cuda(), treat_cond_i[idcs].cuda()), dim=-1), t_span=time_span)
                    pred_batches.append(traj[-1, :, : self.model.D])
                    traj_batches.append(traj[ts, :, : self.model.D])

                pred = torch.cat(pred_batches, dim=0)
                traj = torch.cat(traj_batches, dim=1)
                
                if self.pca is not None and self.dim == 43 and self.pca_space_eval:
                    pred = self.pca.inverse_transform(pred.cpu().numpy())
                    pred = torch.tensor(pred).cuda()
                    
                true = x1_full.float() if self.pca is not None and self.dim == 43 else x1.float()
                
                names, dd = compute_distribution_distances(
                    pred.unsqueeze(1).to(true),
                    true[0].unsqueeze(1),
                )
       
                if prefix == "train":
                    self.train_metrics[culture[0]].append({**dict(zip(names, dd))})
                elif prefix == "val":
                    self.val_metrics[culture[0]].append({**dict(zip(names, dd))})
                elif prefix == "test":
                    self.test_metrics[culture[0]].append({**dict(zip(names, dd))})
                else:
                    raise ValueError(f"unknown prefix: {prefix}")

                # plot in 2d PCA space
                if idx in self.idcs_for_plot and self.pca_for_plot is not None:
                    print("Plotting 2D-PCA predictions ... \n")
                    treat_id = torch.argmax(treat_cond_i).item()
                    treat_name = TREAT_NAMES[self.treatments[treat_id]]
                    self.plot(
                        x0_i.cpu().numpy(),
                        traj.cpu().numpy(),
                        true.squeeze(0).cpu().numpy(),
                        prefix,
                        treat_name=treat_name,
                        tag=f"fm_traj_treat_{treat_name}_{idx.item()}",
                    )

    def plot(self, source, traj, target, prefix, treat_name, tag):
        # Flatten traj to [t*n, d]
        t, n, d = traj.shape
        m, d = target.shape

        traj_flat = traj.reshape(-1, d)

        # Get 2D PCA
        if self.pca_for_plot is not None:
            source_pca = self.pca_for_plot.transform(source)
            traj_pca = self.pca_for_plot.transform(traj_flat).reshape(t, n, 2)
            target_pca = self.pca_for_plot.transform(target)
        else:
            all_data = np.concatenate([source, traj_flat, target], axis=0)
            all_data_pca = PCA(n_components=2).fit_transform(all_data)
            source_pca = all_data_pca[:n]
            traj_pca = all_data_pca[n : n + t * n].reshape(t, n, 2)
            target_pca = all_data_pca[-m:]

        # Create subplots
        fig, axs = plt.subplots(1, 3, figsize=(9, 3))

        # Center the suptitle
        mid = (fig.subplotpars.right + fig.subplotpars.left) / 2
        fig.suptitle(
            f"{treat_name}", fontsize=16, x=mid, y=0.85, verticalalignment="center"
        )

        # Plot source points
        axs[0].scatter(
            source_pca[:, 0],
            source_pca[:, 1],
            c="lightsteelblue",
            label="Source",
            alpha=0.9,
            s=15,
            rasterized=True,
        )
        axs[0].set_title("Source")
        axs[0].set_xlabel("PCA 1")
        axs[0].set_ylabel("PCA 2")
        axs[0].tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)
        axs[0].grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.2)

        # Plot prediction points
        preds = []
        for i in range(n):
            preds.append([traj_pca[-1, i, 0], traj_pca[-1, i, 1]])
        pred = np.stack(preds)
        axs[1].scatter(
            pred[:, 0],
            pred[:, 1],
            c="rosybrown",
            marker="o",
            label="Prediction",
            alpha=0.6,
            s=15,
            rasterized=True,
        )
        axs[1].set_title("Prediction")
        axs[1].set_xlabel("PCA 1")
        axs[1].set_ylabel("PCA 2")
        axs[1].tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)
        axs[1].grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.2)

        # Plot target points
        axs[2].scatter(
            target_pca[:, 0],
            target_pca[:, 1],
            c="navy",
            label="Target",
            alpha=0.3,
            s=15,
            rasterized=True,
        )
        axs[2].set_title("Target")
        axs[2].set_xlabel("PCA 1")
        axs[2].set_ylabel("PCA 2")
        axs[2].tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)
        axs[2].grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.2)

        # Adjust layout
        fig.subplots_adjust(top=0.85)
        plt.tight_layout(rect=[0, 0, 1, 0.95])

        # Save the plot
        fname = f"{sccfm_directory}/figs/trellis_{prefix}_{tag}.pdf"
        os.makedirs(os.path.dirname(fname), exist_ok=True)
        fig.savefig(
            fname, bbox_inches="tight", pad_inches=0.1, transparent=True, dpi=300
        )

        # Optionally, you can log this to wandb or display it directly
        wandb.log({f"{prefix}_{tag}": wandb.Image(fig)})
        plt.close()


class TrellisCGFM(pl.LightningModule):
    def __init__(
        self,
        train_eval_batches,
        lr=1e-4,
        dim=43,
        num_hidden=512,
        num_layers=7,
        base="source",
        integrate_time_steps=500,
        num_train_replica=861,
        num_test_replica=33,
        num_val_replica=33,
        pca_for_plot=None,
        treatments=None,
        num_exp_conditions=927,
        num_treat_conditions=11,
        pca=None,
        pca_space_eval=True,
        run_validation=True,
        name="trellis_cgfm",
        seed=0,
    ) -> None:
        super().__init__()

        # Important: This property controls manual optimization.
        self.automatic_optimization = True

        self.save_hyperparameters(ignore=["train_eval_batches", "pca_for_plot"])

        self.num_exp_conditions = num_train_replica + num_test_replica + num_val_replica
        assert self.num_exp_conditions == 927, "Invalid number of experimental conditions. Should be 927."
        
        self.num_treat_conditions = num_treat_conditions
        self.num_conditions = num_exp_conditions + num_treat_conditions
        
        self.model = Flow(
            D=dim,
            num_hidden=num_hidden,
            num_layers=num_layers,
            num_conditions=num_exp_conditions,
            num_treat_conditions=num_treat_conditions,
            skip_connections=True,
        ).cuda()  # if using small expt data
        
        self.lr = lr
        self.dim = dim
        self.num_hidden = num_hidden
        self.integrate_time_steps = integrate_time_steps
        self.num_train_replica = num_train_replica
        self.num_test_replica = num_test_replica
        self.num_val_replica = num_val_replica
        self.pca = pca 
        self.pca_space_eval = pca_space_eval

        self.run_validation = run_validation
        
        assert base in [
            "source",
            "gaussian",
        ], "Invalid base. Must be either 'source' or 'gaussian'"
        self.base = base
        self.name = name
        
        # eval cell batch rng
        self.rng = np.random.default_rng(seed)
        
        # for training data eval
        if train_eval_batches is not None:
            self.train_eval_batches = train_eval_batches
            self.use_pre_train_eval_batches = True
        else:
            self.num_train_evals = 100
            self.train_evals_count = 0
            self.train_eval_batches = []
            self.use_pre_train_eval_batches = False
        
        #self.num_train_evals = 100 #6
        #self.train_evals_count = 0
        #self.train_eval_batches = []
        
        # for plotting
        self.pca_for_plot = pca_for_plot
        self.treatments = treatments
        n_plot_idcs = min(num_test_replica, num_val_replica)
        interval_plot_idcs = int(n_plot_idcs / 5)
        self.idcs_for_plot = [i for i in range(1, n_plot_idcs, interval_plot_idcs)]

        self.train_metrics = {"PDO": [], "PDOF": [], "F": []}
        self.val_metrics = {"PDO": [], "PDOF": [], "F": []}
        self.test_metrics = {"PDO": [], "PDOF": [], "F": []}

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        return optimizer

    def forward(self, t, x):
        return self.model(t.squeeze(-1), x)

    def compute_loss(self, source_samples, target_samples, cond):
        t = torch.rand_like(source_samples[..., 0, None])

        if self.base == "source":
            x = (1.0 - t) * source_samples + t * target_samples
            u = target_samples - source_samples
            b = self.forward(t, torch.cat((x, cond), dim=-1).float())
            loss = b.norm(dim=-1) ** 2 - 2.0 * (b * u).sum(dim=-1)
        elif self.base == "gaussian":
            z = torch.randn_like(target_samples)
            x = (1.0 - t) * z + t * target_samples
            u = target_samples - z
            b = self.forward(t, torch.cat((x, cond), dim=-1).float())
            loss = ((b - u) ** 2).sum(dim=-1)
        else:
            raise ValueError(f"unknown base: {self.base}")

        loss = loss.mean()
        return loss
        
    def get_exp_cond(self, idx, x0):
        exp_cond = torch.nn.functional.one_hot(
            idx.long(), num_classes=self.num_exp_conditions
        )
        if len(exp_cond.shape) < 2:
            exp_cond = exp_cond.unsqueeze(0)
        exp_cond = exp_cond.unsqueeze(1).expand(-1, x0.shape[1], -1)
        return exp_cond
    
    def training_step(self, batch, batch_idx):
        # save subset of training batches to evaluate on train data later
        if self.use_pre_train_eval_batches is False:
            if (
                self.current_epoch % (self.trainer.check_val_every_n_epoch - 1)
            ) == 0 and self.train_evals_count < self.num_train_evals:
                self.train_eval_batches.append(batch)
                self.train_evals_count += 1

        idx, _, x0, x1, _, _, treat_cond = batch
        exp_cond = self.get_exp_cond(idx, x0)
        
        cond = torch.cat((exp_cond, treat_cond), dim=-1)
        loss = self.compute_loss(
            x0.view(-1, x0.shape[-1]).float(),
            x1.view(-1, x1.shape[-1]).float(),
            cond.view(-1, cond.shape[-1]).float(),
        )
        self.log("train/loss", loss, on_step=False, on_epoch=True)
        return loss

    def training_epoch_end(self, outputs):
        if self.current_epoch == self.trainer.max_epochs - 1:
            for batch in self.train_eval_batches:
                self.eval_batch(batch, prefix="train")

            list_train_metrics = []
            for culture_key in list(self.train_metrics.keys()):
                eval_metrics_mean_train = {
                    k: np.mean([m[k] for m in self.train_metrics[culture_key]])
                    for k in self.train_metrics[culture_key][0]
                }
                for key, value in eval_metrics_mean_train.items():
                    self.log(
                        f"train/{key}-{culture_key}",
                        value,
                        on_step=False,
                        on_epoch=True,
                    )
                list_train_metrics.append(eval_metrics_mean_train)

            avg_train_metrics = dict(pd.DataFrame(list_train_metrics).mean())
            for key, value in avg_train_metrics.items():
                self.log(f"train/{key}-avg", value, on_step=False, on_epoch=True)

            if self.use_pre_train_eval_batches is False:
                self.train_eval_batches = []
                self.train_evals_count = 0

            self.train_metrics = {"PDO": [], "PDOF": [], "F": []}

        # Save a checkpoint of the model
        ckpt_name = (
            "last.ckpt"
            if self.current_epoch == self.trainer.max_epochs - 1
            else "ckpt.ckpt"
        )
        ckpt_path = os.path.join(self.trainer.log_dir, "checkpoints", ckpt_name)
        self.trainer.save_checkpoint(ckpt_path)

    def validation_step(self, batch, batch_idx):
        if self.run_validation:
            self.eval_batch(batch, prefix="val")
        else:
            return None

    def validation_epoch_end(self, outputs):
        if self.run_validation:
            list_val_metrics = []
            for culture_key in list(self.val_metrics.keys()):
                if not self.val_metrics[culture_key]:
                    continue
                eval_metrics_mean_ood = {
                    k: np.mean([m[k] for m in self.val_metrics[culture_key]])
                    for k in self.val_metrics[culture_key][0]
                }
                for key, value in eval_metrics_mean_ood.items():
                    self.log(
                        f"val/{key}-{culture_key}", value, on_step=False, on_epoch=True
                    )
                list_val_metrics.append(eval_metrics_mean_ood)

            avg_val_metrics = dict(pd.DataFrame(list_val_metrics).mean())
            for key, value in avg_val_metrics.items():
                self.log(f"val/{key}-avg", value, on_step=False, on_epoch=True)

            self.val_metrics = {"PDO": [], "PDOF": [], "F": []}
        else:
            return None

    def test_step(self, batch, batch_idx):
        self.eval_batch(batch, prefix="test")

    def test_epoch_end(self, outputs):
        list_test_metrics = []
        for culture_key in list(self.test_metrics.keys()):
            if not self.test_metrics[culture_key]:
                continue
            eval_metrics_mean_ood = {
                k: np.mean([m[k] for m in self.test_metrics[culture_key]])
                for k in self.test_metrics[culture_key][0]
            }
            for key, value in eval_metrics_mean_ood.items():
                self.log(
                    f"test/{key}-{culture_key}", value, on_step=False, on_epoch=True
                )
            list_test_metrics.append(eval_metrics_mean_ood)

        avg_test_metrics = dict(pd.DataFrame(list_test_metrics).mean())
        for key, value in avg_test_metrics.items():
            self.log(f"test/{key}-avg", value, on_step=False, on_epoch=True)

        self.test_metrics = {"PDO": [], "PDOF": [], "F": []}

    def eval_batch(self, batch, prefix):
        if prefix == 'train':
            idx, culture, x0, x1, x1_full, _, treat_cond = batch
            exp_cond = self.get_exp_cond(idx, x0)
        else:
            idx, culture, x0, x1, x1_full, _, treat_cond = batch
            
            if prefix == 'val' or prefix == 'test':
                num_train_conditions = self.num_exp_conditions - (self.num_val_replica + self.num_test_replica)  # for replica split use: - 111 - 103. TODO: make this a hparam.
                exp_cond = (
                    torch.ones((x0.shape[0], x0.shape[1], num_train_conditions)).cuda()
                    / num_train_conditions
                )
                exp_cond = torch.cat(
                    (
                        exp_cond,
                        torch.zeros(
                            (
                                x0.shape[0],
                                x0.shape[1],
                                self.num_val_replica + self.num_test_replica, #33 + 33 for patient split, for replica split use: 111 + 103.
                            )
                        ).cuda(),
                    ),
                    dim=-1,
                )
        
        node = NeuralODE(
            torch_wrapper_flow_cond(self.model),
            solver="dopri5",
            sensitivity="adjoint",
            atol=1e-4,
            rtol=1e-4,
        )

        time_span = torch.linspace(0, 1, self.integrate_time_steps)
        ts_for_plot = self.integrate_time_steps / 5
        ts = [i for i in range(int(ts_for_plot - 1), self.integrate_time_steps, int(ts_for_plot))]

        with torch.no_grad():
            if len(x0.shape) > 3:
                x0 = x0.squeeze(0)
                x1 = x1.squeeze(0)
                x1_full = x1_full.squeeze(0)
                treat_cond = treat_cond.squeeze(0)
                exp_cond = exp_cond.squeeze(0)
                
            for i in range(x0.shape[0]):  # for loop used to allow for replicate batching for eval
                x0_i = x0[i].float()
                
                if self.base == "gaussian":
                    x0_i = torch.randn_like(x0_i)
                    
                exp_cond_i = exp_cond[i]
                treat_cond_i = treat_cond[i].float()
                pred_batches = []
                traj_batches = []
                idcs_batches = np.arange(x0_i.shape[0])
                for j in range(0, x0_i.shape[0], 1024): # smaller batch size for eval to fit in memory
                    idcs = idcs_batches[j : j + 1024]
                    traj = node.trajectory(
                        torch.cat((x0_i[idcs].cuda(), exp_cond_i[idcs].cuda(), treat_cond_i[idcs].cuda()), dim=-1),
                        t_span=time_span,
                    )
                    pred_batches.append(traj[-1, :, : self.model.D])
                    traj_batches.append(traj[ts, :, : self.model.D])

                pred = torch.cat(pred_batches, dim=0)
                traj = torch.cat(traj_batches, dim=1)

                if self.pca is not None and self.dim == 43 and self.pca_space_eval:
                    pred = self.pca.inverse_transform(pred.cpu().numpy())
                    pred = torch.tensor(pred).cuda()
                    
                true = x1_full.float() if self.pca is not None and self.dim == 43 else x1.float()

                names, dd = compute_distribution_distances(
                    pred.unsqueeze(1).to(true),
                    true[0].unsqueeze(1),
                )

                if prefix == "train":
                    self.train_metrics[culture[0]].append({**dict(zip(names, dd))})
                elif prefix == "val":
                    self.val_metrics[culture[0]].append({**dict(zip(names, dd))})
                elif prefix == "test":
                    self.test_metrics[culture[0]].append({**dict(zip(names, dd))})
                else:
                    raise ValueError(f"unknown prefix: {prefix}")
            
                # plot in 2d PCA space
                if idx in self.idcs_for_plot and self.pca_for_plot is not None:
                    print("Plotting 2D-PCA predictions ... \n")
                    treat_id = torch.argmax(treat_cond_i).item()
                    treat_name = TREAT_NAMES[self.treatments[treat_id]]
                    self.plot(
                        x0_i.cpu().numpy(),
                        traj.cpu().numpy(),
                        true.squeeze(0).cpu().numpy(),
                        prefix,
                        treat_name=treat_name,
                        tag=f"cgfm_traj_treat_{treat_name}_{idx.item()}",
                    )

    def plot(self, source, traj, target, prefix, treat_name, tag):             
        # Flatten traj to [t*n, d]
        t, n, d = traj.shape
        m, d = target.shape

        traj_flat = traj.reshape(-1, d)

        # Get 2D PCA
        if self.pca_for_plot is not None:
            source_pca = self.pca_for_plot.transform(source)
            traj_pca = self.pca_for_plot.transform(traj_flat).reshape(t, n, 2)
            target_pca = self.pca_for_plot.transform(target)
        else:
            all_data = np.concatenate([source, traj_flat, target], axis=0)
            all_data_pca = PCA(n_components=2).fit_transform(all_data)
            source_pca = all_data_pca[:n]
            traj_pca = all_data_pca[n:n + t * n].reshape(t, n, 2)
            target_pca = all_data_pca[-m:]

        # Create subplots
        fig, axs = plt.subplots(1, 3, figsize=(9, 3))
        
        # Center the suptitle
        mid = (fig.subplotpars.right + fig.subplotpars.left) / 2
        fig.suptitle(f"{treat_name}", fontsize=16, x=mid, y=0.85, verticalalignment='center')

        # Plot source points
        axs[0].scatter(
            source_pca[:, 0],
            source_pca[:, 1],
            c="lightsteelblue",
            label="Source",
            alpha=0.9,
            s=15,
            rasterized=True,
        )
        axs[0].set_title("Source")
        axs[0].set_xlabel("PCA 1")
        axs[0].set_ylabel("PCA 2")
        axs[0].tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)
        axs[0].grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.2)

        # Plot prediction points
        preds = []
        for i in range(n):
            preds.append([traj_pca[-1, i, 0], traj_pca[-1, i, 1]])
        pred = np.stack(preds)
        axs[1].scatter(
            pred[:, 0],
            pred[:, 1],
            c="rosybrown",
            marker="o",
            label="Prediction",
            alpha=0.6,
            s=15,
            rasterized=True,
        )
        axs[1].set_title("Prediction")
        axs[1].set_xlabel("PCA 1")
        axs[1].set_ylabel("PCA 2")
        axs[1].tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)
        axs[1].grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.2)
        
        # Plot target points
        axs[2].scatter(
            target_pca[:, 0],
            target_pca[:, 1],
            c="navy",
            label="Target",
            alpha=0.3,
            s=15,
            rasterized=True,
        )
        axs[2].set_title("Target")
        axs[2].set_xlabel("PCA 1")
        axs[2].set_ylabel("PCA 2")
        axs[2].tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)
        axs[2].grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.2)

        # Adjust layout
        fig.subplots_adjust(top=0.85)
        plt.tight_layout(rect=[0, 0, 1, 0.95])

        # Save the plot
        fname = f"{sccfm_directory}/figs/trellis_{prefix}_{tag}.pdf"
        os.makedirs(os.path.dirname(fname), exist_ok=True)
        fig.savefig(
            fname, bbox_inches="tight", pad_inches=0.1, transparent=True, dpi=300
        )

        # Optionally, you can log this to wandb or display it directly
        wandb.log({f"{prefix}_{tag}": wandb.Image(fig)})
        plt.close()


class TrellisMFM(pl.LightningModule):
    def __init__(
        self,
        train_eval_batches,
        flow_lr=1e-4,
        gnn_lr=1e-4,
        dim=43,
        num_hidden=512,
        num_layers_decoder=7,
        num_hidden_gnn=512,
        num_layers_gnn=2,
        knn_k=100,
        num_treat_conditions=None,
        num_cell_conditions=None,
        base="source",
        num_train_replica=861,
        num_test_replica=33,
        num_val_replica=33,
        pca_for_plot=None,
        treatments=None,
        integrate_time_steps=500,
        pca=None,
        pca_space_eval=True,
        run_validation=True,
        name="trellis_mfm",
        seed=0,
    ) -> None:
        super().__init__()

        # Important: This property controls manual optimization.
        self.automatic_optimization = False

        self.save_hyperparameters(ignore=["train_eval_batches", "pca_for_plot"])

        self.model = GlobalGNN(
            D=dim,
            num_hidden_decoder=num_hidden,
            num_layers_decoder=num_layers_decoder,
            num_hidden_gnn=num_hidden_gnn,
            num_layers_gnn=num_layers_gnn,
            knn_k=knn_k,
            num_treat_conditions=num_treat_conditions,
            num_cell_conditions=num_cell_conditions,
            skip_connections=True,
        ).cuda()
        
        assert len(list(self.model.parameters())) == len(
            list(self.model.decoder.parameters())
        ) + len(list(self.model.gcn_convs.parameters()))

        self.flow_lr = flow_lr
        self.gnn_lr = gnn_lr
        self.dim = dim
        self.num_hidden = num_hidden
        self.knn_k = knn_k
        self.num_train_replica = num_train_replica
        self.integrate_time_steps = integrate_time_steps
        self.pca = pca
        self.pca_space_eval = pca_space_eval
        
        self.run_validation = run_validation
                
        assert base in [
            "source",
            "gaussian",
        ], "Invalid base. Must be either 'source' or 'gaussian'"
        self.base = base
        self.name = name
        
        # eval cell batch rng
        self.rng = np.random.default_rng(seed)

        # for training data eval
        if train_eval_batches is not None:
            self.train_eval_batches = train_eval_batches
            self.use_pre_train_eval_batches = True
        else:
            self.num_train_evals = 100
            self.train_evals_count = 0
            self.train_eval_batches = []
            self.use_pre_train_eval_batches = False
            
        #self.num_train_evals = 100 #6
        #self.train_evals_count = 0
        #self.train_eval_batches = []
        
        self.embeddings = {}
        
        # for plotting
        self.pca_for_plot = pca_for_plot
        self.treatments = treatments
        n_plot_idcs = min(num_test_replica, num_val_replica)
        interval_plot_idcs = int(n_plot_idcs / 5)
        self.idcs_for_plot = [i for i in range(1, n_plot_idcs, interval_plot_idcs)]
        
        self.train_metrics = {"PDO": [], "PDOF": [], "F": []}
        self.val_metrics = {"PDO": [], "PDOF": [], "F": []}
        self.test_metrics = {"PDO": [], "PDOF": [], "F": []}

    def configure_optimizers(self):
        self.flow_optimizer = torch.optim.Adam(self.model.decoder.parameters(), lr=self.flow_lr)
        self.gnn_optimizer = torch.optim.Adam(
            self.model.gcn_convs.parameters(),
            lr=self.gnn_lr,
        )
        return self.flow_optimizer, self.gnn_optimizer

    def compute_loss(self, embedding, source_samples, target_samples, treat_cond):
        t = torch.rand_like(source_samples[..., 0, None])

        if self.base == "source":
            y = (1.0 - t) * source_samples + t * target_samples
            u = target_samples - source_samples

            b = self.model.flow(
                embedding, t.squeeze(-1), torch.cat((y, treat_cond), dim=-1)
            )
            loss = b.norm(dim=-1) ** 2 - 2.0 * (b * u).sum(dim=-1)
        elif self.base == "gaussian":
            z = torch.randn_like(target_samples)
            y = (1.0 - t) * z + t * target_samples
            u = target_samples - z
            b = self.model.flow(
                embedding, t.squeeze(-1), torch.cat((y, treat_cond), dim=-1)
            )
            loss = ((b - u) ** 2).sum(dim=-1)
        else:
            raise ValueError(f"unknown base: {self.base}")

        loss = loss.mean()
        return loss

    def OLD_get_embeddings(self, idx, source_samples, cond=None):
        if idx.shape[0] > 1: # using batched replicas
            embedding_batch = []
            for i in range(idx.shape[0]):
                if idx[i].item() in self.embeddings:
                    embedding_batch.append(self.embeddings[idx[i].item()].expand(source_samples.shape[1], -1))
                else:
                    embedding = self.model.embed_source(source_samples[i], cond=cond[i])
                    self.embeddings[idx[i].item()] = embedding.detach()
                    embedding_batch.append(embedding.expand(source_samples.shape[1], -1))
            return torch.stack(embedding_batch)
        else: # using sinlge replica
            idx = idx.item()
            if idx in self.embeddings:
                return self.embeddings[idx]
            else:
                embedding = self.model.embed_source(source_samples, cond=cond)
                self.embeddings[idx] = embedding.detach()
                return embedding
            
    def get_embeddings(self, idx, source_samples, cond=None):
        if idx.shape[0] > 1:  # using batched replicas
            embedding_batch = []
            for i in range(idx.shape[0]):
                if idx[i].item() in self.embeddings:
                    embedding_batch.append(self.embeddings[idx[i].item()].expand(source_samples.shape[1], -1))
                else:
                    if cond is not None:
                        embedding = self.model.embed_source(source_samples[i], cond=cond[i]).detach()
                    else:
                        embedding = self.model.embed_source(source_samples[i]).detach()
                    self.embeddings[idx[i].item()] = embedding
                    embedding_batch.append(embedding.expand(source_samples.shape[1], -1))
            return torch.stack(embedding_batch)
        else:  # using single replica
            idx = idx.item()
            if idx in self.embeddings:
                return self.embeddings[idx]
            else:
                if cond is not None:
                    embedding = self.model.embed_source(source_samples, cond=cond).detach()
                else:
                    embedding = self.model.embed_source(source_samples).detach()
                self.embeddings[idx] = embedding
                return embedding

    def flow_step(self, batch):
        idx, _, x0, x1, _, cell_cond, treat_cond = batch
            
        embedding = self.get_embeddings(
            idx, x0.float().squeeze(), cell_cond.float().squeeze()
        )
        
        loss = self.compute_loss(
            embedding.reshape(-1, embedding.shape[-1]),
            x0.reshape(-1, x0.shape[-1]),
            x1.reshape(-1, x1.shape[-1]),
            treat_cond.reshape(-1, treat_cond.shape[-1]),
        )
        
        self.flow_optimizer.zero_grad()
        self.manual_backward(loss)
        self.flow_optimizer.step()
        return loss
    
    def gnn_step(self, batch):
        idx, _, x0, x1, _, cell_cond, treat_cond = batch
            
        embedding = self.model.embed_source(x0.float().squeeze(0), cond=cell_cond.float().squeeze(0))
        
        if len(embedding.shape) > 1:  # when using replica batching
            embedding = embedding.unsqueeze(1).expand(-1, x0.shape[1], -1)
            for i in range(len(idx)):
                self.embeddings[idx[i].item()] = embedding[i].detach()
        else:
            self.embeddings[idx.item()] = embedding.detach()
        
        loss = self.compute_loss(
            embedding.reshape(-1, embedding.shape[-1]),
            x0.reshape(-1, x0.shape[-1]),
            x1.reshape(-1, x1.shape[-1]),
            treat_cond.reshape(-1, treat_cond.shape[-1]),
        )
        
        self.gnn_optimizer.zero_grad()
        self.manual_backward(loss)
        self.gnn_optimizer.step()
        return loss
    
    def training_step(self, batch, batch_idx):
        if (batch_idx + 1) % 2 == 0:
            loss = self.gnn_step(batch)
        else:
            loss = self.flow_step(batch)
        self.log("train/loss", loss, on_step=False, on_epoch=True)
        
        if self.use_pre_train_eval_batches is False:
            if (
                self.current_epoch % (self.trainer.check_val_every_n_epoch - 1)
            ) == 0 and self.train_evals_count < self.num_train_evals:
                self.train_eval_batches.append(batch)
                self.train_evals_count += 1
        return loss

    def training_epoch_end(self, outputs):
        if self.current_epoch == self.trainer.max_epochs - 1:
            for batch in self.train_eval_batches:
                self.eval_batch(batch, prefix="train")

            list_train_metrics = []
            for culture_key in list(self.train_metrics.keys()):
                eval_metrics_mean_train = {
                    k: np.mean([m[k] for m in self.train_metrics[culture_key]])
                    for k in self.train_metrics[culture_key][0]
                }
                for key, value in eval_metrics_mean_train.items():
                    self.log(
                        f"train/{key}-{culture_key}",
                        value,
                        on_step=False,
                        on_epoch=True,
                    )
                list_train_metrics.append(eval_metrics_mean_train)

            avg_train_metrics = dict(pd.DataFrame(list_train_metrics).mean())
            for key, value in avg_train_metrics.items():
                self.log(f"train/{key}-avg", value, on_step=False, on_epoch=True)

            if self.use_pre_train_eval_batches is False:
                self.train_eval_batches = []
                self.train_evals_count = 0

            self.train_metrics = {"PDO": [], "PDOF": [], "F": []}
            
        # Save a checkpoint of the model
        ckpt_name = (
            "last.ckpt"
            if self.current_epoch == self.trainer.max_epochs - 1
            else "ckpt.ckpt"
        )
        ckpt_path = os.path.join(self.trainer.log_dir, "checkpoints", ckpt_name)
        self.trainer.save_checkpoint(ckpt_path)

    def predict_step(self, batch, batch_idx):
        # only used at test time when evaluating models on train data
        if self.train_evals_count <= self.num_train_evals:
            self.eval_batch(batch, prefix="train")
            self.train_evals_count += 1
        else:
            pass

    def predict_epoch_end(self, outputs):
        # only used at test time when evaluating models on train data
        list_val_metrics = []
        for culture_key in list(self.train_metrics.keys()):
            if not self.train_metrics[culture_key]:
                continue
            eval_metrics_mean_ood = {
                k: np.mean([m[k] for m in self.train_metrics[culture_key]])
                for k in self.train_metrics[culture_key][0]
            }
            for key, value in eval_metrics_mean_ood.items():
                self.log(
                    f"train/{key}-{culture_key}", value, on_step=False, on_epoch=True
                )
            list_val_metrics.append(eval_metrics_mean_ood)

        avg_val_metrics = dict(pd.DataFrame(list_val_metrics).mean())
        for key, value in avg_val_metrics.items():
            self.log(f"train/{key}-avg", value, on_step=False, on_epoch=True)

        self.train_metrics = {"PDO": [], "PDOF": [], "F": []}

    def validation_step(self, batch, batch_idx):
        if self.run_validation:
            self.eval_batch(batch, prefix='val')
        else:
            return None
            
    def validation_epoch_end(self, outputs):
        if self.run_validation:
            list_val_metrics = []
            for culture_key in list(self.val_metrics.keys()):
                if not self.val_metrics[culture_key]:
                    continue
                eval_metrics_mean_ood = {
                    k: np.mean([m[k] for m in self.val_metrics[culture_key]])
                    for k in self.val_metrics[culture_key][0]
                }
                for key, value in eval_metrics_mean_ood.items():
                    self.log(
                        f"val/{key}-{culture_key}", value, on_step=False, on_epoch=True
                    )
                list_val_metrics.append(eval_metrics_mean_ood)
            
            avg_val_metrics = dict(pd.DataFrame(list_val_metrics).mean())
            for key, value in avg_val_metrics.items():
                self.log(f"val/{key}-avg", value, on_step=False, on_epoch=True)
            
            self.val_metrics = {"PDO": [], "PDOF": [], "F": []}
        else:
            return None
    
    def test_step(self, batch, batch_idx):
        self.eval_batch(batch, prefix="test")
        
    def test_epoch_end(self, outputs):
        list_test_metrics = []
        for culture_key in list(self.test_metrics.keys()):
            if not self.test_metrics[culture_key]:
                continue
            eval_metrics_mean_ood = {
                k: np.mean([m[k] for m in self.test_metrics[culture_key]])
                for k in self.test_metrics[culture_key][0]
            }
            for key, value in eval_metrics_mean_ood.items():
                self.log(
                    f"test/{key}-{culture_key}", value, on_step=False, on_epoch=True
                )
            list_test_metrics.append(eval_metrics_mean_ood)

        avg_test_metrics = dict(pd.DataFrame(list_test_metrics).mean())
        for key, value in avg_test_metrics.items():
            self.log(f"test/{key}-avg", value, on_step=False, on_epoch=True)
            
        self.test_metrics = {"PDO": [], "PDOF": [], "F": []}

    def eval_batch(self, batch, prefix):
        idx, culture, x0, x1, x1_full, cell_cond, treat_cond = batch
        
        node = NeuralODE(
            torch_wrapper_gnn_flow_cond(self.model),
            solver="dopri5",
            sensitivity="adjoint",
            atol=1e-4,
            rtol=1e-4,
        )

        time_span = torch.linspace(0, 1, self.integrate_time_steps)
        ts_for_plot = self.integrate_time_steps / 5
        ts = [i for i in range(int(ts_for_plot - 1), self.integrate_time_steps, int(ts_for_plot))]

        with torch.no_grad():
            if len(x0.shape) > 3:
                x0 = x0.squeeze(0)
                x1 = x1.squeeze(0)
                x1_full = x1_full.squeeze(0)
                treat_cond = treat_cond.squeeze(0)
                cell_cond = cell_cond.squeeze(0)

            for i in range(x0.shape[0]): # for loop used to allow for replicate batching for eval                    
                x0_i = x0[i].float()
                treat_cond_i = treat_cond[i].float()
                cell_cond_i = cell_cond[i].float()
                
                self.model.update_embedding_for_inference(
                    x0_i.cuda(), cond=cell_cond_i.cuda()
                )
                
                if self.base == "gaussian":
                    x0_i = torch.randn_like(x0_i)
                
                pred_batches = []
                traj_batches = []
                idcs_batches = np.arange(x0_i.shape[0])
                for j in range(0, x0_i.shape[0], 1024):
                    idcs = idcs_batches[j : j + 1024]
                    traj = node.trajectory(
                        torch.cat((x0_i[idcs].cuda(), treat_cond_i[idcs].cuda()), dim=-1).float(), t_span=time_span
                    )
                    pred_batches.append(traj[-1, :, : self.model.D])
                    traj_batches.append(traj[ts, :, : self.model.D])

                pred = torch.cat(pred_batches, dim=0)
                traj = torch.cat(traj_batches, dim=1)

                if self.pca is not None and self.dim == 43 and self.pca_space_eval:
                    pred = self.pca.inverse_transform(pred.cpu().numpy())
                    pred = torch.tensor(pred).cuda()
                    
                true = x1_full.float() if self.pca is not None and self.dim == 43 else x1.float()

                names, dd = compute_distribution_distances(
                    pred.unsqueeze(1).to(true),
                    true[0].unsqueeze(1),
                )
                
                if prefix == 'train':
                    self.train_metrics[culture[0]].append({**dict(zip(names, dd))})
                elif prefix == 'val':
                    self.val_metrics[culture[0]].append({**dict(zip(names, dd))})
                elif prefix == 'test':
                    self.test_metrics[culture[0]].append({**dict(zip(names, dd))})
                else:
                    raise ValueError(f"unknown prefix: {prefix}")

                # plot in 2d PCA space
                if idx in self.idcs_for_plot and self.pca_for_plot is not None:
                    print("Plotting 2D-PCA predictions ... \n")
                    treat_id = torch.argmax(treat_cond_i).item()
                    treat_name = TREAT_NAMES[self.treatments[treat_id]]
                    self.plot(
                        x0_i.cpu().numpy(),
                        traj.cpu().numpy(),
                        true.squeeze(0).cpu().numpy(),
                        prefix,
                        treat_name=treat_name,
                        tag=f"mfm_traj_treat_{treat_name}_{self.knn_k}_{idx.item()}",
                    )

    def plot(self, source, traj, target, prefix, treat_name, tag):
        # Flatten traj to [t*n, d]
        t, n, d = traj.shape
        m, d = target.shape

        traj_flat = traj.reshape(-1, d)

        # Get 2D PCA
        if self.pca_for_plot is not None:
            source_pca = self.pca_for_plot.transform(source)
            traj_pca = self.pca_for_plot.transform(traj_flat).reshape(t, n, 2)
            target_pca = self.pca_for_plot.transform(target)
        else:
            all_data = np.concatenate([source, traj_flat, target], axis=0)
            all_data_pca = PCA(n_components=2).fit_transform(all_data)
            source_pca = all_data_pca[:n]
            traj_pca = all_data_pca[n : n + t * n].reshape(t, n, 2)
            target_pca = all_data_pca[-m:]

        # Create subplots
        fig, axs = plt.subplots(1, 3, figsize=(9, 3))

        # Center the suptitle
        mid = (fig.subplotpars.right + fig.subplotpars.left) / 2
        fig.suptitle(
            f"{treat_name}", fontsize=16, x=mid, y=0.85, verticalalignment="center"
        )

        # Plot source points
        axs[0].scatter(
            source_pca[:, 0],
            source_pca[:, 1],
            c="lightsteelblue",
            label="Source",
            alpha=0.9,
            s=15,
            rasterized=True,
        )
        axs[0].set_title("Source")
        axs[0].set_xlabel("PCA 1")
        axs[0].set_ylabel("PCA 2")
        axs[0].tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)
        axs[0].grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.2)

        # Plot prediction points
        preds = []
        for i in range(n):
            preds.append([traj_pca[-1, i, 0], traj_pca[-1, i, 1]])
        pred = np.stack(preds)
        axs[1].scatter(
            pred[:, 0],
            pred[:, 1],
            c="rosybrown",
            marker="o",
            label="Prediction",
            alpha=0.6,
            s=15,
            rasterized=True,
        )
        axs[1].set_title("Prediction")
        axs[1].set_xlabel("PCA 1")
        axs[1].set_ylabel("PCA 2")
        axs[1].tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)
        axs[1].grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.2)

        # Plot target points
        axs[2].scatter(
            target_pca[:, 0],
            target_pca[:, 1],
            c="navy",
            label="Target",
            alpha=0.3,
            s=15,
            rasterized=True,
        )
        axs[2].set_title("Target")
        axs[2].set_xlabel("PCA 1")
        axs[2].set_ylabel("PCA 2")
        axs[2].tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)
        axs[2].grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.2)

        # Adjust layout
        fig.subplots_adjust(top=0.85)
        plt.tight_layout(rect=[0, 0, 1, 0.95])

        # Save the plot
        fname = f"{sccfm_directory}/figs/trellis_{prefix}_{tag}.pdf"
        os.makedirs(os.path.dirname(fname), exist_ok=True)
        fig.savefig(
            fname, bbox_inches="tight", pad_inches=0.1, transparent=True, dpi=300
        )

        # Optionally, you can log this to wandb or display it directly
        wandb.log({f"{prefix}_{tag}": wandb.Image(fig)})
        plt.close()