"""Lightning module for letters dataset."""

import os
import numpy as np
import ot as pot
import torch
import torch.optim as optim
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pickle 
import wandb

from torchdyn.core import NeuralODE
from util.distribution_distances import compute_distribution_distances
from models.components.mlp import Flow, torch_wrapper, torch_wrapper_flow_cond
from src.models.components.gnn import GlobalGNN

from pathlib import Path

script_directory = Path(__file__).absolute().parent
sccfm_directory = script_directory.parent.parent

class LettersFM(pl.LightningModule):
    def __init__(
        self,
        train_eval_batches,
        lr=1e-4,
        dim=2,
        num_hidden=512,
        num_layers=4,
        skip_connections=False,
        base="source",
        ot_sample=False,
        integrate_time_steps=500,
        name="letters_fm",
    ) -> None:
        super().__init__()
        
        # Important: This property controls manual optimization.
        self.automatic_optimization = True
        
        self.save_hyperparameters(ignore="train_eval_batches")

        self.model = Flow(
            D=2,
            num_hidden=num_hidden,
            num_layers=num_layers,
            skip_connections=skip_connections,
        ).cuda()
        
        self.lr = lr
        self.dim = dim
        self.num_hidden = num_hidden
        self.integrate_time_steps = integrate_time_steps
        self.ot_sample = ot_sample

        assert base in [
            "source",
            "gaussian",
        ], "Invalid base. Must be either 'source' or 'gaussian'"
        self.base = base
        self.name = name
        
        # for training data eval
        if train_eval_batches is not None:
            self.train_eval_batches = train_eval_batches
            self.use_pre_train_eval_batches = True
        else:
            self.num_train_evals = 10
            self.train_evals_count = 0
            self.train_eval_batches = []
            self.use_pre_train_eval_batches = False
        
        self.predict_count = 0

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        return optimizer

    def forward(self, t, x):
        return self.model(t.squeeze(-1), x)

    def compute_loss(self, source_samples, target_samples):
        t = torch.rand_like(source_samples[..., 0, None])

        if self.base == "source":
            x = (1.0 - t) * source_samples + t * target_samples
            u = target_samples - source_samples
            b = self.forward(t, x)
            loss = b.norm(dim=-1) ** 2 - 2.0 * (b * u).sum(dim=-1)
        elif self.base == "gaussian":
            z = torch.randn_like(target_samples)
            x = (1.0 - t) * z + t * target_samples
            u = target_samples - z
            b = self.forward(t, x)
            loss = ((b - u) ** 2).sum(dim=-1)
        else:
            raise ValueError(f"unknown base: {self.base}")

        loss = loss.mean()
        return loss
    
    def sample_ot(self, x0, x1):
        # Resample x0, x1 according to transport matrix
        batch_size = x0.shape[0]
        a, b = pot.unif(x0.size()[0]), pot.unif(x1.size()[0])
        M = torch.cdist(x0, x1) ** 2
        pi = pot.emd(a, b, M.detach().cpu().numpy())
        # Sample random interpolations on pi
        p = pi.flatten()
        p = p / p.sum()
        choices = np.random.choice(pi.shape[0] * pi.shape[1], p=p, size=batch_size)
        i, j = np.divmod(choices, pi.shape[1])
        x0 = x0[i]
        x1 = x1[j]
        return x0, x1

    def training_step(self, batch, batch_idx):
        _, x0, x1 = batch
        
        if self.ot_sample:
            for i in range(x0.shape[0]):
                x0[i], x1[i] = self.sample_ot(x0[i], x1[i])
        
        assert (
            len(x0.shape) == 3
        ), "This was a temporary fix for the dataloader -- TODO: Make the code more gener."
        loss = self.compute_loss(x0.view(-1, x0.shape[-1]), x1.view(-1, x1.shape[-1]))
        self.log("train/loss", loss, on_step=False, on_epoch=True)
        if self.use_pre_train_eval_batches is False:
            if (
                (self.current_epoch % (self.trainer.check_val_every_n_epoch - 1)) == 0
                and self.train_evals_count < self.num_train_evals
            ):
                self.train_eval_batches.append(batch)
                self.train_evals_count += 1
        return loss
    
    def training_epoch_end(self, outputs):
        if self.current_epoch == self.trainer.max_epochs - 1:
            trajs, preds, trues = [], [], []
            for batch in self.train_eval_batches:
                traj, pred, true = self.eval_batch(batch)
                trajs.append(traj[0])
                preds.append(pred[0])
                trues.append(true[0])

            eval_metrics = []
            for i in range(len(trues)):
                pred_samples, target_samples = preds[i], trues[i]
                names, dd = compute_distribution_distances(
                    pred_samples.squeeze(0).unsqueeze(1).to(target_samples),
                    target_samples.squeeze(0).unsqueeze(1),
                    r2_pairwise_feat_corrs=False,
                )
                eval_metrics.append({**dict(zip(names, dd))})
            eval_metrics = {
                k: np.mean([m[k] for m in eval_metrics]) for k in eval_metrics[0]
            }
            for key, value in eval_metrics.items():
                self.log(f"train/{key}", value, on_step=False, on_epoch=True)
            self.plot(
                trajs,
                self.train_eval_batches,
                num_row=6,
                num_step=3,
                tag="fm_train_samples_6_plots",
            )
            self.plot(
                trajs,
                self.train_eval_batches,
                num_row=2,
                num_step=3,
                tag="fm_train_samples_2_plots",
            )
            
            if self.use_pre_train_eval_batches is False:
                self.train_eval_batches = []
                self.train_evals_count = 0
                
        # Save a checkpoint of the model
        ckpt_name = (
            "last.ckpt"
            if self.current_epoch == self.trainer.max_epochs - 1
            else "ckpt.ckpt"
        )
        ckpt_path = os.path.join(self.trainer.log_dir, "checkpoints", ckpt_name)
        self.trainer.save_checkpoint(ckpt_path)

    def predict_step(self, batch, batch_idx):
        # use to return predictions for final plots
        if self.predict_count < 53:
            idx, x0, _ = batch
            trajs, pred, true = self.eval_batch(batch)
            self.predict_count += 1
            return idx, trajs, x0, pred, true
        else:
            pass

    def validation_step(self, batch, batch_idx):
        trajs, pred, true = self.eval_batch(batch)
        eval_metrics_ood = self.compute_eval_metrics(pred, true, aggregate=False)
        eval_metrics_mean_ood = {
            k: np.mean([m[k] for m in eval_metrics_ood]) for k in eval_metrics_ood[0]
        }
        for key, value in eval_metrics_mean_ood.items():
            self.log(f"val/{key}", value, on_step=False, on_epoch=True)
        self.plot(trajs, batch, num_row=6, num_step=3, tag="fm_val_samples_6_plots")
        self.plot(trajs, batch, num_row=2, num_step=3, tag="fm_val_samples_2_plots")

    def test_step(self, batch, batch_idx):
        trajs, pred, true = self.eval_batch(batch)
        eval_metrics_ood = self.compute_eval_metrics(pred, true, aggregate=False)
        eval_metrics_mean_ood = {
            k: np.mean([m[k] for m in eval_metrics_ood]) for k in eval_metrics_ood[0]
        }
        for key, value in eval_metrics_mean_ood.items():
            self.log(f"test/{key}", value, on_step=False, on_epoch=True)
        self.plot(trajs, batch, num_row=6, num_step=3, tag="fm_test_samples_6_plots")
        self.plot(trajs, batch, num_row=2, num_step=3, tag="fm_test_samples_2_plots")

    def compute_eval_metrics(self, pred_samples, target_samples, aggregate=True):
        metrics = []
        for i in range(target_samples.shape[0]):
            names, dd = compute_distribution_distances(
                pred_samples[i].unsqueeze(1).to(target_samples),
                target_samples[i].unsqueeze(1),
                r2_pairwise_feat_corrs=False,
            )
            metrics.append({**dict(zip(names, dd))})
        if aggregate:
            metrics = {k: np.mean([m[k] for m in metrics]) for k in metrics[0]}
        return metrics

    def eval_batch(self, batch):
        _, x0, x1 = batch

        node = NeuralODE(
            torch_wrapper(self.model),
            solver='dopri5', #'rk4'
            sensitivity="adjoint",
            atol=1e-4,
            rtol=1e-4,
        )

        time_span = torch.linspace(0, 1, self.integrate_time_steps)

        with torch.no_grad():
            if len(x0.shape) > 3:
                x0 = x0.squeeze()
                x1 = x1.squeeze()

            trajs, pred = [], []
            for i in range(x0.shape[0]):
                x0_i = x0[i]
                
                if self.base == "gaussian":
                    x0_i = torch.randn_like(x0_i)
                    
                traj = node.trajectory(x0_i, t_span=time_span)
                trajs.append(traj)
                pred.append(traj[-1, :, :])

            trajs = torch.stack(trajs, dim=0)
            pred = torch.stack(pred, dim=0)
            true = x1
            
        return trajs, pred, true

    def plot(
        self, trajs, samples, num_row=6, num_step=5, tag="fm_samples",
    ):  
        if not isinstance(trajs, list):
            trajs = trajs.cpu().detach().numpy()
            _, source, target = samples
        else:
            trajs_tmp = []
            for traj in trajs:
                trajs_tmp.append(traj.cpu().detach().numpy())
            trajs = trajs_tmp

        if self.base == "source":
            num_col = 1 + num_step
        elif self.base == "gaussian":
            num_col = 1 + num_step
        else:
            raise ValueError(f"unknown base: {self.base}")

        fig, axs = plt.subplots(
            num_row,
            num_col,
            figsize=(num_col, num_row),
            gridspec_kw={"wspace": 0.0, "hspace": 0.0},
        )
        axs = axs.reshape(num_row, num_col)

        for i in range(num_row):
            ax = axs[i, 0]

            n = 700
            rng = np.random.default_rng(42)

            if not isinstance(trajs, list):
                idcs = rng.choice(
                    np.arange(source[i].shape[0]),
                    size=min(n, source[i].shape[0]),
                    replace=False,
                )
                source_samples = source[i].cpu().numpy()
                target_samples = target[i].cpu().numpy()
                source_samples = source_samples[idcs]
                target_samples = target_samples[idcs]
            else:
                source_samples = samples[i][1].cpu().numpy()
                target_samples = samples[i][2].cpu().numpy()
                idcs = rng.choice(np.arange(source_samples.shape[1]), size=min(n, source_samples.shape[1]), replace=False)
                source_samples = source_samples[0, idcs]
                target_samples = target_samples[0, idcs]

            ax.scatter(*source_samples.T, s=1, c="#3283FB",rasterized=True)
            ax.set_facecolor((206 / 256, 206 / 256, 229 / 256))

            ax = axs[i, -1]
            ax.scatter(*target_samples.T, s=1, c="#3283FB", rasterized=True)
            ax.set_facecolor((206 / 256, 206 / 256, 229 / 256))

            traj = trajs[i]

            t_step = int(traj.shape[0] / (num_step - 1))
            start_j = 1
            ts = np.arange(t_step, t_step * num_step, t_step)
            for j in range(start_j, num_step):
                t = ts[j - 1] - 1
                offset = 0
                ax = axs[i, j + offset]
                ax.scatter(*traj[t, idcs].T, s=1, c="#3283FB", rasterized=True)
                ax.set_facecolor((206 / 256, 206 / 256, 229 / 256))
                if i == 0:
                    time = t / (t_step * (num_step - 1))
                    ax.set_title(f"t={time:.2f}")

        axs[0, 0].set_title("source")
        axs[0, -1].set_title("target")

        for ax in axs.ravel():
            ax.set_xlim(-4, 4)
            ax.set_ylim(-4, 4)
            ax.set_xticks([])
            ax.set_yticks([])

        fig.tight_layout()
        fname = f"{sccfm_directory}/figs/{tag}.pdf"
        fig.savefig(
            fname, bbox_inches="tight", pad_inches=0.0, transparent=True, dpi=300
        )
        wandb.log({f"imgs/{tag}": wandb.Image(fig)})
        plt.close(fig)


class LettersCGFM(pl.LightningModule):
    def __init__(
        self,
        train_eval_batches,
        lr=1e-4,
        dim=2,
        num_hidden=512,
        num_layers=4,
        skip_connections=False,
        base="source",
        ot_sample=False,
        integrate_time_steps=500,
        num_conditions=262,
        name="letters_cgfm",
    ) -> None:
        super().__init__()

        # Important: This property controls manual optimization.
        self.automatic_optimization = True

        self.save_hyperparameters(ignore="train_eval_batches")

        self.model = Flow(
            D=2,
            num_hidden=512,
            num_layers=num_layers,
            num_conditions=num_conditions,
            skip_connections=skip_connections,
        ).cuda()
        
        self.lr = lr
        self.dim = dim
        self.num_hidden = num_hidden
        self.integrate_time_steps = integrate_time_steps
        self.num_conditions = num_conditions
        self.ot_sample = ot_sample
        
        assert base in [
            "source",
            "gaussian",
        ], "Invalid base. Must be either 'source' or 'gaussian'"
        self.base = base
        self.name = name
            
        # for training data eval
        if train_eval_batches is not None:
            self.train_eval_batches = train_eval_batches
            self.use_pre_train_eval_batches = True
        else:
            self.num_train_evals = 10
            self.train_evals_count = 0
            self.train_eval_batches = []
            self.use_pre_train_eval_batches = False
        
        self.predict_count = 0

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
    
    def sample_ot(self, x0, x1):
        # Resample x0, x1 according to transport matrix
        batch_size = x0.shape[0]
        a, b = pot.unif(x0.size()[0]), pot.unif(x1.size()[0])
        M = torch.cdist(x0, x1) ** 2
        pi = pot.emd(a, b, M.detach().cpu().numpy())
        # Sample random interpolations on pi
        p = pi.flatten()
        p = p / p.sum()
        choices = np.random.choice(pi.shape[0] * pi.shape[1], p=p, size=batch_size)
        i, j = np.divmod(choices, pi.shape[1])
        x0 = x0[i]
        x1 = x1[j]
        return x0, x1

    def training_step(self, batch, batch_idx):
        _, x0, x1, cond = batch
        
        if self.ot_sample:
            for i in range(x0.shape[0]):
                x0[i], x1[i] = self.sample_ot(x0[i], x1[i])
        
        assert (
            len(x0.shape) == 3
        ), "This was a temporary fix for the dataloader -- TODO: Make the code more gener."
        loss = self.compute_loss(
            x0.view(-1, x0.shape[-1]), x1.view(-1, x1.shape[-1]), cond.view(-1, cond.shape[-1]),
        )
        self.log("train/loss", loss, on_step=False, on_epoch=True)
        if self.use_pre_train_eval_batches is False:
            if (
                (self.current_epoch % (self.trainer.check_val_every_n_epoch - 1)) == 0
                and self.train_evals_count < self.num_train_evals
            ):
                self.train_eval_batches.append(batch)
                self.train_evals_count += 1
        return loss

    def training_epoch_end(self, outputs):
        if self.current_epoch == self.trainer.max_epochs - 1:
            trajs, preds, trues = [], [], []
            for batch in self.train_eval_batches:
                traj, pred, true = self.eval_batch(batch)
                trajs.append(traj[0])
                preds.append(pred[0])
                trues.append(true[0])
            
            eval_metrics = []
            for i in range(len(trues)):
                pred_samples, target_samples = preds[i], trues[i]
                names, dd = compute_distribution_distances(
                    pred_samples.squeeze(0).unsqueeze(1).to(target_samples),
                    target_samples.squeeze(0).unsqueeze(1),
                    r2_pairwise_feat_corrs=False,
                )
                eval_metrics.append({**dict(zip(names, dd))})
            eval_metrics = {
                k: np.mean([m[k] for m in eval_metrics]) for k in eval_metrics[0]
            }
            for key, value in eval_metrics.items():
                self.log(f"train/{key}", value, on_step=False, on_epoch=True)
            self.plot(
                trajs,
                self.train_eval_batches,
                num_row=6,
                num_step=3,
                tag="cond_fm_train_samples_6_plots",
            )
            self.plot(
                trajs,
                self.train_eval_batches,
                num_row=2,
                num_step=3,
                tag="cond_fm_train_samples_2_plots",
            )
            
            if self.use_pre_train_eval_batches is False:
                self.train_eval_batches = []
                self.train_evals_count = 0
                
        # Save a checkpoint of the model
        ckpt_name = (
            "last.ckpt"
            if self.current_epoch == self.trainer.max_epochs - 1
            else "ckpt.ckpt"
        )
        ckpt_path = os.path.join(self.trainer.log_dir, "checkpoints", ckpt_name)
        self.trainer.save_checkpoint(ckpt_path)
    
    def predict_step(self, batch, batch_idx):
        # use to return predictions for final plots
        if self.predict_count < 60:
            idx, x0, _ = batch[:3]
            trajs, pred, true = self.eval_batch(batch)
            self.predict_count += 1
            return idx, trajs, x0, pred, true
        else:
            pass

    def validation_step(self, batch, batch_idx):
        trajs, pred, true = self.eval_batch(batch, prefix='val')
        eval_metrics_ood = self.compute_eval_metrics(pred, true, aggregate=False)
        eval_metrics_mean_ood = {
            k: np.mean([m[k] for m in eval_metrics_ood]) for k in eval_metrics_ood[0]
        }
        for key, value in eval_metrics_mean_ood.items():
            self.log(f"val/{key}", value, on_step=False, on_epoch=True)
        self.plot(
            trajs, batch, num_row=6, num_step=3, tag="cond_fm_val_samples_6_plots",
        )
        self.plot(
            trajs, batch, num_row=2, num_step=3, tag="cond_fm_val_samples_2_plots",
        )

    def test_step(self, batch, batch_idx):
        trajs, pred, true = self.eval_batch(batch, prefix='test')
        eval_metrics_ood = self.compute_eval_metrics(pred, true, aggregate=False)
        eval_metrics_mean_ood = {
            k: np.mean([m[k] for m in eval_metrics_ood]) for k in eval_metrics_ood[0]
        }
        for key, value in eval_metrics_mean_ood.items():
            self.log(f"test/{key}", value, on_step=False, on_epoch=True)
        self.plot(
            trajs, batch, num_row=6, num_step=3, tag="cond_fm_test_samples_6_plots",
        )
        self.plot(
            trajs, batch, num_row=2, num_step=3, tag="cond_fm_test_samples_2_plots",
        )

    def compute_eval_metrics(self, pred_samples, target_samples, aggregate=True):
        metrics = []
        for i in range(target_samples.shape[0]):
            names, dd = compute_distribution_distances(
                pred_samples[i].unsqueeze(1).to(target_samples),
                target_samples[i].unsqueeze(1),
                r2_pairwise_feat_corrs=False,
            )
            metrics.append({**dict(zip(names, dd))})
        if aggregate:
            metrics = {k: np.mean([m[k] for m in metrics]) for k in metrics[0]}
        return metrics

    def eval_batch(self, batch, prefix='train'):
        _, x0, x1, cond = batch
        
        if prefix != 'train':
            # average over train one-hot conditions
            cond = torch.cat(
                [
                    torch.ones(
                        (
                            cond.shape[0],
                            cond.shape[1],
                            self.num_conditions - 10 - 10,
                        )
                    ),
                    torch.zeros(
                        (
                            cond.shape[0],
                            cond.shape[1],
                            10 + 10,
                        )
                    ),
                ],
                dim=-1,
            ).to(x0) / (self.num_conditions - 10 - 10)
        
        node = NeuralODE(
            torch_wrapper_flow_cond(self.model),
            solver='dopri5', #rk4
            sensitivity="adjoint",
            atol=1e-4,
            rtol=1e-4,
        )

        time_span = torch.linspace(0, 1, self.integrate_time_steps)

        with torch.no_grad():
            if len(x0.shape) > 3:
                x0 = x0.squeeze()
                x1 = x1.squeeze()

            trajs, pred = [], []
            for i in range(x0.shape[0]):
                x0_i = x0[i]
                
                if self.base == "gaussian":
                    x0_i = torch.randn_like(x0_i)
                    
                cond_i = cond[i]
                traj = node.trajectory(torch.cat((x0_i, cond_i), dim=-1).float(), t_span=time_span)
                trajs.append(traj[:, :, :self.model.D])
                pred.append(traj[-1, :, :self.model.D])

            trajs = torch.stack(trajs, dim=0)
            pred = torch.stack(pred, dim=0)
            true = x1
        return trajs, pred, true

    def plot(
        self, trajs, samples, num_row=6, num_step=5, tag="cond_fm_samples", 
    ):
        if not isinstance(trajs, list):
            trajs = trajs.cpu().detach().numpy()
            _, source, target, _ = samples
        else:
            trajs_tmp = []
            for traj in trajs:
                trajs_tmp.append(traj.cpu().detach().numpy())
            trajs = trajs_tmp

        if self.base == "source":
            num_col = 1 + num_step
        elif self.base == "gaussian":
            num_col = 1 + num_step
        else:
            raise ValueError(f"unknown base: {self.base}")

        fig, axs = plt.subplots(
            num_row,
            num_col,
            figsize=(num_col, num_row),
            gridspec_kw={"wspace": 0.0, "hspace": 0.0},
        )
        axs = axs.reshape(num_row, num_col)

        for i in range(num_row):
            ax = axs[i, 0]

            n = 700
            rng = np.random.default_rng(42)

            if not isinstance(trajs, list):
                idcs = rng.choice(
                    np.arange(source[i].shape[0]),
                    size=min(n, source[i].shape[0]),
                    replace=False,
                )
                source_samples = source[i].cpu().numpy()
                target_samples = target[i].cpu().numpy()
                source_samples = source_samples[idcs]
                target_samples = target_samples[idcs]
            else:
                source_samples = samples[i][1].cpu().numpy()
                target_samples = samples[i][2].cpu().numpy()
                idcs = rng.choice(
                    np.arange(source_samples.shape[0]),
                    size=min(n, source_samples.shape[0]),
                    replace=False,
                )
                idcs = rng.choice(
                    np.arange(source_samples.shape[1]),
                    size=min(n, source_samples.shape[1]),
                    replace=False,
                )
                source_samples = source_samples[0, idcs]
                target_samples = target_samples[0, idcs]
                
            ax.scatter(*source_samples.T, s=1, c="#3283FB", rasterized=True)
            ax.set_facecolor((206 / 256, 206 / 256, 229 / 256))

            ax = axs[i, -1]
            ax.scatter(*target_samples.T, s=1, c="#3283FB", rasterized=True)
            ax.set_facecolor((206 / 256, 206 / 256, 229 / 256))

            traj = trajs[i]

            t_step = int(traj.shape[0] / (num_step - 1))
            start_j = 1
            ts = np.arange(t_step, t_step * num_step, t_step)
            for j in range(start_j, num_step):
                t = ts[j - 1] - 1
                offset = 0
                ax = axs[i, j + offset]
                ax.scatter(*traj[t, idcs].T, s=1, c="#3283FB", rasterized=True)
                ax.set_facecolor((206 / 256, 206 / 256, 229 / 256))
                if i == 0:
                    time = t / (t_step * (num_step - 1))
                    ax.set_title(f"t={time:.2f}")

        axs[0, 0].set_title("source")
        axs[0, -1].set_title("target")

        for ax in axs.ravel():
            ax.set_xlim(-4, 4)
            ax.set_ylim(-4, 4)
            ax.set_xticks([])
            ax.set_yticks([])

        fig.tight_layout()
        fname = f"{sccfm_directory}/figs/{tag}.pdf"
        fig.savefig(
            fname, bbox_inches="tight", pad_inches=0.0, transparent=True, dpi=300
        )
        wandb.log({f"imgs/{tag}": wandb.Image(fig)})
        plt.close(fig)


class LettersMFM(pl.LightningModule):
    def __init__(
        self,
        train_eval_batches,
        flow_lr=1e-4,
        gnn_lr=1e-4,
        dim=2,
        num_hidden=512,
        num_layers_decoder=4,
        num_hidden_gnn=64,
        num_layers_gnn=3,
        knn_k=50,
        skip_connections=True,
        base="source",
        ot_sample=False,
        save_embeddings=False,
        data_for_embed_save=None,
        integrate_time_steps=500,
        name="letters_mfm",
    ) -> None:
        super().__init__()

        # Important: This property controls manual optimization.
        self.automatic_optimization = False

        self.save_hyperparameters()

        self.model = GlobalGNN(
            D=dim,
            num_hidden_decoder=num_hidden,
            num_layers_decoder=num_layers_decoder,
            num_hidden_gnn=num_hidden_gnn,
            num_layers_gnn=num_layers_gnn,
            knn_k=knn_k,
            skip_connections=skip_connections,
        ).cuda()

        assert len(list(self.model.parameters())) == len(
            list(self.model.decoder.parameters())
        ) + len(list(self.model.gcn_convs.parameters()))

        self.flow_lr = flow_lr
        self.gnn_lr = gnn_lr
        self.dim = dim
        self.knn_k = knn_k
        self.num_hidden = num_hidden
        self.integrate_time_steps = integrate_time_steps
        self.ot_sample = ot_sample
        
        self.save_embeddings = save_embeddings
        if self.save_embeddings:
            self.data_for_embed_save = data_for_embed_save
            self.save_population_embeddings()
        
        assert base in [
            "source",
            "gaussian",
        ], "Invalid base. Must be either 'source' or 'gaussian'"
        self.base = base
        self.name = name
        
        # for training data eval
        if train_eval_batches is not None:
            self.train_eval_batches = train_eval_batches
            self.use_pre_train_eval_batches = True
        else:
            self.num_train_evals = 10
            self.train_evals_count = 0
            self.train_eval_batches = []
            self.use_pre_train_eval_batches = False
        
        self.embeddings = {}
        
        self.predict_count = 0

    def configure_optimizers(self):
        # init optimizers
        self.flow_optimizer = torch.optim.Adam(self.model.decoder.parameters(), lr=self.flow_lr)
        self.gnn_optimizer = torch.optim.Adam(
            self.model.gcn_convs.parameters(),
            lr=self.gnn_lr,
        )
        return self.flow_optimizer, self.gnn_optimizer

    def compute_loss(self, embedding, source_samples, target_samples):
        t = torch.rand_like(source_samples[..., 0, None])

        if self.base == "source":
            y = (1.0 - t) * source_samples + t * target_samples
            u = target_samples - source_samples

            b = self.model.flow(embedding, t.squeeze(-1), y)
            loss = b.norm(dim=-1) ** 2 - 2.0 * (b * u).sum(dim=-1)
        elif self.base == "gaussian":
            z = torch.randn_like(target_samples)
            y = (1.0 - t) * z + t * target_samples
            u = target_samples - z
            b = self.model.flow(embedding, t.squeeze(-1), y)
            loss = ((b - u) ** 2).sum(dim=-1)
        else:
            raise ValueError(f"unknown base: {self.base}")

        loss = loss.mean()
        return loss

    def get_embeddings(self, idx, source_samples):
        if idx.shape[0] > 1: # using batched replicas
            embedding_batch = []
            for i in range(idx.shape[0]):
                if idx[i].item() in self.embeddings:
                    embedding_batch.append(self.embeddings[idx[i].item()].expand(source_samples.shape[1], -1))
                else:
                    embedding = self.model.embed_source(source_samples[i]).detach()
                    self.embeddings[idx[i].item()] = embedding
                    embedding_batch.append(
                        embedding.expand(source_samples.shape[1], -1)
                    )
            return torch.stack(embedding_batch)
        else:
            idx = idx.item()
            if idx in self.embeddings:
                return self.embeddings[idx]
            else:
                embedding = self.model.embed_source(source_samples).detach()
                self.embeddings[idx] = embedding
                return embedding
            
    def sample_ot(self, x0, x1):
        # Resample x0, x1 according to transport matrix
        batch_size = x0.shape[0]
        a, b = pot.unif(x0.size()[0]), pot.unif(x1.size()[0])
        M = torch.cdist(x0, x1) ** 2
        pi = pot.emd(a, b, M.detach().cpu().numpy())
        # Sample random interpolations on pi
        p = pi.flatten()
        p = p / p.sum()
        choices = np.random.choice(pi.shape[0] * pi.shape[1], p=p, size=batch_size)
        i, j = np.divmod(choices, pi.shape[1])
        x0 = x0[i]
        x1 = x1[j]
        return x0, x1

    def flow_step(self, batch):
        idx, x0, x1 = batch
        
        if self.ot_sample:
            for i in range(x0.shape[0]):
                x0[i], x1[i] = self.sample_ot(x0[i], x1[i])
                
        embedding = self.get_embeddings(idx, x0)
        loss = self.compute_loss(
            embedding.reshape(-1, embedding.shape[-1]),
            x0.reshape(-1, x0.shape[-1]),
            x1.reshape(-1, x1.shape[-1]),
        )
        self.flow_optimizer.zero_grad()
        self.manual_backward(loss)
        self.flow_optimizer.step()
        return loss
    
    def gnn_step(self, batch):
        idx, x0, x1 = batch
        
        if self.ot_sample:
            for i in range(x0.shape[0]):
                x0[i], x1[i] = self.sample_ot(x0[i], x1[i])
                
        embedding = self.model.embed_source(x0)

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
                (self.current_epoch % (self.trainer.check_val_every_n_epoch - 1)) == 0
                and self.train_evals_count < self.num_train_evals
            ):
                self.train_eval_batches.append(batch) 
                self.train_evals_count += 1
        return loss
    
    def training_epoch_end(self, outputs):
        if self.current_epoch == self.trainer.max_epochs - 1:
            trajs, preds, trues = [], [], []
            for batch in self.train_eval_batches:
                traj, pred, true = self.eval_batch(batch)
                trajs.append(traj[0])
                preds.append(pred[0])
                trues.append(true[0])

            eval_metrics = []
            for i in range(len(trues)):
                pred_samples, target_samples = preds[i], trues[i]
                names, dd = compute_distribution_distances(
                    pred_samples.unsqueeze(1).to(target_samples),
                    target_samples.unsqueeze(1),
                    r2_pairwise_feat_corrs=False,
                )
                eval_metrics.append({**dict(zip(names, dd))})
            eval_metrics = {
                k: np.mean([m[k] for m in eval_metrics]) for k in eval_metrics[0]
            }
            for key, value in eval_metrics.items():
                self.log(f"train/{key}", value, on_step=False, on_epoch=True)
            self.plot(
                trajs,
                self.train_eval_batches,
                num_row=6,
                num_step=3,
                tag="gnn_train_samples_6_plots",
            )
            self.plot(
                trajs,
                self.train_eval_batches,
                num_row=2,
                num_step=3,
                tag="gnn_train_samples_2_plots",
            )
            if self.use_pre_train_eval_batches is False:
                self.train_eval_batches = []
                self.train_evals_count = 0
            
        # Save a checkpoint of the model
        ckpt_name = (
            "last.ckpt"
            if self.current_epoch == self.trainer.max_epochs - 1
            else "ckpt.ckpt"
        )
        ckpt_path = os.path.join(self.trainer.log_dir, "checkpoints", ckpt_name)
        self.trainer.save_checkpoint(ckpt_path)

    def predict_step(self, batch, batch_idx):
        # use to return predictions for final plots
        idx, x0, _ = batch
        if self.predict_count < 60:
            trajs, pred, true = self.eval_batch(batch)
            self.predict_count += 1
            return idx, trajs, x0, pred, true
        else:
            pass

    def validation_step(self, batch, batch_idx):
        trajs, pred, true = self.eval_batch(batch)
        eval_metrics_ood = self.compute_eval_metrics(pred, true, aggregate=False)
        eval_metrics_mean_ood = {
            k: np.mean([m[k] for m in eval_metrics_ood]) for k in eval_metrics_ood[0]
        }
        for key, value in eval_metrics_mean_ood.items():
            self.log(f"val/{key}", value, on_step=False, on_epoch=True)
        self.plot(trajs, batch, num_row=6, num_step=3, tag="gnn_val_samples_6_plots")
        self.plot(trajs, batch, num_row=2, num_step=3, tag="gnn_val_samples_2_plots")

    def test_step(self, batch, batch_idx):
        trajs, pred, true = self.eval_batch(batch)
        eval_metrics_ood = self.compute_eval_metrics(pred, true, aggregate=False)
        eval_metrics_mean_ood = {
            k: np.mean([m[k] for m in eval_metrics_ood]) for k in eval_metrics_ood[0]
        }
        for key, value in eval_metrics_mean_ood.items():
            self.log(f"test/{key}", value, on_step=False, on_epoch=True)
        self.plot(trajs, batch, num_row=6, num_step=3, tag="gnn_test_samples_6_plots")
        self.plot(trajs, batch, num_row=2, num_step=3, tag="gnn_test_samples_2_plots")
    
    def compute_eval_metrics(self, pred_samples, target_samples, aggregate=True):
        metrics = []
        for i in range(target_samples.shape[0]):
            names, dd = compute_distribution_distances(
                pred_samples[i].unsqueeze(1).to(target_samples),
                target_samples[i].unsqueeze(1),
                r2_pairwise_feat_corrs=False,
            )
            metrics.append({**dict(zip(names, dd))})
        if aggregate:
            metrics = {k: np.mean([m[k] for m in metrics]) for k in metrics[0]}
        return metrics

    def eval_batch(self, batch):
        _, x0, x1 = batch
        
        node = NeuralODE(
            torch_wrapper(self.model),
            solver='dopri5', #rk4
            sensitivity="adjoint",
            atol=1e-4,
            rtol=1e-4,
        )

        time_span = torch.linspace(0, 1, self.integrate_time_steps)
 
        with torch.no_grad():
            if len(x0.shape) > 3:
                x0 = x0.squeeze()
                x1 = x1.squeeze()

            trajs, pred = [], []
            for i in range(x0.shape[0]):
                x0_i = x0[i]
                self.model.update_embedding_for_inference(x0_i)
                
                if self.base == "gaussian":
                    x0_i = torch.randn_like(x0_i)
                
                traj = node.trajectory(x0_i, t_span=time_span)
                trajs.append(traj)
                pred.append(traj[-1, :, :])

            trajs = torch.stack(trajs, dim=0) 
            pred = torch.stack(pred, dim=0) 
            true = x1
        return trajs, pred, true

    def plot(self, trajs, samples, num_row=6, num_step=5, tag="gnn_samples"):
        if not isinstance(trajs, list): 
            trajs = trajs.cpu().detach().numpy()
            _, source, target = samples
        else:
            trajs_tmp = []
            for traj in trajs:
                trajs_tmp.append(traj.cpu().detach().numpy())
            trajs = trajs_tmp

        if self.base == "source":
            num_col = 1 + num_step
        elif self.base == "gaussian":
            num_col = 1 + num_step
        else:
            raise ValueError(f"unknown base: {self.base}")

        fig, axs = plt.subplots(
            num_row,
            num_col,
            figsize=(num_col, num_row),
            gridspec_kw={"wspace": 0.0, "hspace": 0.0},
        )
        axs = axs.reshape(num_row, num_col)

        for i in range(num_row):
            ax = axs[i, 0]

            n = 700
            rng = np.random.default_rng(42)
            
            if not isinstance(trajs, list): 
                idcs = rng.choice(np.arange(source[i].shape[0]), size=min(n, source[i].shape[0]), replace=False)
                source_samples = source[i].cpu().numpy()
                target_samples = target[i].cpu().numpy()
                source_samples = source_samples[idcs]
                target_samples = target_samples[idcs]
            else:
                source_samples = samples[i][1].cpu().numpy()
                target_samples = samples[i][2].cpu().numpy()
                idcs = rng.choice(np.arange(source_samples.shape[1]), size=min(n, source_samples.shape[1]), replace=False)
                source_samples = source_samples[0, idcs]
                target_samples = target_samples[0, idcs]
            
            ax.scatter(*source_samples.T, s=1.5, alpha=0.8, c="#3283FB", rasterized=True)
            ax.set_facecolor((206/256, 206/256, 229/256))
            
            ax = axs[i, -1]
            ax.scatter(*target_samples.T, s=1.5, alpha=0.8, c="#3283FB", rasterized=True)
            ax.set_facecolor((206/256, 206/256, 229/256))

            traj = trajs[i]

            t_step = int(traj.shape[0] / (num_step - 1))
            start_j = 1
            ts = np.arange(t_step, t_step * num_step, t_step)
            for j in range(start_j, num_step):
                t = ts[j-1] - 1
                offset = 0
                ax = axs[i, j + offset]
                ax.scatter(
                    *traj[t, idcs].T, s=1.5, alpha=0.8, c="#3283FB", rasterized=True
                )
                ax.set_facecolor((206/256, 206/256, 229/256))
                if i == 0:
                    time = t / (t_step*(num_step - 1))
                    ax.set_title(f"t={time:.2f}")            

        axs[0, 0].set_title("source")
        axs[0, -1].set_title("target")

        for ax in axs.ravel():
            ax.set_xlim(-4, 4)
            ax.set_ylim(-4, 4)
            ax.set_xticks([])
            ax.set_yticks([])

        fig.tight_layout()
        fname = f"{sccfm_directory}/figs/{tag}_{self.knn_k}.pdf"
        os.makedirs(os.path.dirname(fname), exist_ok=True)
        fig.savefig(
            fname, bbox_inches="tight", pad_inches=0.0, transparent=True, dpi=300
        )
        wandb.log({f"imgs/{tag}_{self.knn_k}": wandb.Image(fig)})
        plt.close(fig)
        
    def gif(self, trajs, samples, num_row=3, num_col=3, num_step=250, tag="gnn_samples", title="Train"):
        if not isinstance(trajs, list): 
            trajs = trajs.cpu().detach().numpy()
            _, source, _ = samples  # Don't need target here
        else:
            trajs_tmp = []
            for traj in trajs:
                trajs_tmp.append(traj.cpu().detach().numpy())
            trajs = trajs_tmp

        fig, axs = plt.subplots(
            num_row,
            num_col,
            figsize=(4, 4),
            gridspec_kw={"wspace": 0.0, "hspace": 0.0},
        )
        axs = axs.reshape(num_row * num_col)

        n = 700
        rng = np.random.default_rng(42)

        def init():
            for ax in axs:
                ax.set_xlim(-4, 4)
                ax.set_ylim(-4, 4)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_facecolor((1, 1, 1))  # Set background to white
            return axs

        def update(frame):
            for i in range(num_row * num_col):
                ax = axs[i]
                ax.clear()
                ax.set_xlim(-4, 4)
                ax.set_ylim(-4, 4)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_facecolor((1, 1, 1))  # Keep background white

                # Initialize idcs based on the source samples
                if not isinstance(trajs, list):
                    idcs = rng.choice(np.arange(source[i].shape[0]), size=min(n, source[i].shape[0]), replace=False)
                else:
                    source_samples = samples[i][1].cpu().numpy()
                    idcs = rng.choice(np.arange(source_samples.shape[1]), size=min(n, source_samples.shape[1]), replace=False)

                if frame == 0:
                    if not isinstance(trajs, list):
                        source_samples = source[i].cpu().numpy()[idcs]
                    else:
                        source_samples = source_samples[0, idcs]
                    ax.scatter(*source_samples.T, s=3, c="#3283FB", alpha=0.8, rasterized=True)  # Adjusted size and transparency
                else:
                    traj = trajs[i]
                    t = int(frame * (traj.shape[0] - 1) / (num_step - 1))  # Evenly distribute frames across trajs
                    ax.scatter(*traj[t, idcs].T, s=3, c="#3283FB", alpha=0.8, rasterized=True)  # Adjusted size and transparency

            return axs

        # Add a title above the plots
        fig.suptitle(title, fontsize=16, y=0.93, fontweight="bold")
    
        ani = animation.FuncAnimation(
            fig, update, frames=np.arange(0, num_step), init_func=init, blit=False, interval=1, repeat_delay=1500
        )

        gif_path = f"{sccfm_directory}/figs/{tag}_{self.knn_k}.gif"
        os.makedirs(os.path.dirname(gif_path), exist_ok=True)
        ani.save(gif_path, writer='imagemagick', fps=60)  # High fps for smoother animation
        plt.close(fig)
        
    def save_population_embeddings(self):
        print("Saving population embeddings...")
        for k, v in self.data_for_embed_save.items():    
            for x in v['source']:
                self.model.update_embedding_for_inference(x.to(device='cuda'))
                embeddings = self.model.embedding.detach()
                self.data_for_embed_save[k]['embed'].append(embeddings)
                print(k, x.shape, embeddings.shape)
        
        with open('embeddings_letters_200.pkl', 'wb') as f:
            pickle.dump(self.data_for_embed_save, f)
            
        print("Population embeddings saved.")
        exit()