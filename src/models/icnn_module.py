"""
Adopted from https://github.com/bunnech/cellot/blob/main/cellot/models/cellot.py
"""

from pathlib import Path
import torch
from collections import namedtuple
from models.components.icnn import ICNN

import pytorch_lightning as pl
import yaml as yml

# from absl import flags
import numpy as np
import torch.nn.functional as F
from util.distribution_distances import (
    compute_distribution_distances,
    compute_scalar_mmd,
)

FGPair = namedtuple("FGPair", "f g")

def load_networks(config, **kwargs):
    def unpack_kernel_init_fxn(name="uniform", **kwargs):
        if name == "normal":

            def init(*args):
                return torch.nn.init.normal_(*args, **kwargs)

        elif name == "uniform":

            def init(*args):
                return torch.nn.init.uniform_(*args, **kwargs)

        else:
            raise ValueError

        return init

    print(config)
    kwargs.setdefault("hidden_units", [64] * 4)
    kwargs.update(dict(config.get("model", {})))

    kwargs.pop("name")
    if "latent_dim" in kwargs:
        kwargs.pop("latent_dim")
    fupd = kwargs.pop("f", {})
    gupd = kwargs.pop("g", {})

    fkwargs = kwargs.copy()
    fkwargs.update(fupd)
    fkwargs["kernel_init_fxn"] = unpack_kernel_init_fxn(
        **fkwargs.pop("kernel_init_fxn")
    )

    gkwargs = kwargs.copy()
    gkwargs.update(gupd)
    gkwargs["kernel_init_fxn"] = unpack_kernel_init_fxn(
        **gkwargs.pop("kernel_init_fxn")
    )

    f = ICNN(**fkwargs)
    g = ICNN(**gkwargs)
    return f, g


def load_opts(config, f, g):
    kwargs = dict(config.get("optim", {}))
    assert kwargs.pop("optimizer", "Adam") == "Adam"

    fupd = kwargs.pop("f", {})
    gupd = kwargs.pop("g", {})

    fkwargs = kwargs.copy()
    fkwargs.update(fupd)
    fkwargs["betas"] = (fkwargs.pop("beta1", 0.9), fkwargs.pop("beta2", 0.999))

    gkwargs = kwargs.copy()
    gkwargs.update(gupd)
    gkwargs["betas"] = (gkwargs.pop("beta1", 0.9), gkwargs.pop("beta2", 0.999))

    opts = FGPair(
        f=torch.optim.Adam(f.parameters(), **fkwargs),
        g=torch.optim.Adam(g.parameters(), **gkwargs),
    )

    return opts


def load_cellot_model(config, restore=None, **kwargs):
    f, g = load_networks(config, **kwargs)
    opts = load_opts(config, f, g)

    if restore is not None and Path(restore).exists():
        ckpt = torch.load(restore)
        f.load_state_dict(ckpt["f_state"])
        opts.f.load_state_dict(ckpt["opt_f_state"])

        g.load_state_dict(ckpt["g_state"])
        opts.g.load_state_dict(ckpt["opt_g_state"])

    return (f, g), opts


def compute_loss_g(f, g, source, transport=None):
    if transport is None:
        transport = g.transport(source)

    return f(transport) - torch.multiply(source, transport).sum(-1, keepdim=True)


def compute_g_constraint(g, form=None, beta=0):
    if form is None or form == "None":
        return 0

    if form == "clamp":
        g.clamp_w()
        return 0

    elif form == "fnorm":
        if beta == 0:
            return 0

        return beta * sum(map(lambda w: w.weight.norm(p="fro"), g.W))

    raise ValueError


def compute_loss_f(f, g, source, target, transport=None):
    if transport is None:
        transport = g.transport(source)

    return -f(transport) + f(target)


def compute_w2_distance(f, g, source, target, transport=None):
    if transport is None:
        transport = g.transport(source).squeeze()

    with torch.no_grad():
        Cpq = (source * source).sum(1, keepdim=True) + (target * target).sum(
            1, keepdim=True
        )
        Cpq = 0.5 * Cpq

        cost = (
            f(transport)
            - torch.multiply(source, transport).sum(-1, keepdim=True)
            - f(target)
            + Cpq
        )
        cost = cost.mean()
    return cost


def numerical_gradient(param, fxn, *args, eps=1e-4):
    with torch.no_grad():
        param += eps
    plus = float(fxn(*args))

    with torch.no_grad():
        param -= 2 * eps
    minus = float(fxn(*args))

    with torch.no_grad():
        param += eps

    return (plus - minus) / (2 * eps)


def check_loss(*args):
    for arg in args:
        if torch.isnan(arg):
            raise ValueError


# lightning wrapped cellot
class cellOT_ICNN_PL(pl.LightningModule):
    """Can only test one transport target at a time"""

    def __init__(
        self,
        config="cellot_specs.yaml",
        restore=None,
        target=None,
        loss_fn=F.mse_loss,
        metrics=["r2", "mmd", "corr", "wasserstein"],
        pca=None,
        ivp_batch_size=1024,
        seed=0,
        **kwargs,
    ):
        super().__init__()
        self.config = yml.safe_load(open(config))
        self.lr = self.config["optim"]["lr"]
        self.f, self.g = load_networks(self.config, **kwargs)
        self.loss_fn = loss_fn
        self.metrics = metrics
        self.naming = "ICNN"
        self.name = self.naming
        self.pca = pca
        self.ivp_batch_size = ivp_batch_size
        self.rng = np.random.default_rng(seed)
        self.automatic_optimization = False

        # to device
        self.f = self.f.cpu()
        self.g = self.g.cpu()

        self.opts = None
        self.restore = restore
        self.target = target
        self.train_metrics = {"PDO": [], "PDOF": [], "F": []}
        self.val_metrics = {"PDO": [], "PDOF": [], "F": []}
        self.test_metrics = {"PDO": [], "PDOF": [], "F": []}
        self.train_eval_batches = []

    def configure_optimizers(self):
        self.opts = load_opts(self.config, self.f, self.g)
        return self.opts

    def training_step(self, batch, batch_idx):
        """Got rid of outer loop because already randomized in dataloader"""
        _, _, x0, x1, _, _, _ = batch
        if len(x0.shape) >= 3:
            x0 = x0.squeeze()
            x1 = x1.squeeze()
        if x0.shape[0] != x1.shape[0]:
            min_len = min(x0.shape[0], x1.shape[0])
            x0 = x0[:min_len]
            x1 = x1[:min_len]
        target = x1.cpu()
        source = x0.cpu()
        source.requires_grad_(True)
        transport = self.g.transport(source)
        for _ in range(self.config["training"]["n_inner_iters"]):
            source.requires_grad_(True)

            self.opts.g.zero_grad()
            gl = compute_loss_g(self.f, self.g, source).mean()
            if not self.g.softplus_W_kernels and self.g.fnorm_penalty > 0:
                gl = gl + self.g.penalize_w()

            gl.backward()
            self.opts.g.step()

        self.opts.f.zero_grad()
        fl = compute_loss_f(self.f, self.g, source, target).mean()
        fl.backward()
        self.opts.f.step()
        check_loss(gl, fl)
        self.f.clamp_w()

        transport = transport.detach()
        gl = compute_loss_g(self.f, self.g, source, transport).mean()
        fl = compute_loss_f(self.f, self.g, source, target, transport).mean()
        mmd = compute_scalar_mmd(target.detach().numpy(), transport.detach().numpy())
        self.log("train_gloss", gl)
        self.log("train_floss", fl)
        self.log("train_mmd_cellot", mmd)

        if self.current_epoch > 0:
            self.train_eval_batches.append(batch)

        return gl + fl

    def on_train_epoch_end(self):
        if self.current_epoch > 0:
            for batch in self.train_eval_batches:
                x_pred, ground_truth, treatment = self.intermediate_steps(batch)
                names, dd = compute_distribution_distances(
                    x_pred,
                    ground_truth,
                )
                self.train_metrics[treatment].append({**dict(zip(names, dd))})
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
                self.train_eval_batches = []
                self.train_evals_count = 0
            self.train_metrics = {"PDO": [], "PDOF": [], "F": []}

    def transport_cellot(self, inputs):
        with torch.set_grad_enabled(True):
            inputs = inputs.float().cpu()
            if len(inputs.shape) < 2:
                inputs = inputs.unsqueeze(0)
            outputs = self.g.transport(inputs.requires_grad_(True))
        return outputs

    def validation_step(self, batch, batch_idx):
        x_pred, ground_truth, treatment = self.intermediate_steps(batch)
        names, dd = compute_distribution_distances(
            x_pred,
            ground_truth,
        )

        self.val_metrics[treatment].append({**dict(zip(names, dd))})

    def test_step(self, batch, batch_idx):
        x_pred, x1, treatment = self.intermediate_steps(batch)
        names, dd = compute_distribution_distances(
            x_pred,
            x1,
        )
        self.test_metrics[treatment].append({**dict(zip(names, dd))})

    def intermediate_steps(self, batch):
        """retuen list of results"""
        _, culture, x0, x1, _, _, _ = batch
        x0 = x0.float().cpu()
        x1 = x1.float().cpu()
        ground_truth = []
        total_pred = []
        for i in range(x0.shape[0]):
            x_pred = self.transport_cellot(x0[i])
            ground_truth.append(x1[i])
            total_pred.append(x_pred)
        return total_pred, ground_truth, culture[0]

    def on_validation_epoch_end(self):
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

        self.val_metrics = {"PDO": [], "PDOF": [], "F": []}

    def on_test_epoch_end(self):
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
        self.test_metrics = {"PDO": [], "PDOF": [], "F": []}

    def evaluate_val_mmd(self, target, source):
        """check if loss is correct"""

        source.requires_grad_(True)
        transport = self.g.transport(source)

        transport = transport.detach()
        with torch.no_grad():
            gl = compute_loss_g(self.f, self.g, source, transport).mean()
            dist = compute_w2_distance(self.f, self.g, source, target, transport)
            mmd = compute_scalar_mmd(
                target.detach().numpy(), transport.detach().numpy()
            )

        check_loss(gl, gl, dist)

        return mmd