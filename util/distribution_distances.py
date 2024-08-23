import math
import numpy as np
import torch
from typing import Optional
import ot as pot
from functools import partial
import pandas as pd
from sklearn.metrics.pairwise import rbf_kernel

def wasserstein(
    x0: torch.Tensor,
    x1: torch.Tensor,
    method: Optional[str] = None,
    reg: float = 0.05,
    power: int = 2,
    **kwargs,
) -> float:
    assert power == 1 or power == 2
    # ot_fn should take (a, b, M) as arguments where a, b are marginals and
    # M is a cost matrix
    if method == "exact" or method is None:
        ot_fn = pot.emd2
    elif method == "sinkhorn":
        ot_fn = partial(pot.sinkhorn2, reg=reg)
    else:
        raise ValueError(f"Unknown method: {method}")

    a, b = pot.unif(x0.shape[0]), pot.unif(x1.shape[0])
    if x0.dim() > 2:
        x0 = x0.reshape(x0.shape[0], -1)
    if x1.dim() > 2:
        x1 = x1.reshape(x1.shape[0], -1)
    M = torch.cdist(x0, x1)
    if power == 2:
        M = M**2
    ret = ot_fn(a, b, M.detach().cpu().numpy(), numItermax=1e7)
    if power == 2:
        ret = math.sqrt(ret)
    return ret

def mmd_distance(x, y, gamma):
    xx = rbf_kernel(x, x, gamma)
    xy = rbf_kernel(x, y, gamma)
    yy = rbf_kernel(y, y, gamma)

    return xx.mean() + yy.mean() - 2 * xy.mean()

def compute_scalar_mmd(target, transport, gammas=None):
    if gammas is None:
        gammas = [2, 1, 0.5, 0.1, 0.01, 0.005]

    def safe_mmd(*args):
        try:
            mmd = mmd_distance(*args)
        except ValueError:
            mmd = np.nan
        return mmd

    return np.mean(list(map(lambda x: safe_mmd(target, transport, x), gammas)))

def compute_distances(pred, true):
    """computes distances between vectors."""
    mse = torch.nn.functional.mse_loss(pred, true).item()
    me = math.sqrt(mse)
    return mse, me, torch.nn.functional.l1_loss(pred, true).item()

def compute_distribution_distances(pred, true):
    """computes distances between distributions.
    pred: [batch, times, dims] tensor
    true: [batch, times, dims] tensor or list[batch[i], dims] of length times

    This handles jagged times as a list of tensors.
    return the eval for the last time point
    """
    NAMES = [
        "1-Wasserstein",
        "2-Wasserstein",
        "RBF_MMD",
        "Mean_MSE",
        "Mean_L2",
        "Mean_L1",
        "Median_MSE",
        "Median_L2",
        "Median_L1",
        "r2_pwf_corr",
    ]
    is_jagged = isinstance(true, list)
    pred_is_jagged = isinstance(pred, list)
    dists = []
    to_return = []
    names = []
    filtered_names = [name for name in NAMES if not is_jagged or not name.endswith("MMD")]
    ts = len(pred) if pred_is_jagged else pred.shape[1]

    t = max(ts - 1, 0)
    if pred_is_jagged:
        a = pred[t]
    else:
        a = pred[:, t, :]
    if is_jagged:
        b = true[t]
    else:
        b = true[:, t, :]
    w1 = wasserstein(a, b, power=1)
    w2 = wasserstein(a, b, power=2)
    if not pred_is_jagged and not is_jagged:
        mmd_rbf = compute_scalar_mmd(b.cpu().numpy(), a.cpu().numpy())
    mean_dists = compute_distances(torch.mean(a, dim=0), torch.mean(b, dim=0))
    median_dists = compute_distances(torch.median(a, dim=0)[0], torch.median(b, dim=0)[0])
    
    r2_pairwise_feat_corrs = cellot_corr(a.cpu().numpy(), b.cpu().numpy())
    
    if pred_is_jagged or is_jagged:
        dists.append((w1, w2, *mean_dists, *median_dists, r2_pairwise_feat_corrs))
    else:
        dists.append(
            (w1, w2, mmd_rbf, *mean_dists, *median_dists, r2_pairwise_feat_corrs)
        )

    to_return.extend(np.array(dists).mean(axis=0))
    names.extend(filtered_names)
    return names, to_return

def compute_pairwise_corrs(df):
    corr = df.corr().rename_axis(index="lhs", columns="rhs")
    return (
        corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        .stack()
        .reset_index()
        .set_index(["lhs", "rhs"])
        .squeeze()
    )

def cellot_corr(pred, ground_truth):
    pwct = compute_pairwise_corrs(pd.DataFrame(pred))
    pwci = compute_pairwise_corrs(pd.DataFrame(ground_truth))
    r2_pairwise_feat_corrs = pd.Series(pwct).corr(pd.Series(pwci))
    return r2_pairwise_feat_corrs