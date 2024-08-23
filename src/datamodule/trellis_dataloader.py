"""
Lightning datamodule for the organoid drug-screen (trellis) dataset.
In this code base, we use the name "trellis" as a short hand for this dataset.
"""

import time
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
import pickle
import yaml as yml
import numpy as np
import torch
from sklearn.decomposition import PCA


class trellis_dataset(Dataset):
    def __init__(
        self,
        split,
        data,
        num_components=None,
        ivp_batch_size=None,
        plot_pca=False,
        use_small_exp_num=False,
        control=set(["DMSO", "AH", "H2O"]),
        treatment=["O", "S", "VS", "L", "V", "F", "C", "SF", "CS", "CF", "CSF"],
        culture=["PDO", "PDOF", "F"],
        cell_type=["PDOs", "Fibs"],
        prefix="train",
        pca=None,
        seed=0,
    ):

        self.rng = np.random.default_rng(seed)
        
        self.data = data
        self.ivp_batch_size = ivp_batch_size
        self.plot_pca = plot_pca
        
        self.control = control  # identify x0
        self.treatment = treatment 
        self.culture = culture
        self.cell_type = cell_type
        
        self.num_components = num_components
        self.use_small_exp_num = use_small_exp_num
        self.prefix = prefix

        self.split = self.__filter_control__(split)
        self.exp_idx = self.__lst_exp__(self.split)

        self.pca = pca
        self.pca_for_plot = PCA(n_components=2) # used for plotting.

        print("Constructing {} data ...".format(prefix))
        
        # construct dataset
        start = time.time()
        self.construct_data()
        end = time.time()
        print("done. Time (s):", print(end - start))

        # select train envs/replicas for train-evaluation
        if prefix == "train":
            self.eval_mode = False
            if not self.use_small_exp_num:
                self.idcs_train_eval = np.random.default_rng(42).choice(
                    np.arange(len(self.samples)), size=100, replace=False
                )
            else:
                self.idcs_train_eval = [i for i in range(len(self.samples))]
            
            self.train_eval_replicas = []
            for i in self.idcs_train_eval:
                if self.pca is not None:
                    culture, x0, x1, _, x1_full, cell_cond, treat_cond = self.samples[i]
                    replica = [
                        torch.tensor(i),
                        (culture,),
                        torch.tensor(x0[None, ...]),
                        torch.tensor(x1[None, ...]),
                        torch.tensor(x1_full[None, ...]),
                        torch.tensor(cell_cond[None, ...]),
                        torch.tensor(treat_cond[None, ...]),
                    ]
                    self.train_eval_replicas.append(self.get_eval_cells(replica, eval=True))
                else: 
                    culture, x0, x1, cell_cond, treat_cond = self.samples[i]
                    replica = [
                        torch.tensor(i),
                        (culture,),
                        torch.tensor(x0[None, ...]),
                        torch.tensor(x1[None, ...]),
                        None,
                        torch.tensor(cell_cond[None, ...]),
                        torch.tensor(treat_cond[None, ...]),
                    ]
                    self.train_eval_replicas.append(self.get_eval_cells(replica, eval=True))            
        elif prefix == "val" or prefix == "test":
            self.eval_mode = True
        else:
            raise ValueError("prefix not recognized")

        print("... Data loaded!")

    def construct_data(self):
        self.samples_tmp, self.culture, self.x0, self.x1, self.cell_cond, self.treat_cond = self.select_experiments()
        
        if self.prefix == "train":
            if self.num_components is None:
                self.samples = self.samples_tmp
            else:
                print("Fitting PCA for low-dim representation ...")
                x0_train = np.concatenate(self.x0, axis=0)
                x1_train = np.concatenate(self.x1, axis=0)
                x_train = np.concatenate([x0_train, x1_train], axis=0)
                print(x_train.shape)

                self.pca = PCA(n_components=self.num_components)
                self.pca.fit(x_train)

                print("... PCA fit done!")

                self.samples = self.pca_embed_samples(self.samples_tmp)

            if self.use_small_exp_num:
                idcs = self.rng.choice(
                    np.arange(len(self.samples)), size=6, replace=False
                )
                new_samples = []
                for i in idcs:
                    new_samples.append(self.samples[i])
                self.samples = new_samples
                
            # fit PCA for 2D plotting
            if self.plot_pca:
                print("Fitting PCA for 2D plotting ...")
                xs = []
                for sample in self.samples:
                    _, x0, x1, _, _ = sample
                    xs.append(np.concatenate([x0, x1], axis=0))
                xs = np.concatenate(xs, axis=0)
                print(xs.shape)
                self.pca_for_plot.fit(xs)
                print("... 2D plotting PCA fit done!")

        elif self.prefix == "val":
            if self.num_components is None:
                self.samples = self.samples_tmp
            else:
                self.samples = self.pca_embed_samples(self.samples_tmp)

            if self.use_small_exp_num:
                idcs = [1, 12, 23]
                new_samples = []
                for i in idcs:
                    new_samples.append(self.samples[i])
                self.samples = new_samples

        elif self.prefix == "test":
            if self.num_components is None:
                self.samples = self.samples_tmp
            else:
                self.samples = self.pca_embed_samples(self.samples_tmp)

            if self.use_small_exp_num:
                idcs = [1, 12, 23]
                new_samples = []
                for i in idcs:
                    new_samples.append(self.samples[i])
                self.samples = new_samples

        else:
            raise ValueError("prefix not recognized")

    def select_experiments(self):
        samples_tmp, cultures, sources, targets, cell_conds, treat_conds = [], [], [], [], [], []
        
        for i in range(len(self.split)):
            exp = self.split[i]

            x0_treatment = list(set(exp.keys()).intersection(self.control))[0]
            treatkeys = [key for key in exp.keys() if key not in self.control]
            for t in treatkeys:
                concentration = list(exp[t].keys())
                max_conc = str(max(map(int, concentration)))

                cultures_keys = list(exp[t][max_conc].keys())
                for culture in cultures_keys:
                    
                    x0_pdos_idx, x1_pdos_idx, x0_fibs_idx, x1_fibs_idx = [], [], [], []
                    if culture in ["PDOF", "PDO"]:
                        x0_pdos_idx = exp[x0_treatment]["0"][culture][self.cell_type[0]].copy().tolist()
                        x1_pdos_idx = exp[t][max_conc][culture][self.cell_type[0]].copy().tolist()

                    if culture in ["PDOF", "F"]:
                        x0_fibs_idx = exp[x0_treatment]["0"][culture][self.cell_type[1]].copy().tolist()
                        x1_fibs_idx = exp[t][max_conc][culture][self.cell_type[1]].copy().tolist()
                        
                    # concat x0 and x1 idcs
                    x0_idx = x0_pdos_idx + x0_fibs_idx
                    x1_idx = x1_pdos_idx + x1_fibs_idx

                    # create data
                    x0 = np.array(self.data[x0_idx])
                    x1 = np.array(self.data[x1_idx])
                    
                    # get cell type one-hot encoding for x0 populations
                    x0_cell_pdos_idx = range(0, len(x0_pdos_idx))
                    x0_cell_fibs_idx = range(len(x0_pdos_idx), len(x0_idx))
                    cond_cell = np.zeros((x0.shape[0], len(self.cell_type)))
                    cond_cell[x0_cell_pdos_idx, 0] = 1
                    cond_cell[x0_cell_fibs_idx, 1] = 1

                    # get treatment one-hot encoding 
                    treat_idx = self.treatment.index(t)
                    cond_treat = torch.nn.functional.one_hot(
                        torch.tensor(treat_idx).long(), num_classes=len(self.treatment)
                    )
                    cond_treat = cond_treat.expand(x0.shape[0], -1).detach().numpy()

                    samples_tmp.append(
                        (
                            culture,
                            x0,
                            x1,
                            cond_cell,
                            cond_treat,
                        )
                    )
                    
                    cultures.append(culture)
                    targets.append(x1)
                    cell_conds.append(cond_cell)
                    treat_conds.append(cond_treat)
            sources.append(x0)

        self.num_samples = len(samples_tmp)
        print("{} {} data samples".format(self.num_samples, self.prefix))
        return samples_tmp, cultures, sources, targets, cell_conds, treat_conds

    def pca_embed_samples(self, samples_tmp):
        samples = []
        for sample in samples_tmp:
            culture, x0, x1, cell_cond, treat_cond = sample
            x0_pca = self.pca.transform(x0)
            x1_pca = self.pca.transform(x1)
            
            samples.append(
                (
                    culture,
                    x0_pca,
                    x1_pca,
                    x0,
                    x1,
                    cell_cond,
                    treat_cond,
                )
            )
            
        return samples

    def __filter_control__(self, split):
        split_lst = []
        for ls in split:
            #keyset = set(ls.keys())
            if self.has_empty_element(ls):
                continue
            split_lst.append(ls)
        return split_lst

    def has_empty_element(self, nested_dict):
        for key, value in nested_dict.items():
            if isinstance(value, dict):  # Check if the item is a dictionary
                if not value:  # Check if the dictionary is empty
                    return (
                        True  # Return True immediately upon finding an empty dictionary
                    )
                else:
                    # Recursively check further in the dictionary
                    if self.has_empty_element(value):
                        return True
        return False  # Return False if no empty dictionary is found after checking all items

    def __lst_exp__(self, idx):
        split_idx = {}
        count = 0
        for num, ls in enumerate(self.split):
            keyset = set(ls.keys())
            ctrl_key = list(keyset.intersection(self.control))
            if len(ctrl_key) == 0 or "0" not in ls[ctrl_key[0]]:
                continue
            for key, value in ls.items():
                for conc, ids in value.items():
                    count += 1
                    split_idx[count] = (num, key, conc)
        return split_idx

    def __len__(self):
        return len(self.samples)
    
    def get_eval_cells(self, batch, eval=False):
        # cell batching for evaluation
        idx, culture, x0, x1, x1_full, cell_cond, treat_cond = batch

        if len(x0.shape) < 3:
            x0 = x0[None, ...]
            x1 = x1[None, ...]
            x1_full = x1_full[None, ...] if self.pca is not None else x1_full
            cell_cond = cell_cond[None, ...]
            treat_cond = treat_cond[None, ...]
        
        if eval:
            if x0.shape[1] > 5000 or x1.shape[1] > 5000:
                return (
                    idx,
                    culture,
                    x0[:, :5000, :] if x0.shape[1] > 5000 else x0,
                    x1[:, :5000, :] if x1.shape[1] > 5000 else x1,
                    x1_full[:, :5000, :] if x1.shape[1] > 5000 and self.pca is not None else x1_full,
                    cell_cond[:, :5000, :] if x0.shape[1] > 5000 else cell_cond,
                    treat_cond[:, :5000, :] if x0.shape[1] > 5000 else treat_cond,
                )
            else:
                return idx, culture, x0, x1, x1_full, cell_cond, treat_cond
    
    def unpack_batch(self, batch):
        # cell batching for evaluation
        
        if self.eval_mode:
            batch = self.get_eval_cells(batch, eval=self.eval_mode)
            
        idx, culture, x0, x1, x1_full, cell_cond, treat_cond = batch
        
        # cell batching for training
        if self.ivp_batch_size is not None:
            x0_ivp_idcs = self.rng.choice(
                np.arange(x0.shape[0]), size=self.ivp_batch_size, replace=False
            )
            x1_ivp_idcs = self.rng.choice(
                np.arange(x1.shape[0]), size=self.ivp_batch_size, replace=False
            )
            x0 = x0[x0_ivp_idcs, :]
            x1 = x1[x1_ivp_idcs, :]
            x1_full = x1_full[x1_ivp_idcs, :] if self.pca is not None else None
            cell_cond = cell_cond[x0_ivp_idcs, :]
            treat_cond = treat_cond[x0_ivp_idcs, :]
            return idx, culture, x0, x1, x1_full, cell_cond, treat_cond
        else:
            x1_full = x1_full if self.pca is not None else None
            return idx, culture, x0, x1, x1_full, cell_cond, treat_cond

    def __getitem__(self, idx):
        if self.pca is not None:
            culture, x0, x1, _, x1_full, cell_cond, treat_cond = self.samples[idx]
            replica = [idx, culture, x0, x1, x1_full, cell_cond, treat_cond]
            idx, culture, x0, x1, x1_full, cell_cond, treat_cond = self.unpack_batch(replica)
            return (
                idx,
                culture,
                x0,
                x1,
                x1_full,
                cell_cond,
                treat_cond,
            )
        else:
            culture, x0, x1, cell_cond, treat_cond = self.samples[idx]
            replica = [idx, culture, x0, x1, None, cell_cond, treat_cond]
            idx, culture, x0, x1, _, cell_cond, treat_cond = self.unpack_batch(replica)
            return (
                idx,
                culture,
                x0,
                x1,
                torch.tensor(0), # dummy tensor for x1_full (since x1 = x1_full in this case)
                cell_cond,
                treat_cond,
            )


def custom_collate_fn(batch):
    """Custom collate function to handle variable length tensors and condition data."""
    # Extract fields from the batch
    idxs, cultures, x0s, x1s, x1_fulls, cell_conds, treat_conds = zip(*batch)

    # Determine the actual shapes of x0 and x1
    def resolve_shape(tensor):
        if tensor.ndim == 3 and tensor.shape[0] == 1:
            return tensor.shape[1], tensor.shape[2]  # Return (n, d) for shape [1, n, d]
        elif tensor.ndim == 2:
            return tensor.shape  # Return (n, d) for shape [n, d]
        else:
            raise ValueError(f"Unexpected tensor shape: {tensor.shape}")

    # Get the lengths for x0 and x1 in the batch
    lengths_x0 = [resolve_shape(x0)[0] for x0 in x0s]
    lengths_x1 = [resolve_shape(x1)[0] for x1 in x1s]

    # Get the maximum lengths for x0 and x1 in the batch
    max_n = max(lengths_x0)
    max_m = max(lengths_x1)

    # Determine the dimensions
    d0 = resolve_shape(x0s[0])[1]
    d1 = resolve_shape(x1s[0])[1]
    d1_full = resolve_shape(x1_fulls[0])[1] if x1_fulls[0].ndim == x1s[0].ndim else None

    # Initialize the batch tensors with zeros (or another padding value)
    x0_batch = torch.zeros((len(x0s), max_n, d0))
    x1_batch = torch.zeros((len(x1s), max_m, d1))
    x1_full_batch = torch.zeros((len(x1_fulls), max_m, d1_full)) if d1_full is not None else x1_fulls

    # Initialize the condition batch tensors
    c_cell = resolve_shape(cell_conds[0])[1]
    cell_conds_batch = torch.zeros((len(cell_conds), max_n, c_cell))

    c_treat = resolve_shape(treat_conds[0])[1]
    treat_conds_batch = torch.zeros((len(treat_conds), max_n, c_treat))

    # Initialize zero padding indices if necessary
    needs_padding_x0 = any(length != max_n for length in lengths_x0)
    needs_padding_x1 = any(length != max_m for length in lengths_x1)
    zero_pad_idx_x0 = torch.zeros(len(x0s), dtype=torch.long) if needs_padding_x0 else None
    zero_pad_idx_x1 = torch.zeros(len(x1s), dtype=torch.long) if needs_padding_x1 else None

    # Populate the batch tensors with the actual data
    for i, (x0, x1, x1_full, cell_cond, treat_cond) in enumerate(zip(x0s, x1s, x1_fulls, cell_conds, treat_conds)):
        if x0.ndim == 3 and x0.shape[0] == 1:
            x0 = x0.squeeze(0)
        if x1.ndim == 3 and x1.shape[0] == 1:
            x1 = x1.squeeze(0)
        if d1_full is not None and x1_full.ndim == 3 and x1_full.shape[0] == 1:
            x1_full = x1_full.squeeze(0)
        if cell_cond.ndim == 3 and cell_cond.shape[0] == 1:
            cell_cond = cell_cond.squeeze(0)
        if treat_cond.ndim == 3 and treat_cond.shape[0] == 1:
            treat_cond = treat_cond.squeeze(0)

        x0_batch[i, :x0.shape[0], :] = torch.tensor(x0)
        x1_batch[i, :x1.shape[0], :] = torch.tensor(x1)
        if d1_full is not None:
            x1_full_batch[i, :x1_full.shape[0], :] = torch.tensor(x1_full)
        cell_conds_batch[i, :cell_cond.shape[0], :] = torch.tensor(cell_cond)
        treat_conds_batch[i, :treat_cond.shape[0], :] = torch.tensor(treat_cond)

        if needs_padding_x0:
            zero_pad_idx_x0[i] = x0.shape[0]
        if needs_padding_x1:
            zero_pad_idx_x1[i] = x1.shape[0]

    # Convert idxs and cultures to tensors
    idxs = torch.tensor(idxs)

    if needs_padding_x0 or needs_padding_x1:
        return idxs, cultures, x0_batch, x1_batch, x1_full_batch, cell_conds_batch, treat_conds_batch, zero_pad_idx_x0, zero_pad_idx_x1
    else:
        return idxs, cultures, x0_batch, x1_batch, x1_full_batch, cell_conds_batch, treat_conds_batch


class TrellisDatamodule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size=1,
        ivp_batch_size=1024,
        split="patients",  # ["patients", "replicas"]
        plot_pca=False,
        marker_cols="trellis_marker_col.yaml",
        data_path="trellis_patients_normalized.npy",  # trellis_pdo_fib_normalized_v3.npy
        control=set(["DMSO", "AH", "H2O"]),
        treatment=["O", "S", "VS", "L", "V", "F", "C", "SF", "CS", "CF", "CSF"],
        culture=["PDO", "PDOF", "F"],
        cell_type=["PDOs", "Fibs"],
        name="trellis_replicates",
        num_components=None,
        use_small_exp_num=False,
        seed=0,
    ):
        super().__init__()
        
        self.batch_size = batch_size
        self.ivp_batch_size = ivp_batch_size
        
        assert split in ["patients", "replicas"], "split must be either 'patients' or 'replicas'"
        if split == "patients":
            self.split_source = (
                "data_splits_patients.pickle"
            )
        elif split == "replicas":
            self.split_source = (
                "data_splits_pdo_fib.pickle"
            )
        else:
            raise ValueError("split not recognized")
        
        self.cell_type = cell_type  # filtered cell type
        self.name = name
        self.num_components = num_components
        self.use_small_exp_num = use_small_exp_num
        self.seed = seed
        with open(self.split_source, "rb") as handle:
            self.data_splits = pickle.load(handle)
            # from these split we will get the background cells

        self.marker_cols = list(
            yml.safe_load(open(marker_cols))["marker"]
        )  # dropping one col (because it is missing half the data)
        self.input_dim = len(self.marker_cols)
        self.non_marker_cols = [
            "Treatment",
            "Culture",
            "Date",
            "Patient",
            "Concentration",
            "Replicate",
            "Cell_type",
            "Plate",
            "Batch",
        ]

        self.data = np.load(data_path)[:, :-1]
        
        self.control = control
        self.treatment = treatment
        self.treatment_test = self.treatment
        self.cell_type = cell_type
        self.culture = culture
        
        self.train_dataset = trellis_dataset(
            split=self.data_splits["train"],
            data=self.data,
            ivp_batch_size=self.ivp_batch_size,
            plot_pca=plot_pca,
            num_components=self.num_components,
            use_small_exp_num=self.use_small_exp_num,
            control=self.control,
            treatment=self.treatment,
            culture=self.culture,
            cell_type=self.cell_type,
            prefix="train",
            seed=self.seed,
        )
        self.num_train_replica = self.train_dataset.num_samples
        self.pca_for_plot = self.train_dataset.pca_for_plot if plot_pca else None
        
        self.val_dataset = trellis_dataset(
            split=self.data_splits["val"],
            data=self.data,
            num_components=self.num_components,
            use_small_exp_num=self.use_small_exp_num,
            control=self.control,
            treatment=self.treatment,
            culture=self.culture,
            cell_type=self.cell_type,  
            prefix="val",
            pca=self.train_dataset.pca if self.num_components is not None else None,
            seed=self.seed,
        )
        self.num_val_replica = self.val_dataset.num_samples

        self.test_dataset = trellis_dataset(
            split=self.data_splits["test"],
            data=self.data,
            num_components=self.num_components,
            use_small_exp_num=self.use_small_exp_num,
            control=self.control,
            treatment=self.treatment,
            culture=self.culture,
            cell_type=self.cell_type,  
            prefix="test",
            pca=self.train_dataset.pca if self.num_components is not None else None,
            seed=self.seed,
        )
        self.num_test_replica = self.test_dataset.num_samples

        print("DataModule initialized")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=4,
            shuffle=True,
            collate_fn=custom_collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=1,
            num_workers=4,
            shuffle=False,
            collate_fn=custom_collate_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=1,
            num_workers=4,
            shuffle=False,
            collate_fn=custom_collate_fn,
        )
    
    @property
    def dims(self):
        return {
            "input_dim": self.input_dim,
            "conds": len(self.treatment) + len(self.cell_type),
        }
    