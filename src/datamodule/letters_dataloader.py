"""
This module contains the dataloader for the letters dataset.
"""

import numpy as np
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from PIL import Image, ImageDraw, ImageFont
from collections import defaultdict

import sys
from pathlib import Path

current_directory = Path(__file__).absolute().parent
parent_directory = current_directory.parent
parent_parent_directory = current_directory.parent.parent
sys.path.append(str(parent_directory))

class letters_replica_batch_dataset(Dataset):
    def __init__(
        self,
        noise_scale=0.05,
        source_noise_scale=0.5,
        num_rotations=10,
        ivp_batch_size=None,
        num_train_letters=24,
        conditional=False,
        save_embeddings=False,
        seed=0,
        mode="train",
    ) -> None:

        self.noise_scale = noise_scale
        self.source_noise_scale = source_noise_scale
        self.num_rotations = num_rotations
        self.ivp_batch_size = ivp_batch_size
        self.num_train_letters = num_train_letters
        self.conditional = conditional
        self.save_embeddings = save_embeddings
        self.seed = seed

        if num_rotations == 10:
            self.idcs_train = [2, 11, 23, 31, 45, 52, 61, 72, 83, 94]
        else:
            list_idcs = np.linspace(0, 23*num_rotations, 10).tolist()
            self.idcs_train = [int(x) for x in list_idcs]
        self.train_eval_letters = []
        
        assert mode in [
            "train",
            "val",
            "test",
        ], "Invalid mode. Must be either 'train' or 'val' or 'test'" 
        self.mode = mode
        
        if self.num_train_letters == 24:
            self.alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        # used for diversity ablation
        elif self.num_train_letters == 16:
            self.alphabet = "SQWHEFGAIJKLMNOP"
            self.num_rotations = int((24 * self.num_rotations) / 16)
        elif self.num_train_letters == 8:
            self.alphabet = "SQWHEFGH"
            self.num_rotations = int((24 * self.num_rotations) / 8)
        elif self.num_train_letters == 4:
            self.alphabet = "SQWH"
            self.num_rotations = int((24 * self.num_rotations) / 4)
        elif self.num_train_letters == 2:
            self.alphabet = "SQ"
            self.num_rotations = int((24 * self.num_rotations) / 2)
        elif self.num_train_letters == 1:
            self.alphabet = "S"
            self.num_rotations = int((24 * self.num_rotations) / 1)
        else:
            raise ValueError("Invalid number of train letters specified.")
        
        if self.conditional:
            self.num_train_conditions = (len(self.alphabet) - 2) * self.num_rotations if self.num_train_letters == 24 else len(self.alphabet) * self.num_rotations + 20
            self.num_val_conditions = 10 #self.num_rotations
            self.num_test_conditions = 10 #self.num_rotations
            self.num_conditions = self.num_train_conditions + self.num_val_conditions + self.num_test_conditions
        
        self.samples = self.get_denoising_samples(mode, num_rotations=self.num_rotations, seed=seed)
        self.num_samples = len(self.samples)
        
        if mode == "train":
            for idc in self.idcs_train:
                if self.conditional:
                    x0, x1, conds = self.samples[idc]
                    self.train_eval_letters.append(
                        (None, x0.unsqueeze(0).cuda(), x1.unsqueeze(0).cuda(), conds.unsqueeze(0).cuda())
                    )
                else:
                    x0, x1 = self.samples[idc]
                    self.train_eval_letters.append(
                        (None, x0.unsqueeze(0).cuda(), x1.unsqueeze(0).cuda())
                    )

    def char_sampler(self, char, font_size=100, rotation=0.0):
        # generate samples from a character
        font = ImageFont.truetype(str(parent_parent_directory)+"/arial.ttf", font_size)
        image_size = 2*font_size
        img = Image.new('L', (image_size, image_size), color = 0)
        d = ImageDraw.Draw(img)
        d.text((0,0), char, fill=(255), font=font)

        img = np.array(img)

        img = np.flipud(img)

        # extract samples
        grid = np.indices(img.shape).T
        mask = img > 0
        samples = grid[mask]
        samples = torch.from_numpy(samples).float()

        min_vals = samples.min(axis=0).values
        max_vals = samples.max(axis=0).values
        range_vals = max_vals - min_vals
        samples = (samples - min_vals) / range_vals
        ratio = range_vals[0] / range_vals[1]

        samples = samples * 2 - 1
        samples *= 3.

        if ratio.item() < 1.:
            samples[:, 0] *= ratio
        else:
            samples[:, 1] /= ratio

        samples = samples + self.noise_scale*torch.randn_like(samples)

        samples = samples.cuda()

        M = torch.tensor([
            [np.cos(rotation), -np.sin(rotation)],
            [np.sin(rotation), np.cos(rotation)]
        ], device=samples.device, dtype=samples.dtype)
        samples = samples @ M.T

        return samples

    def get_denoising_samples(self, mode='train', num_rotations=10, seed=0):
        torch.manual_seed(seed)
        np.random.seed(seed)

        samples = []
        if mode == 'train':
            if self.conditional:
                cond = 0
            for target in self.alphabet:
                if target != "Y" and target != "X":
                    for i in range(num_rotations):
                        rotation = 2.0 * torch.pi * torch.rand(1)
                        target_samples = self.char_sampler(
                            target, rotation=rotation.item()
                        )
                        source_samples = (
                            target_samples
                            + self.source_noise_scale * torch.randn_like(target_samples)
                        )
                        if not self.conditional:
                            samples.append((source_samples.cpu(), target_samples.cpu()))
                        else:
                            condition = torch.zeros(
                                (source_samples.shape[0], self.num_conditions)
                            )
                            condition[:, cond] = 1
                            samples.append(
                                (source_samples.cpu(), target_samples.cpu(), condition.cpu())
                            )
                            cond += 1
        elif mode == 'val':
            target = "X"
            num_rotate_val = self.num_rotations if self.save_embeddings else 10
            if self.conditional:
                cond = self.num_train_conditions
            for i in range(num_rotate_val):  # range(num_rotations):
                rotation = 2.0 * torch.pi * torch.rand(1)
                target_samples = self.char_sampler(target, rotation=rotation.item())
                source_samples = (
                    target_samples + self.source_noise_scale * torch.randn_like(target_samples)
                )
                if not self.conditional:
                    samples.append(
                        (source_samples.cpu(), target_samples.cpu())
                    )
                else:
                    condition = torch.zeros((source_samples.shape[0], self.num_conditions))
                    condition[:, cond] = 1
                    samples.append(
                        (source_samples.cpu(), target_samples.cpu(), condition.cpu())
                    )
                    cond += 1
        elif mode == "test":
            target = "Y"
            num_rotate_test = self.num_rotations if self.save_embeddings else 10
            if self.conditional:
                cond = self.num_train_conditions + self.num_val_conditions
            for i in range(num_rotate_test):  # range(num_rotations):
                rotation = 2.0 * torch.pi * torch.rand(1)
                target_samples = self.char_sampler(target, rotation=rotation.item())
                source_samples = (
                    target_samples + self.source_noise_scale * torch.randn_like(target_samples)
                )
                if not self.conditional:
                    samples.append(
                        (source_samples.cpu(), target_samples.cpu())
                    )
                else:
                    condition = torch.zeros((source_samples.shape[0], self.num_conditions))
                    condition[:, cond] = 1
                    samples.append(
                        (source_samples.cpu(), target_samples.cpu(), condition.cpu())
                    )
                    cond += 1
        else:
            raise ValueError('Invalid mode')

        return samples

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        if self.conditional:
            source, target, cond = self.samples[idx] # [1, num_samples, d] one "env" at a time.
            if self.ivp_batch_size is not None:
                if self.mode == "val" or self.mode == "test":
                    return idx, source, target, cond
                ivp_idx = np.random.choice(
                    np.arange(source.shape[0]), size=self.ivp_batch_size, replace=False
                )
                return idx, source[ivp_idx], target[ivp_idx], cond[ivp_idx]  # [1, ivp_bs, d]
            return idx, source, target, cond  
        else:
            source, target = self.samples[idx]  # [1, num_samples, d] one "env" at a time.
            if self.ivp_batch_size is not None:
                if self.mode == "val" or self.mode == "test":
                    return idx, source, target
                ivp_idx = np.random.choice(
                    np.arange(source.shape[0]), size=self.ivp_batch_size, replace=False
                )
                return idx, source[ivp_idx], target[ivp_idx]  # [1, ivp_bs, d]
            return idx, source, target 


def custom_collate_fn(batch, include_cond=False):
    """TODO:
    Training code is not implemented to handle the case where zero padding is necessary. 
    """
    # Extract idx, x0, and x1 from the batch
    if include_cond:
        idxs, x0s, x1s, conds = zip(*batch)
    else:
        idxs, x0s, x1s = zip(*batch)
        conds = None

    # Get the lengths for x0 and x1 in the batch
    lengths_x0 = [x0.shape[0] for x0 in x0s]
    lengths_x1 = [x1.shape[0] for x1 in x1s]

    # Get the maximum lengths for x0 and x1 in the batch
    max_n = max(lengths_x0)
    max_m = max(lengths_x1)

    # Determine the dimension d
    d = x0s[0].shape[-1]

    # Check if zero padding is necessary
    needs_padding_x0 = any(length != max_n for length in lengths_x0)
    needs_padding_x1 = any(length != max_m for length in lengths_x1)

    # Initialize the batch tensors with zeros (or another padding value)
    x0_batch = torch.zeros((len(x0s), max_n, d))
    x1_batch = torch.zeros((len(x1s), max_m, d))

    if needs_padding_x0:
        zero_pad_idx_x0 = torch.zeros(len(x0s), dtype=torch.long)
    if needs_padding_x1:
        zero_pad_idx_x1 = torch.zeros(len(x1s), dtype=torch.long)

    # Populate the batch tensors with the actual data
    for i, (x0, x1) in enumerate(zip(x0s, x1s)):
        x0_batch[i, : x0.shape[0], :] = x0
        x1_batch[i, : x1.shape[0], :] = x1
        if needs_padding_x0:
            zero_pad_idx_x0[i] = x0.shape[0]
        if needs_padding_x1:
            zero_pad_idx_x1[i] = x1.shape[0]

    # Convert idxs to a tensor
    idxs = torch.tensor(idxs)
    
    if include_cond:
        # Determine the shape of conds and initialize the conds_batch tensor with zeros
        c = conds[0].shape[1]
        conds_batch = torch.zeros((len(conds), max_n, c))
        
        if needs_padding_x0:
            zero_pad_idx_cond = torch.zeros(len(conds), dtype=torch.long)

        # Populate the conds_batch tensor with the actual data
        for i, cond in enumerate(conds):
            conds_batch[i, : cond.shape[0], :] = cond
            if needs_padding_x0:
                zero_pad_idx_cond[i] = cond.shape[0]

    if needs_padding_x0 or needs_padding_x1:
        if include_cond:
            return idxs, x0_batch, x1_batch, conds_batch, zero_pad_idx_x0, zero_pad_idx_x1
        else:
            return idxs, x0_batch, x1_batch, zero_pad_idx_x0, zero_pad_idx_x1
    else:
        if include_cond:
            return idxs, x0_batch, x1_batch, conds_batch
        else:
            return idxs, x0_batch, x1_batch


class LettersBatchDatamodule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size=256,
        ivp_batch_size=None,
        noise_scale=0.05,
        source_noise_scale=0.5,
        num_train_letters=24,
        num_rotations=10,
        conditional=False,
        save_embeddings=False,
        seed=0,
        shuffle=None,
    ) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.ivp_batch_size = ivp_batch_size
        self.noise_scale = noise_scale
        self.source_noise_scale = source_noise_scale
        self.num_train_letters = num_train_letters
        self.num_rotations = num_rotations
        self.conditional = conditional
        self.save_embeddings = save_embeddings
        self.seed = seed
        self.shuffle = shuffle

        self.save_hyperparameters(logger=True)

        self.train_dataset = letters_replica_batch_dataset(
            mode="train",
            noise_scale=self.noise_scale,
            source_noise_scale=self.source_noise_scale,
            num_rotations=self.num_rotations,
            ivp_batch_size=self.ivp_batch_size,
            num_train_letters=self.num_train_letters,
            conditional=self.conditional,
            save_embeddings=self.save_embeddings,
            seed=self.seed,
        )

        self.val_dataset = letters_replica_batch_dataset(
            mode="val",
            noise_scale=self.noise_scale,
            source_noise_scale=self.source_noise_scale,
            num_rotations=self.num_rotations,
            ivp_batch_size=self.ivp_batch_size,
            conditional=self.conditional,
            save_embeddings=self.save_embeddings,
            seed=self.seed,
        )

        self.test_dataset = letters_replica_batch_dataset(
            mode="test",
            noise_scale=self.noise_scale,
            source_noise_scale=self.source_noise_scale,
            num_rotations=self.num_rotations,
            ivp_batch_size=self.ivp_batch_size,
            conditional=self.conditional,
            save_embeddings=self.save_embeddings,
            seed=self.seed,
        )

        if save_embeddings: 
            if self.num_rotations == 200:
                alphabet_train = list("ABCDEFGHIJKLMNOPQRSTUVWZ")
                alphabet_val = list("X")
                alphabet_test = list("Y")
                
                print("letters:", alphabet_train)
                print(len(self.train_dataset.samples))
                
                self.data_for_embed_save = defaultdict(lambda: {'source': [], 'embed': []})
                
                for i, (source, target) in enumerate(self.train_dataset.samples):
                    if i % self.num_rotations == 0:
                        s_idc = i // self.num_rotations         
                    letter = alphabet_train[s_idc]
                    self.data_for_embed_save[letter]['source'].append(source)
                    
                for i, (source, target) in enumerate(self.val_dataset.samples):
                    if i % self.num_rotations == 0:
                        s_idc = i // self.num_rotations  
                    letter = alphabet_val[s_idc]
                    self.data_for_embed_save[letter]['source'].append(source)
                    
                for i, (source, target) in enumerate(self.test_dataset.samples):
                    if i % self.num_rotations == 0:
                        s_idc = i // self.num_rotations
                    letter = alphabet_test[s_idc]
                    self.data_for_embed_save[letter]['source'].append(source)

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,  # set to False for testing_pipeline,
            num_workers=4,
            pin_memory=True,
            collate_fn=lambda batch: custom_collate_fn(batch, include_cond=self.conditional),
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=10,  # TODO: fix, this is temp
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            collate_fn=lambda batch: custom_collate_fn(batch, include_cond=self.conditional),
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=10,  # set to 1 for predict_step in testing_pipeline, # TODO: fix, this is temp
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            collate_fn=lambda batch: custom_collate_fn(batch, include_cond=self.conditional),
        )