from dataclasses import dataclass
from typing import Optional

from datasets import Dataset
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Subset

from project.data.common.base_data_module import DataModule
from project.models.common.configs import DataConfig


class KFoldDataModule(DataModule):
    ds_fold: dict
    
    def __init__(self, cfg: DataConfig = DataConfig()):
        super().__init__(cfg=cfg)
        self.ds_fold = {}
    
    def setup_folds(self, num_folds: int) -> None:
        self.num_folds = num_folds
        self.splits = [split for split in KFold(num_folds, shuffle=True).split(range(len(self.ds['train'])))]
    
    def setup_fold_index(self, fold_index: int) -> None:
        train_indices, val_indices = self.splits[fold_index]
        self.ds_fold['train'] = Subset(self.ds['train'], train_indices.tolist())
        self.ds_fold['validation'] = Subset(self.ds['train'], val_indices.tolist())
    
    def train_dataloader(self) -> DataLoader:
        if "train" in self.ds_fold:
            return DataLoader(
                self.ds_fold["train"],
                shuffle=True,
                batch_size=self.batch_size,
                num_workers=self.cfg.num_workers,
                collate_fn=self.collate_fn,
            )
        return super().train_dataloader()
    
    def val_dataloader(self) -> DataLoader:
        if "validation" in self.ds_fold:
            return DataLoader(
                self.ds_fold["validation"],
                shuffle=False,
                batch_size=self.batch_size,
                num_workers=self.cfg.num_workers,
                collate_fn=self.collate_fn,
            )
        return super().val_dataloader()
