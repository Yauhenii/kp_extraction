from typing import Optional, Callable, Dict, Any

from torch.utils.data import DataLoader
import pytorch_lightning as pl

from project.models.common.configs import DataConfig


class DataModule(pl.LightningDataModule):
    def __init__(self, cfg: DataConfig = DataConfig()) -> None:
        super().__init__()
        self.cfg = cfg
        self.ds = None
        self._batch_size = self.cfg.batch_size
    
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.ds["train"],
            batch_size=self.batch_size,
            num_workers=self.cfg.num_workers,
            collate_fn=self.collate_fn,
        )
    
    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.ds["validation"],
            batch_size=self.batch_size,
            num_workers=self.cfg.num_workers,
            collate_fn=self.collate_fn,
        )
    
    def test_dataloader(self) -> Optional[DataLoader]:
        if "test" in self.ds:
            return DataLoader(
                self.ds["test"],
                batch_size=self.batch_size,
                num_workers=self.cfg.num_workers,
                collate_fn=self.collate_fn,
            )
    
    @property
    def batch_size(self):
        return self._batch_size
    
    @batch_size.setter
    def batch_size(self, value):
        self._batch_size = value
    
    @property
    def collate_fn(self) -> Optional[Callable]:
        return None
    
    @property
    def model_data_kwargs(self) -> Dict:
        """Override to provide the model with additional kwargs.

        This is useful to provide the number of classes/pixels to the model or any other data specific args
        Returns: Dict of args
        """
        return {}
