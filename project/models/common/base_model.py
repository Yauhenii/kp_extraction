# source with customization https://github.com/PyTorchLightning/lightning-transformers/blob/0.1.0/lightning_transformers/core/model.py


import io

from neptune.new.types import File
import torchmetrics as tm
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from pytorch_lightning.utilities.types import EPOCH_OUTPUT
from typing import Any, Dict, Optional, Tuple, Union

import pytorch_lightning as pl
import torch
import transformers
from pytorch_lightning.utilities import rank_zero_info, rank_zero_warn

from project.models.common.configs import OptimizerConfig, SchedulerConfig
from project.models.common.hydra_instantiator import Instantiator


class BaseModel(pl.LightningModule):
    """Base class for transformers.

    Provides a few helper functions primarily for optimization.
    """
    
    def __init__(
            self,
            model: torch.nn.Module,
            optimizer: Optional[OptimizerConfig] = None,
            scheduler: Optional[SchedulerConfig] = None,
            instantiator: Optional[Instantiator] = None,
    ):
        super().__init__()
        self.model = model
        self.instantiator = instantiator
        self.optimizer_cfg = optimizer
        self.scheduler_cfg = scheduler
        self.metrics = {}
    
    def _configure_optimizers(self) -> Dict:
        """Prepare optimizer and scheduler."""
        return {
            "optimizer": self.optimizer,
            "lr_scheduler": {"scheduler": self.scheduler, "interval": "step", "frequency": 1},
        }
    
    @property
    def num_training_steps(self) -> int:
        """Total training steps inferred from datamodule and devices."""
        if isinstance(self.trainer.limit_train_batches, int) and self.trainer.limit_train_batches != 0:
            dataset_size = self.trainer.limit_train_batches
        elif isinstance(self.trainer.limit_train_batches, float):
            # limit_train_batches is a percentage of batches
            dataset_size = len(self.trainer.datamodule.train_dataloader())
            dataset_size = int(dataset_size * self.trainer.limit_train_batches)
        else:
            dataset_size = len(self.trainer.datamodule.train_dataloader())
        
        num_devices = max(1, self.trainer.num_gpus, self.trainer.num_processes)
        if self.trainer.tpu_cores:
            num_devices = max(num_devices, self.trainer.tpu_cores)
        
        effective_batch_size = self.trainer.accumulate_grad_batches * num_devices
        max_estimated_steps = (dataset_size // effective_batch_size) * self.trainer.max_epochs
        
        if self.trainer.max_steps and 0 < self.trainer.max_steps < max_estimated_steps:
            return self.trainer.max_steps
        return max_estimated_steps
    
    def compute_warmup(self, num_training_steps: int, num_warmup_steps: Union[int, float]) -> Tuple[int, int]:
        if num_training_steps < 0:
            # less than 0 specifies to infer number of training steps
            num_training_steps = self.num_training_steps
        if isinstance(num_warmup_steps, float):
            # Convert float values to percentage of training steps to use as warmup
            num_warmup_steps *= num_training_steps
        return num_training_steps, num_warmup_steps
    
    def setup(self, stage: Optional[str] = None) -> None:
        self.configure_metrics(stage)
    
    def configure_metrics(self, stage: str) -> Optional[Any]:
        """Override to configure metrics for train/validation/test.

        This is called on fit start to have access to the data module, and initialize any data specific metrics.
        """
        pass
    
    def configure_optimizers(self) -> Dict:
        if self.instantiator is None:
            rank_zero_warn(
                "You haven't specified an optimizer or lr scheduler. "
                "Defaulting to AdamW with an lr of 1e-5 and linear warmup for 10% of steps. "
                "To change this, either use Hydra configs or override ``configure_optimizers`` in the Task."
                "For more information: <todo>"
            )
            self._set_default_optimizer_scheduler()
            return self._configure_optimizers()
        
        self.optimizer = self.instantiator.optimizer(self.model, self.optimizer_cfg)
        # compute_warmup needs the datamodule to be available when `self.num_training_steps`
        # is called that is why this is done here and not in the __init__
        self.scheduler_cfg.num_training_steps, self.scheduler_cfg.num_warmup_steps = self.compute_warmup(
            num_training_steps=self.scheduler_cfg.num_training_steps,
            num_warmup_steps=self.scheduler_cfg.num_warmup_steps,
        )
        rank_zero_info(f"Inferring number of training steps, set to {self.scheduler_cfg.num_training_steps}")
        rank_zero_info(f"Inferring number of warmup steps from ratio, set to {self.scheduler_cfg.num_warmup_steps}")
        self.scheduler = self.instantiator.scheduler(self.scheduler_cfg, self.optimizer)
        return self._configure_optimizers()
    
    def _set_default_optimizer_scheduler(self):
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=1e-5)
        num_training_steps, num_warmup_steps = self.compute_warmup(
            num_training_steps=-1,
            num_warmup_steps=0.1,
        )
        self.scheduler = transformers.get_linear_schedule_with_warmup(
            self.optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps
        )
    
    def on_save_checkpoint(self, checkpoint: Dict[str, Any]):
        # Save tokenizer from datamodule for predictions
        if self.instantiator:
            checkpoint["instantiator"] = self.instantiator
    
    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        self.instantiator = checkpoint.get("instantiator")
    
    def _log_confusion_matrix(self, prefix: str, outputs: EPOCH_OUTPUT):
        if self.logger is None:
            return
        if len(outputs) > 0 and 'preds' in outputs[0] and 'target' in outputs[0] and 'idxs' in outputs[0]:
            preds = torch.cat([tmp['preds'] for tmp in outputs])
            targets = torch.cat([tmp['target'] for tmp in outputs])
            idxs = torch.cat([tmp['idxs'] for tmp in outputs])
            confusion_matrix = tm.functional.confusion_matrix(preds, targets, num_classes=self.num_classes)
            
            diff = targets - preds
            
            # Correct is 0
            # FP is -1
            # FN is 1
            tp_idx = idxs[torch.where(diff == 0)[0]]
            tn_idx = idxs[torch.where(diff != 0)[0]]
            fp_idx = idxs[torch.where(diff == -1)[0]]
            fn_idx = idxs[torch.where(diff == 1)[0]]
            print('Correctly classified: ', len(tp_idx))
            print('Incorrectly classified: ', len(tn_idx))
            print('False positives: ', len(fp_idx))
            print('False negatives: ', len(fn_idx))
            
            if len(tp_idx) > 0:
                self.logger.experiment[f"{prefix}_tp_idx_epoch_{self.current_epoch}"].log(tp_idx.tolist())
            if len(tn_idx) > 0:
                self.logger.experiment[f"{prefix}_tn_idx_epoch_{self.current_epoch}"].log(tn_idx.tolist())
            if len(fp_idx) > 0:
                self.logger.experiment[f"{prefix}_fp_idx_epoch_{self.current_epoch}"].log(fp_idx.tolist())
            if len(fn_idx) > 0:
                self.logger.experiment[f"{prefix}_fn_idx_epoch_{self.current_epoch}"].log(fn_idx.tolist())
            
            df_cm = pd.DataFrame(confusion_matrix.cpu().numpy(), index=range(self.num_classes),
                                 columns=range(self.num_classes))
            plt.figure()
            sn.set(font_scale=1.2)
            sn.heatmap(df_cm, annot=True, annot_kws={"size": 16}, fmt='d')
            buf = io.BytesIO()
            plt.savefig(buf, format='jpeg')
            buf.seek(0)
            im = Image.open(buf)
            self.logger.experiment[f"{prefix}_confusion_epoch_{self.current_epoch}"].log(File.as_image(im))
    
    def common_step(self, prefix: str, batch: Any) -> Dict:
        return {}
    
    def validation_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        self._log_confusion_matrix('val', outputs)
    
    def test_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        self._log_confusion_matrix('test', outputs)
    
    def validation_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Dict:
        return self.common_step("val", batch)
    
    def test_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Dict:
        return self.common_step("test", batch)
    
    def compute_metrics(self, preds, labels, mode="val") -> Dict[str, torch.Tensor]:
        # Not required by all models. Only required for classification
        return {f"{mode}_{k}": metric(preds, labels) for k, metric in self.metrics.items()}
