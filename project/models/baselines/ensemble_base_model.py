from typing import Any, Dict, List

import torch
from pytorch_lightning import LightningModule

from hydra.utils import get_class
from pytorch_lightning.utilities.types import EPOCH_OUTPUT

from project.data.datasets.ARG_KP_2021.task1_evaluator import ArgKP21Task1Evaluator


class EnsembleModel(LightningModule):
    
    def __init__(self, model_cls_type, checkpoint_paths: List[str]) -> None:
        super().__init__()
        model_cls = get_class(model_cls_type)
        self.models = torch.nn.ModuleList([model_cls.load_from_checkpoint(p) for p in checkpoint_paths])
        self.models[0].configure_metrics(None)
        self.arg_kp_validator = ArgKP21Task1Evaluator()
    
    def _forward(self, batch: Dict, predict_with_gpu: bool = False):
        # Compute the averaged predictions over the models.
        if predict_with_gpu:
            for k in batch:
                batch[k] = batch[k].to('cuda:0')
        
        results = [m._forward(batch) for m in self.models]
        all_logits = []
        all_losses = []
        all_outputs = []
        for result in results:
            logits, loss, output = result
            all_logits.append(logits)
            all_losses.append(loss)
            all_outputs.append(output)
        
        logits_averaged = torch.stack(all_logits).mean(0)
        loss_averaged = torch.stack(all_losses).mean(0)
        outputs_averaged = torch.stack(all_outputs).mean(0)
        
        return logits_averaged, loss_averaged, outputs_averaged
    
    def _evaluate(self, stage: str = 'validation') -> None:
        if 'TUNING' in str(self.trainer.state):
            return
        
        if self.trainer.datamodule.cfg.dataset_name == 'ARG_KP_2021':
            map_strict, map_relaxed = self.arg_kp_validator.evaluate(self, self.trainer.datamodule.ds[stage], stage,
                                                                     batch_size=self.trainer.datamodule.batch_size)
            self.log(f"map_strict", map_strict)
            self.log(f"map_relaxed", map_relaxed)
    
    def test_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Dict:
        logits, loss, output = self._forward(batch)
        
        preds = torch.argmax(output, dim=1)
        metric_dict = self.models[0].compute_metrics(preds, batch["labels"])
        self.log_dict(metric_dict, prog_bar=True, on_step=False, on_epoch=True)
        self.log(f"test_loss", loss, prog_bar=True, sync_dist=True)
        return {'loss': loss, 'preds': preds, 'target': batch["labels"], 'logits': logits}
    
    def test_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        super().test_epoch_end(outputs)
        self._evaluate('test')
