# customized version of https://github.com/PyTorchLightning/lightning-transformers/blob/0.1.0/lightning_transformers/core/nlp/model.py

from typing import Optional, Dict, Any

import torch
from hydra.utils import get_class
from pytorch_lightning.utilities.types import EPOCH_OUTPUT
from torchmetrics import Precision, Recall, Accuracy, F1
from transformers import PreTrainedTokenizerBase

from project.models.common.base_model import BaseModel
from project.models.common.configs import OptimizerConfig, SchedulerConfig
from project.models.common.hydra_instantiator import Instantiator
from project.models.common.nlp_configs import HFBackboneConfig


class BaseNLPModel(BaseModel):
    
    def __init__(
            self, backbone: HFBackboneConfig,
            optimizer: OptimizerConfig = OptimizerConfig(),
            scheduler: SchedulerConfig = SchedulerConfig(),
            instantiator: Optional[Instantiator] = None,
            tokenizer: Optional[PreTrainedTokenizerBase] = None,
            downstream_model_type: str = 'transformers.AutoModelForSequenceClassification',
            **model_data_kwargs
    ) -> None:
        self.save_hyperparameters()
        self.backbone = backbone
        model_cls = get_class(downstream_model_type)
        model = model_cls.from_pretrained(backbone.pretrained_model_name_or_path, **model_data_kwargs)
        super().__init__(model=model, optimizer=optimizer, scheduler=scheduler, instantiator=instantiator)
        self._tokenizer = tokenizer
    
    @property
    def tokenizer(self) -> Optional["PreTrainedTokenizerBase"]:
        if (
                self._tokenizer is None
                and hasattr(self, "trainer")  # noqa: W503
                and hasattr(self.trainer, "datamodule")  # noqa: W503
                and hasattr(self.trainer.datamodule, "tokenizer")  # noqa: W503
        ):
            self._tokenizer = self.trainer.datamodule.tokenizer
        return self._tokenizer
    
    @tokenizer.setter
    def tokenizer(self, tokenizer: "PreTrainedTokenizerBase") -> None:
        self._tokenizer = tokenizer
    
    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        if "tokenizer" in checkpoint:
            self.tokenizer = checkpoint["tokenizer"]
    
    @property
    def num_classes(self) -> int:
        if self.trainer is None or self.trainer.datamodule is None or self.trainer.datamodule.labels is None:
            return -1
        return self.trainer.datamodule.labels.num_classes
    
    def configure_metrics(self, _) -> None:
        self.prec = Precision(num_classes=self.num_classes)
        self.recall = Recall(num_classes=self.num_classes)
        self.acc = Accuracy()
        self.f1 = F1()
        self.metrics = {"precision": self.prec, "recall": self.recall, "accuracy": self.acc, "f1": self.f1}
    
    def common_step(self, prefix: str, batch: Any) -> Dict:
        logits, loss, output = self._forward(batch)
        preds = torch.argmax(output, dim=1)
        if batch["labels"] is not None:
            metric_dict = self.compute_metrics(preds, batch["labels"], mode=prefix)
            self.log_dict(metric_dict, prog_bar=True, on_step=False, on_epoch=True)
            self.log(f"{prefix}_loss", loss, prog_bar=True, sync_dist=True)
        return {'loss': loss, 'preds': preds, 'target': batch["labels"], 'logits': logits, "idxs": batch["idx"]}
    
    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        _, loss, _ = self._forward(batch)
        self.log("train_loss", loss)
        return loss
    
    def _evaluate(self, stage: str = 'validation') -> None:
        if 'TUNING' in str(self.trainer.state):
            return
        
        if self.trainer.datamodule.cfg.dataset_name == 'ARG_KP_2021':
            map_strict, map_relaxed = self.arg_kp_validator.evaluate(self, self.trainer.datamodule.ds[stage], stage,
                                                                     batch_size=self.trainer.datamodule.batch_size)
            self.log(f"map_strict", map_strict)
            self.log(f"map_relaxed", map_relaxed)
    
    def validation_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        super().validation_epoch_end(outputs)
        self._evaluate('validation')
    
    def test_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        super().test_epoch_end(outputs)
        self._evaluate('test')
    
    @staticmethod
    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
