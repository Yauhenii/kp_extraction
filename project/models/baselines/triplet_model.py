import torch
from typing import Any, Dict

from torch import nn

from project.data.datasets.ARG_KP_2021.task1_evaluator import ArgKP21Task1Evaluator
from project.models.baselines.losses.triplet import BatchAllTripletLoss
from project.models.common.base_nlp_model import BaseNLPModel

from project.models.common.nlp_configs import HFBackboneConfig


class TripletModel(BaseNLPModel):
    
    def __init__(self, backbone: HFBackboneConfig, cfg: dict, **kwargs):
        super().__init__(backbone=backbone, downstream_model_type='transformers.AutoModel', **kwargs)
        self.margin = 5
        if "margin" in cfg:
            self.margin = cfg["margin"]
        
        self.threshold = 0.5
        if "threshold" in cfg:
            self.threshold = cfg["threshold"]
        
        self.distance_fnc = 'cosine'
        if "distance_fnc" in kwargs:
            self.distance_fnc = kwargs["distance_fnc"]
        
        model_hidden_size = self.model.config.hidden_size
        self.dropout = nn.Dropout(self.model.config.hidden_dropout_prob)
        self.classifier = nn.Linear(model_hidden_size, model_hidden_size // 3)
        self.batch_norm = nn.BatchNorm1d(model_hidden_size)
        self.fc = nn.Sequential(self.dropout, self.batch_norm, self.classifier)
        
        self.criterion = BatchAllTripletLoss(self.distance_fnc, margin=self.margin)
        
        self.arg_kp_validator = ArgKP21Task1Evaluator()
    
    def _forward(self, batch: Dict, predict_with_gpu: bool = False, is_training=False):
        if predict_with_gpu:
            for k in batch:
                batch[k] = batch[k].to('cuda:0')
        
        filtered_batch = dict(
            (k, batch[k]) for k in batch if k not in ['labels', 'idx'] and 'sent1' not in k and 'sent2' not in k)
        
        outputs = self.model(**filtered_batch)
        outputs = BaseNLPModel.mean_pooling(outputs, filtered_batch["attention_mask"])
        # outputs = self.fc(outputs)
        
        # outputs = F.normalize(outputs, p=2, dim=1)
        
        loss = None
        if 'labels' in batch:
            loss = self.criterion(embeddings=outputs, labels=batch["labels"])
        
        if is_training:
            return None, loss, None
        
        sent1_batch = dict(
            (k.replace('_sent1', ''), batch[k]) for k in batch if k not in ['labels', 'idx'] and 'sent1' in k)
        sent2_batch = dict(
            (k.replace('_sent2', ''), batch[k]) for k in batch if k not in ['labels', 'idx'] and 'sent2' in k)
        
        outputs_sent1 = self.model(**sent1_batch)
        outputs_sent1 = BaseNLPModel.mean_pooling(outputs_sent1, sent1_batch["attention_mask"])
        # outputs_sent1 = self.fc(outputs_sent1)
        
        outputs_sent2 = self.model(**sent2_batch)
        outputs_sent2 = BaseNLPModel.mean_pooling(outputs_sent2, sent2_batch["attention_mask"])
        # outputs_sent2 = self.fc(outputs_sent2)
        
        # outputs_sent1 = F.normalize(outputs_sent1, p=2, dim=1)
        # outputs_sent2 = F.normalize(outputs_sent2, p=2, dim=1)
        
        similarity = (self.margin - self.criterion.distance_metric(outputs_sent1, outputs_sent2)) / self.margin
        return torch.diag(similarity), loss, None
    
    def common_step(self, prefix: str, batch: Any) -> Dict:
        preds, loss, _ = self._forward(batch)
        preds = torch.where(preds > self.threshold, 1, 0)
        if batch["labels"] is not None:
            metric_dict = self.compute_metrics(preds, batch["labels"], mode=prefix)
            self.log_dict(metric_dict, prog_bar=True, on_step=False, on_epoch=True)
            self.log(f"{prefix}_loss", loss, prog_bar=True, sync_dist=True)
        return {'loss': loss, 'preds': preds, 'target': batch["labels"], "idxs": batch["idx"]}
    
    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        _, loss, _ = self._forward(batch, is_training=True)
        self.log("train_loss", loss)
        return loss
