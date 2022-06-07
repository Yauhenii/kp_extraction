from typing import Dict

from torch import nn
from torch.nn import CrossEntropyLoss

from project.data.datasets.ARG_KP_2021.task1_evaluator import ArgKP21Task1Evaluator

from project.models.common.base_nlp_model import BaseNLPModel
from project.models.common.nlp_configs import HFBackboneConfig


class ClassificationModel(BaseNLPModel):
    
    def __init__(self, backbone: HFBackboneConfig, **kwargs):
        if 'downstream_model_type' not in kwargs:
            kwargs['downstream_model_type'] = 'transformers.AutoModel'
        super().__init__(backbone=backbone, **kwargs)
        self.num_labels = 2
        self.dropout = nn.Dropout(self.model.config.hidden_dropout_prob)
        self.classifier = nn.Linear(self.model.config.hidden_size, self.num_labels)
        self.sigmoid = nn.Sigmoid()
        self.sentence_embeddings = True
        if 'sentence_embeddings' in backbone:
            self.sentence_embeddings = backbone.sentence_embeddings
        self.arg_kp_validator = ArgKP21Task1Evaluator()
    
    def _forward(self, batch: Dict, predict_with_gpu: bool = False):
        if predict_with_gpu:
            for k in batch:
                batch[k] = batch[k].to('cuda:0')
        
        filtered_batch = dict(
            (k, batch[k]) for k in batch if k not in ['labels', 'idx'] and 'sent1' not in k and 'sent2' not in k)
        outputs = self.model(**filtered_batch)
        pooled_output = outputs['pooler_output']
        if self.sentence_embeddings:
            pooled_output = BaseNLPModel.mean_pooling(outputs, filtered_batch["attention_mask"])
        
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        output = self.sigmoid(logits)
        
        loss = None
        if 'labels' in batch:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), batch['labels'].view(-1))
        
        return logits, loss, output
