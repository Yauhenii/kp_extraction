# source https://github.com/UKPLab/sentence-transformers/blob/master/sentence_transformers/losses/ContrastiveLoss.py

from typing import Iterable, Dict

import torch
import torch.nn.functional as F
from torch import nn, Tensor


def euclidian(x, y):
    return F.pairwise_distance(x, y, p=2)


def manhattan(x, y):
    return F.pairwise_distance(x, y, p=1)


def cos_sim(a: Tensor, b: Tensor):
    return 1 - F.cosine_similarity(a, b)


class ContrastiveLoss(nn.Module):
    
    def __init__(self, distance,
                 margin: float = 0.5):
        super().__init__()
        if distance == "euclidean":
            self.distance_metric = euclidian
        elif distance == "manhattan":
            self.distance_metric = manhattan
        elif distance == "cosine":
            self.distance_metric = cos_sim
        self.margin = margin
    
    def forward(self, sentence_outputs: list, labels: Tensor):
        assert len(sentence_outputs) == 2
        rep_anchor, rep_other = sentence_outputs
        distances = self.distance_metric(rep_anchor, rep_other)
        losses = 0.5 * (
                labels.float() * distances.pow(2) + (1 - labels).float() * F.relu(self.margin - distances).pow(2))
        return losses.mean()
