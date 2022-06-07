from abc import ABC, abstractmethod


class BaseDatasetEvaluator(ABC):
    """A core class for validating a dataset"""
    
    @abstractmethod
    def evaluate(self, **kwargs):
        raise NotImplementedError
