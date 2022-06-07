from abc import ABC, abstractmethod


class BaseDatasetProcessor(ABC):
    """A core class for processing a dataset"""
    
    @abstractmethod
    def process_dataset(self, input_dir: str, subset: str, processed_dir, with_undecided_pairs: bool):
        raise NotImplementedError
