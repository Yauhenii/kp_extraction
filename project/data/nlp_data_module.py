from typing import Any, List, Optional, Union
from datasets import ClassLabel, Dataset, load_dataset, DatasetDict
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from transformers import PreTrainedTokenizerBase

from project.data.common.tokenizer_data_module import TokenizerDataModule
from project.models.common.nlp_configs import HFTransformerDataConfig


class NLPDataModule(TokenizerDataModule):
    cfg: HFTransformerDataConfig
    tokenizer: PreTrainedTokenizerBase
    
    def __init__(
            self, tokenizer: PreTrainedTokenizerBase, cfg: HFTransformerDataConfig = HFTransformerDataConfig()
    ) -> None:
        super().__init__(tokenizer=tokenizer, cfg=cfg)
        self.labels = None
    
    def setup(self, stage: Optional[str] = None):
        dataset = self.load_dataset()
        dataset = self.split_dataset(dataset)
        dataset = self.process_data(dataset, stage=stage)
        self.ds = dataset
    
    def split_dataset(self, dataset: Union[Dataset, DatasetDict]) -> Union[Dataset, DatasetDict]:
        if self.cfg.train_val_split is not None:
            split = dataset["train"].train_test_split(self.cfg.train_val_split)
            dataset["train"] = split["train"]
            dataset["validation"] = split["test"]
        dataset = self._select_samples(dataset)
        return dataset
    
    def _select_samples(self, dataset: Union[Dataset, DatasetDict]) -> Union[Dataset, DatasetDict]:
        samples = (
            ("train", self.cfg.limit_train_samples),
            ("validation", self.cfg.limit_val_samples),
            ("test", self.cfg.limit_test_samples),
        )
        for column_name, n_samples in samples:
            if n_samples is not None and column_name in dataset:
                indices = range(min(len(dataset[column_name]), n_samples))
                dataset[column_name] = dataset[column_name].select(indices)
        return dataset
    
    def process_data(self, dataset: Dataset, stage: Optional[str] = 'train') -> Dataset:
        if stage == 'fit':
            stage = 'train'
        input_feature_fields = ["sentence1", "sentence2"]
        dataset = self.preprocess(
            dataset,
            stage,
            tokenizer=self.tokenizer,
            input_feature_fields=input_feature_fields,
            padding=self.cfg.padding,
            truncation=self.cfg.truncation,
            max_length=self.cfg.max_length
        )
        # encodings for combined pair or each separately
        encodings_cols = [
            "idx",
            "input_ids",
            "attention_mask",
            "token_type_ids",
            "input_ids_sent1",
            "attention_mask_sent1",
            "token_type_ids_sent1",
            "input_ids_sent2",
            "attention_mask_sent2",
            "token_type_ids_sent2",
            "labels"]
        cols_to_keep = [
            x for x in encodings_cols if
            x in dataset[stage].features
        ]
        if dataset[stage].features.get('labels') is not None and not isinstance(dataset[stage].features["labels"], ClassLabel):
            dataset = dataset.class_encode_column("labels")
        
        dataset.set_format("torch", columns=cols_to_keep)
        if dataset[stage].features.get('labels') is not None:
            self.labels = dataset[stage].features["labels"]
        return dataset
    
    def convert_to_features(self,
                            example_batch: Any, _, tokenizer: PreTrainedTokenizerBase, input_feature_fields: List[str],
                            **tokenizer_kwargs
                            ):
        # Either encode single sentence or sentence pairs
        sentence1_encoding = None
        sentence2_encoding = None
        if len(input_feature_fields) > 1:
            texts_or_text_pairs = list(
                zip(example_batch[input_feature_fields[0]], example_batch[input_feature_fields[1]])
            )
            sentence1_encoding = tokenizer(example_batch[input_feature_fields[0]], **tokenizer_kwargs)
            sentence2_encoding = tokenizer(example_batch[input_feature_fields[1]], **tokenizer_kwargs)
        else:
            texts_or_text_pairs = example_batch[input_feature_fields[0]]
        
        features = tokenizer(texts_or_text_pairs, **tokenizer_kwargs)
        if sentence1_encoding is not None:
            for k, v in sentence1_encoding.data.items():
                features.data[f"{k}_sent1"] = v
        if sentence2_encoding is not None:
            for k, v in sentence2_encoding.data.items():
                features.data[f"{k}_sent2"] = v
        # Tokenize the text/text pairs
        return features
    
    def preprocess(self, ds: Dataset, stage: str, **fn_kwargs) -> Dataset:
        ds = ds.map(
            self.convert_to_features,
            batched=True,
            with_indices=True,
            fn_kwargs=fn_kwargs,
        )
        if 'label' in ds.column_names[stage]:
            ds.rename_column_("label", "labels")
        return ds
    
    def load_dataset(self) -> Dataset:
        # Allow custom data files when loading the dataset
        data_files = {}
        if self.cfg.train_file is not None:
            data_files["train"] = self.cfg.train_file
        if self.cfg.validation_file is not None:
            data_files["validation"] = self.cfg.validation_file
        if self.cfg.test_file is not None:
            data_files["test"] = self.cfg.test_file
        
        data_files = data_files if data_files else None
        # Load straight from data files
        if data_files:
            extension = self.cfg.train_file.split(".")[-1]
            return load_dataset(extension, data_files=data_files)
        
        if self.cfg.dataset_name is None:
            raise MisconfigurationException(
                "You have not specified a dataset name or a custom train and validation files"
            )
        
        # Download and load the Huggingface dataset.
        return load_dataset(
            path=self.cfg.dataset_name,
            name=self.cfg.dataset_config_name,
            cache_dir=self.cfg.cache_dir,
            data_files=data_files
        )
